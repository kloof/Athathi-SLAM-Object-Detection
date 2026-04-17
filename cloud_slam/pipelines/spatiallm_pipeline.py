"""SLAM + SpatialLM pipeline.

Runs ICP+IMU SLAM while periodically snapshotting the merged cloud, leveling
+ Manhattan-aligning it, and submitting it to a :class:`SpatialLMClient` for
structured layout inference. Each completed pass is fed into a
:class:`FusedScene` so repeated observations converge on a stable scene.

The actual SpatialLM inference runs in a separate ``~/spatiallm_env`` venv;
this module only talks to the worker via the client. No torch / transformers
import occurs here.
"""

import copy
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as SciRot

from cloud_slam.detectors.scene import Scene
from cloud_slam.detectors.scene_fusion import FusedScene
from cloud_slam.detectors.spatiallm_client import SpatialLMClient
from cloud_slam.frustum import estimate_gravity
from cloud_slam.manhattan import estimate_manhattan_frame
from cloud_slam.pipelines import icp_imu_pipeline
from cloud_slam.room_structure import detect_room

logger = logging.getLogger(__name__)

Z_UP = np.array([0.0, 0.0, 1.0])


def _level_and_align(
    pcd: o3d.geometry.PointCloud,
    gravity_up: np.ndarray,
) -> o3d.geometry.PointCloud:
    """Gravity-level the cloud and snap walls to X/Y axes (non-destructive)."""
    out = copy.deepcopy(pcd)
    if not np.allclose(gravity_up, Z_UP, atol=0.01):
        R_level = SciRot.align_vectors([Z_UP], [gravity_up])[0].as_matrix()
        out.rotate(R_level, center=(0, 0, 0))

    try:
        room_leveled = detect_room(out, Z_UP)
    except Exception as exc:
        logger.debug("detect_room failed during snapshot: %s", exc)
        return out
    if not room_leveled.walls:
        return out

    manhattan_leveled = estimate_manhattan_frame(room_leveled.walls, Z_UP)
    if manhattan_leveled.confidence <= 0.3:
        return out

    mx = manhattan_leveled.R.T[:, 0]
    wall_yaw = np.arctan2(mx[1], mx[0])
    snapped_wall = round(wall_yaw / (np.pi / 2)) * (np.pi / 2)
    residual = wall_yaw - snapped_wall
    if abs(residual) > 0.01:
        R_align = SciRot.from_euler('z', -residual).as_matrix()
        out.rotate(R_align, center=(0, 0, 0))
    return out


def _snapshot_for_inference(
    merged: o3d.geometry.PointCloud,
    gravity_up: np.ndarray,
    snapshot_voxel: float,
) -> Optional[o3d.geometry.PointCloud]:
    """Voxel-downsample + level + align; returns None if too sparse."""
    if len(merged.points) < 500:
        return None
    snap = merged.voxel_down_sample(snapshot_voxel)
    snap = _level_and_align(snap, gravity_up)
    if len(snap.points) < 500:
        return None
    return snap


def run(
    clouds,
    imus,
    images,
    calib,
    voxel_size: float = 0.005,
    refinement_interval_frames: int = 100,
    categories: Optional[List[str]] = None,
    spatiallm_model: str = "manycore-research/SpatialLM1.1-Qwen-0.5B",
    max_points: int = 200000,
    snapshot_voxel: float = 0.02,
    worker_env_python: str = "~/spatiallm_env/bin/python",
    worker_script: Optional[str] = None,
    inference_timeout_s: float = 180.0,
) -> Tuple[o3d.geometry.PointCloud, List[np.ndarray], FusedScene, Dict[str, Any]]:
    """Run SLAM + iterative SpatialLM refinement.

    Parameters
    ----------
    clouds, imus, images, calib : same as :func:`cloud_slam.pipelines.icp_imu_pipeline.run`.
    voxel_size : final merged-cloud voxel size (m).
    refinement_interval_frames : submit a snapshot every N frames during SLAM.
    categories : optional list of SpatialLM object categories to detect.
    spatiallm_model : HF model id / local path passed to the worker.
    max_points : per-snapshot cap; worker downsamples further if exceeded.
    snapshot_voxel : voxel size (m) for the mid-scan snapshots (coarser than
        the final merged voxel_size — snapshots are for inference only).

    Returns
    -------
    (merged_pcd, poses, fused_scene, stats)
    """
    t0 = time.time()
    stats: Dict[str, Any] = {
        "algorithm": "spatiallm+icp-imu",
        "refinement_interval_frames": refinement_interval_frames,
        "n_inference_submitted": 0,
        "n_inference_completed": 0,
        "n_inference_empty": 0,
        "n_inference_error": 0,
        "inference_seconds_total": 0.0,
        "inference_seconds_per_pass": [],
    }

    gravity_up = estimate_gravity(imus)
    stats["gravity_up"] = gravity_up.tolist()

    fused = FusedScene()

    if worker_script is None:
        import os
        worker_script = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "detectors", "spatiallm_worker.py")

    client = SpatialLMClient(
        worker_env_python=worker_env_python,
        worker_script=worker_script,
        model_path=spatiallm_model,
        categories=categories,
        detect_type="all",
        max_points=max_points,
    )

    def _absorb(result: Dict[str, Any]) -> None:
        status = result.get("status", "ok")
        ls = result.get("language_string", "")
        meta = result.get("meta", {})
        seconds = float(meta.get("inference_seconds", 0.0))
        stats["n_inference_completed"] += 1
        stats["inference_seconds_total"] += seconds
        stats["inference_seconds_per_pass"].append(seconds)
        if status == "empty":
            stats["n_inference_empty"] += 1
            logger.warning("SpatialLM pass %s returned empty output "
                           "(issue #81) — ignoring", result.get("job_id"))
            return
        if status == "error":
            stats["n_inference_error"] += 1
            logger.warning("SpatialLM pass %s failed: %s",
                           result.get("job_id"), meta.get("error"))
            return
        scene = Scene.from_language_string(ls)
        if not (scene.walls or scene.bboxes or scene.doors or scene.windows):
            stats["n_inference_empty"] += 1
            logger.warning("SpatialLM pass %s parsed to empty scene", result.get("job_id"))
            return
        delta = fused.update(scene)
        logger.info("fused SpatialLM pass %s: %s", result.get("job_id"), delta)

    logger.info("starting SpatialLM worker (model=%s)", spatiallm_model)
    client.start()

    current_job_id: Optional[str] = None
    last_submitted_frame = -refinement_interval_frames
    # Maintain our own running snapshot cloud in world frame. The main
    # pipeline owns its own `merged` we can't see, so we build a coarse
    # parallel accumulation here (voxel-downsampled after each frame).
    running_snap = o3d.geometry.PointCloud()
    merged = o3d.geometry.PointCloud()
    poses: List[np.ndarray] = []
    slam_stats: Dict[str, Any] = {}

    try:
        def per_frame(frame_idx: int, stamp, xyz, pose, image) -> None:
            nonlocal current_job_id, last_submitted_frame, running_snap

            # Append the new scan to the running snapshot cloud.
            if xyz is not None and len(xyz) > 0:
                frame_pcd = o3d.geometry.PointCloud()
                frame_pcd.points = o3d.utility.Vector3dVector(
                    np.asarray(xyz, dtype=np.float64))
                frame_pcd.transform(pose)
                running_snap += frame_pcd
                # Rebuild-downsample periodically to keep size in check.
                if frame_idx % 25 == 0 and len(running_snap.points) > 50000:
                    running_snap = running_snap.voxel_down_sample(snapshot_voxel)

            # Drain completed jobs early (non-blocking).
            if current_job_id is not None:
                result = client.poll(job_id=current_job_id, timeout=0.0)
                if result is not None:
                    _absorb(result)
                    current_job_id = None

            # Snapshot + submit if interval elapsed and no job in flight.
            if (current_job_id is None
                    and frame_idx - last_submitted_frame >= refinement_interval_frames
                    and frame_idx > 0):
                snap_pcd = running_snap.voxel_down_sample(snapshot_voxel)
                if len(snap_pcd.points) > 500:
                    snap_pcd = _level_and_align(snap_pcd, gravity_up)
                    try:
                        current_job_id = client.submit(snap_pcd)
                        last_submitted_frame = frame_idx
                        stats["n_inference_submitted"] += 1
                        logger.info("submitted SpatialLM snapshot at frame %d "
                                    "(%d pts, job=%s)",
                                    frame_idx, len(snap_pcd.points), current_job_id)
                    except Exception as exc:
                        logger.warning("snapshot submit failed: %s", exc)

        merged, poses, slam_stats = icp_imu_pipeline.run(
            clouds, imus, voxel_size=voxel_size,
            images=images, calib=calib, per_frame_callback=per_frame)

        # Drain any in-flight job before running the final pass.
        if current_job_id is not None:
            try:
                result = client.wait_for(current_job_id,
                                         timeout=inference_timeout_s)
                _absorb(result)
            except TimeoutError as exc:
                logger.warning("in-flight SpatialLM job timed out: %s", exc)
            current_job_id = None

        # One final full-scan pass on the finished merged cloud.
        final_snap = _snapshot_for_inference(merged, gravity_up, snapshot_voxel)
        if final_snap is not None:
            try:
                job_id = client.submit(final_snap)
                stats["n_inference_submitted"] += 1
                logger.info("submitted final SpatialLM pass (%d pts, job=%s)",
                            len(final_snap.points), job_id)
                result = client.wait_for(job_id, timeout=inference_timeout_s)
                _absorb(result)
            except Exception as exc:
                logger.warning("final SpatialLM pass failed: %s", exc)
        else:
            logger.warning("final merged cloud too sparse for SpatialLM snapshot")
    finally:
        client.stop()

    stats.update({k: v for k, v in slam_stats.items() if k not in stats})
    if stats["n_inference_completed"] > 0:
        stats["avg_inference_sec"] = round(
            stats["inference_seconds_total"] / stats["n_inference_completed"], 3)
    else:
        stats["avg_inference_sec"] = 0.0
    stats["total_time_s"] = round(time.time() - t0, 2)
    stats["fused_scene_summary"] = fused.current().summary()
    return merged, poses, fused, stats
