"""Stage 8 — best-view-per-bbox.

For every bbox in ``layout_merged.txt``, pick the single camera frame
that shows the object most clearly and save a cropped JPG plus a JSON
manifest. Strictly additive to the rest of the pipeline: reads stage 0-7
artifacts, writes only into ``<output>/best_views/``.

Design doc: docs/superpowers/specs/2026-04-24-best-views-per-bbox-design.md

## Implementation notes / small deviations

- The spec's ``T_cam_world = calib['T_lidar_cam'] · inv(T_world_lidar)``
  formula is what this module uses verbatim (the calibration dict's
  ``T_lidar_cam`` is actually ``T_cam_from_lidar`` in the convention
  ``p_cam = T @ p_lidar`` — see ``colorizer.load_calibration``).
- Pose interpolation uses SLERP on rotation + LERP on translation
  between the two bracketing trajectory poses. Camera frames whose
  timestamp falls outside the trajectory range are dropped, not
  extrapolated.
- Sharpness is percentile-normalized *within* each bbox's candidate
  pool — so a bbox whose whole candidate pool is blurry can still pick
  a best-of-what-we-have winner without the sharpness term crushing it.
- The occlusion test is the cheap ray-point proximity one from the
  spec (5 cm tolerance, 10 cm closer-than-corner depth); it is
  approximate and a known false-negative mode for very thin geometry.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Weights are a module-level dict so scoring can be re-tuned standalone
# without re-running stages 0-7. Values sum to 1.0.
SCORE_WEIGHTS = {
    "area": 0.35,
    "centering": 0.15,
    "occlusion": 0.30,
    "sharpness": 0.20,
}
SCORE_FLOOR = 0.2
CANDIDATE_TOPK = 30   # prefilter pool size per bbox (cheap -> expensive)
OCCLUSION_RADIUS_M = 0.05
OCCLUSION_DEPTH_MARGIN_M = 0.10
CROP_PAD_FRAC = 0.10   # 10% of AABB width/height each side


# ---------------------------------------------------------------------------
# Pose + transform helpers
# ---------------------------------------------------------------------------

def _read_trajectory_csv(path: Path) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return (timestamps_sec, [(4x4 T_world_lidar raw), ...])."""
    from scipy.spatial.transform import Rotation as R

    timestamps = []
    poses = []
    with open(path) as f:
        header = f.readline()
        assert header.strip().startswith("timestamp"), header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 8:
                continue
            ts = float(parts[0])
            x, y, z = (float(parts[1]), float(parts[2]), float(parts[3]))
            qw, qx, qy, qz = (float(parts[4]), float(parts[5]),
                              float(parts[6]), float(parts[7]))
            T = np.eye(4)
            T[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
            T[:3, 3] = [x, y, z]
            timestamps.append(ts)
            poses.append(T)
    return np.asarray(timestamps, dtype=np.float64), poses


def _build_level_transform(level_rotation: np.ndarray,
                           level_z_shift_m: float) -> np.ndarray:
    """T_level in homogeneous form.

    Matches ``cloud_slam.level.level_points``: rotate the point by
    ``R_level``, then subtract ``z_shift`` from the Z coordinate.
    In matrix form:

        p' = R_level @ p - [0, 0, z_shift]
        T_level = [[R_level, -[0,0,z_shift]^T], [0, 0, 0, 1]]
    """
    T = np.eye(4)
    T[:3, :3] = np.asarray(level_rotation, dtype=np.float64)
    T[:3, 3] = np.array([0.0, 0.0, -float(level_z_shift_m)])
    return T


def _build_yaw_transform(yaw_deg: float) -> np.ndarray:
    """Pure yaw (rotation around world +Z) in homogeneous form."""
    yaw = np.deg2rad(yaw_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = np.array([[c, -s, 0.0],
                          [s,  c, 0.0],
                          [0.0, 0.0, 1.0]])
    return T


def _interpolate_pose(t_query: float,
                      timestamps: np.ndarray,
                      poses: list[np.ndarray]) -> Optional[np.ndarray]:
    """SLERP rotation + LERP translation between bracketing poses.

    Returns None if t_query is outside the trajectory range.
    """
    if len(timestamps) == 0:
        return None
    if t_query < timestamps[0] or t_query > timestamps[-1]:
        return None
    idx = np.searchsorted(timestamps, t_query)
    if idx == 0:
        return poses[0].copy()
    if idx >= len(timestamps):
        return poses[-1].copy()
    t0, t1 = timestamps[idx - 1], timestamps[idx]
    if t1 <= t0:
        return poses[idx - 1].copy()
    alpha = (t_query - t0) / (t1 - t0)
    T0, T1 = poses[idx - 1], poses[idx]

    from scipy.spatial.transform import Rotation as R
    from scipy.spatial.transform import Slerp
    rots = R.from_matrix(np.stack([T0[:3, :3], T1[:3, :3]]))
    slerp = Slerp([0.0, 1.0], rots)
    R_interp = slerp([alpha]).as_matrix()[0]
    t_interp = (1 - alpha) * T0[:3, 3] + alpha * T1[:3, 3]

    out = np.eye(4)
    out[:3, :3] = R_interp
    out[:3, 3] = t_interp
    return out


def _compose_camera_poses(
    trajectory_times: np.ndarray,
    trajectory_poses_raw: list[np.ndarray],
    level_rotation: np.ndarray,
    level_z_shift_m: float,
    manhattan_yaw_deg: float,
    calib_T_lidar_cam: np.ndarray,
    camera_timestamps_ns: list[int],
) -> list[tuple[int, Optional[np.ndarray], Optional[np.ndarray]]]:
    """For each camera t_ns, compute (t_ns, T_world_cam, T_cam_world).

    Missing poses (t outside trajectory range) yield (t_ns, None, None).
    """
    T_level = _build_level_transform(level_rotation, level_z_shift_m)
    T_manhattan = _build_yaw_transform(manhattan_yaw_deg)
    T_world_from_raw = T_manhattan @ T_level

    # Pre-transform trajectory poses into the bbox world frame. We
    # interpolate in the already-transformed frame because slerp on the
    # raw frame then applying T_world_from_raw gives a different
    # (but equally valid) interpolation than transforming first and
    # slerping in-world — for small inter-scan gaps the difference is
    # negligible and transforming first keeps the interp arithmetic in
    # one well-defined reference frame.
    traj_poses_world = [T_world_from_raw @ P for P in trajectory_poses_raw]

    # Camera pose in world: T_world_cam = T_world_lidar @ inv(T_lidar_cam)
    # where calib['T_lidar_cam'] is really T_cam_from_lidar in the convention
    # p_cam = T @ p_lidar (see load_calibration). Hence T_world_cam_inv
    # below follows spec Step 1.
    T_lc = np.asarray(calib_T_lidar_cam, dtype=np.float64)  # T_cam_from_lidar

    out = []
    for t_ns in camera_timestamps_ns:
        t_sec = t_ns * 1e-9
        T_world_lidar = _interpolate_pose(t_sec, trajectory_times,
                                          traj_poses_world)
        if T_world_lidar is None:
            out.append((t_ns, None, None))
            continue
        T_cam_world = T_lc @ np.linalg.inv(T_world_lidar)
        T_world_cam = np.linalg.inv(T_cam_world)
        out.append((t_ns, T_world_cam, T_cam_world))
    return out


# ---------------------------------------------------------------------------
# Bbox helpers
# ---------------------------------------------------------------------------

def _bbox_corners_world(cx: float, cy: float, cz: float,
                        yaw: float, sx: float, sy: float, sz: float
                        ) -> np.ndarray:
    """8 world-frame corners of a rotated-around-Z AABB centered on (cx,cy,cz).

    Corner layout matches embed.py / generic box wireframe conventions.
    """
    hx, hy, hz = 0.5 * sx, 0.5 * sy, 0.5 * sz
    local = np.array([
        [-hx, -hy, -hz],
        [ hx, -hy, -hz],
        [ hx,  hy, -hz],
        [-hx,  hy, -hz],
        [-hx, -hy,  hz],
        [ hx, -hy,  hz],
        [ hx,  hy,  hz],
        [-hx,  hy,  hz],
    ], dtype=np.float64)
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0.0],
                   [s,  c, 0.0],
                   [0.0, 0.0, 1.0]])
    world = (Rz @ local.T).T + np.array([cx, cy, cz])
    return world


def _aabb_intersects_image(pixels_valid: np.ndarray,
                           image_size: tuple[int, int]
                           ) -> tuple[bool, tuple[float, float, float, float]]:
    """Check if the 2D pixel AABB overlaps the image rectangle.

    pixels_valid is (K, 2) of finite pixel coordinates (only in-front
    corners). Returns (overlaps, (x0, y0, x1, y1)) where the AABB may
    extend outside the image. K must be > 0.
    """
    W, H = image_size
    x0 = float(pixels_valid[:, 0].min())
    y0 = float(pixels_valid[:, 1].min())
    x1 = float(pixels_valid[:, 0].max())
    y1 = float(pixels_valid[:, 1].max())
    overlaps = (x1 > 0) and (y1 > 0) and (x0 < W) and (y0 < H)
    return overlaps, (x0, y0, x1, y1)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _area_score(aabb: tuple[float, float, float, float],
                image_size: tuple[int, int]) -> float:
    W, H = image_size
    x0, y0, x1, y1 = aabb
    xa = max(0.0, min(W, x1)) - max(0.0, min(W, x0))
    ya = max(0.0, min(H, y1)) - max(0.0, min(H, y0))
    if xa <= 0 or ya <= 0:
        return 0.0
    area = xa * ya
    return float(min(1.0, area / (W * H)))


def _centering_score(aabb: tuple[float, float, float, float],
                     image_size: tuple[int, int]) -> float:
    W, H = image_size
    x0, y0, x1, y1 = aabb
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    ix, iy = 0.5 * W, 0.5 * H
    d = float(np.hypot(cx - ix, cy - iy))
    half_diag = 0.5 * float(np.hypot(W, H))
    if half_diag <= 0:
        return 0.0
    return float(max(0.0, 1.0 - min(1.0, d / half_diag)))


def _laplacian_var(bgr_image: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _occlusion_score(corners_world: np.ndarray,
                     T_cam_world: np.ndarray,
                     kdtree,
                     voxel_points_world: np.ndarray,
                     radius: float = OCCLUSION_RADIUS_M,
                     depth_margin: float = OCCLUSION_DEPTH_MARGIN_M,
                     ) -> float:
    """Fraction of the 8 corner rays that are unobstructed.

    A corner is "obstructed" iff the voxel cloud contains a point within
    `radius` of the ray (in world space) AND at least `depth_margin`
    closer to the camera (along the ray direction) than the corner
    itself.

    Cheap O(K * queries) loop — 8 per bbox, dominated by kdtree.query.
    """
    if kdtree is None or voxel_points_world is None or len(voxel_points_world) == 0:
        return 1.0

    # Camera origin in world = center of inv(T_cam_world)
    T_world_cam = np.linalg.inv(T_cam_world)
    cam_origin = T_world_cam[:3, 3]

    clear = 0
    for corner in corners_world:
        direction = corner - cam_origin
        corner_dist = float(np.linalg.norm(direction))
        if corner_dist <= 1e-6:
            clear += 1
            continue
        unit_dir = direction / corner_dist

        # Sample along the ray at intervals of `radius`; check each sample
        # for nearby voxels. Stop at (corner_dist - depth_margin).
        max_t = corner_dist - depth_margin
        if max_t <= 0:
            clear += 1
            continue
        n_samples = max(1, int(np.ceil(max_t / radius)))
        ts = np.linspace(radius, max_t, n_samples)
        blocked = False
        for t in ts:
            sample = cam_origin + unit_dir * t
            idx = kdtree.query_ball_point(sample, r=radius)
            if not idx:
                continue
            # Any hit close enough to the ray and strictly in front of
            # the corner is an occluder.
            blocked = True
            break
        if not blocked:
            clear += 1
    return clear / float(len(corners_world))


def _composite(scores: dict) -> float:
    return sum(SCORE_WEIGHTS[k] * scores.get(k, 0.0) for k in SCORE_WEIGHTS)


def _percentile_normalize(values: list[float]) -> list[float]:
    """Map values -> [0,1] via rank percentile (ties -> average rank).

    Returns [0.5, ...] if values is a single element, [0, ..., 1] otherwise.
    """
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [0.5]
    arr = np.asarray(values, dtype=np.float64)
    # rankdata is equivalent and avoids argsort-of-argsort pitfalls; we
    # sidestep the scipy dep and do it by hand since values lists are tiny.
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n)
    # handle ties: average ranks across equal values
    # (cheap because candidate pools are <=30)
    for v in set(arr.tolist()):
        mask = arr == v
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    return (ranks / (n - 1)).tolist()


# ---------------------------------------------------------------------------
# Crop
# ---------------------------------------------------------------------------

def _safe_filename_component(cls: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", cls).strip("_") or "item"


def _pad_and_clip_aabb(aabb: tuple[float, float, float, float],
                       image_size: tuple[int, int],
                       pad_frac: float = CROP_PAD_FRAC
                       ) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = aabb
    w = max(0.0, x1 - x0)
    h = max(0.0, y1 - y0)
    x0 -= pad_frac * w; x1 += pad_frac * w
    y0 -= pad_frac * h; y1 += pad_frac * h
    W, H = image_size
    x0c = int(np.floor(max(0, min(W, x0))))
    y0c = int(np.floor(max(0, min(H, y0))))
    x1c = int(np.ceil(max(0, min(W, x1))))
    y1c = int(np.ceil(max(0, min(H, y1))))
    return x0c, y0c, x1c, y1c


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_best_views(output_dir: Path,
                   mcap_path: Optional[Path] = None,
                   *,
                   calibration_dir: Optional[Path] = None,
                   verbose: bool = True,
                   ) -> dict:
    """Entry point for stage 8. Returns the written manifest dict.

    Args:
        output_dir: Pipeline output directory holding layout_merged.txt,
                    trajectory.csv (in slam/), voxel.ply, and
                    slam/frames_index.json.
        mcap_path: Override for the rosbag path; defaults to the value
                   recorded in frames_index.json.
        calibration_dir: Override for calibration; defaults to the
                         vendored calibration/ dir in the repo.
    """
    from cloud_slam.spatiallm_pipeline.merge import parse_layout
    from cloud_slam.colorizer import load_calibration
    from cloud_slam.mcap_reader import read_frames_by_time_ns

    output_dir = Path(output_dir).resolve()
    best_views_dir = output_dir / "best_views"
    best_views_dir.mkdir(parents=True, exist_ok=True)

    # --- load inputs -------------------------------------------------------
    layout_path = output_dir / "layout_merged.txt"
    if not layout_path.is_file():
        raise FileNotFoundError(f"Missing {layout_path}")
    layout = parse_layout(layout_path)
    bboxes = layout["bboxes"]
    if verbose:
        print(f"[best_views] {len(bboxes)} bboxes from {layout_path.name}")

    traj_path = output_dir / "slam" / "trajectory.csv"
    if not traj_path.is_file():
        raise FileNotFoundError(f"Missing {traj_path}")
    traj_times, traj_poses_raw = _read_trajectory_csv(traj_path)
    if verbose:
        print(f"[best_views] {len(traj_times)} trajectory poses")

    frames_index_path = output_dir / "slam" / "frames_index.json"
    if not frames_index_path.is_file():
        raise FileNotFoundError(f"Missing {frames_index_path} (stage 0 hook)")
    idx = json.loads(frames_index_path.read_text())
    camera_frame_times = [int(f["t_ns"]) for f in idx.get("frames", [])]
    topic = idx.get("topic", "/camera/image_raw/compressed")
    level_rotation = np.asarray(idx.get("level_rotation", np.eye(3).tolist()),
                                dtype=np.float64)
    level_z_shift = float(idx.get("level_z_shift_m", 0.0))
    manhattan_yaw = float(idx.get("manhattan_yaw_deg", 0.0))
    if mcap_path is None:
        rec = idx.get("mcap_path")
        if rec is None:
            raise ValueError("mcap_path not in frames_index.json and not "
                             "provided as CLI arg")
        mcap_path = Path(rec)
    mcap_path = Path(mcap_path)

    # calibration (vendored default matches scripts/rosbag_to_bboxes.py)
    if calibration_dir is None:
        calibration_dir = (Path(__file__).resolve().parent.parent.parent
                           / "calibration")
    calibration_dir = Path(calibration_dir)
    calib = load_calibration(calibration_dir / "intrinsics.yaml",
                             calibration_dir / "extrinsics.yaml")
    image_size = tuple(calib["image_size"])
    K = calib["K"]; D = calib["dist_coeffs"]
    T_lidar_cam = calib["T_lidar_cam"]

    # voxel cloud for occlusion
    voxel_ply = output_dir / "voxel.ply"
    kdtree = None
    voxel_points = None
    if voxel_ply.is_file():
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(str(voxel_ply))
        voxel_points = np.asarray(pcd.points, dtype=np.float64)
        if len(voxel_points) > 0:
            from scipy.spatial import cKDTree
            kdtree = cKDTree(voxel_points)
        if verbose:
            print(f"[best_views] voxel cloud: {len(voxel_points)} points "
                  f"(kdtree built)")
    elif verbose:
        print(f"[best_views] voxel.ply missing — occlusion score defaults to 1.0")

    # camera poses per frame
    camera_poses = _compose_camera_poses(
        traj_times, traj_poses_raw, level_rotation, level_z_shift,
        manhattan_yaw, T_lidar_cam, camera_frame_times)

    # --- per-bbox loop -----------------------------------------------------
    from cloud_slam.projection import project_world_to_image

    entries = []
    sharpness_cache: dict[int, float] = {}
    frame_decode_cache: dict[int, np.ndarray] = {}

    # Pre-resolve the set of winning frame timestamps so we only decode
    # what we actually need.
    winners: list[tuple[dict, int, np.ndarray, tuple, dict, float]] = []

    for bbox_id, (cls, v) in enumerate(bboxes):
        cx, cy, cz, yaw, sx, sy, sz = v
        corners = _bbox_corners_world(cx, cy, cz, yaw, sx, sy, sz)
        center = np.array([cx, cy, cz])

        # Cheap-score pass over all frames
        cheap = []  # list of (frame_idx, score, aabb, area, centering, cam_z)
        for fi, (t_ns, T_wc, T_cw) in enumerate(camera_poses):
            if T_cw is None:
                continue
            # center must be in front of cam
            center_cam = T_cw[:3, :3] @ center + T_cw[:3, 3]
            if center_cam[2] <= 0:
                continue
            pts_cam, pixels, in_front = project_world_to_image(
                corners, T_cw, K, D)
            if not in_front.any():
                continue
            pixels_valid = pixels[in_front]
            overlaps, aabb = _aabb_intersects_image(pixels_valid, image_size)
            if not overlaps:
                continue
            a = _area_score(aabb, image_size)
            c = _centering_score(aabb, image_size)
            cheap.append({
                "frame_idx": fi, "t_ns": t_ns,
                "T_cw": T_cw, "T_wc": T_wc,
                "aabb": aabb, "area": a, "centering": c,
                "cheap_composite": a * c,
                "cam_dist": float(np.linalg.norm(center - T_wc[:3, 3])),
            })

        if not cheap:
            entries.append({
                "bbox_id": bbox_id,
                "class": cls,
                "bbox_3d": [float(x) for x in v],
                "skipped": "never_visible",
            })
            continue

        cheap.sort(key=lambda d: d["cheap_composite"], reverse=True)
        topk = cheap[:CANDIDATE_TOPK]

        # Expensive per-candidate passes: sharpness (decode) + occlusion.
        # Sharpness needs image bytes; keep a cache across bboxes.
        t_ns_needed = sorted({c["t_ns"] for c in topk
                              if c["t_ns"] not in frame_decode_cache})
        if t_ns_needed:
            decoded = read_frames_by_time_ns(str(mcap_path), topic, t_ns_needed)
            for t_ns, blob, _fmt in decoded:
                arr = np.frombuffer(blob, dtype=np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is not None:
                    frame_decode_cache[t_ns] = bgr

        sharpness_values = []
        for c in topk:
            if c["t_ns"] not in sharpness_cache:
                bgr = frame_decode_cache.get(c["t_ns"])
                if bgr is None:
                    sharpness_cache[c["t_ns"]] = 0.0
                else:
                    sharpness_cache[c["t_ns"]] = _laplacian_var(bgr)
            sharpness_values.append(sharpness_cache[c["t_ns"]])

        # Percentile-normalize sharpness within this bbox's pool.
        sharp_pct = _percentile_normalize(sharpness_values)

        # Occlusion per candidate
        for c, sharp_n in zip(topk, sharp_pct):
            occl = _occlusion_score(corners, c["T_cw"], kdtree, voxel_points)
            scores = {
                "area": c["area"],
                "centering": c["centering"],
                "occlusion": occl,
                "sharpness": sharp_n,
            }
            scores["composite"] = _composite(scores)
            c["scores"] = scores

        topk.sort(key=lambda d: d["scores"]["composite"], reverse=True)
        winner = topk[0]

        if winner["scores"]["composite"] < SCORE_FLOOR:
            entries.append({
                "bbox_id": bbox_id,
                "class": cls,
                "bbox_3d": [float(x) for x in v],
                "skipped": "no_candidate_above_floor",
                "scores": winner["scores"],
                "frame_timestamp_ns": int(winner["t_ns"]),
            })
            continue

        # Crop winner
        bgr = frame_decode_cache.get(winner["t_ns"])
        if bgr is None:
            # Re-fetch single frame if it somehow wasn't decoded.
            decoded = read_frames_by_time_ns(
                str(mcap_path), topic, [winner["t_ns"]])
            if not decoded:
                entries.append({
                    "bbox_id": bbox_id,
                    "class": cls,
                    "bbox_3d": [float(x) for x in v],
                    "skipped": "decode_failed",
                    "frame_timestamp_ns": int(winner["t_ns"]),
                })
                continue
            arr = np.frombuffer(decoded[0][1], dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            frame_decode_cache[winner["t_ns"]] = bgr

        x0c, y0c, x1c, y1c = _pad_and_clip_aabb(winner["aabb"], image_size)
        if x1c <= x0c or y1c <= y0c:
            entries.append({
                "bbox_id": bbox_id,
                "class": cls,
                "bbox_3d": [float(x) for x in v],
                "skipped": "crop_empty",
                "frame_timestamp_ns": int(winner["t_ns"]),
                "scores": winner["scores"],
            })
            continue

        crop = bgr[y0c:y1c, x0c:x1c].copy()
        safe_cls = _safe_filename_component(cls)
        fname = f"{safe_cls}_{bbox_id:03d}.jpg"
        fpath = best_views_dir / fname
        cv2.imwrite(str(fpath), crop)

        entries.append({
            "bbox_id": bbox_id,
            "class": cls,
            "bbox_3d": [float(x) for x in v],
            "frame_timestamp_ns": int(winner["t_ns"]),
            "image_path": f"best_views/{fname}",
            "pixel_aabb": [float(x) for x in winner["aabb"]],
            "crop_aabb": [int(x0c), int(y0c), int(x1c), int(y1c)],
            "scores": {k: float(v) for k, v in winner["scores"].items()},
            "camera_distance_m": float(winner["cam_dist"]),
        })

    manifest = {
        "weights": SCORE_WEIGHTS,
        "score_floor": SCORE_FLOOR,
        "entries": entries,
    }
    manifest_path = best_views_dir / "best_views.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    if verbose:
        kept = sum(1 for e in entries if "image_path" in e)
        skipped = len(entries) - kept
        print(f"[best_views] {kept} kept, {skipped} skipped -> "
              f"{manifest_path}")
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Stage 8: best-view-per-bbox cropper")
    p.add_argument("output_dir", type=Path,
                   help="Pipeline output directory (holds layout_merged.txt)")
    p.add_argument("mcap_path", type=Path, nargs="?", default=None,
                   help="Rosbag to reopen (defaults to path in frames_index.json)")
    p.add_argument("--calibration", type=Path, default=None,
                   help="Override calibration dir (default: repo-vendored)")
    args = p.parse_args(argv)

    run_best_views(
        output_dir=args.output_dir,
        mcap_path=args.mcap_path,
        calibration_dir=args.calibration,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
