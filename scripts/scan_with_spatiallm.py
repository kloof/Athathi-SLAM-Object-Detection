#!/usr/bin/env python3
"""SLAM + SpatialLM1.1 iterative scene refinement.

Reads a lidar-camera rosbag, runs ICP+IMU SLAM, periodically invokes
SpatialLM via a worker subprocess (running in ``~/spatiallm_env``) for
structured layout inference, and fuses successive passes into one scene.

Output is a colored PLY of the merged map plus a JSON of the fused scene.
Mirrors the file layout / leveling / floorplan post-processing of the
legacy ``detect_and_slam.py``, but replaces the YOLOE-based object detection
with SpatialLM inference.

Usage:
    python3 scan_with_spatiallm.py /path/to/rosbag /path/to/output /path/to/calibration
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SLAM + SpatialLM1.1 iterative scene refinement")
    parser.add_argument("rosbag", help="Path to rosbag directory or .mcap file")
    parser.add_argument("output", help="Output directory")
    _default_calib = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "calibration")
    parser.add_argument("calibration", nargs="?", default=_default_calib,
                        help=f"Path to calibration directory (default: "
                             f"vendored {_default_calib})")
    parser.add_argument("--categories", type=str, default=None,
                        help="Comma-separated SpatialLM object categories "
                             "(default: all categories)")
    parser.add_argument("--refinement-interval-frames", type=int, default=100,
                        help="Submit a SpatialLM snapshot every N frames "
                             "(default: 100)")
    parser.add_argument("--voxel-size", type=float, default=0.005,
                        help="Final merged-cloud voxel size (m, default 0.005)")
    parser.add_argument("--snapshot-voxel", type=float, default=0.02,
                        help="Snapshot voxel size for mid-scan passes "
                             "(m, default 0.02)")
    parser.add_argument("--spatiallm-model", type=str,
                        default="manycore-research/SpatialLM1.1-Qwen-0.5B",
                        help="HuggingFace model id or local path")
    parser.add_argument("--max-points", type=int, default=200000,
                        help="Per-snapshot point cap (default 200000)")
    parser.add_argument("--worker-python", type=str,
                        default="~/spatiallm_env/bin/python",
                        help="Path to the worker venv python interpreter")
    parser.add_argument("--inference-timeout-s", type=float, default=180.0,
                        help="Per-pass inference timeout (s, default 180)")
    parser.add_argument("--leveling", choices=["legacy", "floor"], default="legacy",
                        help="Leveling strategy (mirrors legacy detect_and_slam.py)")
    parser.add_argument("--min-observations", type=int, default=None,
                        help="Only include fused elements seen in at least N "
                             "SpatialLM passes (drops single-pass noise). "
                             "Applied to objects.json + map_with_boxes.ply. "
                             "Default: no filter. Recommended: 2 for clean output.")
    parser.add_argument("--min-confidence", type=float, default=None,
                        help="Only include fused elements with confidence >= F "
                             "(range 0-1). Default: no filter.")
    parser.add_argument("--log-level", type=str, default="INFO",
                        help="logging level: DEBUG/INFO/WARNING/ERROR")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Defer heavy imports until after arg parsing so --help is fast and safe.
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import open3d as o3d
    from scipy.spatial.transform import Rotation as SciRot

    from cloud_slam.colorizer import load_calibration
    from cloud_slam.mcap_reader import read_mcap
    from cloud_slam.pipelines import spatiallm_pipeline

    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("  SLAM + SpatialLM1.1 Iterative Scene Refinement")
    print("=" * 60)

    calib = load_calibration(
        os.path.join(args.calibration, "intrinsics.yaml"),
        os.path.join(args.calibration, "extrinsics.yaml"),
    )
    print(f"[INFO] Loaded calibration from {args.calibration}")

    t0 = time.time()
    clouds, imus, images = read_mcap(args.rosbag)
    t_read = time.time() - t0
    print(f"[INFO] Read {len(clouds)} clouds + {len(imus)} IMU + {len(images)} "
          f"images in {t_read:.1f}s")

    categories = None
    if args.categories:
        categories = [c.strip() for c in args.categories.split(",") if c.strip()]

    merged, poses, fused, stats = spatiallm_pipeline.run(
        clouds, imus, images, calib,
        voxel_size=args.voxel_size,
        refinement_interval_frames=args.refinement_interval_frames,
        categories=categories,
        spatiallm_model=args.spatiallm_model,
        max_points=args.max_points,
        snapshot_voxel=args.snapshot_voxel,
        worker_env_python=args.worker_python,
        inference_timeout_s=args.inference_timeout_s,
    )
    t_total = time.time() - t0

    # Leveling (legacy path: IMU gravity -> Z-up after SLAM, to match the
    # working visual appearance of the original detect_and_slam output).
    z_up = np.array([0.0, 0.0, 1.0])
    if args.leveling == "legacy":
        from cloud_slam.frustum import estimate_gravity
        gravity_up = estimate_gravity(imus)
        if not np.allclose(gravity_up, z_up, atol=0.01):
            R_level = SciRot.align_vectors([z_up], [gravity_up])[0].as_matrix()
            print(f"[INFO] Leveling scan (gravity was {gravity_up})")
            merged.rotate(R_level, center=(0, 0, 0))

    # Wall-axis alignment on the leveled cloud.
    from cloud_slam.manhattan import estimate_manhattan_frame
    from cloud_slam.room_structure import detect_room
    room_leveled = detect_room(merged, z_up)
    if room_leveled.walls:
        manhattan_leveled = estimate_manhattan_frame(room_leveled.walls, z_up)
        if manhattan_leveled.confidence > 0.3:
            mx = manhattan_leveled.R.T[:, 0]
            wall_yaw = np.arctan2(mx[1], mx[0])
            snapped_wall = round(wall_yaw / (np.pi / 2)) * (np.pi / 2)
            residual = wall_yaw - snapped_wall
            if abs(residual) > 0.01:
                R_align = SciRot.from_euler('z', -residual).as_matrix()
                print(f"[INFO] Aligning room to axes "
                      f"(rotating {np.degrees(-residual):.1f}° around Z)")
                merged.rotate(R_align, center=(0, 0, 0))

    # Write colored map.
    map_path = os.path.join(args.output, "colored_map.ply")
    o3d.io.write_point_cloud(map_path, merged)
    print(f"[INFO] Saved colored map: {map_path} ({len(merged.points)} points)")

    # Refined RANSAC floor-leveler post-process (same as legacy script).
    # Returns the rotation + z-shift applied to merged so we can mirror the
    # same transform on the fused scene below (otherwise bboxes stay at the
    # pre-leveled floor z while the cloud's floor moves to z=0).
    R_floor = np.eye(3)
    dz_floor = 0.0
    try:
        from cloud_slam.level import detect_floor_plane, level_points
        print("[Level] Running RANSAC floor leveler on map...")
        xyz = np.asarray(merged.points)
        result = detect_floor_plane(xyz)
        if result is None:
            raise RuntimeError("No horizontal plane found")
        normal, _ = result
        # Recompute the exact rotation level_points uses so we can apply it
        # to the scene. level_points returns (leveled_pts, angle_deg, floor_z)
        # where the final shift is -floor_z in Z.
        R_floor = SciRot.align_vectors(
            [[0.0, 0.0, 1.0]], normal.reshape(1, 3))[0].as_matrix()
        leveled_xyz, _, floor_z = level_points(xyz, normal)
        dz_floor = -float(floor_z)
        leveled_pcd = o3d.geometry.PointCloud()
        leveled_pcd.points = o3d.utility.Vector3dVector(
            leveled_xyz.astype(np.float64))
        if merged.has_colors():
            leveled_pcd.colors = merged.colors
        leveled_path = os.path.join(args.output, "colored_map_leveled.ply")
        o3d.io.write_point_cloud(leveled_path, leveled_pcd)
        print(f"[Level] Saved leveled map: {leveled_path}")
        # Mutate merged itself to match the leveled frame so downstream
        # overlay generation and floorplan see the final post-level geometry.
        merged = leveled_pcd
    except Exception as exc:
        print(f"[WARNING] level.py post-process failed: {exc}")

    # Floor plan extraction (orthogonal, same call as legacy).
    try:
        from cloud_slam.floorplan import generate_floorplan
        rosbag_parent = os.path.dirname(os.path.abspath(args.rosbag))
        scan_name = os.path.basename(rosbag_parent)
        scan_root = os.path.dirname(rosbag_parent)
        fp_out_dir = os.path.join(scan_root, "results", scan_name)
        os.makedirs(fp_out_dir, exist_ok=True)
        print(f"[Floorplan] Generating floor plan -> {fp_out_dir}")
        _variants, fp_meta = generate_floorplan(
            merged, fp_out_dir, name="floorplan",
            gravity_up=np.array([0.0, 0.0, 1.0]), verbose=False)
        v = fp_meta['variants']
        print(f"[Floorplan] Saved ("
              f"A={v['A_natural']['area_m2']}m²/{v['A_natural']['n_walls']}w, "
              f"D={v['D_refined']['area_m2']}m²/{v['D_refined']['n_walls']}w)")
    except Exception as exc:
        print(f"[WARNING] floorplan.py post-process failed: {exc}")

    # Apply the same floor-level transform to the fused scene so bboxes and
    # walls end up in the same frame as the leveled merged cloud (otherwise
    # they remain at the pre-shift floor z and appear below the visible floor).
    scene = fused.current()
    scene.apply_transform(R_floor, np.array([0.0, 0.0, dz_floor]))
    scene_meta = fused.current_with_metadata()["meta"]

    # Optional confidence-based filter: keep only elements seen in enough
    # passes / above a confidence threshold. Applied to BOTH objects.json and
    # the overlay so the user's selection shows up everywhere consistently.
    if args.min_observations is not None or args.min_confidence is not None:
        def _keep(kind: str, elem_id: int) -> bool:
            for m in scene_meta.get(kind, []):
                if m["id"] == elem_id:
                    if (args.min_observations is not None
                            and m["observations"] < args.min_observations):
                        return False
                    if (args.min_confidence is not None
                            and m["confidence"] < args.min_confidence):
                        return False
                    return True
            return True
        scene.walls = [w for w in scene.walls if _keep("walls", w.id)]
        scene.doors = [d for d in scene.doors if _keep("doors", d.id)]
        scene.windows = [w for w in scene.windows if _keep("windows", w.id)]
        scene.bboxes = [b for b in scene.bboxes if _keep("bboxes", b.id)]
        kept_ids = {k: {e.id for e in getattr(scene, k)}
                    for k in ("walls", "doors", "windows", "bboxes")}
        scene_meta = {k: [m for m in scene_meta.get(k, []) if m["id"] in kept_ids[k]]
                      for k in ("walls", "doors", "windows", "bboxes")}
        print(f"[INFO] Filtered scene "
              f"(min_observations={args.min_observations}, "
              f"min_confidence={args.min_confidence}): {scene.summary()}")

    # Write fused scene + metadata (scene already in leveled frame).
    objects_path = os.path.join(args.output, "objects.json")
    output_json = {
        "format_version": "2.0",
        "coordinate_frame": "world",
        "source": "spatiallm",
        "stats": stats,
        "scene": scene.to_json(),
        "element_meta": scene_meta,
    }
    with open(objects_path, "w") as f:
        json.dump(output_json, f, indent=2)
    print(f"[INFO] Saved fused scene ({scene.summary()}): {objects_path}")

    # Overlay PLY: merged cloud + colored wireframes for walls/doors/windows/bboxes.
    # When --min-observations / --min-confidence are set, low-confidence single-
    # pass hallucinations are hidden from both the overlay and objects.json.
    try:
        from cloud_slam.detectors.scene_visualize import build_overlay
        overlay_pts, overlay_cols = build_overlay(
            scene, element_meta=scene_meta,
            min_observations=args.min_observations,
            min_confidence=args.min_confidence)
        map_pts = np.asarray(merged.points)
        if merged.has_colors():
            map_cols = np.asarray(merged.colors)  # float [0,1]
        else:
            map_cols = np.full((len(map_pts), 3), 0.5)
        overlay_cols_f = overlay_cols.astype(np.float64) / 255.0
        combined = o3d.geometry.PointCloud()
        combined.points = o3d.utility.Vector3dVector(
            np.vstack([map_pts, overlay_pts]))
        combined.colors = o3d.utility.Vector3dVector(
            np.vstack([map_cols, overlay_cols_f]))
        boxes_map_path = os.path.join(args.output, "map_with_boxes.ply")
        o3d.io.write_point_cloud(boxes_map_path, combined)
        print(f"[INFO] Saved map_with_boxes.ply with {len(overlay_pts)} overlay "
              f"points ({len(scene.bboxes)} bboxes, {len(scene.walls)} walls, "
              f"{len(scene.doors)} doors, {len(scene.windows)} windows)")
    except Exception as exc:
        print(f"[WARNING] map_with_boxes.ply generation failed: {exc}")

    # Summary.
    print(f"\n{'='*60}")
    print(f"  DONE — Total: {t_total:.1f}s")
    print(f"  Points: {len(merged.points)}")
    print(f"  Scene: {scene.summary()}")
    print(f"  Inference passes: "
          f"{stats.get('n_inference_completed', 0)} completed, "
          f"{stats.get('n_inference_empty', 0)} empty, "
          f"{stats.get('n_inference_error', 0)} errored")
    print(f"  Avg inference time: "
          f"{stats.get('avg_inference_sec', 0.0):.2f}s")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
