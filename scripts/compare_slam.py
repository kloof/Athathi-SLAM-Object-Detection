#!/usr/bin/env python3
"""Run one SLAM backend on a scan and write a colored-PLY + metrics bundle.

Every registered backend takes (clouds, imus) and returns poses. Downstream
colorization, color-aware voxel reduction, floor leveling, PLY export, and
metrics are shared so results are directly comparable.

Usage:
  python3 scripts/compare_slam.py <rosbag_dir> <output_dir> [calibration_dir] \\
      --backend baseline \\
      [--voxel-size 0.01]

Example:
  python3 scripts/compare_slam.py \\
    "/mnt/c/Users/klof/Desktop/SLAM_test/Scans/scan_20260416_023604/rosbag" \\
    /tmp/slam_compare/01_baseline_icp_imu \\
    "/mnt/c/Users/klof/Desktop/SLAM_test/Scans/scan_20260416_023604/calibration" \\
    --backend baseline
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d

from cloud_slam.mcap_reader import read_mcap
from cloud_slam.slam_backends import BACKENDS, get_backend
from cloud_slam.slam_backends.metrics import compute_all, cross_section_png
from cloud_slam.slam_backends.post_process import (build_map,
                                                   write_trajectory_csv)


def _load_calibration(calib_dir: Path | None):
    if calib_dir is None:
        return None
    intr = calib_dir / "intrinsics.yaml"
    ext = calib_dir / "extrinsics.yaml"
    if not (intr.exists() and ext.exists()):
        print(f"[warn] calibration files not found in {calib_dir}, "
              f"skipping colorization")
        return None
    from cloud_slam.colorizer import load_calibration
    return load_calibration(intr, ext)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Run one SLAM backend and emit a colored-PLY bundle")
    p.add_argument("rosbag", help="Directory or file with MCAP rosbag")
    p.add_argument("output", help="Output directory")
    p.add_argument("calibration", nargs="?", default=None,
                   help="Directory with camera_intrinsics.yaml + extrinsics")
    p.add_argument("--backend", required=True,
                   choices=sorted(BACKENDS),
                   help="Backend name to run")
    p.add_argument("--voxel-size", type=float, default=0.01,
                   help="Final voxel size for the reduced cloud (m)")
    p.add_argument("--no-deskew", action="store_true",
                   help="Skip IMU deskew in post-processing")
    p.add_argument("--no-level", action="store_true",
                   help="Skip floor leveling in post-processing")
    args = p.parse_args()

    rosbag = Path(args.rosbag).resolve()
    outdir = Path(args.output).resolve()
    calib_dir = Path(args.calibration).resolve() if args.calibration else None
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Reading {rosbag}")
    t0 = time.time()
    clouds, imus, images = read_mcap(str(rosbag))
    t_read = time.time() - t0
    print(f"  {len(clouds)} clouds, {len(imus)} IMU, {len(images)} images "
          f"({t_read:.1f}s)")

    calib = _load_calibration(calib_dir)
    print(f"[2/4] Calibration: {'loaded' if calib else 'absent — gray map'}")

    print(f"[3/4] Running backend: {args.backend}")
    backend = get_backend(args.backend)
    result = backend.run(clouds, imus)
    print(f"  {len(result.poses)} poses, runtime {result.runtime_s:.1f}s")

    print(f"[4/4] Post-processing (colorize, voxel-reduce, level)")
    pcd, stats = build_map(
        clouds, imus, result.poses,
        images=images, calib=calib,
        voxel_size=args.voxel_size,
        deskew=not args.no_deskew,
        level_to_floor=not args.no_level,
    )

    ply_path = outdir / "colored_map.ply"
    traj_path = outdir / "trajectory.csv"
    slice_path = outdir / "floor_plus_1m_slice.png"
    metrics_path = outdir / "metrics.json"

    o3d.io.write_point_cloud(str(ply_path), pcd, write_ascii=False)
    write_trajectory_csv(result.poses, clouds, str(traj_path))

    print(f"  Computing accuracy metrics…")
    metric_block = compute_all(pcd, result.poses)
    cross_section_png(pcd, str(slice_path))

    metrics = {
        "backend": result.backend_name,
        "scan": rosbag.name,
        "num_frames": len(clouds),
        "num_imu": len(imus),
        "num_images": len(images),
        "read_mcap_s": round(t_read, 2),
        "backend_runtime_s": round(result.runtime_s, 2),
        "voxel_size_m": args.voxel_size,
        **stats,
        **metric_block,
        **result.extra,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"\nWrote:")
    print(f"  {ply_path}  ({stats['post_final_points']:,} points)")
    print(f"  {traj_path}")
    print(f"  {slice_path}")
    print(f"  {metrics_path}")
    print(f"\nBackend: {result.backend_name}")
    print(f"Runtime (backend only): {result.runtime_s:.1f}s")
    print(f"Colorized fraction: {stats['colorized_fraction']:.1%}")
    print(f"Wall RMSE: {metric_block.get('wall_rmse_m')}")
    print(f"Floor RMSE: {metric_block.get('floor_rmse_m')}")
    print(f"Color/uncolored spread: {metric_block.get('color_uncolored_spread_m')}")
    print(f"Bounding box (m): {stats['bounding_box_m']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
