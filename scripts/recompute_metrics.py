#!/usr/bin/env python3
"""Recompute accuracy metrics on an existing compare-slam output dir.

Reads colored_map.ply + trajectory.csv, runs metrics.compute_all, and
updates metrics.json in place (preserving pre-existing fields). Useful
when metrics logic changes and you don't want to re-run the SLAM.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from cloud_slam.slam_backends.metrics import compute_all, cross_section_png


def _read_trajectory(path: Path) -> list[np.ndarray]:
    poses: list[np.ndarray] = []
    with path.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = {name: i for i, name in enumerate(header)}
        for row in reader:
            T = np.eye(4)
            T[:3, 3] = [float(row[idx[k]]) for k in ("x", "y", "z")]
            q = [float(row[idx[k]]) for k in ("qx", "qy", "qz", "qw")]
            T[:3, :3] = Rotation.from_quat(q).as_matrix()
            poses.append(T)
    return poses


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir", nargs="+",
                    help="One or more compare-slam output directories")
    args = ap.parse_args()

    for raw in args.output_dir:
        d = Path(raw).resolve()
        ply = d / "colored_map.ply"
        traj = d / "trajectory.csv"
        metrics_path = d / "metrics.json"

        if not ply.exists() or not traj.exists():
            print(f"[skip] {d}: missing ply or trajectory")
            continue

        print(f"[recompute] {d}")
        pcd = o3d.io.read_point_cloud(str(ply))
        poses = _read_trajectory(traj)

        metric_block = compute_all(pcd, poses)
        cross_section_png(pcd, str(d / "floor_plus_1m_slice.png"))

        existing = {}
        if metrics_path.exists():
            try:
                existing = json.loads(metrics_path.read_text())
            except Exception:
                pass
        existing.update(metric_block)
        metrics_path.write_text(json.dumps(existing, indent=2))

        print(f"  wall_rmse_m: {metric_block.get('wall_rmse_m')}")
        print(f"  floor_rmse_m: {metric_block.get('floor_rmse_m')}")
        print(f"  color_uncolored_spread_m: {metric_block.get('color_uncolored_spread_m')}")
        print(f"  trajectory_jerk_mean: {metric_block.get('trajectory_jerk_mean')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
