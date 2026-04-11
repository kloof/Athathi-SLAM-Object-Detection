#!/usr/bin/env python3
"""
Merge FAST-LIO2 output PCD files into a single point cloud.
Optionally performs voxel downsampling and statistical outlier removal.

Usage: python3 merge_and_refine.py /path/to/slam_output /path/to/output_dir
"""

import open3d as o3d
import numpy as np
import os
import sys
import glob


def merge_pcd_files(slam_dir):
    """Merge all PCD files from FAST-LIO2 output."""
    pcd_files = sorted(glob.glob(os.path.join(slam_dir, "*.pcd")))

    if not pcd_files:
        # FAST-LIO2 might save a single combined PCD
        pcd_files = sorted(glob.glob(os.path.join(slam_dir, "**/*.pcd"), recursive=True))

    if not pcd_files:
        print("[ERROR] No PCD files found in", slam_dir)
        sys.exit(1)

    print(f"[INFO] Found {len(pcd_files)} PCD file(s)")

    merged = o3d.geometry.PointCloud()

    for i, pcd_file in enumerate(pcd_files):
        pcd = o3d.io.read_point_cloud(pcd_file)
        merged += pcd
        if (i + 1) % 50 == 0 or (i + 1) == len(pcd_files):
            print(f"  Loaded {i+1}/{len(pcd_files)} files ({len(merged.points)} points)")

    return merged


def refine_cloud(pcd, voxel_size=0.005):
    """Downsample and clean up the point cloud."""
    print(f"\n[INFO] Raw points: {len(pcd.points)}")

    # Voxel downsampling (5mm default)
    if voxel_size > 0:
        print(f"[INFO] Voxel downsampling at {voxel_size*1000:.1f}mm...")
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"  After downsampling: {len(pcd.points)} points")

    # Statistical outlier removal
    print("[INFO] Removing statistical outliers...")
    pcd, indices = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"  After outlier removal: {len(pcd.points)} points")

    return pcd


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 merge_and_refine.py /path/to/slam_output /path/to/output_dir")
        sys.exit(1)

    slam_dir = sys.argv[1]
    output_dir = sys.argv[2]

    print("=" * 50)
    print("  Point Cloud Merge & Refine")
    print("=" * 50)

    # Merge PCD files
    merged = merge_pcd_files(slam_dir)

    # Refine
    refined = refine_cloud(merged)

    # Save PCD
    pcd_path = os.path.join(output_dir, "final_map.pcd")
    o3d.io.write_point_cloud(pcd_path, refined)
    print(f"\n[DONE] Saved: {pcd_path}")

    # Save PLY
    ply_path = os.path.join(output_dir, "final_map.ply")
    o3d.io.write_point_cloud(ply_path, refined)
    print(f"[DONE] Saved: {ply_path}")

    print(f"\n  Total points: {len(refined.points)}")
    bbox = refined.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    print(f"  Bounding box: {extent[0]:.2f} x {extent[1]:.2f} x {extent[2]:.2f} m")


if __name__ == "__main__":
    main()
