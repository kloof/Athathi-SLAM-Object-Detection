#!/usr/bin/env python3
"""
Scan Quality Verification Tool

Checks:
1. Bounding box dimensions
2. Point density statistics
3. Wall planarity (RANSAC plane fitting)

Usage: python3 verify_scan.py /path/to/final_map.pcd
"""

import open3d as o3d
import numpy as np
import sys


def load_cloud(path):
    pcd = o3d.io.read_point_cloud(path)
    print(f"Loaded {len(pcd.points)} points from {path}")
    return pcd


def check_bounding_box(pcd):
    print("\n=== Bounding Box ===")
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    print(f"  X: {extent[0]:.3f} m")
    print(f"  Y: {extent[1]:.3f} m")
    print(f"  Z: {extent[2]:.3f} m")


def check_density(pcd):
    print("\n=== Point Density ===")
    distances = pcd.compute_nearest_neighbor_distance()
    distances = np.asarray(distances)
    print(f"  Mean nearest-neighbor distance: {distances.mean()*1000:.2f} mm")
    print(f"  Median: {np.median(distances)*1000:.2f} mm")
    print(f"  Std dev: {distances.std()*1000:.2f} mm")
    print(f"  Points within 5mm of neighbor: {(distances < 0.005).sum() / len(distances) * 100:.1f}%")


def check_planarity(pcd, num_planes=5):
    print("\n=== Wall Planarity (RANSAC) ===")
    remaining = pcd
    for i in range(num_planes):
        if len(remaining.points) < 1000:
            break
        plane_model, inliers = remaining.segment_plane(
            distance_threshold=0.01,
            ransac_n=3,
            num_iterations=1000
        )
        [a, b, c, d] = plane_model
        inlier_cloud = remaining.select_by_index(inliers)

        points = np.asarray(inlier_cloud.points)
        distances = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d)

        print(f"  Plane {i+1}: {len(inliers)} points")
        print(f"    Normal: ({a:.3f}, {b:.3f}, {c:.3f})")
        print(f"    Mean residual: {distances.mean()*1000:.2f} mm")
        print(f"    Max residual:  {distances.max()*1000:.2f} mm")
        print(f"    RMS residual:  {np.sqrt((distances**2).mean())*1000:.2f} mm")

        remaining = remaining.select_by_index(inliers, invert=True)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "final_map.pcd"
    pcd = load_cloud(path)
    check_bounding_box(pcd)
    check_density(pcd)
    check_planarity(pcd)
    print("\nDone.")
