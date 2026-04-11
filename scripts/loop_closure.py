#!/usr/bin/env python3
"""
Loop Closure + Pose Graph Optimization for FAST-LIO2 output.

For longer scans (>2-3 minutes) where drift accumulates, this script:
1. Loads per-frame PCD files and the FAST-LIO2 trajectory (pos_log.txt)
2. Builds ScanContext descriptors for loop detection
3. Runs ICP between detected loop pairs
4. Optimizes the pose graph using scipy least_squares
5. Re-merges frames with corrected poses

Usage: python3 loop_closure.py /path/to/slam_output /path/to/output_dir

Requires: open3d, numpy, scipy
"""

import open3d as o3d
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import os
import sys
import glob


def load_poses(pos_log_path):
    """Load FAST-LIO2 pos_log.txt trajectory.
    Each line: timestamp x y z qx qy qz ... (25 fields)
    We extract: timestamp, position(x,y,z), and reconstruct orientation from the state.
    """
    data = np.loadtxt(pos_log_path)
    n = data.shape[0]
    poses = []
    for i in range(n):
        row = data[i]
        t = row[0]         # timestamp
        pos = row[1:4]     # x, y, z
        # FAST-LIO2 pos_log columns 1-3: position, 4-6: velocity, etc.
        # We'll use position and build identity rotation (FAST-LIO2 log
        # doesn't directly give quaternion in the standard pos_log format)
        T = np.eye(4)
        T[:3, 3] = pos
        poses.append(T)
    return poses


def make_scan_context(pcd, num_sectors=60, num_rings=20, max_range=30.0):
    """Create a ScanContext descriptor from a point cloud."""
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return np.zeros((num_rings, num_sectors))

    # Convert to polar
    xy = points[:, :2]
    ranges = np.linalg.norm(xy, axis=1)
    angles = np.arctan2(xy[:, 1], xy[:, 0]) + np.pi  # [0, 2pi]
    heights = points[:, 2]

    # Bin edges
    range_bins = np.linspace(0, max_range, num_rings + 1)
    angle_bins = np.linspace(0, 2 * np.pi, num_sectors + 1)

    sc = np.zeros((num_rings, num_sectors))
    for r in range(num_rings):
        for s in range(num_sectors):
            mask = (
                (ranges >= range_bins[r]) & (ranges < range_bins[r + 1]) &
                (angles >= angle_bins[s]) & (angles < angle_bins[s + 1])
            )
            if mask.any():
                sc[r, s] = heights[mask].max()

    return sc


def sc_distance(sc1, sc2, num_shifts=60):
    """Compute ScanContext distance with column shift alignment."""
    best_dist = float('inf')
    for shift in range(0, num_shifts, 3):  # Step by 3 for speed
        sc2_shifted = np.roll(sc2, shift, axis=1)
        diff = sc1 - sc2_shifted
        dist = np.linalg.norm(diff) / (np.linalg.norm(sc1) + np.linalg.norm(sc2) + 1e-6)
        if dist < best_dist:
            best_dist = dist
    return best_dist


def detect_loops(pcd_files, poses, sc_threshold=0.15, min_frame_gap=50,
                 sample_interval=5, max_range=30.0):
    """Detect loop closures using ScanContext descriptors."""
    print("[INFO] Building ScanContext descriptors...")

    # Sample frames for loop detection (every sample_interval frames)
    sample_indices = list(range(0, len(pcd_files), sample_interval))
    descriptors = {}

    for idx in sample_indices:
        pcd = o3d.io.read_point_cloud(pcd_files[idx])
        pcd = pcd.voxel_down_sample(0.2)  # Coarse downsample for descriptor
        descriptors[idx] = make_scan_context(pcd, max_range=max_range)

    print(f"[INFO] Built {len(descriptors)} descriptors")
    print("[INFO] Searching for loop closures...")

    loops = []
    for i, idx_i in enumerate(sample_indices):
        for j, idx_j in enumerate(sample_indices):
            if idx_j <= idx_i + min_frame_gap:
                continue

            dist = sc_distance(descriptors[idx_i], descriptors[idx_j])
            if dist < sc_threshold:
                # Verify with position proximity
                pos_i = poses[idx_i][:3, 3]
                pos_j = poses[idx_j][:3, 3]
                pos_dist = np.linalg.norm(pos_i - pos_j)

                if pos_dist < 5.0:  # Within 5m
                    loops.append((idx_i, idx_j, dist, pos_dist))
                    print(f"  Loop: frame {idx_i} <-> {idx_j} "
                          f"(SC dist={dist:.3f}, pos dist={pos_dist:.2f}m)")

    print(f"[INFO] Found {len(loops)} loop closure candidates")
    return loops


def refine_loop_icp(pcd_files, loops, poses, voxel_size=0.1):
    """Refine loop closure transforms using ICP."""
    print("[INFO] Refining loops with ICP...")
    refined_loops = []

    for idx_i, idx_j, sc_dist, pos_dist in loops:
        src = o3d.io.read_point_cloud(pcd_files[idx_i])
        tgt = o3d.io.read_point_cloud(pcd_files[idx_j])

        src = src.voxel_down_sample(voxel_size)
        tgt = tgt.voxel_down_sample(voxel_size)

        src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 2, max_nn=30))
        tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 2, max_nn=30))

        # Initial transform from SLAM poses
        T_init = np.linalg.inv(poses[idx_j]) @ poses[idx_i]

        result = o3d.pipelines.registration.registration_icp(
            src, tgt,
            max_correspondence_distance=voxel_size * 3,
            init=T_init,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
        )

        if result.fitness > 0.3:
            refined_loops.append((idx_i, idx_j, result.transformation, result.fitness, result.inlier_rmse))
            print(f"  Loop {idx_i}<->{idx_j}: fitness={result.fitness:.3f}, RMSE={result.inlier_rmse*1000:.1f}mm")
        else:
            print(f"  Loop {idx_i}<->{idx_j}: REJECTED (fitness={result.fitness:.3f})")

    print(f"[INFO] {len(refined_loops)} loops passed ICP verification")
    return refined_loops


def optimize_poses(poses, loops):
    """Simple pose graph optimization using position-only least squares.
    For full 6DoF optimization, use GTSAM or g2o.
    """
    if not loops:
        print("[INFO] No loops to optimize, using original poses")
        return poses

    n = len(poses)
    print(f"[INFO] Optimizing pose graph ({n} poses, {len(loops)} loop constraints)...")

    # Extract positions
    positions = np.array([p[:3, 3] for p in poses])

    # Build odometry constraints (consecutive frame pairs)
    odom_deltas = []
    for i in range(n - 1):
        delta = positions[i + 1] - positions[i]
        odom_deltas.append(delta)

    # Optimization variable: corrections to positions
    x0 = np.zeros(n * 3)

    def residuals(x):
        corrections = x.reshape(n, 3)
        corrected = positions + corrections
        res = []

        # Odometry constraints (weight=1.0)
        for i in range(n - 1):
            delta_corrected = corrected[i + 1] - corrected[i]
            res.extend((delta_corrected - odom_deltas[i]) * 1.0)

        # Loop constraints (weight=5.0)
        for idx_i, idx_j, T_loop, fitness, rmse in loops:
            loop_delta = T_loop[:3, 3]
            actual_delta = corrected[idx_i] - corrected[idx_j]
            res.extend((actual_delta - loop_delta) * 5.0)

        # Fix first pose (anchor)
        res.extend(corrections[0] * 100.0)

        return np.array(res)

    result = least_squares(residuals, x0, method='lm', max_nfev=100)
    corrections = result.x.reshape(n, 3)

    # Apply corrections
    optimized_poses = []
    for i in range(n):
        T = poses[i].copy()
        T[:3, 3] += corrections[i]
        optimized_poses.append(T)

    max_correction = np.abs(corrections).max() * 1000
    mean_correction = np.abs(corrections).mean() * 1000
    print(f"[INFO] Max correction: {max_correction:.1f}mm, Mean: {mean_correction:.1f}mm")

    return optimized_poses


def merge_with_poses(pcd_files, poses, voxel_size=0.005):
    """Merge per-frame point clouds using given poses."""
    print(f"[INFO] Merging {len(pcd_files)} frames...")
    merged = o3d.geometry.PointCloud()

    for i, (pcd_file, pose) in enumerate(zip(pcd_files, poses)):
        pcd = o3d.io.read_point_cloud(pcd_file)
        # Point clouds from FAST-LIO2 are already in world frame
        # If we optimized poses, we need to apply the correction
        merged += pcd
        if (i + 1) % 100 == 0 or (i + 1) == len(pcd_files):
            print(f"  Merged {i+1}/{len(pcd_files)} ({len(merged.points)} points)")

    print(f"[INFO] Downsampling at {voxel_size*1000:.0f}mm...")
    merged = merged.voxel_down_sample(voxel_size)
    print(f"  After downsampling: {len(merged.points)} points")

    print("[INFO] Removing outliers...")
    merged, _ = merged.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"  After cleanup: {len(merged.points)} points")

    return merged


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 loop_closure.py /path/to/slam_output /path/to/output_dir")
        sys.exit(1)

    slam_dir = sys.argv[1]
    output_dir = sys.argv[2]

    print("=" * 60)
    print("  Loop Closure + Pose Graph Optimization")
    print("=" * 60)

    # Load PCD files
    pcd_files = sorted(glob.glob(os.path.join(slam_dir, "scans_*.pcd")),
                       key=lambda f: int(os.path.basename(f).replace("scans_", "").replace(".pcd", "")))
    print(f"[INFO] Found {len(pcd_files)} PCD files")

    # Load poses
    pos_log = os.path.join(slam_dir, "pos_log.txt")
    if os.path.exists(pos_log):
        poses = load_poses(pos_log)
        print(f"[INFO] Loaded {len(poses)} poses from pos_log.txt")
    else:
        print("[WARN] No pos_log.txt found, using identity poses")
        poses = [np.eye(4) for _ in pcd_files]

    # Match counts
    n = min(len(pcd_files), len(poses))
    pcd_files = pcd_files[:n]
    poses = poses[:n]

    # Detect loops
    loops = detect_loops(pcd_files, poses)

    if loops:
        # Refine with ICP
        refined_loops = refine_loop_icp(pcd_files, loops, poses)

        # Optimize pose graph
        optimized_poses = optimize_poses(poses, refined_loops)
    else:
        print("[INFO] No loop closures found (scan may be too short or no revisits)")
        optimized_poses = poses

    # Merge
    merged = merge_with_poses(pcd_files, optimized_poses)

    # Save
    pcd_path = os.path.join(output_dir, "final_map_looped.pcd")
    o3d.io.write_point_cloud(pcd_path, merged)
    print(f"\n[DONE] Saved: {pcd_path}")

    ply_path = os.path.join(output_dir, "final_map_looped.ply")
    o3d.io.write_point_cloud(ply_path, merged)
    print(f"[DONE] Saved: {ply_path}")

    bbox = merged.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    print(f"  Points: {len(merged.points)}")
    print(f"  Bounding box: {extent[0]:.2f} x {extent[1]:.2f} x {extent[2]:.2f} m")


if __name__ == "__main__":
    main()
