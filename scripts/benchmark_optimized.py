#!/usr/bin/env python3
"""
Benchmark: Original Open3D ICP vs small_gicp with SAME map strategy.
Tests on scan_002 (the larger rosbag that produced final_map_direct_icp).
"""

import time
import numpy as np
import open3d as o3d
import small_gicp
from scipy.spatial.transform import Rotation
import sys
sys.path.insert(0, '/home/klof/lidar_slam/cloud_slam')
from mcap_reader import read_mcap


def run_original_open3d(clouds, imus):
    """Original fast_process.py approach — Open3D point-to-plane ICP."""
    t0 = time.time()

    imu_times = np.array([t for t, _, _ in imus])
    imu_gyro = np.array([g for _, g, _ in imus])

    merged = o3d.geometry.PointCloud()
    T_current = np.eye(4)
    prev_stamp = None
    map_cloud = o3d.geometry.PointCloud()

    for i, (stamp, xyz, timestamps) in enumerate(clouds):
        scan = o3d.geometry.PointCloud()
        scan.points = o3d.utility.Vector3dVector(xyz)

        if len(scan.points) < 10:
            continue

        if prev_stamp is not None and len(map_cloud.points) > 0:
            dt = stamp - prev_stamp
            mask = (imu_times >= prev_stamp) & (imu_times < stamp)
            if mask.any():
                dR = Rotation.from_rotvec(imu_gyro[mask].mean(axis=0) * dt).as_matrix()
            else:
                dR = np.eye(3)

            T_guess = T_current.copy()
            T_guess[:3, :3] = T_current[:3, :3] @ dR

            scan_down = scan.voxel_down_sample(0.1)
            map_down = map_cloud.voxel_down_sample(0.1)
            scan_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
            map_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))

            result = o3d.pipelines.registration.registration_icp(
                scan_down, map_down, 0.3, T_guess,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))

            if result.fitness > 0.1:
                T_current = result.transformation
            else:
                T_current = T_guess

        scan_world = o3d.geometry.PointCloud(scan)
        scan_world.transform(T_current)
        merged += scan_world

        if i % 5 == 0:
            map_cloud = merged.voxel_down_sample(0.05)

        prev_stamp = stamp

    elapsed = time.time() - t0
    merged = merged.voxel_down_sample(0.005)
    merged, _ = merged.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return merged, elapsed


def run_small_gicp_same_map(clouds, imus):
    """small_gicp GICP but with the SAME full-accumulated-map strategy."""
    t0 = time.time()

    imu_times = np.array([t for t, _, _ in imus])
    imu_gyro = np.array([g for _, g, _ in imus])

    merged = o3d.geometry.PointCloud()
    T_current = np.eye(4)
    prev_stamp = None
    map_points = None
    map_tree = None

    for i, (stamp, xyz, timestamps) in enumerate(clouds):
        if len(xyz) < 10:
            continue

        if prev_stamp is not None and map_points is not None:
            dt = stamp - prev_stamp
            mask = (imu_times >= prev_stamp) & (imu_times < stamp)
            if mask.any():
                dR = Rotation.from_rotvec(imu_gyro[mask].mean(axis=0) * dt).as_matrix()
            else:
                dR = np.eye(3)

            T_guess = T_current.copy()
            T_guess[:3, :3] = T_current[:3, :3] @ dR

            source, source_tree = small_gicp.preprocess_points(
                xyz, downsampling_resolution=0.1, num_neighbors=20, num_threads=4)

            result = small_gicp.align(
                map_points, source, map_tree,
                init_T_target_source=np.linalg.inv(T_guess),
                registration_type='GICP',
                max_correspondence_distance=0.3,
                num_threads=4)

            if result.converged:
                T_current = np.linalg.inv(result.T_target_source)
            else:
                T_current = T_guess

        # Same accumulation strategy as original
        scan = o3d.geometry.PointCloud()
        scan.points = o3d.utility.Vector3dVector(xyz)
        scan_world = o3d.geometry.PointCloud(scan)
        scan_world.transform(T_current)
        merged += scan_world

        # Same map rebuild strategy: full accumulated cloud every 5 frames
        if i % 5 == 0:
            map_o3d = merged.voxel_down_sample(0.05)
            map_np = np.asarray(map_o3d.points)
            if len(map_np) > 10:
                map_points, map_tree = small_gicp.preprocess_points(
                    map_np, downsampling_resolution=0.05, num_neighbors=20, num_threads=4)

        prev_stamp = stamp

    elapsed = time.time() - t0
    merged = merged.voxel_down_sample(0.005)
    merged, _ = merged.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return merged, elapsed


def measure_quality(pcd, name):
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()

    remaining = pcd
    rms_values = []
    for _ in range(3):
        if len(remaining.points) < 500:
            break
        plane_model, inliers = remaining.segment_plane(
            distance_threshold=0.01, ransac_n=3, num_iterations=1000)
        a, b, c, d = plane_model
        inlier_cloud = remaining.select_by_index(inliers)
        points = np.asarray(inlier_cloud.points)
        distances = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d)
        rms_values.append(np.sqrt((distances**2).mean()) * 1000)
        remaining = remaining.select_by_index(inliers, invert=True)

    avg_rms = np.mean(rms_values) if rms_values else 0

    print(f"\n  {name}")
    print(f"  Points:       {len(pcd.points)}")
    print(f"  Bounding box: {extent[0]:.2f} x {extent[1]:.2f} x {extent[2]:.2f} m")
    print(f"  Avg wall RMS: {avg_rms:.2f} mm")
    for i, rms in enumerate(rms_values):
        print(f"    Plane {i+1}: {rms:.2f} mm")


# Main
BAG_DIR = "/home/klof/lidar_slam/data/scan_002"
print("Reading MCAP (scan_002, 120MB)...")
t0 = time.time()
clouds, imus = read_mcap(BAG_DIR)
print(f"Read {len(clouds)} clouds + {len(imus)} IMU in {time.time()-t0:.1f}s\n")

print("=" * 60)
print("Running ORIGINAL Open3D ICP + IMU...")
pcd_orig, time_orig = run_original_open3d(clouds, imus)
measure_quality(pcd_orig, f"ORIGINAL Open3D ICP ({time_orig:.1f}s)")
o3d.io.write_point_cloud("/tmp/bench_original.ply", pcd_orig)

print("\n" + "=" * 60)
print("Running small_gicp GICP + IMU (SAME map strategy)...")
pcd_gicp, time_gicp = run_small_gicp_same_map(clouds, imus)
measure_quality(pcd_gicp, f"small_gicp GICP same map ({time_gicp:.1f}s)")
o3d.io.write_point_cloud("/tmp/bench_gicp_samemap.ply", pcd_gicp)

print(f"\n{'=' * 60}")
print(f"  SPEEDUP: {time_orig/time_gicp:.1f}x ({time_orig:.1f}s → {time_gicp:.1f}s)")
print(f"{'=' * 60}")
