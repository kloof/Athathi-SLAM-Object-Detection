#!/usr/bin/env python3
"""
Benchmark SLAM algorithms on existing rosbag data.
Compares: KISS-ICP, small_gicp+IMU, current Open3D ICP+IMU
"""

import time
import struct
import numpy as np
import open3d as o3d
from pathlib import Path
from scipy.spatial.transform import Rotation


# ============================================================
# Shared: Fast MCAP reader
# ============================================================
def read_mcap_fast(bag_dir):
    """Read MCAP and return cloud frames as numpy arrays + IMU data."""
    from mcap_ros2.reader import read_ros2_messages

    bag_path = Path(bag_dir)
    mcap_files = sorted(bag_path.glob("*.mcap"))

    clouds = []  # list of (timestamp, Nx3 float64, Nx1 float64 timestamps)
    imus = []    # list of (timestamp, gyro_xyz, acc_xyz)

    # Point cloud dtype matching Unitree L2 layout (32 bytes/point)
    # x(0) y(4) z(8) _pad(12) intensity(16) ring(20) _pad2(22) time(24) _pad3(28)
    pc_dtype = np.dtype({
        'names': ['x', 'y', 'z', 'intensity', 'ring', 'time'],
        'formats': ['<f4', '<f4', '<f4', '<f4', '<u2', '<f4'],
        'offsets': [0, 4, 8, 16, 20, 24],
        'itemsize': 32
    })

    for mcap_file in mcap_files:
        for msg in read_ros2_messages(str(mcap_file)):
            if msg.channel.topic == "/unilidar/cloud":
                ros_msg = msg.ros_msg
                stamp = ros_msg.header.stamp.sec + ros_msg.header.stamp.nanosec * 1e-9
                data = bytes(ros_msg.data)
                pts = np.frombuffer(data, dtype=pc_dtype)
                xyz = np.column_stack([pts['x'], pts['y'], pts['z']]).astype(np.float64)
                timestamps = pts['time'].astype(np.float64)
                # Filter invalid points
                valid = np.linalg.norm(xyz, axis=1) > 0.05
                clouds.append((stamp, xyz[valid], timestamps[valid]))
            elif msg.channel.topic == "/unilidar/imu":
                ros_msg = msg.ros_msg
                stamp = ros_msg.header.stamp.sec + ros_msg.header.stamp.nanosec * 1e-9
                gyro = np.array([ros_msg.angular_velocity.x, ros_msg.angular_velocity.y, ros_msg.angular_velocity.z])
                acc = np.array([ros_msg.linear_acceleration.x, ros_msg.linear_acceleration.y, ros_msg.linear_acceleration.z])
                imus.append((stamp, gyro, acc))

    clouds.sort(key=lambda x: x[0])
    imus.sort(key=lambda x: x[0])
    return clouds, imus


# ============================================================
# Benchmark A: KISS-ICP
# ============================================================
def benchmark_kiss_icp(clouds):
    """Run KISS-ICP on point cloud frames."""
    from kiss_icp.kiss_icp import KissICP
    from kiss_icp.config import KISSConfig

    config = KISSConfig()
    config.data.max_range = 30.0
    config.data.min_range = 0.05
    config.data.deskew = True
    config.mapping.voxel_size = 0.5

    odometry = KissICP(config=config)
    poses = []

    t_start = time.time()
    for i, (stamp, xyz, timestamps) in enumerate(clouds):
        source, keypoints = odometry.register_frame(xyz, timestamps)
        poses.append(odometry.last_pose.copy())

    elapsed = time.time() - t_start

    # Build merged cloud
    merged = o3d.geometry.PointCloud()
    for (stamp, xyz, ts), pose in zip(clouds, poses):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.transform(pose)
        merged += pcd

    merged = merged.voxel_down_sample(0.005)
    merged, _ = merged.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    return merged, poses, elapsed


# ============================================================
# Benchmark B: small_gicp + IMU
# ============================================================
def benchmark_small_gicp_imu(clouds, imus):
    """Run small_gicp GICP with IMU initial guess."""
    import small_gicp

    imu_times = np.array([t for t, _, _ in imus])
    imu_gyro = np.array([g for _, g, _ in imus])

    T_current = np.eye(4)
    poses = [T_current.copy()]
    map_points = None
    map_tree = None

    t_start = time.time()
    prev_stamp = None

    for i, (stamp, xyz, timestamps) in enumerate(clouds):
        if len(xyz) < 10:
            poses.append(T_current.copy())
            continue

        if prev_stamp is not None and map_points is not None:
            dt = stamp - prev_stamp
            # IMU rotation guess
            mask = (imu_times >= prev_stamp) & (imu_times < stamp)
            if mask.any():
                avg_gyro = imu_gyro[mask].mean(axis=0)
                dR = Rotation.from_rotvec(avg_gyro * dt).as_matrix()
            else:
                dR = np.eye(3)

            T_guess = T_current.copy()
            T_guess[:3, :3] = T_current[:3, :3] @ dR

            # Preprocess source
            source, source_tree = small_gicp.preprocess_points(
                xyz, downsampling_resolution=0.1, num_neighbors=20, num_threads=4)

            # Register
            result = small_gicp.align(
                map_points, source, map_tree,
                init_T_target_source=np.linalg.inv(T_guess),
                registration_type='GICP',
                max_correspondence_distance=0.5,
                num_threads=4
            )

            if result.converged:
                T_current = np.linalg.inv(result.T_target_source)
            else:
                T_current = T_guess
        else:
            pass

        poses.append(T_current.copy())
        prev_stamp = stamp

        # Update map every 5 frames
        if i % 5 == 0:
            # Collect recent points in world frame
            all_pts = []
            start_idx = max(0, len(poses) - 20)
            for j in range(start_idx, len(poses)):
                if j < len(clouds):
                    pts_j = clouds[j][1].copy()
                    pts_world = (poses[j][:3, :3] @ pts_j.T).T + poses[j][:3, 3]
                    all_pts.append(pts_world)
            if all_pts:
                combined = np.vstack(all_pts)
                map_points, map_tree = small_gicp.preprocess_points(
                    combined, downsampling_resolution=0.1, num_neighbors=20, num_threads=4)

    elapsed = time.time() - t_start

    # Build merged cloud
    merged = o3d.geometry.PointCloud()
    for (stamp, xyz, ts), pose in zip(clouds, poses[:len(clouds)]):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.transform(pose)
        merged += pcd

    merged = merged.voxel_down_sample(0.005)
    merged, _ = merged.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    return merged, poses, elapsed


# ============================================================
# Quality metrics
# ============================================================
def measure_quality(pcd, name):
    """Compute quality metrics for a point cloud."""
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()

    # Wall planarity
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
        rms = np.sqrt((distances**2).mean()) * 1000
        rms_values.append(rms)
        remaining = remaining.select_by_index(inliers, invert=True)

    avg_rms = np.mean(rms_values) if rms_values else 0

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Points:       {len(pcd.points)}")
    print(f"  Bounding box: {extent[0]:.2f} x {extent[1]:.2f} x {extent[2]:.2f} m")
    print(f"  Avg wall RMS: {avg_rms:.2f} mm")
    for i, rms in enumerate(rms_values):
        print(f"    Plane {i+1} RMS: {rms:.2f} mm")

    return len(pcd.points), extent, avg_rms


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    BAG_DIR = "/home/klof/lidar_slam/data"

    print("Reading MCAP...")
    t0 = time.time()
    clouds, imus = read_mcap_fast(BAG_DIR)
    print(f"Read {len(clouds)} clouds + {len(imus)} IMU in {time.time()-t0:.1f}s\n")

    # Benchmark KISS-ICP
    print("Running KISS-ICP...")
    kiss_pcd, kiss_poses, kiss_time = benchmark_kiss_icp(clouds)
    measure_quality(kiss_pcd, f"KISS-ICP ({kiss_time:.1f}s)")

    # Benchmark small_gicp + IMU
    print("\nRunning small_gicp + IMU...")
    gicp_pcd, gicp_poses, gicp_time = benchmark_small_gicp_imu(clouds, imus)
    measure_quality(gicp_pcd, f"small_gicp + IMU ({gicp_time:.1f}s)")

    # Load FAST-LIO2 baseline
    baseline_path = "/home/klof/lidar_slam/data/final_map.pcd"
    baseline = o3d.io.read_point_cloud(baseline_path)
    measure_quality(baseline, "FAST-LIO2 baseline (via ROS2)")

    print(f"\n{'='*50}")
    print(f"  SUMMARY")
    print(f"{'='*50}")
    print(f"  KISS-ICP:        {kiss_time:.1f}s, {len(kiss_pcd.points)} pts")
    print(f"  small_gicp+IMU:  {gicp_time:.1f}s, {len(gicp_pcd.points)} pts")
    print(f"  FAST-LIO2:       ~53s (ROS2 playback), {len(baseline.points)} pts")

    # Save outputs for visual comparison
    o3d.io.write_point_cloud("/home/klof/lidar_slam/data/benchmark_kiss.ply", kiss_pcd)
    o3d.io.write_point_cloud("/home/klof/lidar_slam/data/benchmark_gicp.ply", gicp_pcd)
    print("\nSaved: benchmark_kiss.ply, benchmark_gicp.ply")
