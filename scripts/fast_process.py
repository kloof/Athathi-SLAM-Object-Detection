#!/usr/bin/env python3
"""
Direct MCAP → Point Cloud processing — bypasses ROS2 entirely.

Reads the rosbag MCAP file directly, extracts point clouds and IMU data,
and produces a merged point cloud using scan-to-scan ICP registration.

This is MUCH faster than playing the bag through ROS2.

Usage: python3 fast_process.py /path/to/rosbag_dir /path/to/output_dir [/path/to/calibration_dir]
"""

import os
import sys
import struct
import time
import numpy as np
import open3d as o3d
from pathlib import Path


def read_mcap_messages(bag_dir):
    """Read all messages from an MCAP rosbag directory."""
    from mcap_ros2.reader import read_ros2_messages

    bag_path = Path(bag_dir)
    mcap_files = list(bag_path.glob("*.mcap"))
    if not mcap_files:
        print(f"[ERROR] No .mcap files found in {bag_dir}")
        sys.exit(1)

    cloud_msgs = []
    imu_msgs = []

    for mcap_file in sorted(mcap_files):
        for msg in read_ros2_messages(str(mcap_file)):
            if msg.channel.topic == "/unilidar/cloud":
                cloud_msgs.append(msg.ros_msg)
            elif msg.channel.topic == "/unilidar/imu":
                imu_msgs.append(msg.ros_msg)

    return cloud_msgs, imu_msgs


def pointcloud2_to_array(msg):
    """Convert sensor_msgs/PointCloud2 to numpy array of [x,y,z,intensity]."""
    point_step = msg.point_step
    data = bytes(msg.data)
    n_points = msg.width * msg.height

    points = np.zeros((n_points, 4), dtype=np.float32)
    for i in range(n_points):
        offset = i * point_step
        x, y, z = struct.unpack_from('fff', data, offset)
        intensity = struct.unpack_from('f', data, offset + 16)[0]
        points[i] = [x, y, z, intensity]

    # Filter out zero/invalid points
    valid = np.linalg.norm(points[:, :3], axis=1) > 0.05
    return points[valid]


def pointcloud2_to_o3d(msg):
    """Convert sensor_msgs/PointCloud2 to Open3D point cloud."""
    pts = pointcloud2_to_array(msg)
    if len(pts) == 0:
        return o3d.geometry.PointCloud()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[:, :3].astype(np.float64))
    return pcd


def register_scan_to_map(scan, map_cloud, T_init, voxel_size=0.1):
    """Register a scan to the existing map using ICP."""
    if len(map_cloud.points) == 0:
        return T_init, True

    scan_down = scan.voxel_down_sample(voxel_size)
    map_down = map_cloud.voxel_down_sample(voxel_size)

    if len(scan_down.points) < 10 or len(map_down.points) < 10:
        return T_init, False

    scan_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    map_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

    result = o3d.pipelines.registration.registration_icp(
        scan_down, map_down,
        max_correspondence_distance=voxel_size * 3,
        init=T_init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
    )

    return result.transformation, result.fitness > 0.1


def process_with_imu_integration(cloud_msgs, imu_msgs, camera_imgs=None, calib=None):
    """
    Simple IMU-aided scan matching.
    Uses IMU for initial transform guess, then refines with ICP.
    """
    print(f"[INFO] Processing {len(cloud_msgs)} scans with {len(imu_msgs)} IMU messages")

    do_color = camera_imgs is not None and calib is not None and len(camera_imgs) > 0
    if do_color:
        import cv2 as _cv2
        from cloud_slam.colorizer import colorize_cloud, match_nearest_image
        image_timestamps = np.array([t for t, _, _ in camera_imgs])
        print(f"[INFO] Color projection enabled ({len(camera_imgs)} images)")

    # Sort by timestamp
    def get_stamp(msg):
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    cloud_msgs.sort(key=get_stamp)
    imu_msgs.sort(key=get_stamp)

    # Build IMU timestamp -> angular velocity lookup
    imu_times = np.array([get_stamp(m) for m in imu_msgs])
    imu_gyro = np.array([[m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z]
                         for m in imu_msgs])

    # Process scans
    merged = o3d.geometry.PointCloud()
    T_current = np.eye(4)  # Current pose in world frame
    poses = [T_current.copy()]

    prev_stamp = None
    map_cloud = o3d.geometry.PointCloud()
    map_update_interval = 5  # Update map every N frames

    t_start = time.time()

    for i, cloud_msg in enumerate(cloud_msgs):
        stamp = get_stamp(cloud_msg)
        scan = pointcloud2_to_o3d(cloud_msg)

        # Colorize from camera if available
        if do_color and len(scan.points) > 0:
            import cv2 as _cv2
            # M4a: match_nearest_image returns (idx, dt); the dt is
            # unused here (fast_process is a standalone script that
            # doesn't emit the scan_quality block).
            img_idx, _img_dt = match_nearest_image(stamp, image_timestamps)
            if img_idx is not None:
                _, compressed_bytes, _ = camera_imgs[img_idx]
                img_arr = np.frombuffer(compressed_bytes, dtype=np.uint8)
                image = _cv2.imdecode(img_arr, _cv2.IMREAD_COLOR)
                if image is not None:
                    xyz = np.asarray(scan.points)
                    colors = colorize_cloud(xyz, image, calib)
                    scan.colors = o3d.utility.Vector3dVector(colors)

        if len(scan.points) < 10:
            poses.append(T_current.copy())
            continue

        if prev_stamp is not None and len(map_cloud.points) > 0:
            dt = stamp - prev_stamp

            # Integrate gyro for rotation estimate
            mask = (imu_times >= prev_stamp) & (imu_times < stamp)
            if mask.any():
                avg_gyro = imu_gyro[mask].mean(axis=0)
                dtheta = avg_gyro * dt
                # Small angle rotation matrix
                from scipy.spatial.transform import Rotation
                dR = Rotation.from_rotvec(dtheta).as_matrix()
            else:
                dR = np.eye(3)

            # Build initial guess from IMU
            T_guess = T_current.copy()
            T_guess[:3, :3] = T_current[:3, :3] @ dR

            # Refine with ICP
            T_result, success = register_scan_to_map(scan, map_cloud, T_guess)
            if success:
                T_current = T_result
            else:
                T_current = T_guess
        else:
            # First frame
            pass

        # Transform scan to world frame and add to merged cloud
        scan_world = o3d.geometry.PointCloud(scan)
        scan_world.transform(T_current)
        merged += scan_world

        # Update local map periodically
        if i % map_update_interval == 0:
            map_cloud = merged.voxel_down_sample(0.05)

        poses.append(T_current.copy())
        prev_stamp = stamp

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t_start
            fps = (i + 1) / elapsed
            print(f"  Frame {i+1}/{len(cloud_msgs)} ({fps:.0f} fps, {len(merged.points)} pts)")

    elapsed = time.time() - t_start
    print(f"[DONE] Processed {len(cloud_msgs)} frames in {elapsed:.1f}s "
          f"({len(cloud_msgs)/elapsed:.0f} fps)")

    return merged, poses


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 fast_process.py /path/to/rosbag_dir /path/to/output_dir [/path/to/calibration_dir]")
        sys.exit(1)

    bag_dir = sys.argv[1]
    output_dir = sys.argv[2]
    calibration_dir = sys.argv[3] if len(sys.argv) > 3 else None
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  Fast Direct MCAP Processing (no ROS2 playback)")
    print("=" * 60)

    # Load calibration for color projection
    calib = None
    if calibration_dir:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from cloud_slam.colorizer import load_calibration, colorize_cloud, match_nearest_image
        intr = os.path.join(calibration_dir, "intrinsics.yaml")
        extr = os.path.join(calibration_dir, "extrinsics.yaml")
        if os.path.exists(intr) and os.path.exists(extr):
            calib = load_calibration(intr, extr)
            print(f"[INFO] Loaded camera calibration from {calibration_dir}")
            import cv2

    # Read all messages at once (instant — no playback delay)
    t0 = time.time()
    print("[INFO] Reading MCAP file...")
    cloud_msgs, imu_msgs = read_mcap_messages(bag_dir)
    t_read = time.time() - t0
    print(f"[INFO] Read {len(cloud_msgs)} clouds + {len(imu_msgs)} IMU in {t_read:.1f}s")

    # Read camera images if calibration provided
    camera_imgs = []
    if calib:
        from mcap_ros2.reader import read_ros2_messages as read_ros2
        bag_path = Path(bag_dir)
        for mcap_file in sorted(bag_path.glob("*.mcap")):
            for msg in read_ros2(str(mcap_file)):
                if msg.channel.topic == "/camera/image_raw/compressed":
                    ros_msg = msg.ros_msg
                    stamp = ros_msg.header.stamp.sec + ros_msg.header.stamp.nanosec * 1e-9
                    camera_imgs.append((stamp, bytes(ros_msg.data), ros_msg.format))
        camera_imgs.sort(key=lambda x: x[0])
        print(f"[INFO] Read {len(camera_imgs)} camera images")

    # Process
    merged, poses = process_with_imu_integration(
        cloud_msgs, imu_msgs,
        camera_imgs=camera_imgs if calib else None,
        calib=calib
    )

    # Downsample and clean
    print("[INFO] Downsampling at 5mm...")
    merged = merged.voxel_down_sample(0.005)
    print(f"  {len(merged.points)} points")

    print("[INFO] Removing outliers...")
    merged, _ = merged.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"  {len(merged.points)} points")

    # Save
    pcd_path = os.path.join(output_dir, "final_map_fast.pcd")
    ply_path = os.path.join(output_dir, "final_map_fast.ply")
    o3d.io.write_point_cloud(pcd_path, merged)
    o3d.io.write_point_cloud(ply_path, merged)

    bbox = merged.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    total_time = time.time() - t0

    print(f"\n{'='*60}")
    print(f"  DONE — Total: {total_time:.1f}s")
    print(f"  Points: {len(merged.points)}")
    print(f"  Bounding box: {extent[0]:.2f} x {extent[1]:.2f} x {extent[2]:.2f} m")
    print(f"  Saved: {pcd_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
