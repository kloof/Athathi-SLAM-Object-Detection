"""
Open3D point-to-plane ICP + IMU gyro integration.
This is the pipeline that produced the best visual results (final_map_direct_icp).
"""

import time
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation


def register_scan_to_map(scan, map_cloud, T_init, voxel_size=0.1):
    """Register a scan to the existing map using point-to-plane ICP."""
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


def run(clouds, imus, voxel_size=0.005, images=None, calib=None):
    """
    Process point cloud frames with ICP + IMU.

    Args:
        clouds: list of (timestamp, xyz_Nx3_float64, time_offsets_N_float64)
        imus: list of (timestamp, gyro_xyz_3, acc_xyz_3)
        voxel_size: final downsample resolution in meters
        images: list of (timestamp, compressed_bytes, format_str) or None
        calib: dict from colorizer.load_calibration() or None

    Returns:
        merged: Open3D PointCloud (with colors if images+calib provided)
        poses: list of 4x4 numpy arrays
        stats: dict with timing info
    """
    t0 = time.time()

    imu_times = np.array([t for t, _, _ in imus])
    imu_gyro = np.array([g for _, g, _ in imus])

    # Build image timestamp index for color projection
    do_color = images is not None and calib is not None and len(images) > 0
    if do_color:
        import cv2
        from cloud_slam.colorizer import colorize_cloud, match_nearest_image
        image_timestamps = np.array([t for t, _, _ in images])
        color_match_count = 0

    merged = o3d.geometry.PointCloud()
    T_current = np.eye(4)
    poses = [T_current.copy()]

    prev_stamp = None
    map_cloud = o3d.geometry.PointCloud()
    map_update_interval = 5

    for i, (stamp, xyz, timestamps) in enumerate(clouds):
        scan = o3d.geometry.PointCloud()
        scan.points = o3d.utility.Vector3dVector(xyz)

        # Colorize from camera if available
        if do_color:
            img_idx = match_nearest_image(stamp, image_timestamps)
            if img_idx is not None:
                _, compressed_bytes, _ = images[img_idx]
                img_arr = np.frombuffer(compressed_bytes, dtype=np.uint8)
                image = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                if image is not None:
                    colors = colorize_cloud(xyz, image, calib)
                    scan.colors = o3d.utility.Vector3dVector(colors)
                    color_match_count += 1

        if len(scan.points) < 10:
            poses.append(T_current.copy())
            continue

        if prev_stamp is not None and len(map_cloud.points) > 0:
            dt = stamp - prev_stamp

            # IMU gyro integration for rotation initial guess
            mask = (imu_times >= prev_stamp) & (imu_times < stamp)
            if mask.any():
                avg_gyro = imu_gyro[mask].mean(axis=0)
                dtheta = avg_gyro * dt
                dR = Rotation.from_rotvec(dtheta).as_matrix()
            else:
                dR = np.eye(3)

            T_guess = T_current.copy()
            T_guess[:3, :3] = T_current[:3, :3] @ dR

            # Refine with ICP
            T_result, success = register_scan_to_map(scan, map_cloud, T_guess)
            if success:
                T_current = T_result
            else:
                T_current = T_guess

        # Transform scan to world frame and accumulate
        scan_world = o3d.geometry.PointCloud(scan)
        scan_world.transform(T_current)
        merged += scan_world

        # Update local map periodically
        if i % map_update_interval == 0:
            map_cloud = merged.voxel_down_sample(0.05)

        poses.append(T_current.copy())
        prev_stamp = stamp

    t_slam = time.time() - t0

    # Downsample and clean
    merged = merged.voxel_down_sample(voxel_size)
    merged, _ = merged.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    t_total = time.time() - t0

    stats = {
        "algorithm": "icp-imu",
        "slam_time_s": round(t_slam, 2),
        "total_time_s": round(t_total, 2),
        "num_frames": len(clouds),
        "num_points": len(merged.points),
        "colorized": do_color,
    }
    if do_color:
        stats["num_images"] = len(images)
        stats["color_matches"] = color_match_count

    bbox = merged.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    stats["bounding_box_m"] = f"{extent[0]:.2f} x {extent[1]:.2f} x {extent[2]:.2f}"

    return merged, poses, stats
