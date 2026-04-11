"""
Combined SLAM + YOLOE 3D object detection pipeline.

Runs ICP+IMU SLAM with a per-frame callback that detects objects in camera
images, extracts frustum points from lidar, and tracks objects in 3D.
"""

import numpy as np

from cloud_slam.detector import YOLODetector
from cloud_slam.frustum import extract_frustum_points, filter_depth_mad, estimate_gravity
from cloud_slam.tracker_3d import ObjectTracker3D
from cloud_slam.pipelines import icp_imu_pipeline


def run(clouds, imus, images, calib, voxel_size=0.005, detector_config=None):
    """
    Run SLAM + object detection pipeline.

    Args:
        clouds: list of (timestamp, xyz, time_offsets)
        imus: list of (timestamp, gyro, acc)
        images: list of (timestamp, compressed_bytes, format)
        calib: dict from load_calibration()
        voxel_size: final point cloud resolution
        detector_config: dict of kwargs for YOLODetector (optional)

    Returns:
        merged: Open3D PointCloud (colored)
        poses: list of 4x4 poses
        objects: list of detected object dicts
        stats: dict with timing info
    """
    detector = YOLODetector(**(detector_config or {}))
    tracker = ObjectTracker3D()
    gravity_up = estimate_gravity(imus)
    detection_count = 0

    def on_frame(frame_idx, stamp, xyz, pose, image):
        nonlocal detection_count
        if image is None:
            return

        detections = detector.detect_and_track(image)
        for det in detections:
            indices = extract_frustum_points(xyz, det, calib)
            if len(indices) < 3:
                continue

            frustum_pts = xyz[indices]
            frustum_pts = filter_depth_mad(frustum_pts)
            if len(frustum_pts) < 3:
                continue

            # Transform to world frame
            world_pts = (pose[:3, :3] @ frustum_pts.T + pose[:3, 3:4]).T
            center_world = world_pts.mean(axis=0)

            tracker.update(
                det.track_id, center_world, world_pts,
                det.class_name, frame_idx,
            )
            detection_count += 1

    # Run SLAM with detection callback
    merged, poses, stats = icp_imu_pipeline.run(
        clouds, imus, voxel_size=voxel_size,
        images=images, calib=calib,
        per_frame_callback=on_frame,
    )

    # Post-processing
    tracker.merge_fragmented_tracks()
    objects_raw = tracker.get_final_objects(gravity_up=gravity_up)

    # RoomPlan-style refinement
    from cloud_slam.box_refiner import refine_objects
    objects = refine_objects(merged, objects_raw, imus, tracker)

    stats['detection_enabled'] = True
    stats['total_detections'] = detection_count
    stats['objects_raw'] = len(objects_raw)
    stats['objects_refined'] = len(objects)
    stats['tracks_before_merge'] = len(tracker.objects)

    return merged, poses, objects, stats
