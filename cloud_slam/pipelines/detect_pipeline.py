"""
Combined SLAM + YOLOE 3D object detection pipeline.

Runs ICP+IMU SLAM with a per-frame callback that detects objects in camera
images, extracts frustum points from lidar, and tracks objects in 3D.
"""

import numpy as np

from cloud_slam.detector import YOLODetector
from cloud_slam.frustum import extract_frustum_points, filter_depth_mad, estimate_gravity
from cloud_slam.tracker_3d import ObjectTracker3D
from cloud_slam.spatial_memory import SpatialObjectMemory
from cloud_slam.pipelines import icp_imu_pipeline


def run(clouds, imus, images, calib, voxel_size=0.005, detector_config=None,
        leveling_mode="legacy"):
    """
    Run SLAM + object detection pipeline.

    Args:
        clouds: list of (timestamp, xyz, time_offsets)
        imus: list of (timestamp, gyro, acc)
        images: list of (timestamp, compressed_bytes, format)
        calib: dict from load_calibration()
        voxel_size: final point cloud resolution
        detector_config: dict of kwargs for YOLODetector (optional)
        leveling_mode: "legacy" (default) = leveling deferred to the caller
            (scripts/detect_and_slam.py does it after this function returns).
            "floor" = pre-level the merged cloud + tracker buffers via
            cloud_slam.leveling.level_by_floor BEFORE box refinement, so OBBs
            are fit in the already-leveled frame. The "legacy" default keeps
            the working IMU-gravity path completely untouched.

    Returns:
        merged: Open3D PointCloud (colored)
        poses: list of 4x4 poses
        objects: list of detected object dicts
        stats: dict with timing info
    """
    detector = YOLODetector(**(detector_config or {}))
    tracker = ObjectTracker3D()
    memory = SpatialObjectMemory()
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

            # Resolve canonical ID through spatial memory
            canonical_id = memory.lookup_or_create(
                det.track_id, center_world, det.class_name, det.confidence
            )
            resolved_class = memory.get_class(canonical_id)

            tracker.update(
                canonical_id, center_world, world_pts,
                resolved_class, frame_idx, det.confidence,
                gravity_up=gravity_up,
            )
            detection_count += 1

    # Run SLAM with detection callback
    merged, poses, stats = icp_imu_pipeline.run(
        clouds, imus, voxel_size=voxel_size,
        images=images, calib=calib,
        per_frame_callback=on_frame,
    )

    # Post-processing
    tracks_before_merge = len(tracker.objects)
    tracker.merge_fragmented_tracks()

    # Optional pre-leveling: rotate the merged cloud + tracker buffers so the
    # detected floor maps to +Z and (optionally) sits at Z=0. This keeps all
    # downstream OBB fitting in an already-leveled frame.
    #
    # Legacy mode is a no-op here; scripts/detect_and_slam.py does IMU-based
    # leveling AFTER refinement, matching the pre-existing behavior exactly.
    gravity_for_refine = None  # → refine_objects re-estimates from IMU (legacy)
    if leveling_mode == "floor":
        from cloud_slam.leveling import level_by_floor, apply_leveling_to_tracker
        R_level, z_shift, level_method = level_by_floor(merged, imus)
        print(f"[LEVEL] method={level_method}, z_shift={z_shift:+.3f}m")
        if not np.allclose(R_level, np.eye(3), atol=1e-6):
            merged.rotate(R_level, center=(0.0, 0.0, 0.0))
        if abs(z_shift) > 1e-6:
            merged.translate((0.0, 0.0, -z_shift))
        apply_leveling_to_tracker(tracker, R_level, z_shift)
        stats['leveling_mode'] = 'floor'
        stats['leveling_method'] = level_method
        stats['leveling_z_shift'] = float(z_shift)
        gravity_for_refine = np.array([0.0, 0.0, 1.0])
    else:
        stats['leveling_mode'] = 'legacy'

    objects_raw = tracker.get_final_objects(gravity_up=gravity_up)

    # RoomPlan-style refinement
    from cloud_slam.box_refiner import refine_objects
    objects = refine_objects(merged, objects_raw, imus, tracker,
                             gravity_up=gravity_for_refine)

    stats['detection_enabled'] = True
    stats['total_detections'] = detection_count
    stats['objects_raw'] = len(objects_raw)
    stats['objects_refined'] = len(objects)
    stats['tracks_before_merge'] = tracks_before_merge

    return merged, poses, objects, stats
