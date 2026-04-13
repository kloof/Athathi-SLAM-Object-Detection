"""
Combined SLAM + YOLOE 3D object detection pipeline.

Runs ICP+IMU SLAM with a per-frame callback that detects objects in camera
images, extracts frustum points from lidar, and tracks objects in 3D.

Optional vision-based wall labeling: when a `wall_segmenter` is passed, the
per-frame callback additionally runs ADE20K semantic segmentation on each
image, z-buffer-filters the in-frame lidar points, maps pixel classes to our
5-bucket id space (other/wall/window/door/glass), and accumulates
world-frame labeled points in a buffer returned alongside the normal outputs.
The YOLOE object-detection path is untouched.
"""

import numpy as np

from cloud_slam.detector import YOLODetector
from cloud_slam.frustum import extract_frustum_points, filter_depth_mad, estimate_gravity
from cloud_slam.tracker_3d import ObjectTracker3D
from cloud_slam.spatial_memory import SpatialObjectMemory
from cloud_slam.pipelines import icp_imu_pipeline
from cloud_slam.projection import project_lidar_to_camera


def _z_buffer_visible(pts_cam, pixels, image_shape):
    """Per-pixel nearest-point visibility mask (z-buffer occlusion check).

    Lidar often returns points that — if projected into the camera — would
    fall on the same pixel but be occluded by closer geometry. For each
    pixel we keep only the nearest point; the rest are rejected.

    Returns a boolean mask (N,) over the input points.
    """
    H, W = image_shape[:2]
    depths = pts_cam[:, 2]
    u = pixels[:, 0]
    v = pixels[:, 1]

    valid = (depths > 0) & np.isfinite(u) & np.isfinite(v)
    if not valid.any():
        return np.zeros(len(pts_cam), dtype=bool)

    ui = np.where(valid, u, 0).astype(np.int32)
    vi = np.where(valid, v, 0).astype(np.int32)
    valid &= (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    if not valid.any():
        return np.zeros(len(pts_cam), dtype=bool)

    idx_valid = np.where(valid)[0]
    flat = vi[idx_valid] * W + ui[idx_valid]
    d = depths[idx_valid]

    # Sort valid points by depth ascending; np.unique on the sorted keys
    # returns the FIRST occurrence per key — which is the nearest depth.
    order = np.argsort(d, kind='stable')
    _, first_idx = np.unique(flat[order], return_index=True)
    keep_orig = idx_valid[order[first_idx]]

    visible = np.zeros(len(pts_cam), dtype=bool)
    visible[keep_orig] = True
    return visible


def _accumulate_wall_labels(segmenter, xyz, pose, image, calib, buffer):
    """Segment one frame and append (world_xyz, bucket_id) to buffer.

    - Lidar points are in the lidar sensor frame.
    - `image` is a (H, W, 3) uint8 BGR array as emitted by cv2-decoded MCAP.
    - `pose` transforms lidar frame → world frame.
    - `buffer` is a dict {'xyz': list[ndarray], 'labels': list[ndarray]}.
      One append per call that yields at least one non-'other' bucket.
    """
    if image is None or len(xyz) == 0:
        return

    # Mask2Former expects RGB; cv2-decoded images are BGR. `[:, :, ::-1]`
    # alone would yield a view with a negative stride on the channel
    # axis, which AutoImageProcessor → PyTorch rejects ("At least one
    # stride in the given numpy array is negative, and tensors with
    # negative strides are not currently supported"). `ascontiguousarray`
    # copies into a new buffer with strictly positive strides.
    image_rgb = np.ascontiguousarray(image[:, :, ::-1])
    bucket_mask = segmenter.segment(image_rgb)
    if bucket_mask is None:
        return

    pts_cam, pixels, _in_front = project_lidar_to_camera(
        xyz.astype(np.float64), calib)

    visible = _z_buffer_visible(pts_cam, pixels, image.shape)
    if not visible.any():
        return

    ui = pixels[visible, 0].astype(np.int32)
    vi = pixels[visible, 1].astype(np.int32)
    bucket_ids = bucket_mask[vi, ui]  # (K,)

    # Drop 'other' (0) — carries no signal for wall typing.
    non_other = bucket_ids > 0
    if not non_other.any():
        return

    idx_visible = np.where(visible)[0]
    idx_keep = idx_visible[non_other]

    xyz_lidar = xyz[idx_keep]
    R = pose[:3, :3]
    t = pose[:3, 3]
    xyz_world = (R @ xyz_lidar.T + t.reshape(3, 1)).T

    buffer['xyz'].append(xyz_world.astype(np.float32))
    buffer['labels'].append(bucket_ids[non_other].astype(np.uint8))


def run(clouds, imus, images, calib, voxel_size=0.005, detector_config=None,
        leveling_mode="legacy", wall_segmenter=None):
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
        wall_segmenter: optional `WallSegmenter` instance. When provided, each
            frame also runs semantic segmentation; pixel classes are mapped to
            5 buckets (other/wall/window/door/glass); visible lidar points
            are tagged and accumulated in the returned `wall_labels` dict.
            When None, this path is a no-op — YOLOE and D_refined are
            byte-identical to the pre-vision behavior.

    Returns:
        merged: Open3D PointCloud (colored)
        poses: list of 4x4 poses
        objects: list of detected object dicts
        stats: dict with timing info
        wall_labels: dict {'xyz': (M,3) float32, 'labels': (M,) uint8} of
            world-frame lidar points with their vision-assigned bucket id.
            Empty arrays when `wall_segmenter` is None.
    """
    detector = YOLODetector(**(detector_config or {}))
    tracker = ObjectTracker3D()
    memory = SpatialObjectMemory()
    gravity_up = estimate_gravity(imus)
    detection_count = 0

    # Mutable outer-scope accumulator for vision-labeled points. The nested
    # on_frame closure appends into `wall_label_chunks`; after SLAM finishes
    # we concatenate into a single array. Keeping it as a list-of-arrays
    # avoids repeated concat/growing across frames.
    wall_label_chunks = {'xyz': [], 'labels': []}

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

        # Optional vision-based wall labeling — only runs if the segmenter
        # was passed. Completely independent of YOLOE; never modifies the
        # object-tracker state.
        if wall_segmenter is not None:
            _accumulate_wall_labels(
                wall_segmenter, xyz, pose, image, calib, wall_label_chunks)

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

    # Flatten the per-frame vision label chunks into a single pair of
    # arrays. When wall_segmenter was None this stays empty, and the
    # downstream floorplan path sees no labels → zero regression.
    if wall_label_chunks['xyz']:
        wall_labels = {
            'xyz': np.concatenate(wall_label_chunks['xyz'], axis=0),
            'labels': np.concatenate(wall_label_chunks['labels'], axis=0),
        }
    else:
        wall_labels = {
            'xyz': np.zeros((0, 3), dtype=np.float32),
            'labels': np.zeros((0,), dtype=np.uint8),
        }
    # M0a: carry the segmenter's cumulative ADE20K class histogram
    # alongside the bucket labels so the floorplan pipeline can vote on
    # `room.category` without re-running inference. Empty dict when
    # wall_segmenter is None — the floorplan then emits
    # "category": "unknown" / "category_source": "unavailable".
    if wall_segmenter is not None:
        wall_labels['ade_class_counts'] = wall_segmenter.get_ade_class_counts()
    else:
        wall_labels['ade_class_counts'] = {}
    stats['wall_labels_count'] = int(wall_labels['labels'].shape[0])

    return merged, poses, objects, stats, wall_labels
