"""
RoomPlan-style object refinement pipeline.

Orchestrates room structure detection, Manhattan alignment, size priors,
and constraint-based post-processing to produce clean 3D bounding boxes.
"""

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from cloud_slam.room_structure import detect_room
from cloud_slam.manhattan import estimate_manhattan_frame, fit_manhattan_obb
from cloud_slam.frustum import estimate_gravity, fit_gravity_aligned_obb
from cloud_slam.size_priors import (
    assign_dimensions, refine_dimensions, validate_dimensions,
    is_floor_contact, is_wall_adjacent,
)


def refine_objects(merged_pcd, objects_raw, imus, tracker):
    """
    Full RoomPlan-style object refinement pipeline.

    Args:
        merged_pcd: Open3D PointCloud (the colored SLAM map)
        objects_raw: list of raw object dicts from tracker.get_final_objects()
        imus: list of (timestamp, gyro, acc) tuples
        tracker: ObjectTracker3D with per-object point buffers

    Returns:
        objects_refined: list of refined object dicts
    """
    gravity_up = estimate_gravity(imus)

    # Step 1: Room structure detection
    print("[REFINE] Detecting room structure...")
    room = detect_room(merged_pcd, gravity_up)
    print(f"[REFINE] Found floor (h={room.floor_height:.2f}m), "
          f"{len(room.walls)} walls, "
          f"ceiling={'yes' if room.ceiling else 'no'}")

    # Step 2: Manhattan frame
    manhattan = estimate_manhattan_frame(room.walls, gravity_up)
    print(f"[REFINE] Manhattan frame confidence: {manhattan.confidence:.2f}")

    use_manhattan = manhattan.confidence > 0.3

    # Step 3: Per-object refinement
    refined = []
    rejected = 0

    for obj in objects_raw:
        track_id = obj.get('track_id', 0)
        class_name = obj.get('class', 'unknown')

        # Get accumulated points
        pts = None
        if tracker and track_id in tracker.objects:
            pts = tracker.objects[track_id].get_accumulated_points()

        if pts is None or len(pts) < 10:
            continue

        # Skip low-observation tracks (likely noise)
        obs_count = obj.get('num_observations', 0)
        if obs_count < 8:
            rejected += 1
            continue

        # 3a. DBSCAN pre-filter: remove mixed-in points from other objects
        pts = _dbscan_filter(pts)
        if len(pts) < 10:
            continue

        # 3b. Fit OBB (Manhattan-aligned or gravity-only fallback)
        if use_manhattan:
            obb = fit_manhattan_obb(pts, manhattan)
        else:
            obb = fit_gravity_aligned_obb(pts, gravity_up)

        if obb is None:
            continue

        # 3c. Assign width/depth/height from OBB dimensions
        # If width < depth, swap and rotate quaternion 90° to compensate
        raw_dims = obb['dimensions'].copy()
        quat = obb['rotation_quat_xyzw'].copy()

        if raw_dims[1] > raw_dims[0]:
            # Swap width/depth and rotate quaternion 90° around Z
            ordered_dims = np.array([raw_dims[1], raw_dims[0], raw_dims[2]])
            R_box = Rotation.from_quat(quat).as_matrix()
            R_swap = Rotation.from_euler('z', np.pi / 2).as_matrix()
            quat = Rotation.from_matrix(R_box @ R_swap).as_quat()
        else:
            ordered_dims = np.array([raw_dims[0], raw_dims[1], raw_dims[2]])

        obb['rotation_quat_xyzw'] = quat

        # 3d. Bayesian size refinement
        num_points = obj.get('num_points', len(pts))
        conf = obj.get('confidence', 0.5)
        refined_dims, sigma_dev = refine_dimensions(
            ordered_dims, class_name, num_points, conf
        )

        # 3e. Validate
        is_valid, reason = validate_dimensions(refined_dims, class_name)
        if not is_valid:
            rejected += 1
            continue

        # Update object
        obj_refined = dict(obj)
        obj_refined['center'] = obb['center'].tolist()
        obj_refined['dimensions'] = refined_dims.tolist()
        obj_refined['orientation'] = {
            'quaternion': obb['rotation_quat_xyzw'].tolist(),
            'format': 'xyzw',
        }
        obj_refined['obb_confidence'] = obb['confidence']
        obj_refined['num_points'] = obb['num_points']
        obj_refined['raw_dimensions'] = raw_dims.tolist()
        obj_refined['sigma_deviation'] = sigma_dev.tolist()
        refined.append(obj_refined)

    print(f"[REFINE] {len(refined)} objects kept, {rejected} rejected")

    # Step 4: Global post-processing
    if room.floor is not None:
        refined = _snap_to_floor(refined, room.floor_height, gravity_up)

    if room.walls:
        refined = _snap_to_walls(refined, room.walls)

    refined = _enforce_consistent_heights(refined)
    refined = _resolve_intersections(refined)

    return refined


def _dbscan_filter(points, eps=0.10, min_points=3):
    """Keep only the largest DBSCAN cluster."""
    if len(points) < min_points * 2:
        return points

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))

    if labels.max() < 0:
        return points  # no clusters found, keep all

    counts = np.bincount(labels[labels >= 0])
    largest = np.argmax(counts)
    return points[labels == largest]


def _snap_to_floor(objects, floor_height, gravity_up, max_gap=0.20):
    """Snap floor-contact objects so bottom face touches floor."""
    for obj in objects:
        if not is_floor_contact(obj.get('class', '')):
            continue

        dims = np.array(obj['dimensions'])
        center = np.array(obj['center'])
        height = dims[2]

        # Project center onto gravity axis (not hardcoded Z)
        center_h = float(center @ gravity_up)
        current_bottom = center_h - height / 2

        gap = abs(current_bottom - floor_height)
        if gap < max_gap:
            target_h = floor_height + height / 2
            shift = target_h - center_h
            center += shift * gravity_up
            obj['center'] = center.tolist()

    return objects


def _snap_to_walls(objects, walls, threshold=0.15):
    """Snap wall-adjacent objects flush to nearest wall."""
    for obj in objects:
        if not is_wall_adjacent(obj.get('class', '')):
            continue

        center = np.array(obj['center'])
        dims = np.array(obj['dimensions'])
        depth = dims[1]

        for wall in walls:
            # Signed distance from center to wall plane
            dist = center @ wall.normal + wall.offset

            # Gap between object back face and wall
            gap = abs(dist) - depth / 2

            if 0 < gap < threshold:
                # Snap toward wall
                direction = -np.sign(dist) * wall.normal
                center += direction * gap
                obj['center'] = center.tolist()
                break  # snap to nearest wall only

    return objects


def _enforce_consistent_heights(objects):
    """Set height to median for classes with 3+ instances."""
    from collections import defaultdict
    by_class = defaultdict(list)
    for obj in objects:
        by_class[obj.get('class', '')].append(obj)

    for class_name, class_objs in by_class.items():
        if len(class_objs) < 3:
            continue

        heights = [np.array(o['dimensions'])[2] for o in class_objs]
        median_h = float(np.median(heights))

        for obj in class_objs:
            dims = list(obj['dimensions'])
            if abs(dims[2] - median_h) / max(median_h, 0.01) > 0.3:
                dims[2] = median_h
                obj['dimensions'] = dims

    return objects


def _resolve_intersections(objects, iou_threshold=0.15, max_iterations=3):
    """Remove or shrink overlapping boxes."""
    for _ in range(max_iterations):
        changed = False
        objects.sort(key=lambda o: -o.get('confidence', 0.5))
        keep = [True] * len(objects)

        for i in range(len(objects)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(objects)):
                if not keep[j]:
                    continue

                iou = _aabb_iou(objects[i], objects[j])
                if iou > 0.5:
                    keep[j] = False
                    changed = True
                elif iou > iou_threshold:
                    # Shrink lower-confidence box
                    changed = True

        objects = [o for o, k in zip(objects, keep) if k]
        if not changed:
            break

    return objects


def _aabb_iou(a, b):
    """Approximate AABB IoU between two objects."""
    a_c = np.array(a['center'])
    a_d = np.array(a['dimensions'])
    b_c = np.array(b['center'])
    b_d = np.array(b['dimensions'])

    a_min, a_max = a_c - a_d / 2, a_c + a_d / 2
    b_min, b_max = b_c - b_d / 2, b_c + b_d / 2

    inter_min = np.maximum(a_min, b_min)
    inter_max = np.minimum(a_max, b_max)
    inter = np.maximum(0, inter_max - inter_min)
    inter_vol = np.prod(inter)

    a_vol = np.prod(a_d)
    b_vol = np.prod(b_d)
    union_vol = a_vol + b_vol - inter_vol

    return inter_vol / max(union_vol, 1e-10)
