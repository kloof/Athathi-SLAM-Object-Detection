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
        if obs_count < 5:
            rejected += 1
            continue

        # 3a. Remove structural surface points (floor/wall/ceiling)
        n_before = len(pts)
        pts = _remove_structural_points(pts, room, class_name)
        if len(pts) < 10:
            continue

        # 3b. DBSCAN cluster selection (closest to tracked centroid)
        kalman_center = tracker.objects[track_id].x
        pts = _dbscan_filter(pts, kalman_center)
        if len(pts) < 10:
            continue

        # 3c. Fit OBB (Manhattan-aligned or gravity-only fallback)
        if use_manhattan:
            obb = fit_manhattan_obb(pts, manhattan)
        else:
            obb = fit_gravity_aligned_obb(pts, gravity_up)

        if obb is None:
            continue

        # 3d. Assign width/depth/height using gravity alignment
        # OBB contract: dims[i] extends along column i of R_box.
        # assign_dimensions reorders to [width, depth, height] for Bayesian priors,
        # then we map refined dims BACK to rotation-column order. Never touch quaternion.
        raw_dims = obb['dimensions'].copy()
        R_box = Rotation.from_quat(obb['rotation_quat_xyzw']).as_matrix()
        ordered_dims, axis_map = assign_dimensions(raw_dims, R_box, gravity_up)
        w_idx, d_idx, h_idx = axis_map

        # 3e. Bayesian size refinement (on semantically-ordered [width, depth, height])
        num_points = obj.get('num_points', len(pts))
        conf = obj.get('confidence', 0.5)
        refined_dims, sigma_dev = refine_dimensions(
            ordered_dims, class_name, num_points, conf
        )

        # 3f. Validate
        is_valid, reason = validate_dimensions(refined_dims, class_name)
        if not is_valid:
            rejected += 1
            continue

        # Map refined dims back to rotation-column order
        final_dims = np.empty(3)
        final_dims[w_idx] = refined_dims[0]   # width → its original axis
        final_dims[d_idx] = refined_dims[1]   # depth → its original axis
        final_dims[h_idx] = refined_dims[2]   # height → its original axis

        # Update object — quaternion is UNCHANGED from the OBB fitter
        obj_refined = dict(obj)
        obj_refined['center'] = obb['center'].tolist()
        obj_refined['dimensions'] = final_dims.tolist()
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

    # Re-anchor after constraint solving (shrinking can un-anchor from floor/walls)
    if room.floor is not None:
        refined = _snap_to_floor(refined, room.floor_height, gravity_up)
    if room.walls:
        refined = _snap_to_walls(refined, room.walls)

    return refined


def _fit_tracked_yaw_obb(points, gravity_up, tracked_yaw, manhattan=None):
    """
    Fit OBB using the tracker's smoothed yaw orientation.

    Uses accumulated yaw from per-frame PCA observations, snapped to the
    nearest Manhattan direction if a Manhattan frame is available.
    This produces more stable orientations than fitting fresh each time.
    """
    if len(points) < 3:
        return None

    # Build gravity-aligned rotation: gravity→Z
    z_axis = np.array([0.0, 0.0, 1.0])
    g = gravity_up / np.linalg.norm(gravity_up)
    if np.allclose(g, z_axis, atol=0.01):
        R_grav = np.eye(3)
    elif np.allclose(g, -z_axis, atol=0.01):
        R_grav = np.diag([1.0, -1.0, -1.0])
    else:
        R_grav = Rotation.align_vectors([z_axis], [g])[0].as_matrix()

    # Snap tracked yaw to nearest Manhattan direction (0° or 90° in Manhattan frame)
    # CRITICAL: manhattan_yaw must be in the GRAVITY-ALIGNED frame (same as tracked_yaw)
    if manhattan is not None:
        R_m = manhattan.R  # world → Manhattan
        manhattan_x_world = R_m.T[:, 0]  # Manhattan X direction in raw world frame
        manhattan_x_grav = R_grav @ manhattan_x_world  # transform to gravity-aligned frame
        manhattan_yaw = np.arctan2(manhattan_x_grav[1], manhattan_x_grav[0])
        # Snap tracked_yaw to nearest multiple of 90° relative to Manhattan
        delta = tracked_yaw - manhattan_yaw
        snapped_delta = round(delta / (np.pi / 2)) * (np.pi / 2)
        yaw = manhattan_yaw + snapped_delta
    else:
        yaw = tracked_yaw

    # Rotate points so gravity→Z
    pts_aligned = (R_grav @ points.T).T

    # Apply yaw rotation: rotate points by -yaw so OBB is axis-aligned
    c, s = np.cos(-yaw), np.sin(-yaw)
    R_yaw_inv = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    pts_rotated = (R_yaw_inv @ pts_aligned.T).T

    # AABB in the rotated frame
    mins = pts_rotated.min(axis=0)
    maxs = pts_rotated.max(axis=0)
    dims = np.maximum(maxs - mins, 0.02)
    center_rotated = (mins + maxs) / 2

    # Undo rotations to get center in world frame
    R_yaw = Rotation.from_euler('z', yaw).as_matrix()
    center_aligned = R_yaw @ center_rotated
    center_world = R_grav.T @ center_aligned

    # Build final rotation: OBB local → world
    R_full = R_grav.T @ R_yaw
    if np.linalg.det(R_full) < 0:
        R_full[:, 2] *= -1
    quat = Rotation.from_matrix(R_full).as_quat()

    return {
        'center': center_world,
        'dimensions': dims,
        'rotation_quat_xyzw': quat,
        'confidence': 'high' if len(points) >= 30 else 'medium',
        'num_points': len(points),
    }


def _remove_structural_points(pts, room, class_name):
    """Remove points that lie on detected floor, wall, or ceiling planes.

    Doors are exempted because they ARE wall surfaces (depth prior ~5cm).
    Wall-adjacent classes use a tighter wall threshold to preserve their back face.
    """
    if class_name == "door":
        return pts

    structural = np.zeros(len(pts), dtype=bool)

    # Floor filtering (5cm threshold)
    if room.floor is not None:
        dists = np.abs(pts @ room.floor.normal + room.floor.offset)
        structural |= dists < 0.05

    # Ceiling filtering (5cm threshold)
    if room.ceiling is not None:
        dists = np.abs(pts @ room.ceiling.normal + room.ceiling.offset)
        structural |= dists < 0.05

    # Wall filtering (class-specific threshold)
    wall_thresh = 0.05 if class_name in ("shelf", "desk", "monitor") else 0.07
    for wall in room.walls:
        dists = np.abs(pts @ wall.normal + wall.offset)
        structural |= dists < wall_thresh

    filtered = pts[~structural]

    # Safety: if too few points survive, return original (priors will constrain)
    if len(filtered) < 10:
        return pts

    return filtered


def _dbscan_filter(points, kalman_center, eps=0.05, min_points=3):
    """Keep the DBSCAN cluster closest to the tracked object centroid."""
    if len(points) < min_points * 2:
        return points

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))

    if labels.max() < 0:
        return points  # no clusters found, keep all

    n_clusters = labels.max() + 1
    if n_clusters == 1:
        return points[labels == 0]

    # Select cluster whose centroid is closest to the Kalman position estimate
    best_label = 0
    best_dist = np.inf
    best_count = 0
    for lbl in range(n_clusters):
        mask = labels == lbl
        centroid = points[mask].mean(axis=0)
        dist = np.linalg.norm(centroid - kalman_center)
        count = mask.sum()
        if dist < best_dist - 0.05 or (abs(dist - best_dist) <= 0.05 and count > best_count):
            best_dist = dist
            best_label = lbl
            best_count = count

    return points[labels == best_label]


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

        # Project box half-extents onto wall normal to get true face distance
        quat = obj.get('orientation', {}).get('quaternion')
        if quat is not None:
            R_box = Rotation.from_quat(quat).as_matrix()
            half_extents = np.array(dims) / 2
            face_dist = float(np.abs(half_extents @ np.abs(R_box.T @ walls[0].normal)))
        else:
            face_dist = dims[1] / 2  # fallback to depth

        for wall in walls:
            # Signed distance from center to wall plane
            dist = center @ wall.normal + wall.offset

            # Recompute face distance for this specific wall normal
            if quat is not None:
                face_dist = float(np.abs(half_extents @ np.abs(R_box.T @ wall.normal)))

            # Gap between object back face and wall
            gap = abs(dist) - face_dist

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
                    _shrink_to_resolve(objects[i], objects[j])
                    changed = True

        objects = [o for o, k in zip(objects, keep) if k]
        if not changed:
            break

    return objects


def _shrink_to_resolve(dominant, recessive):
    """Shrink the recessive box along the axis of maximum overlap."""
    d_c = np.array(dominant['center'])
    d_d = np.array(dominant['dimensions'])
    r_c = np.array(recessive['center'])
    r_d = np.array(recessive['dimensions'])

    # Find overlap per axis
    for axis in range(3):
        d_lo, d_hi = d_c[axis] - d_d[axis] / 2, d_c[axis] + d_d[axis] / 2
        r_lo, r_hi = r_c[axis] - r_d[axis] / 2, r_c[axis] + r_d[axis] / 2
        overlap = max(0, min(d_hi, r_hi) - max(d_lo, r_lo))
        if overlap <= 0:
            continue

        # Shift recessive center away from dominant and trim extent
        direction = 1.0 if r_c[axis] >= d_c[axis] else -1.0
        shift = overlap / 2
        r_c[axis] += direction * shift
        r_d[axis] = max(r_d[axis] - shift, 0.02)

    recessive['center'] = r_c.tolist()
    recessive['dimensions'] = r_d.tolist()


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
