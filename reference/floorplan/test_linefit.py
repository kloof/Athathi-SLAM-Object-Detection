"""
Test: Fit least-squares lines to contour segments between Douglas-Peucker corners.
This gives us straight walls with precise angles from the actual edge data.

Pipeline:
1. Ceiling mask → contour
2. Douglas-Peucker → approximate corner INDICES on the contour
3. For each edge segment: collect contour points → fit least-squares line
4. Re-intersect consecutive fitted lines → clean corners
5. Result: straight walls, precise measurements, no angle snap needed
"""
import matplotlib
matplotlib.use('Agg')
import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes, gaussian_filter1d
from scipy.signal import find_peaks
from shapely.geometry import Polygon as ShapelyPolygon
import os

OUT = "C:/Users/klof/Desktop/SLAM_test/floorplan/floorplan_tool/output"

def fit_line_to_points(points):
    """Fit a robust line to 2D points using Huber M-estimator.
    Handles outliers automatically (furniture, noise near walls).
    Returns (point_on_line, direction_unit_vector)."""
    pts = points.astype(np.float32).reshape(-1, 1, 2)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
    return np.array([x0, y0], dtype=np.float64), np.array([vx, vy], dtype=np.float64)

def line_intersection(p1, d1, p2, d2):
    """Intersect two lines: p1 + t*d1 and p2 + s*d2. Returns intersection point."""
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-10:
        return None  # parallel
    dp = p2 - p1
    t = (dp[0] * d2[1] - dp[1] * d2[0]) / denom
    return p1 + t * d1

def process(ply_path, name):
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")

    # Load + preprocess
    pcd = o3d.io.read_point_cloud(ply_path)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.voxel_down_sample(voxel_size=0.03)
    pts = np.asarray(pcd.points)

    # Floor/ceiling
    z = pts[:, 2]
    bins = np.arange(z.min(), z.max() + 0.02, 0.02)
    hist, edges = np.histogram(z, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    zhist = gaussian_filter1d(hist.astype(float), sigma=3)
    pks, props = find_peaks(zhist, height=np.max(zhist) * 0.1, distance=10)
    top2 = pks[np.argsort(props['peak_heights'])[-2:]]
    top2 = np.sort(top2)
    floor_z, ceiling_z = centers[top2[0]], centers[top2[1]]
    h = ceiling_z - floor_z

    # Ceiling mask — use upper 30% of room height to catch double ceilings
    z_cutoff = floor_z + h * 0.70
    ceil_mask = (pts[:, 2] > z_cutoff) & (pts[:, 2] < ceiling_z + 0.12)
    ceil_xy = pts[ceil_mask][:, :2]

    RES = 0.03
    x_min, y_min = ceil_xy.min(axis=0) - 0.5
    x_max, y_max = ceil_xy.max(axis=0) + 0.5
    nx = int((x_max - x_min) / RES)
    ny = int((y_max - y_min) / RES)
    grid, xe, ye = np.histogram2d(ceil_xy[:, 0], ceil_xy[:, 1],
                                   bins=[nx, ny], range=[[x_min, x_max], [y_min, y_max]])
    ext = [xe[0], xe[-1], ye[0], ye[-1]]

    if np.any(grid > 0):
        cap = np.percentile(grid[grid > 0], 90)
        g8 = (np.clip(grid, 0, cap) / cap * 255).astype(np.uint8)
    else:
        g8 = np.zeros((nx, ny), dtype=np.uint8)

    _, binary = cv2.threshold(g8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    binary = (binary_fill_holes(binary > 0).astype(np.uint8)) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # Remove outlier clusters
    n_labels, labeled, stats, _ = cv2.connectedComponentsWithStats(binary)
    if n_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        biggest = np.argmax(areas) + 1
        clean = np.zeros_like(binary)
        clean[labeled == biggest] = 255
    else:
        clean = binary
    clean = cv2.erode(clean, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    clean = cv2.dilate(clean, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    # ============================================================
    # STEP 1: Get contour + Douglas-Peucker corners
    # ============================================================
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    contour_pts = contour.reshape(-1, 2)  # (N, 2) in pixel coords (col, row)

    perimeter = cv2.arcLength(contour, True)
    epsilon = 0.012 * perimeter
    simplified = cv2.approxPolyDP(contour, epsilon, True)
    corner_pts = simplified.reshape(-1, 2)  # approximate corner positions

    # Find the INDEX of each corner in the full contour
    corner_indices = []
    for cp in corner_pts:
        dists = np.linalg.norm(contour_pts - cp, axis=1)
        corner_indices.append(np.argmin(dists))
    corner_indices = sorted(corner_indices)

    n_corners = len(corner_indices)
    n_contour = len(contour_pts)
    print(f"Contour: {n_contour} pts, {n_corners} corners detected")

    # ============================================================
    # STEP 2: For each edge segment, fit a least-squares line
    # ============================================================
    fitted_lines = []  # (point_on_line, direction, n_points, segment_points)

    for i in range(n_corners):
        idx_start = corner_indices[i]
        idx_end = corner_indices[(i + 1) % n_corners]

        # Collect contour points for this edge segment
        if idx_end > idx_start:
            segment = contour_pts[idx_start:idx_end + 1]
        else:
            # Wraps around
            segment = np.vstack([contour_pts[idx_start:], contour_pts[:idx_end + 1]])

        if len(segment) < 3:
            continue

        # Convert to real-world coords (col, row) -> (x, y)
        seg_real = np.zeros_like(segment, dtype=float)
        seg_real[:, 0] = x_min + segment[:, 1] * RES  # row -> x
        seg_real[:, 1] = y_min + segment[:, 0] * RES  # col -> y

        # Fit line
        point, direction = fit_line_to_points(seg_real)
        fitted_lines.append((point, direction, len(segment), seg_real))

    print(f"Fitted {len(fitted_lines)} line segments")

    # ============================================================
    # STEP 3: Re-intersect consecutive fitted lines → corners
    # ============================================================
    corners = []
    for i in range(len(fitted_lines)):
        p1, d1, _, _ = fitted_lines[i]
        p2, d2, _, _ = fitted_lines[(i + 1) % len(fitted_lines)]
        pt = line_intersection(p1, d1, p2, d2)
        if pt is not None:
            corners.append(pt)
        else:
            # Parallel — use midpoint between line endpoints
            corners.append((p1 + p2) / 2)

    corners = np.array(corners)
    print(f"Corners after re-intersection: {len(corners)}")

    # Build polygon
    poly = ShapelyPolygon(corners)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.geom_type == 'MultiPolygon':
        poly = max(poly.geoms, key=lambda g: g.area)

    # Extract walls
    coords = np.array(poly.exterior.coords)
    walls = []
    for i in range(len(coords) - 1):
        p1, p2 = coords[i], coords[i + 1]
        length = np.linalg.norm(p2 - p1)
        angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) * 180 / np.pi % 180
        if length > 0.15:
            walls.append((p1, p2, angle, length))

    print(f"Room (line-fit): {poly.area:.1f} m2, {len(walls)} walls")

    # ============================================================
    # STEP 4: Per-wall RANSAC refinement
    # For each wall, grab nearby points, refit line, allow small angle change
    # ============================================================
    print("\nStep 4: Per-wall refinement...")

    from matplotlib.path import Path as MplPath

    # Wall-height points only (skip floor/ceiling)
    z_lo = floor_z + h * 0.20
    z_hi = floor_z + h * 0.80
    wall_height_mask = (pts[:, 2] > z_lo) & (pts[:, 2] < z_hi)
    pts_wall_height = pts[wall_height_mask][:, :2]

    MAX_ANGLE_CHANGE = 3.0  # degrees
    BAND_WIDTH = 0.30  # meters — how far from the wall line to grab points

    refined_lines = []  # (point_on_line, direction, angle, length)
    n_refined = 0

    for p1, p2, angle, length in walls:
        mid = (p1 + p2) / 2
        direction = p2 - p1
        norm_d = np.linalg.norm(direction)
        if norm_d < 1e-6:
            refined_lines.append((mid, np.array([1, 0]), angle, length))
            continue
        direction /= norm_d
        wall_normal = np.array([-direction[1], direction[0]])

        # Select points within a narrow band along this wall
        # 1. Perpendicular distance < BAND_WIDTH
        vecs = pts_wall_height - p1
        perp_dists = np.abs(vecs @ wall_normal)
        along_dists = vecs @ direction

        in_band = (perp_dists < BAND_WIDTH) & \
                  (along_dists > -0.3) & (along_dists < length + 0.3)
        band_pts = pts_wall_height[in_band]

        if len(band_pts) < 10:
            # Not enough points — keep original
            refined_lines.append((mid, direction, angle, length))
            continue

        # Refit line to these points
        new_point, new_dir = fit_line_to_points(band_pts)
        new_angle = np.arctan2(new_dir[1], new_dir[0]) * 180 / np.pi % 180

        # Clamp angle change
        angle_diff = new_angle - angle
        if angle_diff > 90: angle_diff -= 180
        elif angle_diff < -90: angle_diff += 180
        clamped_diff = np.clip(angle_diff, -MAX_ANGLE_CHANGE, MAX_ANGLE_CHANGE)
        final_angle = angle + clamped_diff

        # Use the refitted line's position (perpendicular shift)
        final_rad = final_angle * np.pi / 180
        final_dir = np.array([np.cos(final_rad), np.sin(final_rad)])

        # Perpendicular shift: project new_point onto the old normal
        perp_shift = np.dot(new_point - mid, wall_normal)
        perp_shift = np.clip(perp_shift, -0.15, 0.15)
        final_mid = mid + wall_normal * perp_shift

        refined_lines.append((final_mid, final_dir, final_angle, length))
        n_refined += 1

        if abs(clamped_diff) > 0.1 or abs(perp_shift) > 0.01:
            print(f"  wall {length:.2f}m: angle {angle:.1f} -> {final_angle:.1f} "
                  f"({clamped_diff:+.1f}), shift {perp_shift*100:+.1f}cm "
                  f"({len(band_pts)} pts)")

    print(f"  Refined {n_refined}/{len(walls)} walls")

    # Re-intersect corners
    r_corners = []
    for i in range(len(refined_lines)):
        m1, d1, _, _ = refined_lines[i]
        m2, d2, _, _ = refined_lines[(i + 1) % len(refined_lines)]
        pt = line_intersection(m1, d1, m2, d2)
        if pt is not None:
            r_corners.append(pt)
        else:
            r_corners.append((m1 + m2) / 2)

    r_corners = np.array(r_corners)
    poly_refined = ShapelyPolygon(r_corners)
    if not poly_refined.is_valid:
        poly_refined = poly_refined.buffer(0)
    if poly_refined.geom_type == 'MultiPolygon':
        poly_refined = max(poly_refined.geoms, key=lambda g: g.area)

    walls_refined = []
    rc = np.array(poly_refined.exterior.coords)
    for i in range(len(rc) - 1):
        rp1, rp2 = rc[i], rc[i + 1]
        rl = np.linalg.norm(rp2 - rp1)
        ra = np.arctan2(rp2[1] - rp1[1], rp2[0] - rp1[0]) * 180 / np.pi % 180
        if rl > 0.15:
            walls_refined.append((rp1, rp2, ra, rl))

    print(f"Room (refined): {poly_refined.area:.1f} m2, {len(walls_refined)} walls")

    # ============================================================
    # STEP 5: Micro-merge — merge adjacent walls within 15 degrees
    # ============================================================
    print("\nStep 5: Micro-merge (adjacent walls <15 deg)...")

    MERGE_ANGLE_THRESH = 15.0  # degrees

    def try_merge_adjacent(wall_list):
        """One pass: merge consecutive walls that are nearly parallel."""
        if len(wall_list) < 3:
            return wall_list, False

        merged = []
        changed = False
        i = 0
        while i < len(wall_list):
            p1, p2, a1, l1 = wall_list[i]
            j = (i + 1) % len(wall_list)
            p1b, p2b, a2, l2 = wall_list[j]

            angle_diff = min(abs(a1 - a2), 180 - abs(a1 - a2))

            if angle_diff < MERGE_ANGLE_THRESH and j != 0:
                # Merge: fit one line through both segments' contour band
                combined_start = p1
                combined_end = p2b
                new_mid = (combined_start + combined_end) / 2
                new_dir = combined_end - combined_start
                new_len = np.linalg.norm(new_dir)
                if new_len > 0:
                    new_angle = np.arctan2(new_dir[1], new_dir[0]) * 180 / np.pi % 180
                    merged.append((combined_start, combined_end, new_angle, new_len))
                    print(f"  merged: {l1:.2f}m ({a1:.1f}) + {l2:.2f}m ({a2:.1f}) "
                          f"-> {new_len:.2f}m ({new_angle:.1f}), diff was {angle_diff:.1f} deg")
                    changed = True
                    i += 2
                    continue

            merged.append(wall_list[i])
            i += 1

        # Handle wraparound: check first and last
        if len(merged) >= 2:
            p1a, p2a, a1, l1 = merged[-1]
            p1b, p2b, a2, l2 = merged[0]
            angle_diff = min(abs(a1 - a2), 180 - abs(a1 - a2))
            if angle_diff < MERGE_ANGLE_THRESH:
                combined_start = p1a
                combined_end = p2b
                new_dir = combined_end - combined_start
                new_len = np.linalg.norm(new_dir)
                if new_len > 0:
                    new_angle = np.arctan2(new_dir[1], new_dir[0]) * 180 / np.pi % 180
                    merged[-1] = (combined_start, combined_end, new_angle, new_len)
                    merged.pop(0)
                    print(f"  merged wraparound: {l1:.2f}m + {l2:.2f}m -> {new_len:.2f}m")
                    changed = True

        return merged, changed

    # Iterate until no more merges
    current_walls = list(walls_refined)
    for _ in range(5):  # max 5 passes
        current_walls, did_merge = try_merge_adjacent(current_walls)
        if not did_merge:
            break

    # Rebuild polygon from merged walls using re-intersection
    if len(current_walls) >= 3 and len(current_walls) != len(walls_refined):
        m_edges = []
        for p1, p2, angle, length in current_walls:
            mid = (p1 + p2) / 2
            rad = angle * np.pi / 180
            d = np.array([np.cos(rad), np.sin(rad)])
            m_edges.append((mid, d, length, angle))

        m_corners = []
        for i in range(len(m_edges)):
            m1, d1, _, _ = m_edges[i]
            m2, d2, _, _ = m_edges[(i + 1) % len(m_edges)]
            pt = line_intersection(m1, d1, m2, d2)
            if pt is not None:
                m_corners.append(pt)
            else:
                m_corners.append((m1 + m2) / 2)

        m_corners = np.array(m_corners)
        poly_merged = ShapelyPolygon(m_corners)
        if not poly_merged.is_valid:
            poly_merged = poly_merged.buffer(0)
        if poly_merged.geom_type == 'MultiPolygon':
            poly_merged = max(poly_merged.geoms, key=lambda g: g.area)

        walls_final = []
        mc = np.array(poly_merged.exterior.coords)
        for i in range(len(mc) - 1):
            fp1, fp2 = mc[i], mc[i + 1]
            fl = np.linalg.norm(fp2 - fp1)
            fa = np.arctan2(fp2[1] - fp1[1], fp2[0] - fp1[0]) * 180 / np.pi % 180
            if fl > 0.15:
                walls_final.append((fp1, fp2, fa, fl))

        poly_refined = poly_merged
        walls_refined = walls_final

    print(f"Room (after merge): {poly_refined.area:.1f} m2, {len(walls_refined)} walls")

    # ============================================================
    # STEP 6: Post-processing — fix geometric artifacts
    # ============================================================
    print("\nStep 6: Post-processing...")

    def interior_angle(w_prev, w_curr):
        """Compute interior angle at the junction between two consecutive walls."""
        d1 = w_prev[1] - w_prev[0]  # direction of previous wall (toward corner)
        d2 = w_curr[1] - w_curr[0]  # direction of current wall (away from corner)
        d1 = d1 / np.linalg.norm(d1)
        d2 = d2 / np.linalg.norm(d2)
        # Interior angle: angle between -d1 (incoming) and d2 (outgoing)
        cos_a = np.dot(-d1, d2)
        cos_a = np.clip(cos_a, -1, 1)
        return np.degrees(np.arccos(cos_a))

    def remove_short_walls(wall_list, min_length=0.30):
        """Remove walls shorter than min_length, re-intersect neighbors."""
        if len(wall_list) < 4:
            return wall_list, False
        changed = False
        new_walls = []
        skip = set()
        for i in range(len(wall_list)):
            if i in skip:
                continue
            _, _, _, length = wall_list[i]
            if length < min_length:
                skip.add(i)
                changed = True
                continue
            new_walls.append(wall_list[i])
        return new_walls, changed

    def collapse_acute_angles(wall_list, min_angle=20.0):
        """Merge walls at corners with interior angle < min_angle."""
        if len(wall_list) < 4:
            return wall_list, False
        changed = False
        new_walls = list(wall_list)
        i = 0
        while i < len(new_walls) and len(new_walls) >= 4:
            prev_idx = (i - 1) % len(new_walls)
            angle = interior_angle(new_walls[prev_idx], new_walls[i])
            if angle < min_angle:
                # Merge wall[prev] and wall[i] into one
                p1 = new_walls[prev_idx][0]
                p2 = new_walls[i][1]
                d = p2 - p1
                l = np.linalg.norm(d)
                a = np.arctan2(d[1], d[0]) * 180 / np.pi % 180
                new_walls[prev_idx] = (p1, p2, a, l)
                new_walls.pop(i)
                changed = True
                # Don't increment i — check the new wall against its next neighbor
            else:
                i += 1
        return new_walls, changed

    def collinearity_merge(wall_list, angle_thresh=8.0, perp_thresh=0.10):
        """Merge consecutive walls that are nearly collinear AND close together."""
        if len(wall_list) < 4:
            return wall_list, False
        changed = False
        new_walls = []
        i = 0
        while i < len(wall_list):
            if i == len(wall_list) - 1:
                new_walls.append(wall_list[i])
                i += 1
                continue
            p1a, p2a, a1, l1 = wall_list[i]
            p1b, p2b, a2, l2 = wall_list[i + 1]
            angle_diff = min(abs(a1 - a2), 180 - abs(a1 - a2))
            # Perpendicular offset between the two wall lines
            d = p2a - p1a
            nd = np.linalg.norm(d)
            if nd > 1e-6:
                normal = np.array([-d[1], d[0]]) / nd
                perp = abs(np.dot(p1b - p1a, normal))
            else:
                perp = 999
            if angle_diff < angle_thresh and perp < perp_thresh:
                # Merge
                combined_d = p2b - p1a
                combined_l = np.linalg.norm(combined_d)
                combined_a = np.arctan2(combined_d[1], combined_d[0]) * 180 / np.pi % 180
                new_walls.append((p1a, p2b, combined_a, combined_l))
                i += 2
                changed = True
            else:
                new_walls.append(wall_list[i])
                i += 1
        return new_walls, changed

    def rebuild_polygon(wall_list):
        """Re-intersect consecutive walls to get clean corners, rebuild polygon."""
        if len(wall_list) < 3:
            return wall_list, None

        edges = []
        for p1, p2, angle, length in wall_list:
            mid = (p1 + p2) / 2
            rad = angle * np.pi / 180
            d = np.array([np.cos(rad), np.sin(rad)])
            edges.append((mid, d))

        corners = []
        for i in range(len(edges)):
            m1, d1 = edges[i]
            m2, d2 = edges[(i + 1) % len(edges)]
            pt = line_intersection(m1, d1, m2, d2)
            if pt is not None:
                corners.append(pt)
            else:
                corners.append((m1 + m2) / 2)

        corners = np.array(corners)
        poly = ShapelyPolygon(corners)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.geom_type == 'MultiPolygon':
            poly = max(poly.geoms, key=lambda g: g.area)

        new_walls = []
        coords = np.array(poly.exterior.coords)
        for i in range(len(coords) - 1):
            wp1, wp2 = coords[i], coords[i + 1]
            wl = np.linalg.norm(wp2 - wp1)
            wa = np.arctan2(wp2[1] - wp1[1], wp2[0] - wp1[0]) * 180 / np.pi % 180
            if wl > 0.05:
                new_walls.append((wp1, wp2, wa, wl))

        return new_walls, poly

    # Run post-processing passes
    current = list(walls_refined)
    area_before = poly_refined.area

    for pass_num in range(5):
        n_before = len(current)

        current, c1 = remove_short_walls(current, min_length=0.30)
        if c1:
            current, poly_refined = rebuild_polygon(current)

        current, c2 = collapse_acute_angles(current, min_angle=20.0)
        if c2:
            current, poly_refined = rebuild_polygon(current)

        current, c3 = collinearity_merge(current, angle_thresh=8.0, perp_thresh=0.10)
        if c3:
            current, poly_refined = rebuild_polygon(current)

        if not (c1 or c2 or c3):
            break

        print(f"  pass {pass_num+1}: {n_before} -> {len(current)} walls")

    walls_refined = current

    # Area sanity check
    area_after = poly_refined.area if poly_refined else 0
    area_change = abs(area_after - area_before) / area_before * 100 if area_before > 0 else 0
    if area_change > 25:
        print(f"  WARNING: area changed {area_change:.1f}% ({area_before:.1f} -> {area_after:.1f} m2)")

    # Final stats
    if walls_refined:
        shortest = min(w[3] for w in walls_refined)
        angles = []
        for i in range(len(walls_refined)):
            prev = (i - 1) % len(walls_refined)
            angles.append(interior_angle(walls_refined[prev], walls_refined[i]))
        min_angle = min(angles) if angles else 180
        print(f"  Shortest wall: {shortest:.2f}m, Min interior angle: {min_angle:.1f} deg")

    print(f"Room (final): {poly_refined.area:.1f} m2, {len(walls_refined)} walls")

    # ============================================================
    # VISUALIZATIONS
    # ============================================================

    # Background grid
    res_v = 0.02
    xmn, ymn = pts[:, :2].min(0) - 0.5
    xmx, ymx = pts[:, :2].max(0) + 0.5
    nxv, nyv = int((xmx - xmn) / res_v), int((ymx - ymn) / res_v)
    gv, xev, yev = np.histogram2d(pts[:, 0], pts[:, 1], bins=[nxv, nyv],
                                   range=[[xmn, xmx], [ymn, ymx]])
    p93 = np.percentile(gv[gv > 0], 93) if np.any(gv > 0) else 1
    g8v = (np.clip(gv, 0, p93) / p93 * 255).astype(np.uint8)
    extv = [xev[0], xev[-1], yev[0], yev[-1]]

    # 6-panel pipeline
    fig, axes = plt.subplots(2, 3, figsize=(21, 14), dpi=150)

    # Ceiling mask + DP corners
    axes[0, 0].imshow(clean.T, origin='lower', cmap='gray', extent=ext)
    dp_real = np.zeros((len(corner_pts), 2), dtype=float)
    dp_real[:, 0] = x_min + corner_pts[:, 1] * RES
    dp_real[:, 1] = y_min + corner_pts[:, 0] * RES
    dp_closed = np.vstack([dp_real, dp_real[0:1]])
    axes[0, 0].plot(dp_closed[:, 0], dp_closed[:, 1], 'r-', linewidth=1.5)
    axes[0, 0].plot(dp_real[:, 0], dp_real[:, 1], 'ro', markersize=6)
    axes[0, 0].set_title(f'1. DP corners ({n_corners})', fontsize=11)

    # Fitted lines
    axes[0, 1].set_facecolor('white')
    colors = plt.cm.tab10(np.linspace(0, 1, len(fitted_lines)))
    for j, (pt, d, n, seg_real) in enumerate(fitted_lines):
        axes[0, 1].scatter(seg_real[:, 0], seg_real[:, 1], s=1, color=colors[j], alpha=0.5)
        t_vals = (seg_real - pt) @ d
        lp1 = pt + d * t_vals.min()
        lp2 = pt + d * t_vals.max()
        axes[0, 1].plot([lp1[0], lp2[0]], [lp1[1], lp2[1]], '-', color=colors[j], linewidth=2.5)
    axes[0, 1].set_title(f'2. Least-squares lines ({len(fitted_lines)})', fontsize=11)

    # Line-fit polygon
    axes[0, 2].set_facecolor('white')
    if hasattr(poly, 'exterior'):
        rx, ry = poly.exterior.xy
        axes[0, 2].fill(rx, ry, color='#E8F5E9', alpha=0.5)
        axes[0, 2].plot(rx, ry, 'k-', linewidth=2.5)
    axes[0, 2].scatter(corners[:, 0], corners[:, 1], s=60, c='red', zorder=5,
                        edgecolors='white', linewidth=1.5)
    axes[0, 2].set_title(f'3. Line-fit: {poly.area:.1f} m2', fontsize=11)

    # Per-wall point bands
    axes[1, 0].imshow(g8v.T, origin='lower', cmap='gray_r', extent=extv, alpha=0.3)
    for p1, p2, angle, length in walls:
        mid = (p1 + p2) / 2
        d = (p2 - p1); nd = np.linalg.norm(d)
        if nd < 1e-6: continue
        d /= nd; wn = np.array([-d[1], d[0]])
        vecs = pts_wall_height - p1
        perp = np.abs(vecs @ wn)
        along = vecs @ d
        in_band = (perp < BAND_WIDTH) & (along > -0.3) & (along < length + 0.3)
        bp = pts_wall_height[in_band]
        if len(bp) > 0:
            axes[1, 0].scatter(bp[:, 0], bp[:, 1], s=0.5, alpha=0.3)
    if hasattr(poly, 'exterior'):
        rx, ry = poly.exterior.xy
        axes[1, 0].plot(rx, ry, 'r-', linewidth=1.5, alpha=0.7)
    axes[1, 0].set_title(f'4. Wall-band points', fontsize=11)

    # Refined polygon
    axes[1, 1].set_facecolor('white')
    if hasattr(poly_refined, 'exterior'):
        rx2, ry2 = poly_refined.exterior.xy
        axes[1, 1].fill(rx2, ry2, color='#E3F2FD', alpha=0.5)
        axes[1, 1].plot(rx2, ry2, 'b-', linewidth=2.5, label='Refined')
    if hasattr(poly, 'exterior'):
        rx, ry = poly.exterior.xy
        axes[1, 1].plot(rx, ry, 'r--', linewidth=1, alpha=0.5, label='Before')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].set_title(f'5. RANSAC refined: {poly_refined.area:.1f} m2', fontsize=11)

    # Final overlay
    axes[1, 2].imshow(g8v.T, origin='lower', cmap='gray_r', extent=extv, alpha=0.3)
    if hasattr(poly_refined, 'exterior'):
        rx2, ry2 = poly_refined.exterior.xy
        axes[1, 2].fill(rx2, ry2, color='#4CAF50', alpha=0.15)
    for p1, p2, a, l in walls_refined:
        axes[1, 2].plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=2.5)
        if l > 0.3:
            mid = (p1 + p2) / 2
            d = p2 - p1
            nm_d = np.linalg.norm(d)
            perp = np.array([-d[1], d[0]]) / nm_d * 0.18
            axes[1, 2].text(mid[0]+perp[0], mid[1]+perp[1], f'{l:.2f}m',
                            ha='center', fontsize=7, color='yellow', fontweight='bold',
                            rotation=a if a <= 90 else a - 180)
    axes[1, 2].set_title(f'6. Final: {len(walls_refined)} walls', fontsize=11)

    for ax in axes.flat:
        ax.set_aspect('equal'); ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')

    fig.suptitle(f'{name}: Line-Fit + RANSAC Refinement', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT}/{name}_linefit.png', bbox_inches='tight')
    plt.close()

    return poly_refined, walls_refined


# Run on all leveled files
import glob
for f in sorted(glob.glob("../result_leveled_*.ply")):
    name = os.path.splitext(os.path.basename(f))[0]
    process(f, name)

print(f"\nAll results in {OUT}/")
