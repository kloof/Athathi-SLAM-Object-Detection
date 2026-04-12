#!/usr/bin/env python3
"""
floorplan — Extract 2D floor plans from LiDAR point cloud scans.

Traces the ceiling boundary, simplifies to straight walls, snaps to 45-degree
angles, and exports PNG + DXF.

Usage:
    python floorplan.py input.ply
    python floorplan.py input.ply -o output_dir/
    python floorplan.py input.ply --epsilon 0.015 --snap 45 --resolution 0.03
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import cv2
import ezdxf
from ezdxf.enums import TextEntityAlignment
import numpy as np
import open3d as o3d
from scipy.ndimage import binary_fill_holes, gaussian_filter1d
from scipy.signal import find_peaks
from shapely.geometry import Polygon as ShapelyPolygon


# ============================================================
# CORE FUNCTIONS
# ============================================================

def load_and_preprocess(path, voxel_size=0.03, sor_neighbors=20, sor_std=2.0):
    """Load PLY/PCD, apply statistical outlier removal + voxel downsample."""
    pcd = o3d.io.read_point_cloud(path)
    n_raw = len(pcd.points)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=sor_neighbors, std_ratio=sor_std)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    pts = np.asarray(pcd.points)
    return pts, n_raw


def detect_floor_ceiling(pts, bin_width=0.02):
    """Detect floor and ceiling Z from histogram peaks."""
    z = pts[:, 2]
    bins = np.arange(z.min(), z.max() + bin_width, bin_width)
    hist, edges = np.histogram(z, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    zhist = gaussian_filter1d(hist.astype(float), sigma=3)
    pks, props = find_peaks(zhist, height=np.max(zhist) * 0.1, distance=10)

    if len(pks) >= 2:
        top2 = pks[np.argsort(props['peak_heights'])[-2:]]
        top2 = np.sort(top2)
        floor_z, ceiling_z = centers[top2[0]], centers[top2[1]]
    else:
        floor_z = np.percentile(z, 5)
        ceiling_z = np.percentile(z, 95)

    return floor_z, ceiling_z


def extract_ceiling_points(pts, ceiling_z, band=0.12, floor_z=None):
    """Extract ceiling points. Handles multi-level ceilings by taking
    all points in the upper portion of the room, not just a narrow band."""
    if floor_z is not None:
        h = ceiling_z - floor_z
        # Take everything above 70% of room height — catches double ceilings
        z_cutoff = floor_z + h * 0.70
        mask = (pts[:, 2] > z_cutoff) & (pts[:, 2] < ceiling_z + band)
    else:
        mask = (pts[:, 2] > ceiling_z - band) & (pts[:, 2] < ceiling_z + band)
    return pts[mask][:, :2]


def ceiling_to_binary(ceil_xy, resolution=0.03, close_kernel=11, open_kernel=5):
    """Project ceiling points to 2D binary mask with morphological cleanup."""
    x_min, y_min = ceil_xy.min(axis=0) - 0.5
    x_max, y_max = ceil_xy.max(axis=0) + 0.5
    nx = int((x_max - x_min) / resolution)
    ny = int((y_max - y_min) / resolution)

    grid, xe, ye = np.histogram2d(
        ceil_xy[:, 0], ceil_xy[:, 1],
        bins=[nx, ny], range=[[x_min, x_max], [y_min, y_max]])

    # Normalize + Otsu threshold
    if np.any(grid > 0):
        cap = np.percentile(grid[grid > 0], 90)
        g8 = (np.clip(grid, 0, cap) / cap * 255).astype(np.uint8)
    else:
        g8 = np.zeros((nx, ny), dtype=np.uint8)

    _, binary = cv2.threshold(g8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphology: close scan-line gaps, fill holes, remove noise
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel)))
    binary = (binary_fill_holes(binary > 0).astype(np.uint8)) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel)))

    return binary, g8, xe, ye, resolution, x_min, y_min


def remove_outlier_clusters(binary):
    """Keep only the largest connected component."""
    n_labels, labeled, stats, _ = cv2.connectedComponentsWithStats(binary)
    if n_labels <= 1:
        return binary, 0

    areas = stats[1:, cv2.CC_STAT_AREA]
    biggest = np.argmax(areas) + 1
    clean = np.zeros_like(binary)
    clean[labeled == biggest] = 255

    # Smooth jagged edges
    clean = cv2.erode(clean, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    clean = cv2.dilate(clean, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    return clean, n_labels - 2


def trace_and_simplify(binary, epsilon_ratio=0.012):
    """Trace contour and simplify with Douglas-Peucker."""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)

    perimeter = cv2.arcLength(contour, True)
    epsilon = epsilon_ratio * perimeter
    simplified = cv2.approxPolyDP(contour, epsilon, True)

    return contour, simplified


def pixel_to_real(px_coords, x_min, y_min, resolution):
    """Convert pixel contour coords to real-world (swap col/row -> x/y)."""
    real = np.zeros_like(px_coords, dtype=float)
    real[:, 0] = x_min + px_coords[:, 1] * resolution  # row -> x
    real[:, 1] = y_min + px_coords[:, 0] * resolution  # col -> y
    return real


def detect_corners(binary, g8, x_min, y_min, resolution, max_corners=20,
                    quality=0.02, min_distance_m=0.5):
    """Detect corners directly from the ceiling binary mask using Shi-Tomasi.

    Returns real-world corner coordinates ordered as a polygon.
    """
    min_distance_px = int(min_distance_m / resolution)

    # Shi-Tomasi corner detection on the binary mask edges
    edges = cv2.Canny(binary, 50, 150)
    # Dilate edges slightly so corners are detected at wall intersections
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    corners_px = cv2.goodFeaturesToTrack(
        edges, maxCorners=max_corners, qualityLevel=quality,
        minDistance=min_distance_px, blockSize=7)

    if corners_px is None or len(corners_px) < 3:
        return None

    corners_px = corners_px.reshape(-1, 2)

    # Filter: keep only corners that are ON the contour boundary
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)

    on_boundary = []
    for cx, cy in corners_px:
        # Check distance to contour — should be very close
        dist = abs(cv2.pointPolygonTest(contour, (float(cx), float(cy)), True))
        if dist < 8:  # within 8 pixels of boundary
            on_boundary.append([cx, cy])

    if len(on_boundary) < 3:
        return None

    on_boundary = np.array(on_boundary)

    # Convert to real-world (swap col/row -> x/y)
    real = np.zeros_like(on_boundary, dtype=float)
    real[:, 0] = x_min + on_boundary[:, 1] * resolution  # row -> x
    real[:, 1] = y_min + on_boundary[:, 0] * resolution  # col -> y

    # Order corners as a polygon (by angle from centroid)
    centroid = real.mean(axis=0)
    angles = np.arctan2(real[:, 1] - centroid[1], real[:, 0] - centroid[0])
    order = np.argsort(angles)
    real = real[order]

    return real, on_boundary


def snap_polygon_to_angles(coords, snap_angles_deg=None):
    """Snap each polygon edge to nearest angle, re-intersect for clean corners."""
    if snap_angles_deg is None:
        snap_angles_deg = np.array([0, 45, 90, 135])

    n = len(coords)
    edges = []

    for i in range(n):
        p1, p2 = coords[i], coords[(i + 1) % n]
        d = p2 - p1
        length = np.linalg.norm(d)
        if length < 0.1:
            continue

        angle = np.arctan2(d[1], d[0]) * 180 / np.pi % 180
        diffs = [min(abs(angle - sa), 180 - abs(angle - sa)) for sa in snap_angles_deg]
        best = np.argmin(diffs)
        snap = snap_angles_deg[best]

        rad = snap * np.pi / 180
        direction = np.array([np.cos(rad), np.sin(rad)])
        mid = (p1 + p2) / 2
        edges.append((mid, direction, length, snap))

    # Re-intersect consecutive edges
    corners = []
    for i in range(len(edges)):
        mid1, dir1, _, _ = edges[i]
        mid2, dir2, _, _ = edges[(i + 1) % len(edges)]

        denom = dir1[0] * dir2[1] - dir1[1] * dir2[0]
        if abs(denom) < 1e-10:
            corners.append((mid1 + mid2) / 2)
            continue

        dp = mid2 - mid1
        t = (dp[0] * dir2[1] - dp[1] * dir2[0]) / denom
        corners.append(mid1 + t * dir1)

    return np.array(corners), edges


def build_room_polygon(corners):
    """Build a valid Shapely polygon from corners."""
    poly = ShapelyPolygon(corners)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.geom_type == 'MultiPolygon':
        poly = max(poly.geoms, key=lambda g: g.area)
    return poly


def extract_walls(poly, min_length=0.15):
    """Extract wall segments from polygon exterior."""
    coords = np.array(poly.exterior.coords)
    walls = []
    for i in range(len(coords) - 1):
        p1, p2 = coords[i], coords[i + 1]
        length = np.linalg.norm(p2 - p1)
        angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) * 180 / np.pi % 180
        if length > min_length:
            walls.append((p1, p2, angle, length))
    return walls


# ============================================================
# RANSAC REFINEMENT
# ============================================================

def detect_ransac_walls(pts, floor_z, ceiling_z, room_poly,
                        max_iter=80, min_len=0.25):
    """Run RANSAC on points inside the room polygon to find precise wall planes."""
    from matplotlib.path import Path as MplPath

    # Clip points to room boundary
    boundary_path = MplPath(np.array(room_poly.buffer(0.3).exterior.coords))
    inside = boundary_path.contains_points(pts[:, :2])
    pts_inside = pts[inside]

    # Build point cloud for RANSAC
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_inside)

    remaining = pcd
    ransac_walls = []

    for _ in range(max_iter):
        if len(remaining.points) < 50:
            break
        best_inliers, best_model = [], None
        for thresh in [0.02, 0.04, 0.06, 0.08]:
            model, inliers = remaining.segment_plane(thresh, 3, 1000)
            if len(inliers) > len(best_inliers):
                best_inliers, best_model = inliers, model
        if not best_inliers or len(best_inliers) < 10:
            break

        a, b, c, d = best_model
        normal = np.array([a, b, c])
        normal /= np.linalg.norm(normal)

        if abs(normal[2]) < 0.3:  # vertical wall
            wc = remaining.select_by_index(best_inliers)
            wp = np.asarray(wc.points)[:, :2]
            if len(wp) >= 10:
                wpcd = o3d.geometry.PointCloud()
                wpcd.points = o3d.utility.Vector3dVector(
                    np.hstack([wp, np.zeros((len(wp), 1))]))
                labels = np.array(wpcd.cluster_dbscan(eps=0.5, min_points=10))
                for lbl in range(max(labels.max(), 0) + 1):
                    cl = wp[labels == lbl]
                    if len(cl) < 10:
                        continue
                    # PCA fit
                    mean = cl.mean(axis=0)
                    centered = cl - mean
                    cov = np.cov(centered.T)
                    eigvals, eigvecs = np.linalg.eigh(cov)
                    direction = eigvecs[:, np.argmax(eigvals)]
                    proj = centered @ direction
                    p1 = mean + direction * proj.min()
                    p2 = mean + direction * proj.max()
                    length = np.linalg.norm(p2 - p1)
                    angle = np.arctan2(direction[1], direction[0]) * 180 / np.pi % 180
                    # Perpendicular tightness
                    perp_dir = eigvecs[:, np.argmin(eigvals)]
                    perp_spread = (centered @ perp_dir).ptp()
                    if length >= min_len:
                        ransac_walls.append({
                            'p1': p1, 'p2': p2, 'angle': angle, 'length': length,
                            'n_pts': len(cl), 'perp_spread': perp_spread,
                            'midpoint': mean.copy(),
                        })

        remaining = remaining.select_by_index(best_inliers, invert=True)

    return ransac_walls


def refine_walls_with_ransac(walls, ransac_walls, angle_flex=3.0, max_shift=0.3):
    """For each ceiling-trace wall, find the matching RANSAC wall and refine.

    - Replaces the wall position with the RANSAC wall's precise position
    - Uses the RANSAC wall's actual angle (within angle_flex degrees of original)
    - Shifts the wall perpendicular to align with RANSAC data

    Args:
        walls: list of (p1, p2, angle, length) from ceiling trace
        ransac_walls: list of dicts from detect_ransac_walls
        angle_flex: max degrees a refined angle can deviate from snapped angle
        max_shift: max perpendicular shift in meters to accept a RANSAC match
    """
    if not ransac_walls:
        return walls

    refined = []
    for p1, p2, angle, length in walls:
        mid = (p1 + p2) / 2
        direction = p2 - p1
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            refined.append((p1, p2, angle, length))
            continue
        direction = direction / norm
        wall_normal = np.array([-direction[1], direction[0]])

        # Find best matching RANSAC wall
        best_match = None
        best_score = float('inf')

        for rw in ransac_walls:
            # Angle compatibility
            angle_diff = min(abs(angle - rw['angle']), 180 - abs(angle - rw['angle']))
            if angle_diff > 20:  # too different, not the same wall
                continue

            # Perpendicular distance from RANSAC midpoint to ceiling-trace wall line
            perp_dist = abs(np.dot(rw['midpoint'] - p1, wall_normal))
            if perp_dist > max_shift:
                continue

            # Along-wall overlap check
            proj_start = np.dot(rw['p1'] - mid, direction)
            proj_end = np.dot(rw['p2'] - mid, direction)
            rw_center_proj = (proj_start + proj_end) / 2
            overlap_dist = abs(rw_center_proj)

            # Score: lower = better match (prefer close + aligned + overlapping)
            score = perp_dist + angle_diff * 0.01 + overlap_dist * 0.1

            if score < best_score:
                best_score = score
                best_match = rw

        if best_match is not None:
            # Refine angle: use RANSAC angle but clamp deviation
            raw_angle = best_match['angle']
            angle_deviation = raw_angle - angle
            # Handle wraparound
            if angle_deviation > 90:
                angle_deviation -= 180
            elif angle_deviation < -90:
                angle_deviation += 180
            clamped_deviation = np.clip(angle_deviation, -angle_flex, angle_flex)
            refined_angle = angle + clamped_deviation

            # Refine position: shift wall perpendicular to match RANSAC
            perp_shift = np.dot(best_match['midpoint'] - mid, wall_normal)
            clamped_shift = np.clip(perp_shift, -max_shift, max_shift)
            new_mid = mid + wall_normal * clamped_shift

            # Rebuild wall with refined angle and position
            rad = refined_angle * np.pi / 180
            new_dir = np.array([np.cos(rad), np.sin(rad)])
            half = length / 2
            new_p1 = new_mid - new_dir * half
            new_p2 = new_mid + new_dir * half

            refined.append((new_p1, new_p2, refined_angle, length))
        else:
            # No RANSAC match — keep original
            refined.append((p1, p2, angle, length))

    return refined


def re_intersect_corners(walls):
    """After refining wall positions/angles, re-intersect consecutive walls
    to get clean corners again."""
    if len(walls) < 3:
        return walls

    # Build directions
    edges = []
    for p1, p2, angle, length in walls:
        mid = (p1 + p2) / 2
        rad = angle * np.pi / 180
        direction = np.array([np.cos(rad), np.sin(rad)])
        edges.append((mid, direction, length, angle))

    # Re-intersect consecutive edges
    corners = []
    for i in range(len(edges)):
        mid1, dir1, _, _ = edges[i]
        mid2, dir2, _, _ = edges[(i + 1) % len(edges)]

        denom = dir1[0] * dir2[1] - dir1[1] * dir2[0]
        if abs(denom) < 1e-10:
            corners.append((mid1 + mid2) / 2)
            continue

        dp = mid2 - mid1
        t = (dp[0] * dir2[1] - dp[1] * dir2[0]) / denom
        corners.append(mid1 + t * dir1)

    # Rebuild walls from new corners
    new_walls = []
    for i in range(len(corners)):
        p1 = corners[i]
        p2 = corners[(i + 1) % len(corners)]
        length = np.linalg.norm(p2 - p1)
        angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) * 180 / np.pi % 180
        if length > 0.1:
            new_walls.append((np.array(p1), np.array(p2), angle, length))

    return new_walls


# ============================================================
# EXPORT FUNCTIONS
# ============================================================

def export_dxf(walls, room_poly, output_path):
    """Export floor plan to DXF."""
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()
    doc.layers.add("WALLS", color=7)
    doc.layers.add("ROOM", color=3)
    doc.layers.add("DIMS", color=2)

    for p1, p2, a, l in walls:
        msp.add_line((float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1])),
                     dxfattribs={"layer": "WALLS"})
        if l > 0.3:
            mid = (p1 + p2) / 2
            d = p2 - p1
            nm = np.linalg.norm(d)
            perp = np.array([-d[1], d[0]]) / nm * 0.2
            t = msp.add_text(f"{l:.2f}m", height=0.1, dxfattribs={"layer": "DIMS"})
            t.set_placement((float(mid[0] + perp[0]), float(mid[1] + perp[1])),
                            align=TextEntityAlignment.MIDDLE_CENTER)

    if hasattr(room_poly, 'exterior'):
        coords = list(room_poly.exterior.coords)
        msp.add_lwpolyline([(float(x), float(y)) for x, y in coords],
                           close=True, dxfattribs={"layer": "ROOM"})
        t = msp.add_text(f"Area: {room_poly.area:.1f}m2", height=0.15,
                         dxfattribs={"layer": "ROOM"})
        t.set_placement((float(room_poly.centroid.x), float(room_poly.centroid.y)),
                        align=TextEntityAlignment.MIDDLE_CENTER)

    doc.saveas(output_path)


def export_pngs(walls, room_poly, pts, binary, clean, g8, xe, ye,
                real_coords, snapped_coords, output_dir, name,
                floor_z, ceiling_z):
    """Generate all PNG visualizations."""
    h = ceiling_z - floor_z
    ext = [xe[0], xe[-1], ye[0], ye[-1]]

    # 01: Pipeline steps
    fig, axes = plt.subplots(2, 3, figsize=(21, 14), dpi=150)
    axes[0, 0].imshow(g8.T, origin='lower', cmap='hot', extent=ext)
    axes[0, 0].set_title('Ceiling Density')
    axes[0, 1].imshow(binary.T, origin='lower', cmap='gray', extent=ext)
    axes[0, 1].set_title('After Morphology')
    axes[0, 2].imshow(clean.T, origin='lower', cmap='gray', extent=ext)
    axes[0, 2].set_title('Outliers Removed')

    rc = np.vstack([real_coords, real_coords[0:1]])
    axes[1, 0].imshow(clean.T, origin='lower', cmap='gray', extent=ext)
    axes[1, 0].plot(rc[:, 0], rc[:, 1], 'r-', linewidth=2)
    axes[1, 0].plot(real_coords[:, 0], real_coords[:, 1], 'ro', markersize=5)
    axes[1, 0].set_title(f'Simplified ({len(real_coords)} vertices)')

    sc = np.vstack([snapped_coords, snapped_coords[0:1]])
    axes[1, 1].set_facecolor('white')
    axes[1, 1].plot(sc[:, 0], sc[:, 1], 'b-', linewidth=2)
    axes[1, 1].plot(snapped_coords[:, 0], snapped_coords[:, 1], 'bo', markersize=5)
    axes[1, 1].plot(rc[:, 0], rc[:, 1], 'r--', linewidth=1, alpha=0.5)
    axes[1, 1].set_title(f'Angle Snapped ({len(snapped_coords)} corners)')

    axes[1, 2].set_facecolor('white')
    if hasattr(room_poly, 'exterior'):
        rx, ry = room_poly.exterior.xy
        axes[1, 2].fill(rx, ry, color='#E8F5E9', alpha=0.5)
        axes[1, 2].plot(rx, ry, 'k-', linewidth=2.5)
    axes[1, 2].set_title(f'Final: {room_poly.area:.1f} m2, {len(walls)} walls')

    for ax in axes.flat:
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
    fig.suptitle('Ceiling Trace Pipeline', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{name}_pipeline.png', bbox_inches='tight')
    plt.close()

    # 02: Floor plan
    fig, ax = plt.subplots(figsize=(14, 14), dpi=200)
    ax.set_facecolor('#FAFAFA')
    if hasattr(room_poly, 'exterior'):
        rx, ry = room_poly.exterior.xy
        ax.fill(rx, ry, color='#E8F5E9', alpha=0.5)
    for p1, p2, a, l in walls:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=3.5, solid_capstyle='round')
        if l > 0.3:
            mid = (p1 + p2) / 2
            d = p2 - p1
            nm = np.linalg.norm(d)
            perp = np.array([-d[1], d[0]]) / nm * 0.18
            ax.text(mid[0] + perp[0], mid[1] + perp[1], f'{l:.2f}m',
                    ha='center', fontsize=7, color='#444',
                    rotation=a if a <= 90 else a - 180)
    ax.text(room_poly.centroid.x, room_poly.centroid.y,
            f"Area: {room_poly.area:.1f} m2\nCeiling: {ceiling_z:.2f}m\nHeight: {h:.2f}m",
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.4'))
    ax.grid(True, alpha=0.06, color='#4488CC')
    ax.set_axisbelow(True)
    ap = np.array([w[0] for w in walls] + [w[1] for w in walls])
    sx, sy = ap[:, 0].min() - 0.3, ap[:, 1].min() - 0.8
    ax.plot([sx, sx + 1], [sy, sy], 'k-', linewidth=3)
    ax.plot([sx, sx], [sy - 0.08, sy + 0.08], 'k-', linewidth=2)
    ax.plot([sx + 1, sx + 1], [sy - 0.08, sy + 0.08], 'k-', linewidth=2)
    ax.text(sx + 0.5, sy + 0.15, '1 m', ha='center', fontsize=9, fontweight='bold')
    ax.set_title(f'Floor Plan - {len(walls)} walls', fontsize=14, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{name}_floorplan.png', bbox_inches='tight')
    plt.close()

    # 03: Overlay on raw
    res_v = 0.02
    xmn, ymn = pts[:, :2].min(0) - 0.5
    xmx, ymx = pts[:, :2].max(0) + 0.5
    nxv, nyv = int((xmx - xmn) / res_v), int((ymx - ymn) / res_v)
    gv, xev, yev = np.histogram2d(pts[:, 0], pts[:, 1], bins=[nxv, nyv],
                                   range=[[xmn, xmx], [ymn, ymx]])
    p93 = np.percentile(gv[gv > 0], 93) if np.any(gv > 0) else 1
    g8v = (np.clip(gv, 0, p93) / p93 * 255).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(14, 14), dpi=200)
    ax.imshow(g8v.T, origin='lower', cmap='gray_r',
              extent=[xev[0], xev[-1], yev[0], yev[-1]], alpha=0.3)
    if hasattr(room_poly, 'exterior'):
        rx, ry = room_poly.exterior.xy
        ax.fill(rx, ry, color='#4CAF50', alpha=0.2)
    for p1, p2, a, l in walls:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=2.5)
    ax.set_title('Floor Plan on Raw Point Cloud', fontsize=14, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{name}_overlay.png', bbox_inches='tight')
    plt.close()


# ============================================================
# PIPELINE
# ============================================================

def run(input_path, output_dir, resolution=0.03, epsilon=0.012,
        snap_angle=45, voxel_size=0.03, ceil_band=0.12,
        close_kernel=11, angle_flex=3.0, verbose=True):
    """Run the full ceiling-trace floor plan extraction pipeline.

    Produces 3 variants:
      A) Ceiling trace with strict angle snap (baseline)
      B) Ceiling trace with flexible angles (snap +/- angle_flex degrees)
      C) RANSAC-refined positions with strict angle snap
    """
    os.makedirs(output_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(input_path))[0]
    t0 = time.time()

    if verbose:
        print(f"floorplan: {name}")
        print("=" * 50)

    # Load
    pts, n_raw = load_and_preprocess(input_path, voxel_size)
    if verbose:
        print(f"Loaded: {n_raw} -> {len(pts)} pts")

    # Floor/ceiling
    floor_z, ceiling_z = detect_floor_ceiling(pts)
    h = ceiling_z - floor_z
    if verbose:
        print(f"Floor: {floor_z:.2f}m, Ceiling: {ceiling_z:.2f}m, Height: {h:.2f}m")

    # Ceiling points
    ceil_xy = extract_ceiling_points(pts, ceiling_z, ceil_band, floor_z=floor_z)
    if verbose:
        print(f"Ceiling points: {len(ceil_xy)}")

    # Binary mask
    binary, g8, xe, ye, res, x_min, y_min = ceiling_to_binary(
        ceil_xy, resolution, close_kernel)

    # Remove outliers
    clean, n_removed = remove_outlier_clusters(binary)
    if verbose:
        print(f"Outlier clusters removed: {n_removed}")

    # Trace + simplify
    raw_contour, simplified = trace_and_simplify(clean, epsilon)
    px_coords = simplified.reshape(-1, 2).astype(float)
    real_coords = pixel_to_real(px_coords, x_min, y_min, res)
    if verbose:
        print(f"Contour: {len(raw_contour)} -> {len(real_coords)} vertices")

    # Corner detection from ceiling mask
    corner_result = detect_corners(clean, g8, x_min, y_min, res)
    if corner_result is not None:
        corner_coords_real, corner_coords_px = corner_result
        if verbose:
            print(f"Corners detected: {len(corner_coords_real)}")
    else:
        corner_coords_real = None
        corner_coords_px = None
        if verbose:
            print("Corner detection: not enough corners found")

    # ============================================================
    # VARIANT A: No snap — raw simplified contour (natural angles)
    # ============================================================
    if verbose:
        print(f"\n--- Variant A: No snap (natural angles) ---")
    poly_a = build_room_polygon(real_coords)
    walls_a = extract_walls(poly_a)
    if verbose:
        print(f"  {poly_a.area:.1f} m2, {len(walls_a)} walls")

    # ============================================================
    # VARIANT B: Corner detection (Shi-Tomasi)
    # ============================================================
    if verbose:
        print(f"\n--- Variant B: Corner detection ---")
    if corner_coords_real is not None and len(corner_coords_real) >= 3:
        poly_b = build_room_polygon(corner_coords_real)
        walls_b = extract_walls(poly_b)
    else:
        # Fallback to contour
        poly_b = poly_a
        walls_b = walls_a
    if verbose:
        print(f"  {poly_b.area:.1f} m2, {len(walls_b)} walls")

    # ============================================================
    # VARIANT C: Angle snap (for comparison)
    # ============================================================
    if verbose:
        print(f"\n--- Variant C: {snap_angle}-deg snap (for comparison) ---")
    snap_angles = np.arange(0, 180, snap_angle) if snap_angle > 0 else None
    snapped_c, _ = snap_polygon_to_angles(real_coords, snap_angles)
    poly_c = build_room_polygon(snapped_c)
    walls_c = extract_walls(poly_c)
    if verbose:
        print(f"  {poly_c.area:.1f} m2, {len(walls_c)} walls")

    # ============================================================
    # EXPORT
    # ============================================================
    variants = {
        'A_natural': (walls_a, poly_a, 'Natural angles (no snap)'),
        'B_corners': (walls_b, poly_b, 'Corner detection'),
        'C_snapped': (walls_c, poly_c, f'{snap_angle}-deg snap'),
    }

    # Pipeline PNG
    export_pngs(walls_a, poly_a, pts, binary, clean, g8, xe, ye,
                real_coords, real_coords, output_dir, name, floor_z, ceiling_z)

    # Comparison PNG with point cloud background
    res_v = 0.02
    xmn, ymn = pts[:, :2].min(0) - 0.5
    xmx, ymx = pts[:, :2].max(0) + 0.5
    nxv, nyv = int((xmx - xmn) / res_v), int((ymx - ymn) / res_v)
    gv, xev, yev = np.histogram2d(pts[:, 0], pts[:, 1], bins=[nxv, nyv],
                                   range=[[xmn, xmx], [ymn, ymx]])
    p93 = np.percentile(gv[gv > 0], 93) if np.any(gv > 0) else 1
    g8v = (np.clip(gv, 0, p93) / p93 * 255).astype(np.uint8)
    extv = [xev[0], xev[-1], yev[0], yev[-1]]

    colors = ['lime', 'cyan', 'orange']
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), dpi=200)
    for idx, (key, (walls, poly, label)) in enumerate(variants.items()):
        ax = axes[idx]
        ax.imshow(g8v.T, origin='lower', cmap='gray_r', extent=extv, alpha=0.3)
        if hasattr(poly, 'exterior'):
            rx, ry = poly.exterior.xy
            ax.fill(rx, ry, color=colors[idx], alpha=0.15)
        for p1, p2, a, l in walls:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=2.5, solid_capstyle='round')
            if l > 0.3:
                mid = (p1 + p2) / 2
                d = p2 - p1
                nm = np.linalg.norm(d)
                perp = np.array([-d[1], d[0]]) / nm * 0.15
                ax.text(mid[0]+perp[0], mid[1]+perp[1], f'{l:.2f}m',
                        ha='center', fontsize=6, color='yellow', fontweight='bold',
                        rotation=a if a <= 90 else a-180)
        # Show detected corners for variant B
        if key == 'B_corners' and corner_coords_real is not None:
            ax.scatter(corner_coords_real[:, 0], corner_coords_real[:, 1],
                       s=80, c='red', marker='o', zorder=5, edgecolors='white', linewidth=1.5)
        ax.set_title(f'{label}\n{poly.area:.1f} m2, {len(walls)} walls', fontsize=11, fontweight='bold')
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    fig.suptitle('3 Variants on Point Cloud', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{name}_comparison.png', bbox_inches='tight')
    plt.close()

    # Corner detection detail PNG
    ext_ceil = [xe[0], xe[-1], ye[0], ye[-1]]
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), dpi=150)
    axes[0].imshow(g8.T, origin='lower', cmap='hot', extent=ext_ceil)
    axes[0].set_title('Ceiling Density')
    # Show edges + detected corners
    edges_img = cv2.Canny(clean, 50, 150)
    axes[1].imshow(edges_img.T, origin='lower', cmap='gray', extent=ext_ceil)
    if corner_coords_real is not None:
        axes[1].scatter(corner_coords_real[:, 0], corner_coords_real[:, 1],
                        s=100, c='red', marker='o', zorder=5, edgecolors='white', linewidth=2)
    axes[1].set_title(f'Edges + Corners ({len(corner_coords_real) if corner_coords_real is not None else 0})')
    # Final polygon from corners
    axes[2].set_facecolor('white')
    if hasattr(poly_b, 'exterior'):
        rx, ry = poly_b.exterior.xy
        axes[2].fill(rx, ry, color='#E8F5E9', alpha=0.5)
        axes[2].plot(rx, ry, 'k-', linewidth=2.5)
    if corner_coords_real is not None:
        axes[2].scatter(corner_coords_real[:, 0], corner_coords_real[:, 1],
                        s=80, c='red', marker='o', zorder=5)
    axes[2].set_title(f'Corner-Based Polygon\n{poly_b.area:.1f} m2')
    for ax in axes:
        ax.set_aspect('equal'); ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    fig.suptitle('Corner Detection Pipeline', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{name}_corners.png', bbox_inches='tight')
    plt.close()

    # DXF for each
    for key, (walls, poly, label) in variants.items():
        export_dxf(walls, poly, f'{output_dir}/{name}_{key}.dxf')

    # Metadata
    elapsed = time.time() - t0
    meta = {
        'input': input_path,
        'n_points_raw': n_raw,
        'n_points_processed': len(pts),
        'floor_z': round(float(floor_z), 3),
        'ceiling_z': round(float(ceiling_z), 3),
        'room_height': round(float(h), 3),
        'n_outlier_clusters_removed': n_removed,
        'n_corners_detected': len(corner_coords_real) if corner_coords_real is not None else 0,
        'variants': {},
        'processing_time_s': round(elapsed, 1),
    }
    for key, (walls, poly, label) in variants.items():
        meta['variants'][key] = {
            'label': label,
            'area_m2': round(float(poly.area), 2),
            'n_walls': len(walls),
            'walls': [{'length_m': round(float(l), 3), 'angle_deg': round(float(a), 1)}
                      for _, _, a, l in walls],
        }
    with open(f'{output_dir}/{name}_metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    if verbose:
        print(f"\nDone in {elapsed:.1f}s -> {output_dir}/")
        for key, (walls, poly, label) in variants.items():
            print(f"  {key}: {poly.area:.1f} m2, {len(walls)} walls - {label}")

    return variants, meta


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        prog='floorplan',
        description='Extract 2D floor plans from LiDAR point cloud scans.')
    parser.add_argument('input', help='Input PLY or PCD file')
    parser.add_argument('-o', '--output', default='output/', help='Output directory')
    parser.add_argument('--resolution', type=float, default=0.03,
                        help='Grid resolution in meters (default: 0.03)')
    parser.add_argument('--epsilon', type=float, default=0.012,
                        help='Contour simplification ratio (default: 0.012)')
    parser.add_argument('--snap', type=float, default=45,
                        help='Angle snap increment in degrees (default: 45, 0=disable)')
    parser.add_argument('--voxel', type=float, default=0.03,
                        help='Voxel downsample size (default: 0.03)')
    parser.add_argument('--ceil-band', type=float, default=0.12,
                        help='Ceiling Z band +/- meters (default: 0.12)')
    parser.add_argument('--close-kernel', type=int, default=11,
                        help='Morphology close kernel size (default: 11)')
    parser.add_argument('--angle-flex', type=float, default=3.0,
                        help='Max angle deviation from snap grid in degrees (default: 3.0)')
    parser.add_argument('-q', '--quiet', action='store_true')
    args = parser.parse_args()

    run(args.input, args.output,
        resolution=args.resolution, epsilon=args.epsilon,
        snap_angle=args.snap, voxel_size=args.voxel,
        ceil_band=args.ceil_band, close_kernel=args.close_kernel,
        angle_flex=args.angle_flex, verbose=not args.quiet)


if __name__ == '__main__':
    main()
