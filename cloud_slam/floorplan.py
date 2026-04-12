"""
Floor-plan extraction from a leveled indoor point cloud.

Ported from `reference/floorplan/floorplan.py` with one substantive change:
the reference's fragile `detect_floor_ceiling()` (top-2 Z-histogram peaks) is
replaced with `detect_floor_ceiling_robust()`, which uses the same RANSAC +
IMU-prior plane detection that the rest of the SLAM pipeline already relies
on (`cloud_slam.room_structure.detect_room`). Furniture peaks no longer win
against the real ceiling.

DXF export from the reference tool is intentionally not ported — this module
emits PNG + JSON only.

Public entry points:
    detect_floor_ceiling_robust(pcd, gravity_up=None, imus=None)
    generate_floorplan(pcd, output_dir, name="floorplan", *, ...)
"""

import json
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import cv2
import numpy as np
import open3d as o3d
from scipy.ndimage import binary_fill_holes, gaussian_filter1d
from scipy.signal import find_peaks
from shapely.geometry import Polygon as ShapelyPolygon

from cloud_slam.room_structure import detect_room
from cloud_slam.frustum import estimate_gravity


# ============================================================
# FLOOR / CEILING DETECTION (the robust replacement)
# ============================================================

def _plane_inliers_xy(pts, plane, distance_thresh=0.15):
    """Return the XY coords of cloud points within `distance_thresh` of plane."""
    signed = (pts - plane.centroid) @ plane.normal
    mask = np.abs(signed) < distance_thresh
    return pts[mask][:, :2]


def _convex_hull_area(xy_points):
    """Area of the 2D convex hull of a point set. Returns 0.0 for degenerate input."""
    if xy_points is None or len(xy_points) < 3:
        return 0.0
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(xy_points)
        return float(hull.volume)  # in 2D, ConvexHull.volume is the area
    except Exception:
        return 0.0


def _select_ceiling_plane(pts, room, distance_thresh=0.15, verbose=False):
    """Of the two horizontal planes detected by detect_room, return the one
    that is the actual ceiling.

    Rationale: ceilings span the entire room (uninterrupted horizontal
    surface). Floors are partially occluded by furniture (beds, tables,
    sofas sit on the floor, hiding it). The plane with the larger XY
    convex-hull coverage is the ceiling.

    This heuristic is robust to leveling-sign inversion: even if the cloud
    is upside-down and detect_room swaps the floor/ceiling labels, picking
    by coverage picks the physically-correct surface.

    Args:
        pts: (N, 3) array of cloud points (full resolution).
        room: RoomStructure with .floor and .ceiling Plane objects.
        distance_thresh: band used to collect inliers for each plane (m).

    Returns:
        (ceiling_plane, floor_plane, swapped_bool, info_dict)
        where info_dict has hull areas for both candidates.
    """
    cand_ceil = room.ceiling   # detect_room's label
    cand_floor = room.floor

    if cand_ceil is None:
        # Nothing to swap with
        return None, cand_floor, False, {'ceil_area': 0.0, 'floor_area': 0.0}
    if cand_floor is None:
        # Only a ceiling candidate — trust detect_room
        return cand_ceil, None, False, {'ceil_area': 0.0, 'floor_area': 0.0}

    xy_ceil = _plane_inliers_xy(pts, cand_ceil, distance_thresh)
    xy_floor = _plane_inliers_xy(pts, cand_floor, distance_thresh)
    area_ceil = _convex_hull_area(xy_ceil)
    area_floor = _convex_hull_area(xy_floor)

    info = {'ceil_area': area_ceil, 'floor_area': area_floor}

    # Swap (i.e. treat detect_room's "floor" as the real ceiling) only when
    # the floor-labelled plane's XY hull is SIGNIFICANTLY larger — not just
    # marginally. In well-scanned rooms with open floor space, the floor's
    # hull can be a few percent larger than the ceiling's just by chance,
    # which is NOT evidence of leveling inversion. The Unitree L2 inverted
    # case shows a ratio of ~1.4x; a threshold of 1.25x excludes the 1.05-
    # 1.07x noise while catching real inversions.
    #
    # Root-caused via 4-agent investigation (2026-04-12): previously any
    # `area_floor > area_ceil` triggered a swap and a 3 m Z sign-flip that
    # emptied the wall-band on two well-leveled test scans.
    SWAP_HULL_RATIO = 1.25
    ratio = (area_floor / area_ceil) if area_ceil > 1e-6 else 1.0
    if area_floor > area_ceil * SWAP_HULL_RATIO:
        if verbose:
            print(f"[FLOORPLAN] auto-swapped floor/ceiling — leveling "
                  f"inverted (ceil hull={area_ceil:.2f} m², "
                  f"floor hull={area_floor:.2f} m², ratio={ratio:.2f} "
                  f"> {SWAP_HULL_RATIO:.2f}).")
        return cand_floor, cand_ceil, True, info

    if verbose:
        print(f"[FLOORPLAN] ceiling plane: hull={area_ceil:.2f} m² "
              f"(floor candidate hull={area_floor:.2f} m², ratio={ratio:.2f} "
              f"≤ {SWAP_HULL_RATIO:.2f} — no swap).")
    return cand_ceil, cand_floor, False, info


def detect_floor_ceiling_robust(pcd, gravity_up=None, imus=None,
                                 bin_width=0.02, verbose=False):
    """Robust floor/ceiling Z using RANSAC + IMU gravity.

    Gravity resolution order:
        explicit `gravity_up` → `estimate_gravity(imus)` → `[0, 0, 1]`.

    Tier 1: detect_room(pcd, gravity_up) returns the lowest horizontal
            plane (floor) and the highest horizontal plane above
            floor + 1.0 m (ceiling). When both are populated, use them.
    Tier 2: floor hit but no ceiling (open-ceiling scans, tall rooms).
            Ceiling_z = 98th percentile of (points · gravity_up) above
            floor_z + 1.0 m.
    Tier 3: detect_room fails entirely (fewer than ~100 horizontal
            points). Fall back to the original Z-histogram top-2 peaks
            for parity with the reference tool. Only reached on
            pathological inputs.

    Args:
        pcd:         Open3D PointCloud.
        gravity_up:  (3,) unit vector for "up". If None, try imus, else [0,0,1].
        imus:        list of (t, gyro, acc) — IMU prior when gravity_up is None.
        bin_width:   Z-histogram bin size for Tier 3 fallback.
        verbose:     Print which tier produced the answer.

    Returns:
        (floor_z, ceiling_z, ceiling_plane_or_none). The third element is
        the RANSAC ceiling Plane when Tier 1 succeeds (so callers can do
        point-to-plane masking); None for Tier 2/3 fallbacks.

    The Tier 1 path also auto-selects between detect_room's floor- and
    ceiling-labelled planes by XY coverage, so the returned ceiling is
    the physically-correct surface even if the leveling is sign-inverted
    (see _select_ceiling_plane).
    """
    # Resolve gravity
    if gravity_up is None:
        if imus:
            try:
                gravity_up = np.asarray(estimate_gravity(imus), dtype=float)
            except Exception:
                gravity_up = np.array([0.0, 0.0, 1.0])
        else:
            gravity_up = np.array([0.0, 0.0, 1.0])
    gravity_up = np.asarray(gravity_up, dtype=float)
    n = np.linalg.norm(gravity_up)
    if n < 1e-6:
        gravity_up = np.array([0.0, 0.0, 1.0])
    else:
        gravity_up = gravity_up / n

    pts = np.asarray(pcd.points)
    z_along = pts @ gravity_up  # scalar height along gravity

    # ---- Tier 1: detect_room ----
    try:
        room = detect_room(pcd, gravity_up=gravity_up)
    except Exception as e:
        if verbose:
            print(f"[FLOORPLAN] detect_room raised: {e}")
        room = None

    if room is not None and room.floor is not None and room.ceiling is not None:
        # Auto-pick the real ceiling by XY coverage. Robust to leveling
        # inversion: picks the physical ceiling even if detect_room's
        # "floor"/"ceiling" labels are swapped.
        ceiling_plane, floor_plane, swapped, _info = _select_ceiling_plane(
            pts, room, verbose=verbose)
        z_ceiling = float(ceiling_plane.centroid @ gravity_up)
        z_floor = float(floor_plane.centroid @ gravity_up)
        # For display consistency, always report floor_z < ceiling_z.
        # When leveling is inverted, the physical ceiling is at the lower
        # Z in the leveled frame; we flip signs so the reported heights are
        # physically sensible (floor below, ceiling above). The returned
        # `ceiling_plane` still references the physical ceiling surface so
        # that point-to-plane masking works on the real cloud.
        if swapped:
            floor_z = float(-z_floor)      # physical floor — make positive up
            ceiling_z = float(-z_ceiling)  # physical ceiling — make positive up
        else:
            floor_z = z_floor
            ceiling_z = z_ceiling
        if verbose:
            print(f"[FLOORPLAN] floor/ceiling via RANSAC (Tier 1): "
                  f"floor={floor_z:.3f}, ceiling={ceiling_z:.3f}"
                  f"{' (Z-flipped for display — leveling is inverted)' if swapped else ''}")
        return floor_z, ceiling_z, ceiling_plane

    # ---- Tier 2: floor only — percentile above floor + 1 m ----
    # If RANSAC found a good floor, preserve it. Only the ceiling falls back.
    if room is not None and room.floor is not None:
        floor_z = float(room.floor_height)
        above = z_along[z_along > floor_z + 1.0]
        if above.size >= 50:
            ceiling_z = float(np.percentile(above, 98))
            if verbose:
                print(f"[FLOORPLAN] ceiling via percentile (Tier 2): "
                      f"floor={floor_z:.3f}, ceiling={ceiling_z:.3f}")
            return floor_z, ceiling_z, None
        # Too few points above floor+1m (pathological scan). Keep the RANSAC
        # floor but use the 99th percentile of the full Z range as ceiling.
        if z_along.size >= 50:
            ceiling_z = float(np.percentile(z_along, 99))
            if ceiling_z - floor_z > 0.5:  # at least half-meter to be usable
                if verbose:
                    print(f"[FLOORPLAN] ceiling via 99th-pct fallback (Tier 2b): "
                          f"floor={floor_z:.3f} (RANSAC), ceiling={ceiling_z:.3f}")
                return floor_z, ceiling_z, None

    # ---- Tier 3: histogram fallback (reference-tool parity) ----
    if z_along.size == 0:
        raise ValueError("empty point cloud")
    bins = np.arange(z_along.min(), z_along.max() + bin_width, bin_width)
    hist, edges = np.histogram(z_along, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    zhist = gaussian_filter1d(hist.astype(float), sigma=3)
    pks, props = find_peaks(zhist, height=np.max(zhist) * 0.1, distance=10)
    if len(pks) >= 2:
        top2 = pks[np.argsort(props['peak_heights'])[-2:]]
        top2 = np.sort(top2)
        floor_z = float(centers[top2[0]])
        ceiling_z = float(centers[top2[1]])
    else:
        floor_z = float(np.percentile(z_along, 5))
        ceiling_z = float(np.percentile(z_along, 95))

    if verbose:
        print(f"[FLOORPLAN] floor/ceiling via histogram fallback (Tier 3): "
              f"floor={floor_z:.3f}, ceiling={ceiling_z:.3f}")
    return floor_z, ceiling_z, None


# ============================================================
# CORE GEOMETRIC FUNCTIONS (1:1 port from reference)
# ============================================================

def _voxel_downsample(pcd, voxel_size=0.03, sor_neighbors=20, sor_std=2.0):
    """Statistical outlier removal + voxel downsample. In-memory equivalent
    of the reference tool's `load_and_preprocess`."""
    n_raw = len(pcd.points)
    pcd2, _ = pcd.remove_statistical_outlier(nb_neighbors=sor_neighbors, std_ratio=sor_std)
    pcd2 = pcd2.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(pcd2.points), n_raw


def extract_ceiling_points(pts, ceiling_z, band=0.15, floor_z=None,
                            ceiling_plane=None):
    """Extract ceiling-level points projected to XY.

    If `ceiling_plane` is provided (the RANSAC ceiling plane from
    detect_room), use point-to-plane distance — robust to residual leveling
    tilt and correct when the plane's normal is not exactly [0, 0, 1].
    Otherwise fall back to a tight Z-band centred on `ceiling_z`.

    The previous implementation used a 70 %-of-room-height cutoff, which
    produced a ~1 m thick band that included tops of tall furniture
    (67.9 % contamination measured on the reference scan). That heuristic
    is intentionally removed.

    Args:
        pts:            (N, 3) cloud points.
        ceiling_z:      ceiling height along gravity (for the Z-band fallback).
        band:           half-thickness of the mask in meters. Default 0.15.
        floor_z:        unused in the current implementation; kept for
                        backwards compatibility.
        ceiling_plane:  Plane object from room_structure.detect_room with
                        `normal` and `centroid` attributes. Optional.

    Returns:
        (M, 2) XY coordinates of the masked points.
    """
    del floor_z  # unused; retained for backwards compatibility

    if ceiling_plane is not None:
        # Point-to-plane distance (preferred path).
        signed = (pts - ceiling_plane.centroid) @ ceiling_plane.normal
        mask = np.abs(signed) < band
    else:
        # Tight Z-band fallback (Tier 2/3 of detect_floor_ceiling_robust).
        mask = np.abs(pts[:, 2] - ceiling_z) < band
    return pts[mask][:, :2]


def ceiling_to_binary(ceil_xy, resolution=0.03, close_kernel=11, open_kernel=5):
    """Project ceiling XY points to a binary mask and clean with morphology."""
    x_min, y_min = ceil_xy.min(axis=0) - 0.5
    x_max, y_max = ceil_xy.max(axis=0) + 0.5
    nx = int((x_max - x_min) / resolution)
    ny = int((y_max - y_min) / resolution)

    grid, xe, ye = np.histogram2d(
        ceil_xy[:, 0], ceil_xy[:, 1],
        bins=[nx, ny], range=[[x_min, x_max], [y_min, y_max]])

    if np.any(grid > 0):
        cap = np.percentile(grid[grid > 0], 90)
        g8 = (np.clip(grid, 0, cap) / cap * 255).astype(np.uint8)
    else:
        g8 = np.zeros((nx, ny), dtype=np.uint8)

    _, binary = cv2.threshold(g8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel)))
    binary = (binary_fill_holes(binary > 0).astype(np.uint8)) * 255
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel)))

    return binary, g8, xe, ye, resolution, x_min, y_min


def remove_outlier_clusters(binary):
    """Keep only the largest connected component; soften jagged edges."""
    n_labels, labeled, stats, _ = cv2.connectedComponentsWithStats(binary)
    if n_labels <= 1:
        return binary, 0

    areas = stats[1:, cv2.CC_STAT_AREA]
    biggest = np.argmax(areas) + 1
    clean = np.zeros_like(binary)
    clean[labeled == biggest] = 255
    clean = cv2.erode(clean, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    clean = cv2.dilate(clean, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    return clean, n_labels - 2


def trace_and_simplify(binary, epsilon_ratio=0.012):
    """Contour trace + Douglas-Peucker simplification."""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    epsilon = epsilon_ratio * perimeter
    simplified = cv2.approxPolyDP(contour, epsilon, True)
    return contour, simplified


def pixel_to_real(px_coords, x_min, y_min, resolution):
    real = np.zeros_like(px_coords, dtype=float)
    real[:, 0] = x_min + px_coords[:, 1] * resolution
    real[:, 1] = y_min + px_coords[:, 0] * resolution
    return real


def detect_corners(binary, g8, x_min, y_min, resolution, max_corners=20,
                    quality=0.02, min_distance_m=0.5):
    """Shi-Tomasi corner detection on the ceiling mask edges."""
    min_distance_px = int(min_distance_m / resolution)
    edges = cv2.Canny(binary, 50, 150)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    corners_px = cv2.goodFeaturesToTrack(
        edges, maxCorners=max_corners, qualityLevel=quality,
        minDistance=min_distance_px, blockSize=7)
    if corners_px is None or len(corners_px) < 3:
        return None

    corners_px = corners_px.reshape(-1, 2)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)

    on_boundary = []
    for cx, cy in corners_px:
        dist = abs(cv2.pointPolygonTest(contour, (float(cx), float(cy)), True))
        if dist < 8:
            on_boundary.append([cx, cy])
    if len(on_boundary) < 3:
        return None

    on_boundary = np.array(on_boundary)
    real = np.zeros_like(on_boundary, dtype=float)
    real[:, 0] = x_min + on_boundary[:, 1] * resolution
    real[:, 1] = y_min + on_boundary[:, 0] * resolution

    centroid = real.mean(axis=0)
    angles = np.arctan2(real[:, 1] - centroid[1], real[:, 0] - centroid[0])
    order = np.argsort(angles)
    real = real[order]

    return real, on_boundary


def snap_polygon_to_angles(coords, snap_angles_deg=None):
    """Snap each polygon edge to nearest listed angle, then re-intersect."""
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
    poly = ShapelyPolygon(corners)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.geom_type == 'MultiPolygon':
        poly = max(poly.geoms, key=lambda g: g.area)
    return poly


def extract_walls(poly, min_length=0.15):
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
# VISUALIZATION (PNG export only — no DXF by design)
# ============================================================

def _export_pipeline_pngs(walls, room_poly, pts, binary, clean, g8, xe, ye,
                          real_coords, snapped_coords, output_dir, name,
                          floor_z, ceiling_z):
    """Generate the pipeline / floorplan / overlay PNGs."""
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
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=3.5,
                solid_capstyle='round')
        if l > 0.3:
            mid = (p1 + p2) / 2
            d = p2 - p1
            nm = np.linalg.norm(d)
            perp = np.array([-d[1], d[0]]) / nm * 0.18
            ax.text(mid[0] + perp[0], mid[1] + perp[1], f'{l:.2f}m',
                    ha='center', fontsize=7, color='#444',
                    rotation=a if a <= 90 else a - 180)
    ax.text(room_poly.centroid.x, room_poly.centroid.y,
            f"Area: {room_poly.area:.1f} m2\nCeiling: {ceiling_z:.2f}m\n"
            f"Height: {h:.2f}m",
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

    # 03: Overlay on raw point cloud
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


def _export_refined_png(walls_d, poly_d, walls_d_meta, pts,
                         output_dir, name, floor_z, ceiling_z):
    """Dedicated refined-polygon PNG with per-wall snap-kind coloring."""
    h = ceiling_z - floor_z

    res_v = 0.02
    xmn, ymn = pts[:, :2].min(0) - 0.5
    xmx, ymx = pts[:, :2].max(0) + 0.5
    nxv, nyv = int((xmx - xmn) / res_v), int((ymx - ymn) / res_v)
    gv, xev, yev = np.histogram2d(pts[:, 0], pts[:, 1], bins=[nxv, nyv],
                                   range=[[xmn, xmx], [ymn, ymx]])
    p93 = np.percentile(gv[gv > 0], 93) if np.any(gv > 0) else 1
    g8v = (np.clip(gv, 0, p93) / p93 * 255).astype(np.uint8)
    extv = [xev[0], xev[-1], yev[0], yev[-1]]

    # Per-snap-kind color (used when no vision type is available OR
    # the wall's type is 'wall'/'unknown').
    kind_color = {
        'dominant': '#2E7D32',    # green: learned-dominant snap
        'manhattan': '#1565C0',   # blue: 0°/90° snap
        'diagonal': '#FF6F00',    # orange: 45°/135° snap
        'hex': '#8E24AA',         # purple: 30°/60° snap
        'free': '#D84315',        # red: preserved at fitted angle
        'fallback_a': '#616161',  # grey: fell back to variant A
    }
    # Per-type color — overrides kind_color for window/door/glass. A
    # wall typed as 'wall' or 'unknown' falls through to kind_color so
    # the snap provenance stays visible.
    type_color = {
        'window': '#00BFFF',   # cyan
        'door':   '#FF7F00',   # orange (distinct from diagonal orange)
        'glass':  '#ADD8E6',   # light cyan
    }

    # Detect whether any wall has a vision-derived type — decides whether
    # to draw the second legend (types) at all.
    has_any_type = any(
        isinstance(m, dict) and 'type' in m for m in walls_d_meta
    ) if walls_d_meta else False

    fig, ax = plt.subplots(figsize=(14, 14), dpi=200)
    ax.imshow(g8v.T, origin='lower', cmap='gray_r', extent=extv, alpha=0.25)
    ax.set_facecolor('#FAFAFA')
    if hasattr(poly_d, 'exterior'):
        rx, ry = poly_d.exterior.xy
        ax.fill(rx, ry, color='#E8F5E9', alpha=0.4)

    legend_kinds = set()      # snap-kind legend entries (green/blue/...)
    legend_types = set()      # type legend entries (cyan/orange/...)
    type_handles = []         # keep matplotlib Line2D refs for the 2nd legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    # Track which feature-class markers we drew so the features legend can
    # aggregate them (dashed-line markers drawn on walls that carry a
    # feature but are not themselves typed as that class).
    feature_markers_seen = set()
    for idx, (p1, p2, a, l) in enumerate(walls_d):
        kind = 'free'
        wtype = None
        wfeatures = []
        if idx < len(walls_d_meta) and walls_d_meta[idx] is not None:
            kind = walls_d_meta[idx].get('snapped_to', 'free')
            wtype = walls_d_meta[idx].get('type')
            wfeatures = walls_d_meta[idx].get('features', []) or []

        # Primary line color: type color wins for window/door/glass;
        # otherwise fall through to the snap-kind color so the kind is
        # still visible.
        if wtype in type_color:
            color = type_color[wtype]
            # Track for the types legend (not the kinds legend).
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '-',
                    color=color, linewidth=4.0, solid_capstyle='round')
            if wtype not in legend_types:
                type_handles.append(
                    Line2D([0], [0], color=color, linewidth=4.0,
                           label=wtype))
                legend_types.add(wtype)
        else:
            color = kind_color.get(kind, '#000000')
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '-',
                    color=color, linewidth=3.5, solid_capstyle='round',
                    label=kind if kind not in legend_kinds else None)
            legend_kinds.add(kind)

        # Secondary-feature overlay: for each feature present on this
        # wall, draw a short dashed overlay in the feature's color
        # centered on the wall midpoint (25 % of wall length, capped at
        # 1.0 m). Also list features in the length-label text.
        if wfeatures and l > 0.3:
            d = p2 - p1
            nm = np.linalg.norm(d)
            udir = d / nm if nm > 1e-6 else np.array([1.0, 0.0])
            mid = (p1 + p2) / 2
            dash_len = min(1.0, 0.25 * l)
            # Stack features: slight perpendicular offset per feature so
            # multiple features on one wall don't overlap.
            perp = np.array([-udir[1], udir[0]])
            for j, feat in enumerate(wfeatures):
                if feat not in type_color:
                    continue
                offset = (j - (len(wfeatures) - 1) / 2.0) * 0.08
                a_pt = mid - udir * (dash_len / 2) + perp * offset
                b_pt = mid + udir * (dash_len / 2) + perp * offset
                ax.plot([a_pt[0], b_pt[0]], [a_pt[1], b_pt[1]],
                        linestyle='--', color=type_color[feat],
                        linewidth=2.8, solid_capstyle='round')
                feature_markers_seen.add(feat)

        if l > 0.3:
            mid = (p1 + p2) / 2
            d = p2 - p1
            nm = np.linalg.norm(d)
            perp = np.array([-d[1], d[0]]) / nm * 0.18
            feature_str = ''
            if wfeatures:
                feature_str = ' [' + ','.join(wfeatures) + ']'
            ax.text(mid[0] + perp[0], mid[1] + perp[1],
                    f'{l:.2f}m{feature_str}',
                    ha='center', fontsize=7, color='#333',
                    rotation=a if a <= 90 else a - 180)

    ax.text(poly_d.centroid.x, poly_d.centroid.y,
            f"RANSAC-refined\nArea: {poly_d.area:.1f} m2\n"
            f"Ceiling: {ceiling_z:.2f}m\nHeight: {h:.2f}m\n"
            f"Walls: {len(walls_d)}",
            ha='center', va='center', fontsize=11, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.4'))
    ax.grid(True, alpha=0.06, color='#4488CC')
    ax.set_axisbelow(True)
    ax.set_title(f'Refined Floor Plan — {len(walls_d)} walls',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    # Primary legend: snap kinds (angle source)
    first_legend = ax.legend(loc='upper right', fontsize=9, framealpha=0.9,
                              title='Wall angle source')
    # Secondary legend: vision types + features.
    # Primary-type walls get a solid line; feature overlays get a dashed
    # line in the same color — combined here so the user sees the full
    # palette.
    combined_type_handles = list(type_handles)
    for feat in sorted(feature_markers_seen):
        if feat in legend_types:
            continue  # already in solid-line legend
        combined_type_handles.append(
            Line2D([0], [0], color=type_color[feat], linewidth=2.8,
                   linestyle='--', label=f'{feat} (feature)'))
    if combined_type_handles:
        ax.add_artist(first_legend)
        ax.legend(handles=combined_type_handles, loc='lower right',
                  fontsize=9, framealpha=0.9, title='Vision type')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{name}_refined.png', bbox_inches='tight')
    plt.close()


def _export_comparison_png(variants, pts, output_dir, name,
                           corner_coords_real):
    """Variant side-by-side over the raw point cloud (supports 3 or 4)."""
    res_v = 0.02
    xmn, ymn = pts[:, :2].min(0) - 0.5
    xmx, ymx = pts[:, :2].max(0) + 0.5
    nxv, nyv = int((xmx - xmn) / res_v), int((ymx - ymn) / res_v)
    gv, xev, yev = np.histogram2d(pts[:, 0], pts[:, 1], bins=[nxv, nyv],
                                   range=[[xmn, xmx], [ymn, ymx]])
    p93 = np.percentile(gv[gv > 0], 93) if np.any(gv > 0) else 1
    g8v = (np.clip(gv, 0, p93) / p93 * 255).astype(np.uint8)
    extv = [xev[0], xev[-1], yev[0], yev[-1]]

    n_variants = len(variants)
    colors = ['lime', 'cyan', 'orange', 'magenta'][:n_variants]
    fig, axes = plt.subplots(1, n_variants, figsize=(8 * n_variants, 8), dpi=200)
    if n_variants == 1:
        axes = [axes]
    for idx, (key, (walls, poly, label)) in enumerate(variants.items()):
        ax = axes[idx]
        ax.imshow(g8v.T, origin='lower', cmap='gray_r', extent=extv, alpha=0.3)
        if hasattr(poly, 'exterior'):
            rx, ry = poly.exterior.xy
            ax.fill(rx, ry, color=colors[idx], alpha=0.15)
        for p1, p2, a, l in walls:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=2.5,
                    solid_capstyle='round')
            if l > 0.3:
                mid = (p1 + p2) / 2
                d = p2 - p1
                nm = np.linalg.norm(d)
                perp = np.array([-d[1], d[0]]) / nm * 0.15
                ax.text(mid[0] + perp[0], mid[1] + perp[1], f'{l:.2f}m',
                        ha='center', fontsize=6, color='yellow',
                        fontweight='bold',
                        rotation=a if a <= 90 else a - 180)
        if key == 'B_corners' and corner_coords_real is not None:
            ax.scatter(corner_coords_real[:, 0], corner_coords_real[:, 1],
                       s=80, c='red', marker='o', zorder=5,
                       edgecolors='white', linewidth=1.5)
        ax.set_title(f'{label}\n{poly.area:.1f} m2, {len(walls)} walls',
                     fontsize=11, fontweight='bold')
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
    fig.suptitle(f'{n_variants} Variants on Point Cloud', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{name}_comparison.png', bbox_inches='tight')
    plt.close()


def _export_corners_png(clean, g8, xe, ye, poly_b, corner_coords_real,
                        output_dir, name):
    """Corner-detection detail PNG."""
    ext_ceil = [xe[0], xe[-1], ye[0], ye[-1]]
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), dpi=150)
    axes[0].imshow(g8.T, origin='lower', cmap='hot', extent=ext_ceil)
    axes[0].set_title('Ceiling Density')
    edges_img = cv2.Canny(clean, 50, 150)
    axes[1].imshow(edges_img.T, origin='lower', cmap='gray', extent=ext_ceil)
    if corner_coords_real is not None:
        axes[1].scatter(corner_coords_real[:, 0], corner_coords_real[:, 1],
                        s=100, c='red', marker='o', zorder=5,
                        edgecolors='white', linewidth=2)
    n_corners = len(corner_coords_real) if corner_coords_real is not None else 0
    axes[1].set_title(f'Edges + Corners ({n_corners})')
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
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
    fig.suptitle('Corner Detection Pipeline', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{name}_corners.png', bbox_inches='tight')
    plt.close()


# ============================================================
# WALL REFINEMENT PIPELINE (variant D_refined)
# ============================================================
# Research-grade refinement of the simplified ceiling polygon:
#   Stage 1: per-edge wall-band RANSAC + Huber LO-refine
#   Stage 2: learn dominant directions (weighted histogram, architectural prior)
#   Stage 3: tolerance-gated tiered snap
#   Stage 4: merge collinear neighbours
#   Stage 5: vertex translation (constrained LSQ, fallback to re-intersection)
#   Stage 6: gap-split segments
#
# Grounded in:
#   - ZInD (80% Manhattan, 15% diagonal, 5% rare) — architectural prior strengths
#   - Cloud2BIM arXiv:2503.11498 — 3° collinear-merge tolerance
#   - Chum et al. 2003 LO-RANSAC — local Huber refinement step
#   - Structure-preserving Simplification arXiv:2408.06814 — vertex translation

# Fixed snap priors and their pull strengths (from ZInD frequencies).
# Angles in radians, on [0, π) (edges are undirected).
_MANHATTAN_ANGLES = np.array([0.0, np.pi / 2])        # 0°, 90°
_DIAGONAL_ANGLES = np.array([np.pi / 4, 3 * np.pi / 4])  # 45°, 135°
_HEX_ANGLES = np.array([np.pi / 6, np.pi / 3,
                        2 * np.pi / 3, 5 * np.pi / 6])    # 30, 60, 120, 150°


def _wrap_angle_pi(a):
    """Wrap a scalar or array of angles to [0, π)."""
    return np.mod(a, np.pi)


def _angle_circ_diff(a, b):
    """Smallest absolute difference between two angles on [0, π)."""
    d = np.abs(_wrap_angle_pi(a) - _wrap_angle_pi(b))
    return np.minimum(d, np.pi - d)


def _fit_line_huber(points_xy):
    """Robust line fit via cv2.fitLine(DIST_HUBER). Returns (direction, point)."""
    if len(points_xy) < 2:
        return None, None
    pts = points_xy.astype(np.float32).reshape(-1, 1, 2)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
    direction = np.array([vx, vy], dtype=np.float64)
    direction /= max(np.linalg.norm(direction), 1e-9)
    return direction, np.array([x0, y0], dtype=np.float64)


def _point_line_residuals(points_xy, direction, point_on_line):
    """Perpendicular distance from each 2D point to the given line."""
    perp = np.array([-direction[1], direction[0]])
    return (points_xy - point_on_line) @ perp


def _adaptive_ransac_threshold(residuals):
    """MAD-based threshold: 2.5 * 1.4826 * MAD, clamped to [0.02, 0.06]."""
    mad = np.median(np.abs(residuals - np.median(residuals)))
    sigma = 1.4826 * mad
    return float(np.clip(2.5 * sigma, 0.02, 0.06))


# ----- Stage 1: per-edge wall-band RANSAC + Huber LO-refine -----

def _stage1_per_edge_ransac(walls_in, wall_band_xy, edge_band=0.25,
                             min_inliers=20, max_iter=200, verbose=False):
    """For each polygon edge, fit a robust line to nearby wall-band points.

    Args:
        walls_in: list of (p1, p2, angle_deg, length_m) from `extract_walls`.
        wall_band_xy: (N, 2) XY coords of wall-band points (projected cloud
                      at mid-height — avoids floor/ceiling overlap).
        edge_band: perpendicular search distance around each edge (m).
        min_inliers: minimum inliers to accept a refined fit.

    Returns:
        list of dicts with keys:
          'p1', 'p2'        — refined endpoints along the fitted line
          'direction'       — unit vector along the wall
          'point_on_line'   — any point on the fitted line
          'inliers_xy'      — (K, 2) inlier points
          'residual'        — mean |perpendicular distance| of inliers
          'length'          — length of the refined segment
          'angle_deg'       — angle on [0, 180)
          'confidence'      — float in [0, 1] (roughly inliers / edge-extent)
          'fallback'        — True if original segment is used (no good fit)
    """
    refined = []

    for p1, p2, angle_deg, length in walls_in:
        edge_vec = np.asarray(p2) - np.asarray(p1)
        edge_len = float(np.linalg.norm(edge_vec))
        if edge_len < 1e-6:
            refined.append({
                'p1': np.asarray(p1, dtype=float),
                'p2': np.asarray(p2, dtype=float),
                'direction': np.array([1.0, 0.0]),
                'point_on_line': np.asarray(p1, dtype=float),
                'inliers_xy': np.zeros((0, 2)),
                'residual': 0.0,
                'length': edge_len,
                'angle_deg': angle_deg,
                'confidence': 0.0,
                'fallback': True,
            })
            continue
        edge_dir = edge_vec / edge_len
        edge_perp = np.array([-edge_dir[1], edge_dir[0]])
        mid = (np.asarray(p1) + np.asarray(p2)) / 2

        # Gather wall-band points within edge_band of the infinite edge line
        # AND roughly within the edge's along-extent (with slack).
        rel = wall_band_xy - mid
        perp_dist = np.abs(rel @ edge_perp)
        along_dist = np.abs(rel @ edge_dir)
        slack = 0.5  # extra half-metre at each end to capture wall extension
        mask = (perp_dist < edge_band) & (along_dist < edge_len / 2 + slack)
        candidates = wall_band_xy[mask]

        if len(candidates) < min_inliers:
            # Too few points — keep original edge as fallback
            refined.append({
                'p1': np.asarray(p1, dtype=float),
                'p2': np.asarray(p2, dtype=float),
                'direction': edge_dir,
                'point_on_line': mid,
                'inliers_xy': candidates,
                'residual': 0.0,
                'length': edge_len,
                'angle_deg': angle_deg,
                'confidence': float(len(candidates)) / max(int(edge_len / 0.05), 1),
                'fallback': True,
            })
            continue

        # Initial Huber fit on candidates
        direction, point_on_line = _fit_line_huber(candidates)
        if direction is None:
            refined.append({
                'p1': np.asarray(p1, dtype=float),
                'p2': np.asarray(p2, dtype=float),
                'direction': edge_dir,
                'point_on_line': mid,
                'inliers_xy': candidates,
                'residual': 0.0,
                'length': edge_len,
                'angle_deg': angle_deg,
                'confidence': 0.0,
                'fallback': True,
            })
            continue

        # LO-RANSAC: iterate to tighten the inlier set using MAD threshold
        inliers = candidates
        for _ in range(3):
            residuals = _point_line_residuals(inliers, direction, point_on_line)
            thresh = _adaptive_ransac_threshold(residuals)
            inlier_mask = np.abs(residuals) < thresh
            if inlier_mask.sum() < min_inliers:
                break
            inliers = inliers[inlier_mask]
            new_dir, new_pt = _fit_line_huber(inliers)
            if new_dir is None:
                break
            direction, point_on_line = new_dir, new_pt

        # Project inliers onto fitted line to find endpoints
        if len(inliers) < min_inliers:
            # Keep original; mark low-confidence
            refined.append({
                'p1': np.asarray(p1, dtype=float),
                'p2': np.asarray(p2, dtype=float),
                'direction': edge_dir,
                'point_on_line': mid,
                'inliers_xy': candidates,
                'residual': 0.0,
                'length': edge_len,
                'angle_deg': angle_deg,
                'confidence': float(len(candidates)) / max(int(edge_len / 0.05), 1),
                'fallback': True,
            })
            continue

        t = (inliers - point_on_line) @ direction
        t_min, t_max = float(t.min()), float(t.max())
        new_p1 = point_on_line + direction * t_min
        new_p2 = point_on_line + direction * t_max
        new_length = float(np.linalg.norm(new_p2 - new_p1))
        # Angle on [0, 180)
        new_angle = float(np.degrees(np.arctan2(direction[1], direction[0])) % 180)

        residuals_final = _point_line_residuals(inliers, direction, point_on_line)
        residual = float(np.mean(np.abs(residuals_final)))
        # Confidence: how full is the expected edge length?
        # (inliers span at least 60% of original edge → high confidence)
        confidence = min(1.0, new_length / max(edge_len * 0.6, 0.1))
        confidence *= min(1.0, len(inliers) / max(edge_len / 0.03, 20))

        refined.append({
            'p1': new_p1,
            'p2': new_p2,
            'direction': direction,
            'point_on_line': point_on_line,
            'inliers_xy': inliers,
            'residual': residual,
            'length': new_length,
            'angle_deg': new_angle,
            'confidence': float(confidence),
            'fallback': False,
        })

    if verbose:
        n_good = sum(1 for r in refined if not r['fallback'])
        print(f"  [Stage 1] {n_good}/{len(refined)} edges refined "
              f"(mean residual={np.mean([r['residual'] for r in refined if not r['fallback']]) if n_good else 0:.4f} m)")
    return refined


# ----- Stage 2: learn dominant directions -----

def _stage2_dominant_directions(refined_edges, k_max=3, bandwidth_deg=2.0,
                                 min_separation_deg=12.0, min_edge_length=0.3,
                                 prior_weights=None, verbose=False):
    """Length-weighted circular histogram on [0, π) with NMS.

    Architectural prior (from ZInD frequencies) seeds the histogram with
    pseudo-counts at 0°/45°/90°/135° with strengths {1.0, 0.3, 1.0, 0.3}
    relative to total perimeter length.
    """
    if prior_weights is None:
        prior_weights = {0.0: 1.0, 45.0: 0.3, 90.0: 1.0, 135.0: 0.3}

    # Collect voting edges (length > min threshold)
    votes = [(r['angle_deg'], r['length']) for r in refined_edges
             if r['length'] >= min_edge_length and not r['fallback']]
    total_len = sum(l for _, l in votes) + 1e-9

    # 1° bin histogram on [0, 180)
    hist = np.zeros(180)
    for angle_deg, length in votes:
        bin_idx = int(angle_deg) % 180
        hist[bin_idx] += length

    # Seed with architectural prior (pseudo-counts scaled to total length)
    for ang_deg, strength in prior_weights.items():
        bin_idx = int(ang_deg) % 180
        hist[bin_idx] += total_len * strength * 0.10

    # Gaussian smooth on the circular domain
    # (pad with copies to simulate wrapping, then trim)
    sigma = bandwidth_deg
    padded = np.concatenate([hist[-10:], hist, hist[:10]])
    smoothed = gaussian_filter1d(padded, sigma=sigma, mode='constant')[10:190]

    # NMS: find peaks above 10% of max, separated by >= min_separation_deg
    peaks = []
    order = np.argsort(-smoothed)
    min_height = 0.10 * smoothed.max()
    for idx in order:
        if smoothed[idx] < min_height:
            break
        ang = float(idx)
        if all(min(abs(ang - p), 180 - abs(ang - p)) >= min_separation_deg
               for p in peaks):
            peaks.append(ang)
        if len(peaks) >= k_max:
            break

    peaks_sorted = sorted(peaks)
    if verbose:
        print(f"  [Stage 2] dominant directions (deg): "
              f"{[round(p, 1) for p in peaks_sorted]}")
    return np.array(peaks_sorted)


# ----- Stage 3: tolerance-gated tiered snap -----

def _stage3_tiered_snap(refined_edges, dominant_deg, tol_dominant=7.0,
                        tol_diagonal=4.0, tol_hex=3.0, verbose=False):
    """Snap each refined edge angle to the closest in-tolerance direction.

    Priority order: dominant peaks (Stage 2) > architectural priors > no snap.
    Returns list of dicts (same shape as refined_edges) with 'angle_deg' and
    new 'snapped_to' key in {"dominant", "manhattan", "diagonal", "hex", "free"}.
    """
    out = []
    for e in refined_edges:
        ang = e['angle_deg']
        if e['fallback']:
            # Cannot trust the angle — leave untouched, mark free
            out.append({**e, 'snapped_to': 'free'})
            continue

        # Try dominant peaks first (broadest tolerance)
        best_target = None
        best_kind = None
        best_diff = np.inf
        for p in dominant_deg:
            d = min(abs(ang - p), 180 - abs(ang - p))
            if d < tol_dominant and d < best_diff:
                best_diff, best_target, best_kind = d, p, 'dominant'

        if best_target is None:
            # Tier 2: architectural priors (only if not already a dominant peak)
            def _not_already_dominant(p):
                return not any(
                    min(abs(p - dp), 180 - abs(p - dp)) < 1e-3
                    for dp in dominant_deg)

            for p in np.degrees(_MANHATTAN_ANGLES):
                if not _not_already_dominant(p):
                    continue
                d = min(abs(ang - p), 180 - abs(ang - p))
                if d < tol_dominant and d < best_diff:  # same tol as Tier 1
                    best_diff, best_target, best_kind = d, p, 'manhattan'
            for p in np.degrees(_DIAGONAL_ANGLES):
                if not _not_already_dominant(p):
                    continue
                d = min(abs(ang - p), 180 - abs(ang - p))
                if d < tol_diagonal and d < best_diff:
                    best_diff, best_target, best_kind = d, p, 'diagonal'
            for p in np.degrees(_HEX_ANGLES):
                if not _not_already_dominant(p):
                    continue
                d = min(abs(ang - p), 180 - abs(ang - p))
                if d < tol_hex and d < best_diff:
                    best_diff, best_target, best_kind = d, p, 'hex'

        if best_target is None:
            out.append({**e, 'snapped_to': 'free'})
            continue

        # Rebuild direction from snapped angle, keep point_on_line the same
        new_dir = np.array([np.cos(np.radians(best_target)),
                            np.sin(np.radians(best_target))])
        # Recompute endpoints: project original inliers onto the snapped line
        pol = e['point_on_line']
        if len(e['inliers_xy']) > 0:
            t = (e['inliers_xy'] - pol) @ new_dir
            t_min, t_max = float(t.min()), float(t.max())
        else:
            # Fallback: use old p1/p2 projected
            t1 = float((e['p1'] - pol) @ new_dir)
            t2 = float((e['p2'] - pol) @ new_dir)
            t_min, t_max = min(t1, t2), max(t1, t2)
        new_p1 = pol + new_dir * t_min
        new_p2 = pol + new_dir * t_max
        out.append({**e,
                    'direction': new_dir,
                    'p1': new_p1, 'p2': new_p2,
                    'angle_deg': float(best_target),
                    'length': float(np.linalg.norm(new_p2 - new_p1)),
                    'snapped_to': best_kind})

    if verbose:
        counts = {}
        for e in out:
            counts[e['snapped_to']] = counts.get(e['snapped_to'], 0) + 1
        print(f"  [Stage 3] snap tallies: {counts}")
    return out


# ----- Stage 4: merge collinear neighbours -----

def _stage4_merge_collinear(edges, angle_tol_deg=3.0, gap_max=1.2,
                             verbose=False):
    """Walk polygon; merge consecutive edges that are collinear continuations.

    Criteria: angle diff < angle_tol_deg AND perpendicular offset within the
    noise level AND along-line gap < gap_max. Refits the merged edge by
    Huber on the union of inliers.
    """
    if len(edges) < 2:
        return edges

    merged = []
    i = 0
    n = len(edges)
    while i < n:
        current = edges[i]
        # Try to extend by consuming the next edge if collinear
        j = (i + 1) % n
        while j != i:
            nxt = edges[j]
            # Angle check
            dang = min(abs(current['angle_deg'] - nxt['angle_deg']),
                       180 - abs(current['angle_deg'] - nxt['angle_deg']))
            if dang > angle_tol_deg:
                break
            # Perpendicular offset: midpoint of nxt onto current line
            mid_next = (nxt['p1'] + nxt['p2']) / 2
            perp = np.array([-current['direction'][1], current['direction'][0]])
            off = abs((mid_next - current['point_on_line']) @ perp)
            sigma_est = max(current['residual'], 0.02)
            if off > max(sigma_est * 3, 0.10):
                break
            # Along-line gap between projected endpoints
            t_cur_max = float((current['p2'] - current['point_on_line'])
                              @ current['direction'])
            t_next_min = float((nxt['p1'] - current['point_on_line'])
                               @ current['direction'])
            gap = t_next_min - t_cur_max
            if gap > gap_max:
                break
            # All checks passed — merge
            if nxt['fallback'] or current['fallback']:
                break
            combined_inliers = np.vstack([current['inliers_xy'],
                                          nxt['inliers_xy']])
            direction, pol = _fit_line_huber(combined_inliers)
            if direction is None:
                break
            t = (combined_inliers - pol) @ direction
            new_p1 = pol + direction * float(t.min())
            new_p2 = pol + direction * float(t.max())
            new_angle = float(np.degrees(np.arctan2(direction[1],
                                                     direction[0])) % 180)
            # Replace current with merged
            current = {
                'p1': new_p1, 'p2': new_p2,
                'direction': direction,
                'point_on_line': pol,
                'inliers_xy': combined_inliers,
                'residual': float(np.mean(np.abs(
                    _point_line_residuals(combined_inliers, direction, pol)))),
                'length': float(np.linalg.norm(new_p2 - new_p1)),
                'angle_deg': new_angle,
                'confidence': min(current.get('confidence', 0) + nxt.get('confidence', 0), 1.0),
                'fallback': False,
                'snapped_to': current.get('snapped_to', 'free'),
            }
            # Consume the merged edge and don't wrap around indefinitely
            i = j
            j = (j + 1) % n
            if j == i:
                break
        merged.append(current)
        i += 1

    if verbose:
        print(f"  [Stage 4] merged {len(edges)} → {len(merged)} edges")
    return merged


# ----- Stage 5: vertex translation (constrained LSQ + fallback) -----

def _stage5_vertex_translation(edges, verbose=False):
    """Replace each polygon vertex with the intersection of neighbour lines.

    Fallback to pairwise intersection when lines are near-parallel (denom
    below threshold).
    """
    if len(edges) < 3:
        return edges

    new_corners = []
    n = len(edges)
    for i in range(n):
        e1 = edges[i]
        e2 = edges[(i + 1) % n]
        d1, p1 = e1['direction'], e1['point_on_line']
        d2, p2 = e2['direction'], e2['point_on_line']
        denom = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(denom) < 1e-6:
            # Near-parallel; use the midpoint of e1.p2 and e2.p1
            new_corners.append((e1['p2'] + e2['p1']) / 2)
            continue
        dp = p2 - p1
        t = (dp[0] * d2[1] - dp[1] * d2[0]) / denom
        new_corners.append(p1 + t * d1)

    # Rebuild edges with new corners, keeping direction + snapped_to
    out = []
    for i in range(n):
        p1 = new_corners[i - 1] if i > 0 else new_corners[-1]
        p2 = new_corners[i]
        length = float(np.linalg.norm(p2 - p1))
        if length < 1e-6:
            continue
        dir_new = (p2 - p1) / length
        angle_new = float(np.degrees(np.arctan2(dir_new[1], dir_new[0])) % 180)
        src = edges[i]
        out.append({
            'p1': p1, 'p2': p2,
            'direction': dir_new,
            'point_on_line': (p1 + p2) / 2,
            # Stash the pre-Stage-5 RANSAC-fitted direction and anchor so
            # Stage 7 can project inliers in the frame they were fitted in.
            # Stage 5's `direction`/`point_on_line` above are derived from
            # the line intersection, NOT from the data — using them for
            # inlier projection introduces a subtle coordinate-frame bias.
            'ransac_direction': np.asarray(src['direction']).copy(),
            'ransac_point_on_line': np.asarray(src['point_on_line']).copy(),
            'inliers_xy': src['inliers_xy'],
            'residual': src['residual'],
            'length': length,
            'angle_deg': angle_new,
            'confidence': src.get('confidence', 0.0),
            'fallback': src.get('fallback', False),
            'snapped_to': src.get('snapped_to', 'free'),
        })

    if verbose:
        print(f"  [Stage 5] vertex translation complete ({len(out)} edges)")
    return out


# ----- Stage 7: length-constrained corner adjustment -----

def _stage7_length_constrain(edges, slack=0.30, verbose=False):
    """Pull/push each shared corner so neither adjacent wall over- or
    under-extends beyond its data support.

    Stage 5 computes corners as line-line intersections of the snapped wall
    directions, which can place corners far outside the physical wall
    extent (over-extension). Stage 7 uses each wall's Stage-1 RANSAC
    inliers to compute a data extent [t_min, t_max] and clamps the corner's
    projection onto each wall to that extent (± slack).

    Projections use `ransac_direction` and `ransac_point_on_line` (stashed
    by Stage 5) because Stage 5 overwrites `direction`/`point_on_line` with
    intersection-derived values that are geometrically valid but in a
    different frame from the RANSAC fit.

    Safeguards (from 4-agent review):
      - Adaptive slack: `min(slack, max(0.05, 0.1 * wall_length))` — tight
        for short walls to avoid over-trimming into near-zero length.
      - Zero-length guard: skip the corner move if it would collapse a
        wall below 0.10 m.
      - Per-corner sign check: skip the move if it would flip a wall's
        p1→p2 orientation relative to its direction (bad geometry).
      - Final polygon validity check (Shapely): if the post-Stage-7 polygon
        self-intersects or invalidates, revert ALL moves for that polygon
        and leave Stage 5's output unchanged (honest: better no refinement
        than silently-corrupted geometry).
    """
    n = len(edges)
    if n < 3:
        return edges, {'n_moved': 0, 'reverted': False}

    # Snapshot for potential revert
    snapshot = [{
        'p1': np.asarray(e['p1']).copy(),
        'p2': np.asarray(e['p2']).copy(),
        'length': float(e['length']),
        'angle_deg': float(e['angle_deg']),
    } for e in edges]

    n_moved = 0
    shifts = []

    for i in range(n):
        e_curr = edges[i]
        e_next = edges[(i + 1) % n]
        if e_curr.get('fallback', False) or e_next.get('fallback', False):
            continue

        # Use pre-Stage-5 RANSAC frame (stashed by Stage 5)
        dir_i = np.asarray(e_curr.get('ransac_direction',
                                       e_curr['direction']))
        pol_i = np.asarray(e_curr.get('ransac_point_on_line',
                                       e_curr['point_on_line']))
        dir_j = np.asarray(e_next.get('ransac_direction',
                                       e_next['direction']))
        pol_j = np.asarray(e_next.get('ransac_point_on_line',
                                       e_next['point_on_line']))

        in_i = e_curr.get('inliers_xy', np.zeros((0, 2)))
        in_j = e_next.get('inliers_xy', np.zeros((0, 2)))
        if len(in_i) < 10 or len(in_j) < 10:
            continue

        # C0 is the shared corner between wall i and wall i+1
        C0 = np.asarray(e_curr['p2'])

        # Current along-wall parameters of C0
        t_i = float((C0 - pol_i) @ dir_i)
        t_j = float((C0 - pol_j) @ dir_j)

        # Data extent from inliers (in the RANSAC frame)
        t_i_data = (in_i - pol_i) @ dir_i
        t_j_data = (in_j - pol_j) @ dir_j
        t_i_min, t_i_max = float(t_i_data.min()), float(t_i_data.max())
        t_j_min, t_j_max = float(t_j_data.min()), float(t_j_data.max())

        # Adaptive slack: tight for short walls, capped at `slack` for long
        len_i = float(np.linalg.norm(e_curr['p2'] - e_curr['p1']))
        len_j = float(np.linalg.norm(e_next['p2'] - e_next['p1']))
        slack_eff = min(slack, max(0.05, 0.10 * max(len_i, len_j)))

        t_i_cl = float(np.clip(t_i, t_i_min - slack_eff, t_i_max + slack_eff))
        t_j_cl = float(np.clip(t_j, t_j_min - slack_eff, t_j_max + slack_eff))

        # No change needed if both clamps are no-ops
        if abs(t_i_cl - t_i) < 1e-6 and abs(t_j_cl - t_j) < 1e-6:
            continue

        # Two clamped candidate corner positions (one per wall's line)
        cand_i = pol_i + dir_i * t_i_cl
        cand_j = pol_j + dir_j * t_j_cl
        C_new = 0.5 * (cand_i + cand_j)

        # Safeguard: zero-length guard
        new_len_curr = float(np.linalg.norm(C_new - e_curr['p1']))
        new_len_next = float(np.linalg.norm(e_next['p2'] - C_new))
        if new_len_curr < 0.10 or new_len_next < 0.10:
            continue

        # Safeguard: per-corner sign check — ensure wall orientation
        # (p1 → p2 along direction) is preserved after the move.
        old_t_p1_curr = float((np.asarray(e_curr['p1']) - pol_i) @ dir_i)
        new_t_p2_curr = float((C_new - pol_i) @ dir_i)
        if new_t_p2_curr <= old_t_p1_curr:
            continue  # flipping the wall — abort this corner
        old_t_p2_next = float((np.asarray(e_next['p2']) - pol_j) @ dir_j)
        new_t_p1_next = float((C_new - pol_j) @ dir_j)
        if new_t_p1_next >= old_t_p2_next:
            continue

        # Commit the corner move
        shift = float(np.linalg.norm(C_new - C0))
        e_curr['p2'] = C_new
        e_next['p1'] = C_new
        e_curr['length'] = new_len_curr
        e_next['length'] = new_len_next
        n_moved += 1
        shifts.append(shift)

    # Final polygon validity check. If Shapely can't build a valid polygon,
    # revert ALL moves and keep Stage 5's output — better unchanged than
    # silently-corrupted.
    reverted = False
    if n_moved > 0:
        try:
            corners = [np.asarray(e['p1']) for e in edges]
            poly_check = ShapelyPolygon(corners)
            if (not poly_check.is_valid) or poly_check.area < 1e-3:
                # Revert
                for e, snap in zip(edges, snapshot):
                    e['p1'] = snap['p1']
                    e['p2'] = snap['p2']
                    e['length'] = snap['length']
                    e['angle_deg'] = snap['angle_deg']
                reverted = True
                if verbose:
                    print(f"  [Stage 7] polygon invalid after {n_moved} "
                          "corner moves — reverted all moves")
        except Exception as exc:
            # Any Shapely exception → revert defensively
            for e, snap in zip(edges, snapshot):
                e['p1'] = snap['p1']
                e['p2'] = snap['p2']
                e['length'] = snap['length']
                e['angle_deg'] = snap['angle_deg']
            reverted = True
            if verbose:
                print(f"  [Stage 7] Shapely exception ({exc}) — reverted")

    stats = {'n_moved': n_moved if not reverted else 0,
             'reverted': reverted}
    if verbose and not reverted:
        if n_moved > 0:
            max_shift = max(shifts)
            mean_shift = sum(shifts) / len(shifts)
            print(f"  [Stage 7] length-constrained {n_moved}/{n} corners "
                  f"(max shift={max_shift:.3f}m, "
                  f"mean shift={mean_shift:.3f}m)")
        else:
            print(f"  [Stage 7] no corners needed length-constraining")
    return edges, stats


# ----- Stage 6: gap-split segments -----

def _stage6_gap_split(edges, gap_thresh=0.3, min_segment=0.2, verbose=False):
    """For each edge, project inliers onto the line; split at gaps > gap_thresh.

    Each resulting sub-segment is emitted as a separate wall. Does not modify
    edges with no inliers or those marked fallback.
    """
    out = []
    for e in edges:
        if e.get('fallback', True) or len(e.get('inliers_xy', [])) < 5:
            out.append(e)
            continue
        direction = e['direction']
        pol = e['point_on_line']
        t = (e['inliers_xy'] - pol) @ direction
        order = np.argsort(t)
        t_sorted = t[order]
        # Find gaps
        diffs = np.diff(t_sorted)
        split_idx = np.where(diffs > gap_thresh)[0]
        if len(split_idx) == 0:
            out.append(e)
            continue
        # Split into runs
        starts = [0] + list(split_idx + 1)
        ends = list(split_idx + 1) + [len(t_sorted)]
        for s, ep in zip(starts, ends):
            run = t_sorted[s:ep]
            if len(run) < 5:
                continue
            seg_len = float(run[-1] - run[0])
            if seg_len < min_segment:
                continue
            p1 = pol + direction * float(run[0])
            p2 = pol + direction * float(run[-1])
            out.append({**e,
                        'p1': p1, 'p2': p2,
                        'length': seg_len,
                        'inliers_xy': e['inliers_xy'][order[s:ep]]})
    if verbose:
        print(f"  [Stage 6] gap-split: {len(edges)} → {len(out)} segments")
    return out


# ----- Top-level orchestration for variant D_refined -----

def _collect_wall_band(pts_3d, floor_z, ceiling_z, margin=0.3,
                        return_mask=False):
    """Return (M, 2) XY coords of points in the wall-band Z slice.

    Band: Z in [floor_z + margin, ceiling_z - margin] — the middle of the
    wall, avoiding floor/ceiling clutter.

    The `floor_z`/`ceiling_z` args may be sign-flipped by the caller for
    display purposes (when leveling-inversion was detected). We detect
    this by checking whether they lie within the actual pts Z range; if
    not, we fall back to percentile-derived bounds. This makes the
    wall-band extraction robust regardless of how the caller reports
    floor/ceiling heights.

    If `return_mask` is True, additionally returns the boolean mask into
    `pts_3d` — useful when a parallel per-point array (e.g. vision class
    labels) must be sliced in lockstep.
    """
    z = pts_3d[:, 2]
    if z.size < 100:
        xy = pts_3d[:, :2] if z.size else np.zeros((0, 2))
        if return_mask:
            mask = np.ones(z.size, dtype=bool) if z.size else np.zeros(0, dtype=bool)
            return xy, mask
        return xy

    z_pts_lo = float(np.percentile(z, 2))
    z_pts_hi = float(np.percentile(z, 98))

    lo_provided = min(float(floor_z), float(ceiling_z))
    hi_provided = max(float(floor_z), float(ceiling_z))

    # Check if the provided Z range overlaps with the actual cloud Z range.
    # If there's no overlap, the caller likely sign-flipped the values for
    # display — use percentile bounds instead.
    overlap_lo = max(lo_provided, z_pts_lo)
    overlap_hi = min(hi_provided, z_pts_hi)
    if overlap_hi - overlap_lo < 0.5:
        # No meaningful overlap — fall back to pts percentiles.
        lo_raw, hi_raw = z_pts_lo, z_pts_hi
    else:
        lo_raw, hi_raw = lo_provided, hi_provided

    m = margin
    if hi_raw - lo_raw < 2 * margin + 0.2:
        # Short room — shrink the margin
        m = max(0.10, (hi_raw - lo_raw) / 4)

    lo, hi = lo_raw + m, hi_raw - m
    mask = (z > lo) & (z < hi)
    if return_mask:
        return pts_3d[mask][:, :2], mask
    return pts_3d[mask][:, :2]


# ----- Vision helpers (Tier 1 per-wall type, Tier 3 count sanity) -----

_VISION_BUCKET_NAMES = ('other', 'wall', 'window', 'door', 'glass')


def _classify_wall_from_endpoints(p1, p2, pts_3d, point_labels,
                                   band_width=0.30, length_slack=0.30,
                                   min_support=10,
                                   special_threshold=0.45,
                                   wall_threshold=0.35,
                                   return_features=False):
    """Classify a wall segment by its *dominant* vision bucket, plus
    optional secondary features.

    Labels mean "this wall IS a ___" (not "has a ___"). Rationale:
    labeling a 5 m interior wall as 'door' just because 0.9 m of it is
    a doorway is misleading — the wall as a whole is still a wall, with
    a door in it. The door is already represented in `objects.json` via
    the 3D YOLOE detector, so we don't need to re-encode it as a wall
    type.

    A wall is tagged 'door' / 'window' / 'glass' only when that class
    is the clear majority (≥45 % of labeled band points). Walls with
    plain wall dominant (≥35 %) get 'wall'. Everything else is
    'unknown'.

    When `return_features=True`, also returns a list of secondary
    features seen at ≥10 % in the band — e.g. `('wall', ['door'])` for
    a regular wall with a door embedded.

    Args:
        p1, p2:            wall endpoints in XY (2,).
        pts_3d:            (N, 3) point cloud (same frame as refine_walls).
        point_labels:      (N,) uint8 bucket ids aligned with pts_3d.
        band_width:        perpendicular tolerance (m).
        length_slack:      extra half-metre at each end.
        min_support:       minimum non-'other' labeled points required.
        special_threshold: fraction to label wall as window/door/glass
                           (default 0.45 — clear majority).
        wall_threshold:    fraction to label as plain 'wall'. Lower than
                           `special_threshold` because walls can have
                           features (doors/windows) eating into the
                           majority.
        return_features:   if True, return (primary_label, feature_list).

    Returns:
        primary_label (str) — one of
          {'wall', 'window', 'door', 'glass', 'unknown'}.
        Or, when return_features=True,
          (primary_label, features) where features is a sorted list of
          strings drawn from {'wall','window','door','glass'} that each
          occupy ≥10 % of the band.
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    edge_vec = p2 - p1
    edge_len = float(np.linalg.norm(edge_vec))
    if edge_len < 1e-6:
        return ('unknown', []) if return_features else 'unknown'
    edge_dir = edge_vec / edge_len
    edge_perp = np.array([-edge_dir[1], edge_dir[0]])
    mid = (p1 + p2) / 2

    rel = pts_3d[:, :2] - mid
    perp_dist = np.abs(rel @ edge_perp)
    along_dist = np.abs(rel @ edge_dir)
    mask = (perp_dist < band_width) & (along_dist < edge_len / 2 + length_slack)
    if not mask.any():
        return ('unknown', []) if return_features else 'unknown'

    labels = np.asarray(point_labels)[mask]
    non_other = labels >= 1
    if non_other.sum() < min_support:
        return ('unknown', []) if return_features else 'unknown'

    labels = labels[non_other]
    counts = np.bincount(labels, minlength=len(_VISION_BUCKET_NAMES))
    total = int(counts.sum())
    if total == 0:
        return ('unknown', []) if return_features else 'unknown'

    wall_frac = counts[1] / total
    window_frac = counts[2] / total
    door_frac = counts[3] / total
    glass_frac = counts[4] / total

    # Primary label: clear majority wins. Specials outrank 'wall' only
    # when the special class is dominant.
    if glass_frac >= special_threshold:
        primary = 'glass'
    elif door_frac >= special_threshold:
        primary = 'door'
    elif window_frac >= special_threshold:
        primary = 'window'
    elif wall_frac >= wall_threshold:
        primary = 'wall'
    else:
        primary = 'unknown'

    if not return_features:
        return primary

    # Secondary features: anything ≥10 % and distinct from the primary.
    feature_threshold = 0.10
    feature_fracs = {
        'wall': wall_frac, 'window': window_frac,
        'door': door_frac, 'glass': glass_frac,
    }
    features = sorted(
        name for name, f in feature_fracs.items()
        if f >= feature_threshold and name != primary
    )
    return primary, features


def _compute_vision_wall_stats(pts_3d, point_labels,
                                grid_resolution=0.05, min_cluster_px=50):
    """Tier-3 sanity diagnostics on wall-class vision labels.

    Counts total wall-class points and how many connected XY blobs they
    form on a coarse occupancy grid. This is a log-only diagnostic; we
    do NOT override D_refined's wall count.

    Returns a dict:
        {'wall_point_count': int, 'wall_blob_count': int}
    """
    from scipy.ndimage import label as ndi_label

    if point_labels is None or len(point_labels) != len(pts_3d):
        return {'wall_point_count': 0, 'wall_blob_count': 0}

    wall_mask = np.asarray(point_labels) == 1
    n_pts = int(wall_mask.sum())
    if n_pts < min_cluster_px:
        return {'wall_point_count': n_pts, 'wall_blob_count': 0}

    xy = pts_3d[wall_mask, :2]
    x_min, y_min = xy.min(axis=0) - grid_resolution
    x_max, y_max = xy.max(axis=0) + grid_resolution
    W = max(int(np.ceil((x_max - x_min) / grid_resolution)), 1)
    H = max(int(np.ceil((y_max - y_min) / grid_resolution)), 1)

    grid = np.zeros((H, W), dtype=np.uint8)
    ui = np.clip(((xy[:, 0] - x_min) / grid_resolution).astype(np.int32), 0, W - 1)
    vi = np.clip(((xy[:, 1] - y_min) / grid_resolution).astype(np.int32), 0, H - 1)
    grid[vi, ui] = 1

    _, n_blobs = ndi_label(grid)
    return {'wall_point_count': n_pts, 'wall_blob_count': int(n_blobs)}


def _lookup_vision_labels_for_pts(pts, wall_labels_buffer, max_distance=0.5):
    """For each point in `pts`, look up the nearest vision-labeled bucket.

    `wall_labels_buffer` is the dict returned by detect_pipeline.run():
        {'xyz': (M, 3) float32, 'labels': (M,) uint8}.
    Returns a (len(pts),) uint8 array of bucket ids; points with no
    labeled neighbor within `max_distance` get bucket 0 (other).
    """
    from scipy.spatial import cKDTree

    xyz_labeled = wall_labels_buffer.get('xyz')
    lbl = wall_labels_buffer.get('labels')
    if xyz_labeled is None or lbl is None or len(lbl) == 0:
        return np.zeros(len(pts), dtype=np.uint8)

    tree = cKDTree(xyz_labeled)
    d, idx = tree.query(pts, k=1, distance_upper_bound=max_distance)
    out = np.zeros(len(pts), dtype=np.uint8)
    valid = idx < len(lbl)
    out[valid] = lbl[idx[valid]]
    return out


def refine_walls(walls_in, pts_3d, floor_z, ceiling_z, *,
                 point_labels=None,
                 edge_band=0.25, min_inliers=20, k_dominant=3,
                 tol_dominant=7.0, tol_diagonal=4.0, tol_hex=3.0,
                 merge_angle_tol=3.0, gap_thresh=0.3, verbose=False):
    """Run the full 6-stage wall refinement pipeline.

    Returns (refined_walls, meta_per_wall) where refined_walls is a list
    of (p1, p2, angle_deg, length) tuples compatible with `extract_walls`,
    and meta_per_wall is a list of dicts with 'snapped_to', 'residual',
    'confidence' for each output wall.

    If the refinement fails (e.g., not enough wall-band points), returns
    (walls_in, [None, ...]) as a safe fallback.

    Optional `point_labels` is a (N,) uint8 array of bucket ids aligned
    with `pts_3d` (0=other, 1=wall, 2=window, 3=door, 4=glass). When
    provided, the wall-band points are pre-filtered to wall-like classes
    (1..4) before Stage 1 RANSAC — Agent-4's "least invasive" Tier 2
    recommendation. Results in cleaner inlier sets in cluttered corners
    without touching the Stage 1-7 algorithms themselves.
    """
    if floor_z is None or ceiling_z is None:
        return walls_in, [None] * len(walls_in)

    # Normalize floor/ceiling ordering (after our sign-flip convention,
    # they can be in either order; pick the pair that spans the room).
    z_lo = min(float(floor_z), float(ceiling_z))
    z_hi = max(float(floor_z), float(ceiling_z))

    # Tier 2 — class-filter wall-band when vision labels are available.
    if point_labels is not None and len(point_labels) == len(pts_3d):
        wall_band_xy, wall_band_mask = _collect_wall_band(
            pts_3d, z_lo, z_hi, return_mask=True)
        wall_band_labels = np.asarray(point_labels)[wall_band_mask]
        keep = wall_band_labels >= 1  # drop 'other' (clutter/furniture)
        n_before = len(wall_band_xy)
        wall_band_xy = wall_band_xy[keep]
        if verbose:
            n_after = len(wall_band_xy)
            print(f"[D_refined][vision] wall-band filter: "
                  f"{n_before} → {n_after} pts "
                  f"({100.0 * n_after / max(n_before, 1):.0f}% kept)")
    else:
        wall_band_xy = _collect_wall_band(pts_3d, z_lo, z_hi)

    if verbose:
        print(f"[D_refined] wall-band: {len(wall_band_xy)} points "
              f"from Z in [{z_lo + 0.3:.2f}, {z_hi - 0.3:.2f}]")

    if len(wall_band_xy) < 100:
        if verbose:
            print("[D_refined] too few wall-band points — falling back to A")
        return walls_in, [None] * len(walls_in)

    # Stage 1
    refined = _stage1_per_edge_ransac(
        walls_in, wall_band_xy, edge_band=edge_band,
        min_inliers=min_inliers, verbose=verbose)

    # Stage 2
    dominant = _stage2_dominant_directions(
        refined, k_max=k_dominant, verbose=verbose)

    # Stage 3
    snapped = _stage3_tiered_snap(
        refined, dominant, tol_dominant=tol_dominant,
        tol_diagonal=tol_diagonal, tol_hex=tol_hex, verbose=verbose)

    # Stage 4
    merged = _stage4_merge_collinear(
        snapped, angle_tol_deg=merge_angle_tol, verbose=verbose)

    # Stage 5
    translated = _stage5_vertex_translation(merged, verbose=verbose)

    # Stage 7: length-constrain corners to data-supported extent
    # (per-wall: pull corners inward if they over-extend beyond the RANSAC
    # inlier extent; push outward if the wall has data reaching beyond).
    # Uses the `ransac_direction` / `ransac_point_on_line` stashed by
    # Stage 5. Safe by construction: reverts on Shapely invalidation.
    constrained, stage7_stats = _stage7_length_constrain(
        translated, slack=0.30, verbose=verbose)

    # Stage 6 (gap-split) is intentionally NOT applied to the polygon
    # topology — it would produce disconnected segments that break Shapely
    # polygon construction. Gap-split is useful as diagnostic info (where
    # are the doors?) but not for the returned outline. Keeping the code
    # for potential future use on a separate per-wall list.
    if verbose:
        _ = _stage6_gap_split(constrained, gap_thresh=gap_thresh,
                              verbose=verbose)

    # Convert to (p1, p2, angle, length) tuples + meta
    out_walls = []
    out_meta = []
    for e in constrained:
        if e['length'] < 0.10:
            continue
        # Compute data-extent length for audit (Stage-1 inlier span
        # projected onto the pre-Stage-5 RANSAC direction).
        length_data = 0.0
        try:
            inliers = e.get('inliers_xy', np.zeros((0, 2)))
            if len(inliers) >= 10:
                rdir = np.asarray(e.get('ransac_direction', e['direction']))
                rpol = np.asarray(e.get('ransac_point_on_line',
                                         e['point_on_line']))
                t = (inliers - rpol) @ rdir
                length_data = float(t.max() - t.min())
        except Exception:
            length_data = 0.0

        out_walls.append((np.asarray(e['p1']), np.asarray(e['p2']),
                          float(e['angle_deg']), float(e['length'])))
        out_meta.append({
            'snapped_to': e.get('snapped_to', 'free'),
            'residual_m': round(float(e.get('residual', 0.0)), 4),
            'confidence': round(float(e.get('confidence', 0.0)), 3),
            'length_m_data_extent': round(length_data, 3),
        })
    return out_walls, out_meta


# ============================================================
# PUBLIC ENTRY POINT
# ============================================================

def generate_floorplan(pcd, output_dir, name="floorplan", *,
                       gravity_up=None, imus=None,
                       resolution=0.03, epsilon=0.012, snap_angle=45,
                       voxel_size=0.03, ceil_band=0.12,
                       close_kernel=11, angle_flex=3.0,
                       wall_labels=None, verbose=True):
    """Extract a 2D floor plan from an in-memory point cloud.

    Writes to `output_dir` (PNG only — DXF export is intentionally omitted):
      {name}_metadata.json       — floor/ceiling Z, walls, areas, variants
      {name}_pipeline.png        — density / mask / contour stages
      {name}_floorplan.png       — final room with dimensions
      {name}_overlay.png         — plan on top of raw point cloud
      {name}_comparison.png      — A/B/C variants side by side
      {name}_corners.png         — corner-detection detail
      {name}_refined.png         — D_refined walls (type-colored if vision
                                    wall_labels was supplied)

    Floor/ceiling detection is robust: uses `detect_room()` (RANSAC + IMU
    gravity prior) rather than histogram peaks, so furniture no longer
    beats the real ceiling.

    Optional `wall_labels` is a dict
        {'xyz': (M, 3) float32, 'labels': (M,) uint8}
    of world-frame lidar points tagged with vision bucket ids (bucket
    layout: 0 other / 1 wall / 2 window / 3 door / 4 glass). When
    provided and non-empty, three vision augmentations activate:
      Tier 1 — per-wall `type` label in the metadata.
      Tier 2 — wall-band filter (drops 'other' clutter) before RANSAC.
      Tier 3 — diagnostic wall-point / wall-blob counts in the metadata.
    When None or empty, behavior is byte-identical to the vision-less path.

    Returns:
        (variants, meta) where variants is a dict of
        {"A_natural"|"B_corners"|"C_snapped"|"D_refined":
            (walls, poly, label)}.
    """
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    if verbose:
        print(f"floorplan: {name}")
        print("=" * 50)

    # --- Robust floor/ceiling detection (the substantive change) ---
    # Tier 1 also returns the ceiling RANSAC Plane so the mask can use
    # point-to-plane distance instead of a Z-band. Tiers 2/3 return None.
    floor_z, ceiling_z, ceiling_plane = detect_floor_ceiling_robust(
        pcd, gravity_up=gravity_up, imus=imus, verbose=verbose)
    h = ceiling_z - floor_z

    if h <= 0.0:
        raise ValueError(
            f"detect_floor_ceiling_robust returned non-positive height: "
            f"floor_z={floor_z:.3f}, ceiling_z={ceiling_z:.3f}")

    # --- Preprocess: SOR + voxel downsample, in XYZ ---
    pts, n_raw = _voxel_downsample(pcd, voxel_size=voxel_size)
    if verbose:
        print(f"Loaded: {n_raw} -> {len(pts)} pts")
        print(f"Floor: {floor_z:.2f}m, Ceiling: {ceiling_z:.2f}m, Height: {h:.2f}m")

    # --- Vision label lookup (optional; no-op when wall_labels is None) ---
    # `pts_labels` ends up as None when vision is absent or contributed no
    # wall-like (bucket >= 1) signal — so every downstream vision hook
    # short-circuits and the pipeline behaves byte-identically to the
    # vision-less path.
    pts_labels = None
    if wall_labels is not None and len(wall_labels.get('labels', [])) > 0:
        _tentative = _lookup_vision_labels_for_pts(pts, wall_labels)
        if np.any(_tentative >= 1):
            pts_labels = _tentative
            if verbose:
                n_labeled = int((_tentative >= 1).sum())
                print(f"[vision] matched {n_labeled}/{len(pts)} downsampled "
                      f"points against {len(wall_labels['labels'])} "
                      f"labeled frame points")
        elif verbose:
            print("[vision] wall_labels contained no wall-like buckets — "
                  "skipping vision augmentations")

    # Mask is a tight band around the ceiling only (no more 70%-of-room
    # cutoff — that included furniture tops). Uses point-to-plane distance
    # when a RANSAC ceiling plane is available; falls back to a Z-band.
    ceil_xy = extract_ceiling_points(pts, ceiling_z, band=ceil_band,
                                     ceiling_plane=ceiling_plane)
    if verbose:
        print(f"Ceiling points: {len(ceil_xy)}")

    binary, g8, xe, ye, res, x_min, y_min = ceiling_to_binary(
        ceil_xy, resolution, close_kernel)

    clean, n_removed = remove_outlier_clusters(binary)
    if verbose:
        print(f"Outlier clusters removed: {n_removed}")

    raw_contour, simplified = trace_and_simplify(clean, epsilon)
    px_coords = simplified.reshape(-1, 2).astype(float)
    real_coords = pixel_to_real(px_coords, x_min, y_min, res)
    if verbose:
        print(f"Contour: {len(raw_contour)} -> {len(real_coords)} vertices")

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

    # ---- Variant A: natural (raw simplified contour) ----
    if verbose:
        print("\n--- Variant A: No snap (natural angles) ---")
    poly_a = build_room_polygon(real_coords)
    walls_a = extract_walls(poly_a)
    if verbose:
        print(f"  {poly_a.area:.1f} m2, {len(walls_a)} walls")

    # ---- Variant B: corner detection (Shi-Tomasi) ----
    if verbose:
        print("\n--- Variant B: Corner detection ---")
    if corner_coords_real is not None and len(corner_coords_real) >= 3:
        poly_b = build_room_polygon(corner_coords_real)
        walls_b = extract_walls(poly_b)
    else:
        poly_b = poly_a
        walls_b = walls_a
    if verbose:
        print(f"  {poly_b.area:.1f} m2, {len(walls_b)} walls")

    # ---- Variant C: angle snap (for comparison) ----
    if verbose:
        print(f"\n--- Variant C: {snap_angle}-deg snap ---")
    snap_angles = np.arange(0, 180, snap_angle) if snap_angle > 0 else None
    snapped_c, _ = snap_polygon_to_angles(real_coords, snap_angles)
    poly_c = build_room_polygon(snapped_c)
    walls_c = extract_walls(poly_c)
    if verbose:
        print(f"  {poly_c.area:.1f} m2, {len(walls_c)} walls")

    # ---- Variant D: RANSAC-refined walls with tolerance-gated snap ----
    if verbose:
        print("\n--- Variant D: RANSAC refinement + tolerance snap ---")
    walls_d, walls_d_meta = refine_walls(
        walls_a, pts, floor_z, ceiling_z,
        point_labels=pts_labels,
        verbose=verbose)
    if len(walls_d) >= 3:
        # Build polygon from refined wall endpoints (corners from wall order)
        corners_d = np.array([w[0] for w in walls_d])
        poly_d = build_room_polygon(corners_d)
        # If gap-split produced extra segments, extract_walls re-derives
        # wall segments from the polygon (may collapse into fewer segments).
        walls_d_clean = extract_walls(poly_d)
        # Keep the per-wall meta aligned by index-truncation (best effort;
        # extract_walls may drop short edges).
        walls_d_meta_clean = walls_d_meta[:len(walls_d_clean)]
        # Pad with empty dicts if extract_walls produced more segments than
        # the refiner's meta (rare, but possible after gap-merge).
        while len(walls_d_meta_clean) < len(walls_d_clean):
            walls_d_meta_clean.append({'snapped_to': 'free',
                                       'residual_m': 0.0,
                                       'confidence': 0.0})
    else:
        # Too few refined walls — fall back to A_natural
        poly_d = poly_a
        walls_d_clean = walls_a
        walls_d_meta_clean = [{'snapped_to': 'fallback_a',
                                'residual_m': 0.0, 'confidence': 0.0}
                               for _ in walls_a]
    if verbose:
        print(f"  {poly_d.area:.1f} m2, {len(walls_d_clean)} walls")

    # --- Tier 1: per-wall vision type label on D_refined walls ---
    # Runs only when pts_labels was computed above (i.e. vision gave useful
    # signal). Attaches 'type' (the wall's dominant vision bucket) and
    # 'features' (secondary classes seen at ≥10% along the band) to each
    # walls_d_meta_clean entry. No effect on wall geometry.
    if pts_labels is not None:
        for i, (p1, p2, _a, _l) in enumerate(walls_d_clean):
            if i >= len(walls_d_meta_clean):
                break
            primary, features = _classify_wall_from_endpoints(
                p1, p2, pts, pts_labels, return_features=True)
            walls_d_meta_clean[i]['type'] = primary
            if features:
                walls_d_meta_clean[i]['features'] = features
        if verbose:
            type_tally = {}
            for m in walls_d_meta_clean:
                t = m.get('type', 'unknown')
                type_tally[t] = type_tally.get(t, 0) + 1
            print(f"  [vision] wall types: {type_tally}")

    # --- Tier 3: vision diagnostics (log-only; does NOT modify D_refined) ---
    vision_stats = None
    if pts_labels is not None:
        vision_stats = _compute_vision_wall_stats(pts, pts_labels)
        if verbose:
            print(f"  [vision] wall_point_count="
                  f"{vision_stats['wall_point_count']}, "
                  f"wall_blob_count={vision_stats['wall_blob_count']}, "
                  f"D_refined n_walls={len(walls_d_clean)}")

    variants = {
        'A_natural': (walls_a, poly_a, 'Natural angles (no snap)'),
        'B_corners': (walls_b, poly_b, 'Corner detection'),
        'C_snapped': (walls_c, poly_c, f'{snap_angle}-deg snap'),
        'D_refined': (walls_d_clean, poly_d, 'RANSAC-refined + tolerance snap'),
    }

    # ---- PNG exports ----
    # Pass snapped_c (not real_coords twice) so the "Angle Snapped" pipeline
    # panel actually shows the snapped polygon. The reference tool passed
    # real_coords twice here — likely a bug in the reference; we correct it.
    _export_pipeline_pngs(walls_a, poly_a, pts, binary, clean, g8, xe, ye,
                          real_coords, snapped_c, output_dir, name,
                          floor_z, ceiling_z)
    _export_comparison_png(variants, pts, output_dir, name, corner_coords_real)
    _export_corners_png(clean, g8, xe, ye, poly_b, corner_coords_real,
                        output_dir, name)
    _export_refined_png(walls_d_clean, poly_d, walls_d_meta_clean, pts,
                        output_dir, name, floor_z, ceiling_z)

    # ---- Metadata JSON ----
    elapsed = time.time() - t0
    meta = {
        'n_points_raw': int(n_raw),
        'n_points_processed': int(len(pts)),
        'floor_z': round(float(floor_z), 3),
        'ceiling_z': round(float(ceiling_z), 3),
        'room_height': round(float(h), 3),
        'n_outlier_clusters_removed': int(n_removed),
        'n_corners_detected':
            int(len(corner_coords_real)) if corner_coords_real is not None else 0,
        'variants': {},
        'processing_time_s': round(elapsed, 1),
    }
    # Top-level vision diagnostics — present only when wall_labels was used.
    if vision_stats is not None:
        meta['vision_model'] = 'mask2former-swin-large-ade-semantic'
        meta['vision_wall_point_count'] = int(
            vision_stats['wall_point_count'])
        meta['vision_wall_blob_count'] = int(
            vision_stats['wall_blob_count'])
    for key, (walls, poly, label) in variants.items():
        wall_entries = []
        for idx, (_, _, a, l) in enumerate(walls):
            entry = {'length_m': round(float(l), 3),
                     'angle_deg': round(float(a), 1)}
            # D_refined: add per-wall snap kind + residual + confidence
            if (key == 'D_refined' and idx < len(walls_d_meta_clean)
                    and walls_d_meta_clean[idx] is not None):
                m = walls_d_meta_clean[idx]
                entry['snapped_to'] = m.get('snapped_to', 'free')
                entry['residual_m'] = m.get('residual_m', 0.0)
                entry['confidence'] = m.get('confidence', 0.0)
                # Stage 7 audit: data-extent length from RANSAC inliers
                # (compare to `length_m` to see how much Stage 7 trimmed
                # the wall to match the point cloud).
                entry['length_m_data_extent'] = m.get(
                    'length_m_data_extent', 0.0)
                # Vision Tier 1: per-wall type — only present when wall_labels
                # was supplied and classification produced a result.
                if 'type' in m:
                    entry['type'] = m['type']
                if 'features' in m:
                    entry['features'] = m['features']
            elif key == 'D_refined':
                # Fell back to A_natural — meta is None
                entry['snapped_to'] = 'fallback_a'
            wall_entries.append(entry)
        meta['variants'][key] = {
            'label': label,
            'area_m2': round(float(poly.area), 2),
            'n_walls': int(len(walls)),
            'walls': wall_entries,
        }
    with open(f'{output_dir}/{name}_metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    if verbose:
        print(f"\nDone in {elapsed:.1f}s -> {output_dir}/")
        for key, (walls, poly, label) in variants.items():
            print(f"  {key}: {poly.area:.1f} m2, {len(walls)} walls - {label}")

    return variants, meta
