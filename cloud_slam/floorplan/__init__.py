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

The original single-file module was split into a package for M0d. The
orchestrator (`generate_floorplan`) and the core geometric helpers stayed
here; the PNG export lives in `render`, the wall-refinement pipeline in
`refine`, the JSON schema builder in `schema`, and configs in `config`.
`openings` is reserved for M3. See `docs/plans/roomplan-quality.md`.
"""

import os
import time

import numpy as np
from scipy.ndimage import binary_fill_holes, gaussian_filter1d
from scipy.signal import find_peaks
from shapely.geometry import Polygon as ShapelyPolygon

import cv2

from cloud_slam.room_structure import detect_room
from cloud_slam.frustum import estimate_gravity

from .render import (
    _export_pipeline_pngs,
    _export_refined_png,
    _export_comparison_png,
    _export_corners_png,
)
from .refine import (
    refine_walls,
    _classify_wall_from_endpoints,
    _compute_vision_wall_stats,
    _lookup_vision_labels_for_pts,
    _stage8_polygon_closure,
)
from .config import Stage8Config
from .schema import (
    SCHEMA_VERSION,
    build_floorplan_metadata,
    write_floorplan_metadata,
)


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
# PUBLIC ENTRY POINT
# ============================================================

def generate_floorplan(pcd, output_dir, name="floorplan", *,
                       gravity_up=None, imus=None,
                       resolution=0.03, epsilon=0.012, snap_angle=45,
                       voxel_size=0.03, ceil_band=0.12,
                       close_kernel=11, angle_flex=3.0,
                       wall_labels=None,
                       calibration_info=None,
                       run_stage8=True,
                       emit_stage8_diagnostics=False,
                       verbose=True):
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
        {'xyz': (M, 3) float32, 'labels': (M,) uint8,
         'ade_class_counts': dict[int,int] (optional — M0a room vote)}
    of world-frame lidar points tagged with vision bucket ids (bucket
    layout: 0 other / 1 wall / 2 window / 3 door / 4 glass). When
    provided and non-empty, three vision augmentations activate:
      Tier 1 — per-wall `type` label in the metadata.
      Tier 2 — wall-band filter (drops 'other' clutter) before RANSAC.
      Tier 3 — diagnostic wall-point / wall-blob counts in the metadata.
    When None or empty, behavior is byte-identical to the vision-less path.

    The optional `ade_class_counts` entry (raw ADE20K class id →
    pixel-count histogram across the scan) drives the M0a `room.category`
    vote. When absent or the segmenter was disabled, the metadata emits
    `{"room": {"category": "unknown", ..., "category_source": "unavailable"}}`.

    Optional `calibration_info` is a dict with keys `method`,
    `calibration_date`, and `age_days` (any subset). Missing values fall
    back to the M0a defaults inside `schema._build_calibration_block`.

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

    # --- Stage 8: polygon-closure solver (M1a) ---
    # Redistributes the residual closure gap Δ = Σ (p2 − p1) across
    # corners via confidence-weighted constrained least squares. Closed-
    # form KKT first, SLSQP + demotion cascade on degenerate snap
    # configurations. Strict no-op on already-closed polygons (|Δ| <
    # no_op_threshold_mm) so the default path remains byte-identical to
    # the pre-M1a baseline. Disable via --no-stage8 → run_stage8=False.
    #
    # Diagnostics emission policy:
    #   - When `emit_stage8_diagnostics` is False (default), the JSON
    #     keeps its pre-M1a layout on no-op runs (no `stage8` key),
    #     but surfaces a `stage8` block when the solver actually moved
    #     walls so the user can see what was redistributed.
    #   - When True, always emit (even on no-op) — useful for A/B
    #     benchmarks and integration tests.
    #   - `--no-stage8` (`run_stage8=False`) skips the solver entirely,
    #     guaranteeing zero schema drift.
    stage8_diagnostics = None
    if run_stage8 and len(walls_d_clean) >= 3:
        # Always compute diagnostics internally — they're <100 floats and
        # the orchestrator decides whether to surface them. This also keeps
        # the no-op / non-no-op dispatch below a pure metadata decision.
        walls_d_clean, raw_diag = _stage8_polygon_closure(
            walls_d_clean, walls_d_meta_clean,
            config=Stage8Config(),
            emit_diagnostics=True,
            verbose=verbose,
        )
        # Rebuild the polygon from the closed walls so the variants dict
        # and the refined-PNG renderer see the updated geometry.
        try:
            corners_closed = np.array([w[0] for w in walls_d_clean])
            poly_d = build_room_polygon(corners_closed)
        except Exception:
            # Any polygon-construction failure → keep the Stage-7 polygon.
            # Stage 8's walls still describe a closed ring of corners;
            # this just means the Shapely polygon couldn't be re-built.
            pass
        if verbose and raw_diag is not None:
            print(f"  [Stage 8] solver={raw_diag.get('solver_used')}, "
                  f"|Δ|={raw_diag.get('closure_gap_mm', 0.0)} mm, "
                  f"demotions={len(raw_diag.get('demotions_cascade', []))}")
        # Decide whether to surface diagnostics in the JSON.
        # Explicit opt-in → always emit. Default → emit only when Stage 8
        # did non-trivial work (solver_used != 'noop').
        if raw_diag is not None and (
                emit_stage8_diagnostics
                or raw_diag.get('solver_used') != 'noop'):
            stage8_diagnostics = raw_diag

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
    # M0a room-vote input — pulled from wall_labels when the segmenter
    # exposed a per-scan ADE histogram; otherwise None → "unavailable".
    ade_class_counts = None
    if wall_labels is not None:
        ade_class_counts = wall_labels.get('ade_class_counts')
    meta = build_floorplan_metadata(
        n_raw=n_raw, pts=pts, floor_z=floor_z, ceiling_z=ceiling_z, h=h,
        n_removed=n_removed, corner_coords_real=corner_coords_real,
        variants=variants, walls_d_meta_clean=walls_d_meta_clean,
        vision_stats=vision_stats, elapsed=elapsed,
        calibration_info=calibration_info,
        ade_class_counts=ade_class_counts,
        stage8_diagnostics=stage8_diagnostics)
    write_floorplan_metadata(meta, output_dir, name)

    if verbose:
        print(f"\nDone in {elapsed:.1f}s -> {output_dir}/")
        for key, (walls, poly, label) in variants.items():
            print(f"  {key}: {poly.area:.1f} m2, {len(walls)} walls - {label}")

    return variants, meta


__all__ = [
    'generate_floorplan',
    'detect_floor_ceiling_robust',
    'refine_walls',
    'extract_walls',
    'extract_ceiling_points',
    'ceiling_to_binary',
    'remove_outlier_clusters',
    'trace_and_simplify',
    'pixel_to_real',
    'detect_corners',
    'snap_polygon_to_angles',
    'build_room_polygon',
    'SCHEMA_VERSION',
    'Stage8Config',
]
