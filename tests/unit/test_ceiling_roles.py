"""M5a: ceiling-plane role assignment.

Tests `_build_ceiling_planes` in cloud_slam.floorplan — the helper that
attaches 'main' / 'raised' / 'lower_step' role labels to each ceiling
plane in the ceiling_planes set. Role logic:
  - 'main': the plane with the largest XY footprint (area_m2).
            Exactly one plane gets this role.
  - 'raised': height_m > main.height_m + 0.05m.
  - 'lower_step': height_m < main.height_m - 0.05m.

The floorplan layer computes roles (not schema.py) so schema.py stays
pure serialization. build_floorplan_metadata just stores what the
caller passes.
"""
import numpy as np
import pytest

from cloud_slam.floorplan import _build_ceiling_planes
from cloud_slam.floorplan.schema import build_floorplan_metadata
from cloud_slam.room_structure import Plane, RoomStructure


def _mk_plane(z, n_inliers, xy_extent, n_pts=None):
    """Synthesize a horizontal Plane centred at z with enough geometry
    to make `_plane_inliers_xy` + `_convex_hull_area` produce a
    specific area. xy_extent = half-side of a square footprint (m).
    """
    # Plane stores only normal/offset/centroid/num_inliers. The caller
    # passes `pts` separately to _build_ceiling_planes, so we build
    # matching pts here.
    return Plane(
        normal=np.array([0.0, 0.0, 1.0]),
        offset=float(-z),
        centroid=np.array([0.0, 0.0, float(z)]),
        num_inliers=int(n_inliers),
    ), xy_extent


def _mk_pts_for_planes(plane_extent_pairs, step=0.05):
    """Generate a single (N, 3) pts array covering all planes. Each
    plane gets a grid of points in its [-extent, extent]^2 footprint.
    """
    all_pts = []
    for plane, extent in plane_extent_pairs:
        z = float(plane.centroid[2])
        xs = np.arange(-extent, extent, step)
        ys = np.arange(-extent, extent, step)
        xx, yy = np.meshgrid(xs, ys)
        pts = np.stack(
            [xx.ravel(), yy.ravel(), np.full(xx.size, z)], axis=1)
        all_pts.append(pts)
    return np.concatenate(all_pts)


def _mk_room(plane_extent_pairs, floor_z=0.0):
    """Build a RoomStructure with the given ceiling planes + a stub floor."""
    floor = Plane(
        normal=np.array([0.0, 0.0, 1.0]),
        offset=float(-floor_z),
        centroid=np.array([0.0, 0.0, float(floor_z)]),
        num_inliers=5000,
    )
    return RoomStructure(
        floor=floor,
        ceiling=plane_extent_pairs[-1][0],    # legacy: highest plane
        ceiling_planes=[p for p, _e in plane_extent_pairs],
        gravity_up=np.array([0.0, 0.0, 1.0]),
        floor_height=float(floor_z),
        ceiling_height=float(plane_extent_pairs[-1][0].centroid[2]),
    )


def test_main_role_picked_by_largest_footprint():
    """Main ceiling (4x4m = 16 m²) + raised tray (1.5x1.5m = 2.25 m²):
    the larger footprint is 'main', the higher one is 'raised'."""
    main = _mk_plane(3.0, n_inliers=5000, xy_extent=2.0)    # 4x4 = 16 m²
    raised = _mk_plane(3.3, n_inliers=500, xy_extent=0.75)  # 1.5x1.5 ≈ 2.25 m²
    planes = [main, raised]
    pts = _mk_pts_for_planes(planes)
    room = _mk_room(planes)

    out = _build_ceiling_planes(room, pts, np.array([0.0, 0.0, 1.0]))

    by_h = {round(e['height_m'], 1): e for e in out}
    assert 3.0 in by_h
    assert 3.3 in by_h
    assert by_h[3.0]['role'] == 'main', (
        f"Expected 3.0m (larger footprint) = main; got {by_h[3.0]['role']}. "
        f"areas: 3.0→{by_h[3.0]['area_m2']}, 3.3→{by_h[3.3]['area_m2']}")
    assert by_h[3.3]['role'] == 'raised', (
        f"Expected 3.3m (above main by >5cm) = raised; "
        f"got {by_h[3.3]['role']}")


def test_lower_step_role_for_stepped_half():
    """Stepped ceiling: main at 3.0m + lower half at 2.6m (both large).
    Main (larger footprint) is 'main'; the lower plane is 'lower_step'."""
    main = _mk_plane(3.0, n_inliers=5000, xy_extent=2.0)   # 4x4 = 16 m²
    lower = _mk_plane(2.6, n_inliers=1000, xy_extent=1.0)  # 2x2 = 4 m²
    planes = [main, lower]
    pts = _mk_pts_for_planes(planes)
    room = _mk_room(planes)

    out = _build_ceiling_planes(room, pts, np.array([0.0, 0.0, 1.0]))
    by_h = {round(e['height_m'], 1): e for e in out}
    assert by_h[3.0]['role'] == 'main'
    assert by_h[2.6]['role'] == 'lower_step'


def test_exactly_one_main():
    """Across a 3-plane ceiling, exactly one is 'main' (the largest)."""
    p1 = _mk_plane(3.0, n_inliers=5000, xy_extent=2.0)    # 16 m²
    p2 = _mk_plane(3.2, n_inliers=300, xy_extent=0.5)     # 1 m²
    p3 = _mk_plane(2.8, n_inliers=200, xy_extent=0.5)     # 1 m²
    planes = [p1, p2, p3]
    pts = _mk_pts_for_planes(planes)
    room = _mk_room(planes)

    out = _build_ceiling_planes(room, pts, np.array([0.0, 0.0, 1.0]))
    roles = [e['role'] for e in out]
    assert roles.count('main') == 1, (
        f"Expected exactly one 'main' role; got {roles}")


def test_build_floorplan_metadata_stores_roles_verbatim():
    """schema.py stays pure serialization: whatever roles the caller
    pre-computed, build_floorplan_metadata stores verbatim."""
    from shapely.geometry import Polygon as ShapelyPolygon

    ceiling_planes_in = [
        {'height_m': 3.0, 'area_m2': 16.0, 'n_inliers': 5000,
         'role': 'main'},
        {'height_m': 3.3, 'area_m2': 2.0, 'n_inliers': 500,
         'role': 'raised'},
    ]
    p0 = np.array([0.0, 0.0])
    p1 = np.array([4.0, 0.0])
    p2 = np.array([4.0, 3.0])
    p3 = np.array([0.0, 3.0])
    walls = [
        (p0, p1, 0.0, 4.0),
        (p1, p2, 90.0, 3.0),
        (p2, p3, 180.0, 4.0),
        (p3, p0, 270.0, 3.0),
    ]
    poly = ShapelyPolygon([p0, p1, p2, p3])
    variants = {
        'A_natural': (walls, poly, 'A'),
        'B_corners': (walls, poly, 'B'),
        'C_snapped': (walls, poly, 'C'),
        'D_refined': (walls, poly, 'D'),
    }
    meta = build_floorplan_metadata(
        n_raw=100, pts=np.zeros((10, 3)),
        floor_z=0.0, ceiling_z=3.3, h=3.3,
        n_removed=0, corner_coords_real=np.zeros((4, 2)),
        variants=variants,
        walls_d_meta_clean=[{} for _ in walls],
        vision_stats=None, elapsed=0.5,
        ceiling_planes=ceiling_planes_in,
    )
    assert 'ceiling_planes' in meta
    roles = {round(p['height_m'], 1): p['role']
             for p in meta['ceiling_planes']}
    assert roles[3.0] == 'main'
    assert roles[3.3] == 'raised'


def test_default_ceiling_planes_is_empty_list():
    """When the caller doesn't supply ceiling_planes, the key is still
    emitted as an empty list so consumers can unconditionally iterate."""
    from shapely.geometry import Polygon as ShapelyPolygon

    p0 = np.array([0.0, 0.0])
    p1 = np.array([4.0, 0.0])
    p2 = np.array([4.0, 3.0])
    p3 = np.array([0.0, 3.0])
    walls = [
        (p0, p1, 0.0, 4.0),
        (p1, p2, 90.0, 3.0),
        (p2, p3, 180.0, 4.0),
        (p3, p0, 270.0, 3.0),
    ]
    poly = ShapelyPolygon([p0, p1, p2, p3])
    variants = {
        'A_natural': (walls, poly, 'A'),
        'B_corners': (walls, poly, 'B'),
        'C_snapped': (walls, poly, 'C'),
        'D_refined': (walls, poly, 'D'),
    }
    meta = build_floorplan_metadata(
        n_raw=100, pts=np.zeros((10, 3)),
        floor_z=0.0, ceiling_z=2.5, h=2.5,
        n_removed=0, corner_coords_real=np.zeros((4, 2)),
        variants=variants,
        walls_d_meta_clean=[{} for _ in walls],
        vision_stats=None, elapsed=0.5,
    )
    assert meta['ceiling_planes'] == []
