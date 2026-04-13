"""M4a: dropped-ceiling-feature handling in detect_room."""
import numpy as np
import open3d as o3d
import pytest

from cloud_slam.room_structure import detect_room


def _synth_horizontal_plane(z, x_range=(-2, 2), y_range=(-2, 2), step=0.05):
    """Generate a grid of points on a horizontal plane at height z."""
    xs = np.arange(x_range[0], x_range[1], step)
    ys = np.arange(y_range[0], y_range[1], step)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, z)], axis=1)
    # Normals point UP (away from gravity)
    normals = np.tile([0., 0., 1.], (len(pts), 1))
    return pts, normals


def _synth_horizontal_plane_with_normal_down(z, x_range=(-2, 2), y_range=(-2, 2), step=0.05):
    """Ceiling plane: normals point DOWN (toward gravity)."""
    pts, _ = _synth_horizontal_plane(z, x_range, y_range, step)
    normals = np.tile([0., 0., -1.], (len(pts), 1))
    return pts, normals


def _make_pcd(point_clouds_with_normals):
    """Combine multiple (pts, normals) pairs into one Open3D PointCloud."""
    pts = np.concatenate([p for p, _ in point_clouds_with_normals])
    normals = np.concatenate([n for _, n in point_clouds_with_normals])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd


def test_ceiling_picked_above_dropped_soffit():
    """Floor + dropped perimeter soffit + actual ceiling.
    Soffit covers the perimeter (more inliers); ceiling covers the
    central area (fewer inliers, but still > 50 with the new threshold).
    The ceiling must still win."""
    floor_pts = _synth_horizontal_plane(0.0, x_range=(-3, 3), y_range=(-3, 3), step=0.04)
    # Soffit: perimeter band, ring shape — many points
    soffit_pts_outer = _synth_horizontal_plane(2.7, x_range=(-3, 3), y_range=(-3, 3), step=0.04)
    soffit_inner = _synth_horizontal_plane(2.7, x_range=(-2, 2), y_range=(-2, 2), step=0.04)
    # Subtract inner from outer to make a ring
    outer_pts, outer_n = soffit_pts_outer
    inner_pts, _ = soffit_inner
    inner_set = {tuple(p) for p in inner_pts}
    keep_mask = [tuple(p) not in inner_set for p in outer_pts]
    soffit_pts = (outer_pts[keep_mask], outer_n[keep_mask])
    # Real ceiling (smaller, central area)
    ceiling_pts = _synth_horizontal_plane_with_normal_down(3.0, x_range=(-1.5, 1.5), y_range=(-1.5, 1.5), step=0.04)

    pcd = _make_pcd([floor_pts, soffit_pts, ceiling_pts])

    room = detect_room(pcd, gravity_up=np.array([0, 0, 1]), voxel_size=0.05)

    assert room.floor is not None
    assert room.ceiling is not None
    # Ceiling should be at ~3.0m (the highest plane), NOT at ~2.7m (the soffit)
    assert abs(room.ceiling_height - 3.0) < 0.1, \
        f"Expected ceiling near 3.0m, got {room.ceiling_height:.2f}m " \
        f"(soffit at 2.7m may have won — bug present)"


def test_iteration_cap_supports_10_planes():
    """Pile up many small horizontal planes, verify the loop finds 10."""
    # Floor + 9 small intermediate planes + ceiling = 11 total.
    # The 5-iteration cap would miss the ceiling (or earlier planes).
    # The 10-iteration cap should reach the ceiling.
    planes_data = []
    planes_data.append(_synth_horizontal_plane(0.0, x_range=(-3, 3), y_range=(-3, 3), step=0.05))
    for i in range(9):
        z = 0.5 + i * 0.2
        planes_data.append(_synth_horizontal_plane(z, x_range=(-0.5, 0.5), y_range=(-0.5, 0.5), step=0.05))
    planes_data.append(_synth_horizontal_plane_with_normal_down(3.5, x_range=(-2, 2), y_range=(-2, 2), step=0.05))
    pcd = _make_pcd(planes_data)
    room = detect_room(pcd, gravity_up=np.array([0, 0, 1]), voxel_size=0.05)
    assert room.ceiling is not None
    assert abs(room.ceiling_height - 3.5) < 0.15, \
        f"Iteration cap may have stopped before reaching the actual ceiling. Got {room.ceiling_height:.2f}m"


def test_low_inlier_ceiling_now_passes():
    """Real ceiling has only ~50 inliers — was filtered with min=100, now passes."""
    floor_pts = _synth_horizontal_plane(0.0, x_range=(-2, 2), y_range=(-2, 2), step=0.04)  # ~2500 pts
    soffit_pts = _synth_horizontal_plane(2.5, x_range=(-2, 2), y_range=(-2, 2), step=0.04)  # ~2500 pts
    # Real ceiling: tiny — only ~60 points
    ceiling_pts = _synth_horizontal_plane_with_normal_down(3.0, x_range=(-0.4, 0.4), y_range=(-0.4, 0.4), step=0.05)
    pcd = _make_pcd([floor_pts, soffit_pts, ceiling_pts])
    room = detect_room(pcd, gravity_up=np.array([0, 0, 1]), voxel_size=0.05)
    # With min_inliers=50, the small ceiling should be detected, and as the highest, should win.
    assert room.ceiling is not None
    assert abs(room.ceiling_height - 3.0) < 0.15


def test_intermediate_horiz_surfaces_soffit():
    """M5a: actual architectural soffits (small features below the main
    ceiling) should surface via RoomStructure.below_ceiling_features
    (and its alias, intermediate_horiz).

    Under M5a, a plane above floor+1m with a footprint comparable to the
    ceiling is a ceiling-region plane (tray / stepped ceiling), NOT a
    soffit. So this test uses a SMALL soffit (0.6x0.6m area) below a
    LARGE ceiling (4x4m) — the main-proxy inlier count picks the big
    ceiling as main, and the small soffit correctly lands in
    below_ceiling_features.
    """
    floor_pts = _synth_horizontal_plane(0.0, x_range=(-2, 2), y_range=(-2, 2), step=0.05)
    # Small soffit: 0.6 x 0.6 m → ~144 pts. Large ceiling: 4 x 4 m →
    # ~6400 pts. Inlier-count proxy picks the ceiling as main.
    soffit_pts = _synth_horizontal_plane(2.3, x_range=(-0.3, 0.3), y_range=(-0.3, 0.3), step=0.05)
    ceiling_pts = _synth_horizontal_plane_with_normal_down(3.0, x_range=(-2, 2), y_range=(-2, 2), step=0.05)
    pcd = _make_pcd([floor_pts, soffit_pts, ceiling_pts])
    room = detect_room(pcd, gravity_up=np.array([0, 0, 1]), voxel_size=0.05)
    # Ceiling was correctly picked at the top
    assert room.ceiling is not None
    assert abs(room.ceiling_height - 3.0) < 0.15
    # The soffit should surface as a below-ceiling feature (not ceiling,
    # not floor). `intermediate_horiz` is the M5a alias for
    # `below_ceiling_features`.
    heights = [float(p.centroid @ np.array([0, 0, 1]))
               for p in room.below_ceiling_features]
    assert any(abs(h - 2.3) < 0.15 for h in heights), \
        f"Expected a soffit at ~2.3m in below_ceiling_features; got heights {heights}"
    # M5a alias: intermediate_horiz == below_ceiling_features.
    heights_alias = [float(p.centroid @ np.array([0, 0, 1]))
                      for p in room.intermediate_horiz]
    assert heights == heights_alias


def test_multi_level_ceiling_reported_as_set():
    """M5a: Room with main ceiling at 3.0m (large) + raised tray at
    3.3m (small). Both planes should appear in room.ceiling_planes; the
    single 'ceiling' (back-compat) is the highest one (3.3m)."""
    # Floor at 0.0 (big, 4x4m)
    floor_pts = _synth_horizontal_plane(
        0.0, x_range=(-2, 2), y_range=(-2, 2), step=0.05)
    # Main ceiling at 3.0m (big, 4x4m)
    main_ceil_pts = _synth_horizontal_plane_with_normal_down(
        3.0, x_range=(-2, 2), y_range=(-2, 2), step=0.05)
    # Raised tray at 3.3m (small, 1.5x1.5m)
    raised_pts = _synth_horizontal_plane_with_normal_down(
        3.3, x_range=(-0.75, 0.75), y_range=(-0.75, 0.75), step=0.05)

    pcd = _make_pcd([floor_pts, main_ceil_pts, raised_pts])
    room = detect_room(pcd, gravity_up=np.array([0, 0, 1]), voxel_size=0.05)

    assert room.floor is not None
    assert len(room.ceiling_planes) >= 2, (
        f"Expected ≥2 ceiling planes (main + tray); got "
        f"{len(room.ceiling_planes)}")

    # Both heights should be present (within voxel tolerance)
    heights = sorted(float(p.centroid @ np.array([0, 0, 1]))
                     for p in room.ceiling_planes)
    # Find a plane near 3.0 and a plane near 3.3 in the set.
    near_30 = any(abs(h - 3.0) < 0.15 for h in heights)
    near_33 = any(abs(h - 3.3) < 0.15 for h in heights)
    assert near_30, f"No ceiling plane near 3.0m in {heights}"
    assert near_33, f"No ceiling plane near 3.3m in {heights}"

    # ceiling_height is the MAX — keeps wall-band/opening detection
    # unclipped (walls extend up to the highest ceiling point).
    assert abs(room.ceiling_height - 3.3) < 0.15
