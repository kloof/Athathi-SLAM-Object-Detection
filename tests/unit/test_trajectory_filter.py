"""M4b: trajectory-containment polygon filter tests.

The floorplan pipeline builds a convex hull of the SLAM trajectory XY,
buffers by 1 m, and drops any wall whose midpoint falls outside that
buffered polygon. The tests below exercise the filter directly by
running `generate_floorplan` on small synthetic clouds with varied
trajectories — the wall-exclusion counts are read back from the
returned metadata.
"""
from __future__ import annotations

import numpy as np
import open3d as o3d
import pytest

from tests.fixtures.synthetic.walls_closed_rectangle import (
    walls_closed_rectangle,
)


def _build_pcd_from_walls(walls, floor_z=0.0, ceiling_z=2.5,
                           density_per_m=400):
    """Rasterize a list of (p1, p2, angle, length) walls into a dense
    3D point cloud spanning [floor_z, ceiling_z]. Adds a floor plane so
    detect_floor_ceiling_robust works reliably. Returns an
    o3d.geometry.PointCloud.
    """
    pts = []
    rng = np.random.default_rng(0)
    # Floor + ceiling baselines — detect_floor_ceiling_robust needs them.
    x_lo = min(min(p1[0], p2[0]) for p1, p2, _a, _l in walls) - 0.5
    x_hi = max(max(p1[0], p2[0]) for p1, p2, _a, _l in walls) + 0.5
    y_lo = min(min(p1[1], p2[1]) for p1, p2, _a, _l in walls) - 0.5
    y_hi = max(max(p1[1], p2[1]) for p1, p2, _a, _l in walls) + 0.5
    n_floor = 3000
    floor_xy = rng.uniform(low=[x_lo, y_lo], high=[x_hi, y_hi],
                            size=(n_floor, 2))
    floor_z_pts = np.column_stack([
        floor_xy,
        np.full(n_floor, floor_z) + rng.normal(0, 0.005, n_floor),
    ])
    ceil_xy = rng.uniform(low=[x_lo, y_lo], high=[x_hi, y_hi],
                           size=(n_floor, 2))
    ceil_z_pts = np.column_stack([
        ceil_xy,
        np.full(n_floor, ceiling_z) + rng.normal(0, 0.005, n_floor),
    ])
    pts.append(floor_z_pts)
    pts.append(ceil_z_pts)
    # Walls — uniform sampling in (t, z) plus small perpendicular jitter.
    for p1, p2, _a, length in walls:
        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        edge = p2 - p1
        L = float(np.linalg.norm(edge))
        if L < 1e-6:
            continue
        d = edge / L
        perp = np.array([-d[1], d[0]])
        n_t = max(int(density_per_m * L), 40)
        n_z = max(int(density_per_m * (ceiling_z - floor_z)), 40)
        ts = np.linspace(0.0, L, n_t)
        zs = np.linspace(floor_z + 0.02, ceiling_z - 0.02, n_z)
        TT, ZZ = np.meshgrid(ts, zs, indexing='xy')
        TT = TT.ravel()
        ZZ = ZZ.ravel()
        jitter = rng.normal(0.0, 0.005, len(TT))
        xy = (p1[None, :] + TT[:, None] * d[None, :]
              + jitter[:, None] * perp[None, :])
        wall_pts = np.column_stack([xy, ZZ])
        pts.append(wall_pts)
    all_pts = np.concatenate(pts, axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts)
    return pcd


def _run_floorplan(pcd, poses, tmp_path):
    from cloud_slam.floorplan import generate_floorplan
    _variants, meta = generate_floorplan(
        pcd, str(tmp_path), name="traj_test",
        gravity_up=np.array([0.0, 0.0, 1.0]),
        wall_labels=None, poses=poses, verbose=False,
    )
    return meta


def test_walls_inside_hull_kept(tmp_path):
    """All 4 walls of a rectangle scanned from inside → 0 excluded."""
    walls = walls_closed_rectangle(length=4.0, width=3.0)
    pcd = _build_pcd_from_walls(walls, floor_z=0.0, ceiling_z=2.5)
    # Trajectory spans the rectangle interior, hugging close to each
    # wall (0.5 m inset) so the 1 m buffered hull comfortably clears
    # every wall midpoint outward.
    poses = np.array([
        [0.5, 0.5, 0.0],
        [3.5, 0.5, 0.0],
        [3.5, 2.5, 0.0],
        [0.5, 2.5, 0.0],
        [2.0, 1.5, 0.0],
    ])
    meta = _run_floorplan(pcd, poses, tmp_path)
    # No walls should be excluded — the buffered hull contains all
    # rectangle midpoints.
    assert meta.get('walls_excluded_phantom', 0) == 0, (
        f"unexpected excluded walls: {meta.get('excluded_walls')}")
    assert meta['variants']['D_refined']['n_walls'] >= 3


def test_walls_outside_hull_excluded(tmp_path):
    """A phantom wall 10 m from the rectangle is dropped; the 4
    rectangle walls are kept.

    The synthetic cloud has a 4-wall rectangle plus one 'phantom' wall
    segment far outside the trajectory hull. Only the rectangle walls
    fall within the 1 m-buffered hull; the phantom is excluded.
    """
    walls = walls_closed_rectangle(length=4.0, width=3.0)
    phantom = (np.array([12.0, 12.0]), np.array([14.0, 12.0]), 0.0, 2.0)
    walls_with_phantom = walls + [phantom]
    pcd = _build_pcd_from_walls(walls_with_phantom, floor_z=0.0,
                                 ceiling_z=2.5)
    poses = np.array([
        [2.0, 1.5, 0.0],
        [1.0, 0.5, 0.0],
        [3.0, 0.5, 0.0],
        [3.0, 2.5, 0.0],
        [1.0, 2.5, 0.0],
    ])
    meta = _run_floorplan(pcd, poses, tmp_path)
    # At least one wall must be excluded (the phantom's midpoint at
    # (13, 12) sits far outside the (0..4, 0..3) + 1m hull).
    excluded = meta.get('excluded_walls', [])
    assert len(excluded) >= 1, (
        f"phantom not excluded — excluded_walls={excluded}")
    # And walls_excluded_phantom count must match the list length.
    assert meta.get('walls_excluded_phantom', 0) == len(excluded)


def test_buffer_zone_walls_kept(tmp_path):
    """A wall just outside the raw convex hull but within the 1 m
    buffer stays in the filtered set.

    Trajectory hugs a ~2x1 inner rectangle; the 4x3 outer walls are
    up to 1 m outside the hull — the 1 m buffer is exactly enough to
    keep them in.
    """
    walls = walls_closed_rectangle(length=4.0, width=3.0)
    pcd = _build_pcd_from_walls(walls, floor_z=0.0, ceiling_z=2.5)
    # Trajectory is a 2x1 box centered at the room center — walls at
    # (x=0, x=4, y=0, y=3) are all within 1 m of the hull.
    poses = np.array([
        [1.5, 1.0, 0.0],
        [2.5, 1.0, 0.0],
        [2.5, 2.0, 0.0],
        [1.5, 2.0, 0.0],
        [2.0, 1.5, 0.0],
    ])
    meta = _run_floorplan(pcd, poses, tmp_path)
    # All 4 rectangle walls should be within the 1 m-buffered hull,
    # so nothing excluded.
    assert meta.get('walls_excluded_phantom', 0) == 0, (
        f"walls within buffer wrongly excluded: "
        f"{meta.get('excluded_walls')}")
