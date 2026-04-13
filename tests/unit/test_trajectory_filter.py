"""M4b: trajectory-containment polygon filter tests.

The floorplan pipeline builds a convex hull of the SLAM trajectory XY,
buffers by TRAJECTORY_HULL_BUFFER_M (= 2.5 m), and drops any wall whose
midpoint falls outside that buffered polygon. A connectivity rescue
pass then re-admits walls whose both endpoints fall within
TRAJECTORY_RESCUE_ENDPOINT_EPS_M (= 0.2 m) of a kept-wall endpoint.
The tests below exercise the filter directly by running
`generate_floorplan` on small synthetic clouds with varied
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
    # wall (0.5 m inset) so the 2.5 m buffered hull comfortably clears
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
    """Phantom walls whose midpoints sit beyond the 2.5 m buffer AND
    whose endpoints don't both connect to kept walls are excluded.

    The synthetic cloud is a hexagonal room (lower rectangle + angled
    upper half). Trajectory stays in the lower rectangle only; the
    three upper walls sit well beyond the 2.5 m buffered hull and
    their endpoints at the top (non-sharing corners) don't match any
    kept-wall endpoints, so the connectivity rescue cannot re-admit
    them.
    """
    # Hexagonal-ish room: rectangle (0..4, 0..5) with angled upper
    # section narrowing to a top at y=7. Six walls; trajectory visits
    # only the lower rectangle.
    walls = [
        (np.array([0.0, 0.0]), np.array([4.0, 0.0]), 0.0, 4.0),
        (np.array([4.0, 0.0]), np.array([4.0, 5.0]), 90.0, 5.0),
        (np.array([4.0, 5.0]), np.array([3.0, 7.0]), 117.0,
         float(np.sqrt(5))),
        (np.array([3.0, 7.0]), np.array([1.0, 7.0]), 180.0, 2.0),
        (np.array([1.0, 7.0]), np.array([0.0, 5.0]), 243.0,
         float(np.sqrt(5))),
        (np.array([0.0, 5.0]), np.array([0.0, 0.0]), 270.0, 5.0),
    ]
    pcd = _build_pcd_from_walls(walls, floor_z=0.0, ceiling_z=2.5)
    # Trajectory within the lower rectangle (0.5..3.5, 0.5..2.5).
    # Buffered hull (2.5 m) reaches y ~ 5, so the upper walls are
    # initially excluded. The two diagonal walls share one kept
    # endpoint each (the east/west wall-top) but their OTHER endpoint
    # is (3, 7) / (1, 7) — not shared with any kept wall, so the
    # connectivity rescue does not re-admit them. Likewise the top
    # wall's endpoints are both at y=7 — not kept.
    poses = np.array([
        [0.5, 0.5, 0.0],
        [3.5, 0.5, 0.0],
        [3.5, 2.5, 0.0],
        [0.5, 2.5, 0.0],
        [2.0, 1.5, 0.0],
        [2.0, 2.5, 0.0],
    ])
    meta = _run_floorplan(pcd, poses, tmp_path)
    # At least one wall must be excluded. Expected: 3 upper walls.
    excluded = meta.get('excluded_walls', [])
    assert len(excluded) >= 1, (
        f"upper-room phantoms not excluded — excluded_walls={excluded}")
    # And walls_excluded_phantom count must match the list length.
    assert meta.get('walls_excluded_phantom', 0) == len(excluded)


def test_buffer_zone_walls_kept(tmp_path):
    """Walls just outside the raw hull but within the 2.5 m buffer stay in.

    Trajectory hugs a tight ~1x1 inner box; the 4x3 outer walls are
    up to 2 m outside the hull — the 2.5 m buffer is large enough to
    keep them in. (With the prior 1 m buffer they would have been
    dropped.)
    """
    walls = walls_closed_rectangle(length=4.0, width=3.0)
    pcd = _build_pcd_from_walls(walls, floor_z=0.0, ceiling_z=2.5)
    # Trajectory is a ~1x1 box centered at the room center — walls at
    # (x=0, x=4, y=0, y=3) are up to 2 m from the hull. 2.5 m buffer
    # keeps them all.
    poses = np.array([
        [1.5, 1.0, 0.0],
        [2.5, 1.0, 0.0],
        [2.5, 2.0, 0.0],
        [1.5, 2.0, 0.0],
        [2.0, 1.5, 0.0],
    ])
    meta = _run_floorplan(pcd, poses, tmp_path)
    # All 4 rectangle walls should be within the 2.5 m-buffered hull,
    # so nothing excluded.
    assert meta.get('walls_excluded_phantom', 0) == 0, (
        f"walls within buffer wrongly excluded: "
        f"{meta.get('excluded_walls')}")


def test_connectivity_rescue_keeps_closing_walls(tmp_path):
    """A wall just beyond the 2.5 m buffer whose endpoints close the
    polygon by connecting to kept walls is rescued.

    Setup: 4-wall rectangle (0..6, 0..3). Trajectory sits far to the
    south-west so that the north wall (y=3) midpoint is ~3 m beyond
    the buffered hull — initially excluded. But the north wall's
    endpoints are (0,3) and (6,3), which coincide with endpoints of
    the west wall ((0,0)-(0,3)) and the east wall ((6,0)-(6,3)), which
    are kept by the buffered hull check. The connectivity rescue must
    re-admit the north wall so the polygon closes.
    """
    # Long thin rectangle so that the trajectory, placed along the
    # south side only, leaves the north wall far from the hull.
    walls = walls_closed_rectangle(length=6.0, width=3.0)
    pcd = _build_pcd_from_walls(walls, floor_z=0.0, ceiling_z=2.5)
    # Trajectory: scanner walked only the south leg (y ~= 0.5),
    # covering the full length (x = 0..6). The convex hull of these
    # poses is a tiny sliver around y=0.5; with a 2.5 m buffer the
    # hull reaches y ~= 3.0. The north-wall midpoint at y=3 sits
    # right on the boundary, so we push the box further to put the
    # north wall JUST beyond the buffer. We use a slightly taller
    # box for clearance.
    walls_tall = walls_closed_rectangle(length=6.0, width=5.5)
    pcd = _build_pcd_from_walls(walls_tall, floor_z=0.0, ceiling_z=2.5)
    poses = np.array([
        [0.5, 0.5, 0.0],
        [2.0, 0.5, 0.0],
        [3.5, 0.5, 0.0],
        [5.5, 0.5, 0.0],
        [3.0, 0.8, 0.0],
    ])
    # With this trajectory (y ~= 0.5..0.8), the convex hull is a
    # thin strip near y=0.5. With a 2.5 m buffer the hull reaches
    # y ~= 3.3. The north wall midpoint is at y=5.5 — well beyond
    # the buffer — so it is initially excluded. Its endpoints
    # (0, 5.5) and (6, 5.5) coincide with the west-wall endpoint
    # (0, 5.5) and the east-wall endpoint (6, 5.5), so the
    # connectivity rescue re-admits it.
    meta = _run_floorplan(pcd, poses, tmp_path)
    # After rescue, all 4 rectangle walls should be kept.
    excluded = meta.get('excluded_walls', [])
    kept = meta['variants']['D_refined']['n_walls']
    assert kept >= 4, (
        f"connectivity rescue failed: kept={kept}, "
        f"excluded_walls={excluded}")
    assert meta.get('walls_excluded_phantom', 0) == 0, (
        f"connectivity rescue did not fully close the polygon — "
        f"excluded_walls={excluded}")
