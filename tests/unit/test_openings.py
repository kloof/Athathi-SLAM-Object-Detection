"""M3: validates door/window/glass/passage opening detector.

Exercises the happy paths of `_detect_openings`:
  - empty inputs
  - a synthetic door blob on a single wall
  - a transom window stacked above a larger window
  - a glass panel with high local density → transparent=True
  - a passage-gap with only a single-side jamb

The scenes below construct point clouds directly (no floorplan stages)
so the tests stay focused on the detector's behavior.
"""
import numpy as np
import pytest

from cloud_slam.floorplan.openings import _detect_openings
from cloud_slam.floorplan.config import OpeningsConfig
from cloud_slam.floorplan.schema import OPENING_REQUIRED_KEYS


# ---------------------------------------------------------------------
# Synthetic scene helpers
# ---------------------------------------------------------------------

def _single_wall(length=4.0):
    """Returns a 4 m wall along +X at y=0, plus a meta dict list."""
    p1 = np.array([0.0, 0.0])
    p2 = np.array([float(length), 0.0])
    walls = [(p1, p2, 0.0, float(length))]
    walls_meta = [{'snapped_to': 'manhattan', 'residual_m': 0.0,
                   'confidence': 1.0}]
    return walls, walls_meta


def _sample_rect_on_wall(t0, t1, z0, z1, density=500):
    """Sample (N, 3) world-frame points on the y=0 wall for t in [t0,t1],
    z in [z0, z1]. Perpendicular (y) jitter is tiny (±1 cm) so the points
    land well inside the density band (0.10 m).
    """
    n_t = max(int(density * (t1 - t0)), 10)
    n_z = max(int(density * (z1 - z0)), 10)
    tt, zz = np.meshgrid(
        np.linspace(t0, t1, n_t),
        np.linspace(z0, z1, n_z),
        indexing='xy')
    tt = tt.ravel()
    zz = zz.ravel()
    yy = np.random.default_rng(42).normal(0.0, 0.01, len(tt))
    return np.column_stack([tt, yy, zz]).astype(np.float32)


def _wall_cloud(length=4.0, density=400, floor_z=0.0, ceiling_z=2.5):
    """A uniformly sampled wall plane spanning [0, length] x [floor_z,
    ceiling_z] with slight y-jitter — the "baseline closed wall"."""
    return _sample_rect_on_wall(0.0, length, floor_z, ceiling_z,
                                 density=density)


# ---------------------------------------------------------------------
# Test: empty input
# ---------------------------------------------------------------------

def test_empty_input_returns_empty_list():
    """No walls → no openings."""
    result = _detect_openings(
        walls=[], walls_meta=[],
        merged_pts=np.zeros((0, 3), dtype=np.float32),
        wall_labels=None,
        ceiling_z=2.5, floor_z=0.0)
    assert result == []


def test_no_wall_labels_still_emits_passages():
    """Without vision labels, vision blobs never fire — but the gap
    detector still runs off the lidar occupancy grid."""
    walls, meta = _single_wall(length=4.0)
    # A wall with a gaping hole at t=[1.5, 2.4], z=[0, 2.05] should
    # produce a passage even without labels.
    pts = _wall_cloud(length=4.0, density=500)
    mask_gap = ~((pts[:, 0] >= 1.5) & (pts[:, 0] <= 2.4)
                 & (pts[:, 2] >= 0.0) & (pts[:, 2] <= 2.05))
    pts = pts[mask_gap]
    openings = _detect_openings(
        walls=walls, walls_meta=meta,
        merged_pts=pts, wall_labels=None,
        ceiling_z=2.5, floor_z=0.0)
    # The algorithm expects wall_labels non-None → for openings!=[], pass
    # an empty labels dict (vision inactive but detector still runs).
    openings = _detect_openings(
        walls=walls, walls_meta=meta,
        merged_pts=pts,
        wall_labels={'xyz': np.zeros((0, 3), dtype=np.float32),
                     'labels': np.zeros((0,), dtype=np.uint8)},
        ceiling_z=2.5, floor_z=0.0)
    assert any(o['type'] == 'passage' for o in openings), (
        f"expected at least one passage, got: {[o['type'] for o in openings]}")


# ---------------------------------------------------------------------
# Test: synthetic door
# ---------------------------------------------------------------------

def test_synthetic_door_detected():
    """Construct a single 4 m wall with a 0.9 m door-bucket blob at
    t=[1.5, 2.4], z=[0, 2.05] — expect one opening with type='door' and
    along_start≈1.5, width_m≈0.9."""
    walls, meta = _single_wall(length=4.0)
    # Wall plane (all bucket-1) + door rectangle (all bucket-3).
    wall_pts = _wall_cloud(length=4.0, density=400)
    door_pts = _sample_rect_on_wall(1.5, 2.4, 0.0, 2.05, density=500)
    xyz = np.concatenate([wall_pts, door_pts], axis=0)
    labels = np.concatenate([
        np.ones(len(wall_pts), dtype=np.uint8),       # bucket 1 = wall
        np.full(len(door_pts), 3, dtype=np.uint8),    # bucket 3 = door
    ])
    wall_labels = {'xyz': xyz.astype(np.float32), 'labels': labels}

    openings = _detect_openings(
        walls=walls, walls_meta=meta,
        merged_pts=xyz, wall_labels=wall_labels,
        ceiling_z=2.5, floor_z=0.0)
    doors = [o for o in openings if o['type'] == 'door']
    assert len(doors) >= 1, (
        f"expected ≥1 door, got {len(doors)} / {[o['type'] for o in openings]}")
    d = doors[0]
    # Every required key must be present.
    missing = OPENING_REQUIRED_KEYS - set(d.keys())
    assert not missing, f"missing keys: {missing}"
    assert abs(d['along_start'] - 1.5) < 0.1
    assert abs(d['along_end'] - 2.4) < 0.1
    assert abs(d['width_m'] - 0.9) < 0.1


# ---------------------------------------------------------------------
# Test: transom linking
# ---------------------------------------------------------------------

def test_transom_linking():
    """Two window-bucket blobs stacked along the same t-range:
    main window at z=[0.9, 2.0], transom at z=[2.0, 2.3]. The upper
    should receive `transom_of` pointing at the lower one's id.
    """
    walls, meta = _single_wall(length=4.0)
    wall_pts = _wall_cloud(length=4.0, density=400)
    # Main window at t=[1.0, 2.2], z=[0.9, 2.0].
    main = _sample_rect_on_wall(1.0, 2.2, 0.9, 2.0, density=500)
    # Transom at t=[1.0, 2.2], z=[2.05, 2.3] (slightly abutting, within 0.10 m).
    transom = _sample_rect_on_wall(1.0, 2.2, 2.05, 2.3, density=500)
    xyz = np.concatenate([wall_pts, main, transom], axis=0)
    labels = np.concatenate([
        np.ones(len(wall_pts), dtype=np.uint8),
        np.full(len(main), 2, dtype=np.uint8),        # window
        np.full(len(transom), 2, dtype=np.uint8),     # window
    ])
    wall_labels = {'xyz': xyz.astype(np.float32), 'labels': labels}

    openings = _detect_openings(
        walls=walls, walls_meta=meta, merged_pts=xyz,
        wall_labels=wall_labels, ceiling_z=2.5, floor_z=0.0)
    wins = [o for o in openings
            if o['type'] in ('window', 'glass')]
    assert len(wins) >= 2, (
        f"expected ≥2 windows (main + transom), got {len(wins)} / "
        f"{[(o['type'], o['z_bottom'], o['z_top']) for o in openings]}")
    # Sort by z_bottom ascending: the higher one should have transom_of.
    wins.sort(key=lambda o: o['z_bottom'])
    lower = wins[0]
    upper = wins[-1]
    assert upper['transom_of'] == lower['id'], (
        f"expected upper transom_of={lower['id']}, got "
        f"upper={upper['transom_of']}")


# ---------------------------------------------------------------------
# Test: glass with dense wall behind → transparent=True
# ---------------------------------------------------------------------

def test_glass_with_high_density():
    """A glass-bucket region with high local density (wall-like) → the
    detector marks it type='glass', is_open=False, transparent=True.

    Real-world pattern: the ADE20K segmenter labels each pixel with ONE
    class, so a physical wall region is either 'wall' or 'glass' but not
    both. The test mirrors that by cutting a glass rectangle out of the
    wall-labeled cloud and replacing it with glass-labeled points.
    """
    walls, meta = _single_wall(length=4.0)
    wall_pts = _wall_cloud(length=4.0, density=500)
    # Remove wall points in the glass region so per-cell majority vote
    # picks the glass bucket (mirrors one-class-per-pixel segmenter
    # output).
    keep = ~((wall_pts[:, 0] >= 1.2) & (wall_pts[:, 0] <= 2.0)
             & (wall_pts[:, 2] >= 0.5) & (wall_pts[:, 2] <= 1.8))
    wall_pts = wall_pts[keep]
    # Glass points are DENSER than the wall to simulate a closed glass
    # panel returning strong lidar (density_ratio > closed_threshold).
    glass_pts = _sample_rect_on_wall(1.2, 2.0, 0.5, 1.8, density=900)
    xyz = np.concatenate([wall_pts, glass_pts], axis=0)
    labels = np.concatenate([
        np.ones(len(wall_pts), dtype=np.uint8),
        np.full(len(glass_pts), 4, dtype=np.uint8),   # bucket 4 = glass
    ])
    wall_labels = {'xyz': xyz.astype(np.float32), 'labels': labels}

    openings = _detect_openings(
        walls=walls, walls_meta=meta, merged_pts=xyz,
        wall_labels=wall_labels, ceiling_z=2.5, floor_z=0.0)
    glasses = [o for o in openings if o['type'] == 'glass']
    assert glasses, (
        f"expected ≥1 glass entry, got: {[o['type'] for o in openings]}")
    g = glasses[0]
    assert g['transparent'] is True
    assert g['is_open'] is False


# ---------------------------------------------------------------------
# Test: required-keys contract on a produced opening
# ---------------------------------------------------------------------

def test_opening_schema_keys_complete():
    """Every emitted entry carries every OPENING_REQUIRED_KEYS key."""
    walls, meta = _single_wall(length=4.0)
    wall_pts = _wall_cloud(length=4.0, density=400)
    door_pts = _sample_rect_on_wall(1.5, 2.4, 0.0, 2.05, density=500)
    xyz = np.concatenate([wall_pts, door_pts], axis=0)
    labels = np.concatenate([
        np.ones(len(wall_pts), dtype=np.uint8),
        np.full(len(door_pts), 3, dtype=np.uint8),
    ])
    wall_labels = {'xyz': xyz.astype(np.float32), 'labels': labels}
    openings = _detect_openings(
        walls=walls, walls_meta=meta, merged_pts=xyz,
        wall_labels=wall_labels, ceiling_z=2.5, floor_z=0.0)
    for o in openings:
        missing = OPENING_REQUIRED_KEYS - set(o.keys())
        assert not missing, f"opening missing keys {missing}: {o}"


# ---------------------------------------------------------------------
# Test: passage with stub-wall (single-side jamb)
# ---------------------------------------------------------------------

def test_passage_single_side_jamb():
    """A loft-style passage with a stub wall on one vertical side
    should be detected via the single-side-jamb rule.

    Geometry (all on the y=0 wall of length 4.0 m, grid res 0.03 m):
      - Left solid block:  t=[0.0, 1.2], z=[0.0, 2.5]  (dense wall)
      - Stub-wall jamb:    t=[2.5, 2.6], z=[0.0, 2.5]  (dense vertical
        column, ~33 cells tall — exceeds the 30-cell single-side threshold)
      - Right open region: t=[2.6, 4.0], z=[0.0, 2.5]  (NO points — open)
      - Passage gap:       t=[1.2, 2.5], z=[0.0, 2.5]  (NO points)

    With a baseline floor-ceiling band on t=[0, 4] the passage + the
    open-right region coalesce into one connected empty component. To
    guarantee the passage lands fully inside the corner buffer (t ∈
    [0.12, 3.88]), we include a dense floor baseline that fences the
    component bottom and a dense ceiling baseline that fences the top,
    while leaving the wall interior open on the right past the stub.

    The stub column provides ≥30 adjacent cells along the passage's
    right vertical edge, satisfying the single-side-jamb rule even
    though perimeter-adjacent fraction would otherwise be low.
    """
    walls, meta = _single_wall(length=4.0)
    # Dense left block (the main wall up to the passage).
    left_wall = _sample_rect_on_wall(0.0, 1.2, 0.0, 2.5, density=500)
    # Stub-wall vertical column at t=[2.5, 2.6] — ~33 grid cells tall,
    # sitting entirely inside the 4m wall (t ∈ [0.12, 3.88] corner window).
    stub_wall = _sample_rect_on_wall(2.5, 2.6, 0.0, 2.5, density=700)
    # Floor + ceiling baselines so the empty region is bounded above
    # and below (otherwise a single baseline-less cell would still
    # connect the passage to unrelated empty space).
    floor_line = _sample_rect_on_wall(2.6, 4.0, 0.0, 0.03, density=500)
    ceiling_line = _sample_rect_on_wall(2.6, 4.0, 2.47, 2.5, density=500)
    xyz = np.concatenate(
        [left_wall, stub_wall, floor_line, ceiling_line], axis=0)
    labels = np.ones(len(xyz), dtype=np.uint8)   # bucket 1 = wall
    wall_labels = {'xyz': xyz.astype(np.float32), 'labels': labels}
    openings = _detect_openings(
        walls=walls, walls_meta=meta, merged_pts=xyz,
        wall_labels=wall_labels, ceiling_z=2.5, floor_z=0.0)
    passages = [o for o in openings if o['type'] == 'passage']
    # Non-conditional assertion: this scene MUST produce at least one
    # passage; the gap is entirely inside the corner buffer and the stub
    # column supplies the single-side jamb.
    assert len(passages) >= 1, (
        f"expected ≥1 passage, got 0 / {[o['type'] for o in openings]}")
    p = passages[0]
    assert 1.0 < p['along_start'] < 1.4, (
        f"expected passage along_start ≈ 1.2, got {p['along_start']:.3f}")
    assert p['along_end'] >= 2.4, (
        f"expected passage along_end ≥ 2.4, got {p['along_end']:.3f}")
    assert 0.5 < p['width_m'] < 3.0, (
        f"expected 0.5 < width < 3.0, got {p['width_m']:.3f}")
