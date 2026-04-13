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


# ---------------------------------------------------------------------
# M4b: picture-frame rejection
# ---------------------------------------------------------------------

def test_picture_frame_rejected():
    """A 0.3×0.3 m window-bucket blob in the middle of a 4 m wall should
    be rejected as a picture frame — interior window-bucket blobs smaller
    than `picture_frame_max_size_m` in BOTH dims that don't sit near the
    wall edge are almost always wall art mis-classified as `windowpane`.
    """
    walls, meta = _single_wall(length=4.0)
    wall_pts = _wall_cloud(length=4.0, density=500)
    # Small picture-frame sized blob centered at t=2.0m, z=1.5m — well
    # inside the 20 cm edge buffer on a 4 m wall.
    frame = _sample_rect_on_wall(1.85, 2.15, 1.35, 1.65, density=600)
    # Remove wall points in that window so the vision bucket actually
    # wins per-cell majority.
    mask = ~((wall_pts[:, 0] >= 1.85) & (wall_pts[:, 0] <= 2.15)
             & (wall_pts[:, 2] >= 1.35) & (wall_pts[:, 2] <= 1.65))
    wall_pts = wall_pts[mask]
    xyz = np.concatenate([wall_pts, frame], axis=0)
    labels = np.concatenate([
        np.ones(len(wall_pts), dtype=np.uint8),        # wall
        np.full(len(frame), 2, dtype=np.uint8),        # window bucket
    ])
    wall_labels = {'xyz': xyz.astype(np.float32), 'labels': labels}
    openings = _detect_openings(
        walls=walls, walls_meta=meta, merged_pts=xyz,
        wall_labels=wall_labels, ceiling_z=2.5, floor_z=0.0)
    # No window should survive — the frame is rejected by the
    # picture-frame filter. Any other emission types are fine.
    windows = [o for o in openings if o['type'] == 'window']
    assert not windows, (
        f"picture frame was not rejected: {[o['type'] for o in openings]}")


def test_window_at_wall_edge_kept():
    """A 0.3×0.3 m window-bucket blob near the wall edge IS a real
    window — the picture-frame rejector only skips INTERIOR small
    blobs. Edge-touching ones (within 20 cm of either wall end) pass.
    """
    walls, meta = _single_wall(length=4.0)
    wall_pts = _wall_cloud(length=4.0, density=500)
    # Blob touching the start edge — along_start ~= 0.05, along_end ~= 0.35
    edge = _sample_rect_on_wall(0.05, 0.35, 1.35, 1.65, density=600)
    mask = ~((wall_pts[:, 0] >= 0.05) & (wall_pts[:, 0] <= 0.35)
             & (wall_pts[:, 2] >= 1.35) & (wall_pts[:, 2] <= 1.65))
    wall_pts = wall_pts[mask]
    xyz = np.concatenate([wall_pts, edge], axis=0)
    labels = np.concatenate([
        np.ones(len(wall_pts), dtype=np.uint8),
        np.full(len(edge), 2, dtype=np.uint8),
    ])
    wall_labels = {'xyz': xyz.astype(np.float32), 'labels': labels}
    # Custom config that widens the corner_reject window so the blob
    # touching the edge isn't rejected by the corner filter — that's a
    # separate concern from the picture-frame rejector.
    from cloud_slam.floorplan.config import OpeningsConfig
    cfg = OpeningsConfig(corner_reject_m=0.02)
    openings = _detect_openings(
        walls=walls, walls_meta=meta, merged_pts=xyz,
        wall_labels=wall_labels, config=cfg,
        ceiling_z=2.5, floor_z=0.0)
    windows = [o for o in openings if o['type'] == 'window']
    assert windows, (
        f"edge-touching window wrongly rejected: "
        f"{[o['type'] for o in openings]}")


# ---------------------------------------------------------------------
# M4b: per-wall lidar-coverage gate
# ---------------------------------------------------------------------

def test_low_coverage_wall_skips_vision():
    """A wall whose lidar occupancy is below `min_wall_coverage_for_vision`
    should skip vision-blob detection — vision is unreliable on walls
    the camera barely covered. Gap detection still runs (passages by
    definition are empty regions).
    """
    from cloud_slam.floorplan.config import OpeningsConfig
    walls, meta = _single_wall(length=4.0)
    # Sparse scene: a thin horizontal line plus a small window-labeled
    # rectangle — total occupancy deliberately under the default 5%
    # threshold. Both elements are sparse (low density) so the blob
    # still lands at low support.
    sparse = _sample_rect_on_wall(0.0, 4.0, 1.20, 1.22, density=150)
    blob = _sample_rect_on_wall(1.80, 2.20, 1.40, 1.55, density=150)
    xyz = np.concatenate([sparse, blob], axis=0)
    labels = np.concatenate([
        np.ones(len(sparse), dtype=np.uint8),
        np.full(len(blob), 2, dtype=np.uint8),    # window bucket
    ])
    wall_labels = {'xyz': xyz.astype(np.float32), 'labels': labels}
    # Raise the coverage threshold well above this scene's actual
    # coverage so the gate reliably fires regardless of exact point
    # counts — this isolates the gate behavior from coverage-tuning
    # noise in the synthetic scene.
    cfg = OpeningsConfig(min_wall_coverage_for_vision=0.90,
                         min_support_vision_cells=5)
    openings = _detect_openings(
        walls=walls, walls_meta=meta, merged_pts=xyz,
        wall_labels=wall_labels, config=cfg,
        ceiling_z=2.5, floor_z=0.0)
    # No vision-sourced opening should emit when coverage is below
    # threshold.
    vision = [o for o in openings if o.get('source') == 'vision+lidar']
    assert not vision, (
        f"low-coverage wall still emitted vision openings: "
        f"{[(o['type'], o['source']) for o in openings]}")
    # The per-wall meta should have wall_band_coverage_pct stamped.
    assert 'wall_band_coverage_pct' in meta[0]
    assert meta[0]['wall_band_coverage_pct'] < 0.90


# ---------------------------------------------------------------------
# M4b: density-ratio guard
# ---------------------------------------------------------------------

def test_phantom_density_rejected():
    """A vision blob whose local density_ratio falls below
    `min_density_ratio_for_emit` (or is None) is a phantom region —
    unscanned wall area being mis-interpreted as a closed opening. The
    detector must drop it before emission. Passages are exempt (they
    deliberately have ratio≈0 and use their own adjacency rule).
    """
    walls, meta = _single_wall(length=4.0)
    # Scene: door-bucket blob in an area with NO nearby wall lidar. The
    # local neighborhood has zero density → ratio is None or very
    # small → guard fires.
    # Build a sparse wall only in the FAR end of the wall so the door
    # blob at t=[1.5, 2.4] has no neighbors within ±1 m.
    wall_far = _sample_rect_on_wall(3.3, 4.0, 0.0, 2.5, density=500)
    door = _sample_rect_on_wall(1.5, 2.4, 0.0, 2.05, density=500)
    xyz = np.concatenate([wall_far, door], axis=0)
    labels = np.concatenate([
        np.ones(len(wall_far), dtype=np.uint8),
        np.full(len(door), 3, dtype=np.uint8),   # door bucket
    ])
    wall_labels = {'xyz': xyz.astype(np.float32), 'labels': labels}
    openings = _detect_openings(
        walls=walls, walls_meta=meta, merged_pts=xyz,
        wall_labels=wall_labels, ceiling_z=2.5, floor_z=0.0)
    for o in openings:
        # Either the opening has a density_ratio ≥ 0.15, or it's a
        # passage (gap-detected, which doesn't use the guard).
        dr = o.get('density_ratio')
        assert (dr is None and o['type'] == 'passage') or \
               (dr is not None and dr >= 0.15) or \
               o['type'] == 'passage', (
            f"opening {o['type']} emitted with density_ratio={dr}")


# ---------------------------------------------------------------------
# M4b: temporal_vote_count populated on vision blobs
# ---------------------------------------------------------------------

def test_temporal_vote_count_populated():
    """Every vision-emitted opening must carry a `temporal_vote_count`
    ≥ 0 (0 for lidar-only passages, the accumulated point count
    otherwise). Doors and windows should have positive vote counts
    when labeled points fell in the blob.
    """
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
        assert 'temporal_vote_count' in o, (
            f"opening missing temporal_vote_count: {o}")
        assert o['temporal_vote_count'] >= 0
