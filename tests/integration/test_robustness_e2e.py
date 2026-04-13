"""M4c integration test: schema 2.2 + telemetry fields end-to-end.

Exercises build_floorplan_metadata with realistic inputs and asserts
the M4a/M4b schema additions appear. Does not require real MCAP data.
"""
import numpy as np
import pytest
from shapely.geometry import Polygon as ShapelyPolygon

from cloud_slam.floorplan.schema import (
    SCHEMA_VERSION,
    build_floorplan_metadata,
    OPENING_TYPES,
)


def _make_rectangle_variants():
    """Build a 4-wall closed-rectangle variants dict for every variant key.

    Each variant is (walls, poly, label). `walls` is a list of
    (p1, p2, angle_deg, length_m) tuples matching extract_walls().
    """
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
    return {
        'A_natural': (walls, poly, 'Natural angles (no snap)'),
        'B_corners': (walls, poly, 'Corner detection'),
        'C_snapped': (walls, poly, '45-deg snap'),
        'D_refined': (walls, poly, 'RANSAC-refined + tolerance snap'),
    }, walls


def _make_walls_d_meta(n_walls, *, with_coverage=True, with_curved=False):
    """Build a walls_d_meta_clean list — one dict per wall."""
    out = []
    for _ in range(n_walls):
        entry = {
            'snapped_to': 'manhattan',
            'residual_m': 0.02,
            'confidence': 0.9,
            'length_m_data_extent': 4.0,
            'curved': with_curved,
        }
        if with_coverage:
            entry['wall_band_coverage_pct'] = 0.85
        out.append(entry)
    return out


def test_schema_version_is_2_2():
    assert SCHEMA_VERSION == "2.2"


def test_opening_types_has_5_with_mirror():
    assert "mirror" in OPENING_TYPES
    assert len(OPENING_TYPES) == 5


def test_m4_full_metadata_shape():
    """Build metadata with all M4a+M4b inputs; assert the full schema
    2.2 surface appears."""
    variants, walls = _make_rectangle_variants()
    walls_d_meta = _make_walls_d_meta(len(walls), with_curved=True)
    pts = np.random.default_rng(0).uniform(-1.0, 1.0, (200, 3))
    pts[:, 2] *= 2.0  # span floor..ceiling
    vision_health = {
        'frame_quality_pct': 42.5,
        'frames_seen': 120,
        'total_pixels_seen': 10_000_000,
    }
    time_sync_dts = [i * 0.010 for i in range(1, 21)]  # 10..200 ms
    wall_frames_seen = {0: 80, 1: 5, 2: 60, 3: 8}  # 2 low-coverage walls
    secondary_ceiling_features = [
        {'height_m': 2.7, 'n_inliers': 800, 'area_m2': 3.2},
    ]
    excluded_walls = [
        {'id': 99, 'uuid': 'abcd' * 8, 'p1': [10.0, 10.0], 'p2': [11.0, 10.0]},
    ]
    openings = [{
        'id': 'open_0', 'type': 'mirror', 'wall_id': 0,
        'source': 'vision-override-mirror', 'confidence': 0.72,
    }]
    meta = build_floorplan_metadata(
        n_raw=1000, pts=pts, floor_z=0.0, ceiling_z=2.5, h=2.5,
        n_removed=3, corner_coords_real=np.zeros((4, 2)),
        variants=variants, walls_d_meta_clean=walls_d_meta,
        vision_stats=None, elapsed=12.3,
        openings=openings,
        vision_health=vision_health,
        time_sync_dts=time_sync_dts,
        time_sync_dropped=2,
        wall_frames_seen=wall_frames_seen,
        secondary_ceiling_features=secondary_ceiling_features,
        excluded_walls=excluded_walls,
    )
    # --- top-level schema contract ---
    assert meta['schema_version'] == '2.2'
    # --- scan_quality block (M4a) ---
    sq = meta['scan_quality']
    assert sq['vision']['frame_quality_pct'] == 42.5
    assert sq['vision']['frames_seen'] == 120
    assert sq['time_sync']['p50_ms'] is not None
    assert sq['time_sync']['p95_ms'] is not None
    assert sq['time_sync']['p99_ms'] is not None
    assert sq['time_sync']['frames_dropped'] == 2
    # Two walls have < 10 frames_seen (walls 1 and 3).
    assert sq['walls_camera_coverage']['n_walls_with_low_coverage'] == 2
    # --- M4a: secondary_ceiling_features ---
    assert isinstance(meta['secondary_ceiling_features'], list)
    assert len(meta['secondary_ceiling_features']) == 1
    assert meta['secondary_ceiling_features'][0]['height_m'] == 2.7
    # --- M4b: trajectory-containment excluded walls ---
    assert isinstance(meta['excluded_walls'], list)
    assert len(meta['excluded_walls']) == 1
    assert meta['walls_excluded_phantom'] == 1
    # --- Per-wall additions on D_refined ---
    d_walls = meta['variants']['D_refined']['walls']
    assert len(d_walls) == 4
    for w in d_walls:
        assert 'curved' in w
        assert 'frames_seen_count' in w
        assert 'wall_band_coverage_pct' in w
    # frames_seen_count threaded through from the dict we passed.
    fs = {w['id']: w['frames_seen_count'] for w in d_walls}
    assert fs == wall_frames_seen
    # D_refined openings are surfaced
    assert meta['variants']['D_refined']['openings'] == openings


def test_m4_metadata_with_no_vision():
    """When wall_labels is None / no vision, scan_quality.vision has
    null or 0 values but the block still exists."""
    variants, walls = _make_rectangle_variants()
    walls_d_meta = _make_walls_d_meta(len(walls))
    pts = np.zeros((10, 3))
    meta = build_floorplan_metadata(
        n_raw=10, pts=pts, floor_z=0.0, ceiling_z=2.5, h=2.5,
        n_removed=0, corner_coords_real=np.zeros((4, 2)),
        variants=variants, walls_d_meta_clean=walls_d_meta,
        vision_stats=None, elapsed=1.0,
        vision_health=None,
        time_sync_dts=None,
        time_sync_dropped=0,
        wall_frames_seen=None,
    )
    sq = meta['scan_quality']
    assert 'vision' in sq
    assert 'time_sync' in sq
    assert 'walls_camera_coverage' in sq
    assert sq['vision']['frame_quality_pct'] is None
    assert sq['vision']['frames_seen'] == 0
    assert sq['time_sync']['p50_ms'] is None
    assert sq['time_sync']['p95_ms'] is None
    assert sq['time_sync']['p99_ms'] is None
    # No vision → M4b feature lists default to empty
    assert meta['secondary_ceiling_features'] == []
    assert meta['excluded_walls'] == []
    assert meta['walls_excluded_phantom'] == 0


def test_m4_metadata_back_compat():
    """Pre-M4 consumers reading the M4 JSON should find ALL old keys
    at their original paths."""
    variants, walls = _make_rectangle_variants()
    walls_d_meta = _make_walls_d_meta(len(walls))
    pts = np.zeros((50, 3))
    meta = build_floorplan_metadata(
        n_raw=50, pts=pts, floor_z=0.0, ceiling_z=2.5, h=2.5,
        n_removed=1, corner_coords_real=np.zeros((4, 2)),
        variants=variants, walls_d_meta_clean=walls_d_meta,
        vision_stats=None, elapsed=2.5,
    )
    # --- Pre-M4 root keys unchanged ---
    assert meta['floor_z'] == 0.0
    assert meta['ceiling_z'] == 2.5
    assert meta['room_height'] == 2.5
    assert meta['n_points_raw'] == 50
    assert meta['n_points_processed'] == 50
    assert meta['n_outlier_clusters_removed'] == 1
    assert meta['processing_time_s'] == 2.5
    # --- Pre-M4 variant shape unchanged ---
    for key in ('A_natural', 'B_corners', 'C_snapped', 'D_refined'):
        v = meta['variants'][key]
        assert 'label' in v
        assert 'area_m2' in v
        assert 'n_walls' in v
        assert 'walls' in v
        # Per-wall pre-M4 keys
        for w in v['walls']:
            assert 'id' in w
            assert 'uuid' in w
            assert 'length_m' in w
            assert 'angle_deg' in w
            assert 'p1' in w and 'p2' in w
    # --- M3 openings key still at D_refined.openings ---
    assert meta['variants']['D_refined']['openings'] == []
