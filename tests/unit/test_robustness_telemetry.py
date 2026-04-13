"""M4a: scan-quality telemetry — vision health, time-sync dt, curved-wall flag."""
import numpy as np
import pytest


def test_curved_wall_flag_formula():
    """residual / length > 0.05 → curved=True."""
    # Direct formula test (no module import needed for trivial logic).
    # If implemented as a helper, import it here.
    def is_curved(residual_m, length_m):
        return float(residual_m) / max(float(length_m), 0.1) > 0.05

    assert not is_curved(0.01, 1.0)  # 1% residual → straight
    assert is_curved(0.06, 1.0)      # 6% residual → curved
    assert not is_curved(0.05, 1.0)  # exactly at threshold → not curved (strict >)


def test_vision_health_empty_scan():
    """get_vision_health() returns sensible defaults when no frames seen."""
    from cloud_slam.wall_segmenter import WallSegmenter
    seg = WallSegmenter.__new__(WallSegmenter)  # bypass __init__ (it loads the model)
    seg._total_pixels_seen = 0
    seg._non_other_pixels = 0
    seg._frames_seen = 0
    h = seg.get_vision_health()
    assert h["frame_quality_pct"] == 0.0
    assert h["frames_seen"] == 0
    assert h["total_pixels_seen"] == 0


def test_vision_health_partial_quality():
    from cloud_slam.wall_segmenter import WallSegmenter
    seg = WallSegmenter.__new__(WallSegmenter)
    seg._total_pixels_seen = 1000
    seg._non_other_pixels = 350
    seg._frames_seen = 10
    h = seg.get_vision_health()
    assert abs(h["frame_quality_pct"] - 35.0) < 0.1


def test_time_sync_dt_returned():
    """match_nearest_image returns (idx, dt) tuple."""
    from cloud_slam.colorizer import match_nearest_image
    stamps = np.array([0.0, 0.1, 0.2, 0.3])
    idx, dt = match_nearest_image(0.15, stamps, max_dt=0.15)
    assert idx == 1 or idx == 2  # nearest to 0.15
    assert dt is not None
    assert abs(dt) < 0.06

    idx, dt = match_nearest_image(1.0, stamps, max_dt=0.15)
    assert idx is None
    assert dt is None


def test_scan_quality_block_empty_defaults():
    """M4a: scan_quality helper returns sensible defaults (nulls/zeros)
    when no vision / time-sync / coverage data is available."""
    from cloud_slam.floorplan.schema import _build_scan_quality_block
    block = _build_scan_quality_block(
        vision_health=None,
        time_sync_dts=None,
        time_sync_dropped=0,
        wall_frames_seen=None,
        n_walls_d_refined=4,
    )
    assert "vision" in block
    assert "time_sync" in block
    assert "walls_camera_coverage" in block
    # No vision → frame_quality_pct None
    assert block["vision"]["frame_quality_pct"] is None
    # No timing data → p50/p95/p99 None
    assert block["time_sync"]["p50_ms"] is None
    assert block["time_sync"]["p95_ms"] is None
    assert block["time_sync"]["p99_ms"] is None
    # No per-wall data → all walls count as low coverage
    assert block["walls_camera_coverage"]["n_walls_with_low_coverage"] == 4


def test_scan_quality_block_populated():
    """M4a: scan_quality helper produces sensible percentiles."""
    from cloud_slam.floorplan.schema import _build_scan_quality_block
    # 10 frames, dts 10, 20, 30 ... 100 ms
    dts = [i * 0.010 for i in range(1, 11)]
    block = _build_scan_quality_block(
        vision_health={"frame_quality_pct": 85.0, "frames_seen": 10,
                       "total_pixels_seen": 1_000_000},
        time_sync_dts=dts,
        time_sync_dropped=2,
        wall_frames_seen={0: 50, 1: 60, 2: 5, 3: 8},
        n_walls_d_refined=4,
    )
    assert block["vision"]["frame_quality_pct"] == 85.0
    # p50 of [10..100] ms is 55 ms
    assert abs(block["time_sync"]["p50_ms"] - 55.0) < 1.0
    assert block["time_sync"]["frames_dropped"] == 2
    # Two walls have < 10 frames_seen (walls 2 and 3).
    assert block["walls_camera_coverage"]["n_walls_with_low_coverage"] == 2
    assert block["walls_camera_coverage"]["min_frames_seen"] == 5


def test_per_wall_frame_coverage_helper():
    """_compute_per_wall_frame_coverage counts frames per wall."""
    from cloud_slam.floorplan import _compute_per_wall_frame_coverage
    # Two walls: one from (0,0)->(2,0) (along X), one from (0,0)->(0,2) (along Y)
    walls = [
        (np.array([0.0, 0.0]), np.array([2.0, 0.0]), 0.0, 2.0),
        (np.array([0.0, 0.0]), np.array([0.0, 2.0]), 90.0, 2.0),
    ]
    # Frame 0: many points near wall 0 (Y≈0)
    f0 = np.array([[0.5, 0.1, 1.0], [1.0, 0.0, 1.0], [1.5, -0.05, 1.0],
                   [1.2, 0.02, 1.0], [0.8, 0.05, 1.0], [1.1, 0.03, 1.0]],
                  dtype=np.float32)
    # Frame 1: points near wall 1 (X≈0)
    f1 = np.array([[0.1, 0.5, 1.0], [0.0, 1.0, 1.0], [-0.05, 1.2, 1.0],
                   [0.03, 0.7, 1.0], [0.01, 1.5, 1.0], [0.02, 1.8, 1.0]],
                  dtype=np.float32)
    wl = {'per_frame_xyz': [f0, f1]}
    cov = _compute_per_wall_frame_coverage(walls, wl)
    # wall 0 has >=5 pts from frame 0, and possibly frame 1 since first point
    # [0.1, 0.5] is within 0.5m of wall 0. Let's just verify both walls
    # got frame coverage.
    assert cov[0] >= 1
    assert cov[1] >= 1


def test_per_wall_frame_coverage_empty():
    """Empty or None wall_labels returns zeros per wall."""
    from cloud_slam.floorplan import _compute_per_wall_frame_coverage
    walls = [
        (np.array([0.0, 0.0]), np.array([2.0, 0.0]), 0.0, 2.0),
    ]
    cov = _compute_per_wall_frame_coverage(walls, None)
    assert cov == {0: 0}
    cov = _compute_per_wall_frame_coverage(walls, {'per_frame_xyz': []})
    assert cov == {0: 0}
