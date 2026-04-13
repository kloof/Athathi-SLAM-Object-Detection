"""Validates the M0a expanded wall JSON schema.

Covers the constants exposed at `cloud_slam.floorplan.schema`. The actual
`build_floorplan_metadata` output is exercised end-to-end via the
detect_and_slam pipeline run (see docs/plans/roomplan-quality.md M0a
verification step) — too much set-up (walls + polygons + point-buffer
data) to be worth a unit-level stub here.
"""
from cloud_slam.floorplan.schema import (
    OPENING_REQUIRED_KEYS,
    OPENING_TYPES,
)


def test_opening_required_keys_has_18():
    # 18 keys: 5 identity (id/uuid/type/wall_id/wall_uuid) +
    # 4 extent (along_start/along_end/z_bottom/z_top) +
    # 4 geometry (width_m/height_m/center_xy/transform_4x4) +
    # 5 state (source/is_open/transparent/transom_of/confidence).
    # (Task spec listed the frozenset contents verbatim but asserted 17
    # in the sample test — that was a miscount; the actual set has 18.)
    assert len(OPENING_REQUIRED_KEYS) == 18


def test_opening_types_has_5():
    # M4b-ext added `mirror` to the vocabulary.
    assert OPENING_TYPES == frozenset(
        {"door", "window", "glass", "passage", "mirror"})


def test_opening_types_includes_mirror():
    """M4b-ext: mirror is a valid opening type."""
    from cloud_slam.floorplan.schema import OPENING_TYPES
    assert 'mirror' in OPENING_TYPES
    assert len(OPENING_TYPES) == 5  # door, window, glass, passage, mirror


def test_vote_room_category_empty():
    from cloud_slam.floorplan.schema import _vote_room_category
    assert _vote_room_category({}) == ("unknown", 0.0, "unavailable")
    assert _vote_room_category(None) == ("unknown", 0.0, "unavailable")


def test_vote_room_category_clean_win():
    from cloud_slam.floorplan.schema import _vote_room_category
    cat, conf, src = _vote_room_category({7: 1000, 23: 50})  # bed dominates
    assert cat == "bedroom"
    assert conf > 0.9
    assert src == "ade20k_vote"


def test_vote_room_category_below_threshold():
    from cloud_slam.floorplan.schema import _vote_room_category
    # 5 equally-weighted signatures each get 20% — below 40% threshold
    counts = {7: 100, 23: 100, 37: 100, 50: 100, 65: 100}
    cat, conf, src = _vote_room_category(counts)
    assert cat == "unknown"
    assert conf <= 0.4
    assert src == "ade20k_vote"


def test_wall_uuid_determinism():
    from cloud_slam.floorplan.schema import _wall_uuid
    # Same geometry → same UUID
    assert _wall_uuid([0.0, 0.0], [1.0, 1.0]) == _wall_uuid([0.0, 0.0], [1.0, 1.0])
    # Different geometry → different UUID
    assert _wall_uuid([0.0, 0.0], [1.0, 1.0]) != _wall_uuid([0.0, 0.0], [1.0, 2.0])
    # 32 hex chars
    assert len(_wall_uuid([0.0, 0.0], [1.0, 1.0])) == 32


# ---- M0c: calibration block pass-through ----------------------------------

def test_calibration_block_default_is_coarse_null_iou():
    """No calibration_info → nulls for IoU, 'coarse' tier, 'manual' method."""
    from cloud_slam.floorplan.schema import _build_calibration_block
    block = _build_calibration_block(None)
    assert block["method"] == "manual_visual_alignment"
    assert block["reprojection_iou_mean"] is None
    assert block["reprojection_iou_min"] is None
    assert block["reprojection_iou_frames_checked"] == 0
    assert block["accuracy_tier"] == "coarse"
    assert block["time_sync_max_dt_ms"] == 150


def test_calibration_block_passes_through_iou_and_tier():
    """M0c: IoU + tier in calibration_info land in the block verbatim."""
    from cloud_slam.floorplan.schema import _build_calibration_block
    info = {
        "method": "target_based",
        "calibration_date": "2026-04-01",
        "age_days": 12,
        "reprojection_iou_mean": 0.87,
        "reprojection_iou_min": 0.74,
        "reprojection_iou_frames_checked": 16,
        "accuracy_tier": "tight",
    }
    block = _build_calibration_block(info)
    assert block["method"] == "target_based"
    assert block["date"] == "2026-04-01"
    assert block["age_days"] == 12
    assert block["reprojection_iou_mean"] == 0.87
    assert block["reprojection_iou_min"] == 0.74
    assert block["reprojection_iou_frames_checked"] == 16
    assert block["accuracy_tier"] == "tight"
