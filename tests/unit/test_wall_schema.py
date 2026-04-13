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


def test_opening_types_has_4():
    assert OPENING_TYPES == frozenset({"door", "window", "glass", "passage"})


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
