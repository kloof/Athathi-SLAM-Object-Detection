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
