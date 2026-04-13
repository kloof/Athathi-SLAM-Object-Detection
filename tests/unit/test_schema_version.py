"""Validates the SCHEMA_VERSION constant.

M0a introduced the 2.0 baseline; M3 bumped to 2.1 when D_refined.openings
(doors / windows / glass / passages) joined the per-variant walls list.
M4a bumped to 2.2 when the top-level `scan_quality` block,
`secondary_ceiling_features`, per-wall `frames_seen_count`, and
`curved` fields were added.
M5a bumped to 2.3 when `ceiling_planes` (the ceiling-as-a-set array
with per-plane `role` labels) joined the top-level schema.
"""
from cloud_slam.floorplan import SCHEMA_VERSION


def test_schema_version_is_string():
    assert isinstance(SCHEMA_VERSION, str)


def test_schema_version_is_2_0_or_higher():
    major, minor = SCHEMA_VERSION.split(".")
    assert int(major) >= 2, (
        "Schema version must be 2.0 or higher — M0a introduced the "
        f"2.0 baseline; got {SCHEMA_VERSION}"
    )


def test_schema_version_is_2_3():
    """M5a bumped the schema version to 2.3 — the JSON now emits a
    top-level `ceiling_planes` array with per-plane role labels
    ("main" / "raised" / "lower_step"). Multi-level ceilings (tray,
    stepped, cathedral) surface as a set rather than being relegated
    to `secondary_ceiling_features`."""
    assert SCHEMA_VERSION == "2.3"
