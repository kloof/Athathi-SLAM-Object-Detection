"""Validates the SCHEMA_VERSION constant.

M0a introduced the 2.0 baseline; M3 bumped to 2.1 when D_refined.openings
(doors / windows / glass / passages) joined the per-variant walls list.
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


def test_schema_version_is_2_1():
    """M3 bumped the schema version to 2.1 — the JSON now emits
    D_refined.openings alongside D_refined.walls."""
    assert SCHEMA_VERSION == "2.1"
