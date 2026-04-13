"""Validates the SCHEMA_VERSION constant introduced in M0d (will be bumped by later milestones)."""
from cloud_slam.floorplan import SCHEMA_VERSION


def test_schema_version_is_string():
    assert isinstance(SCHEMA_VERSION, str)


def test_schema_version_is_2_0_or_higher():
    major, minor = SCHEMA_VERSION.split(".")
    assert int(major) >= 2, (
        "Schema version must be 2.0 or higher — M0a introduced the "
        f"2.0 baseline; got {SCHEMA_VERSION}"
    )
