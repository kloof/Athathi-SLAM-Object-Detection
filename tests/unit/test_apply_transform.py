"""M0b: validates _apply_transform_to_buffers() applies (R, t) in lockstep
to merged, wall_labels, objects (center + orientation.quaternion), and lines.

Currently a placeholder — will be activated when M0b lands the helper."""
import pytest

pytestmark = pytest.mark.skip(reason="M0b helper _apply_transform_to_buffers not yet implemented")


def test_identity_is_noop():
    pass


def test_all_buffers_transform_in_lockstep():
    pass


def test_quaternion_composes_correctly():
    pass


def test_none_buffers_skipped():
    pass
