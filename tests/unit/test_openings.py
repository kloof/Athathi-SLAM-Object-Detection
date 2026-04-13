"""M3: validates door/window/passage opening detector.

Currently a placeholder — will be activated when M3 lands the openings
module."""
import pytest

pytestmark = pytest.mark.skip(reason="M3 openings detector not yet implemented")


def test_door_width_height_filter():
    # Openings outside door_width_m_range / door_height_m_range must
    # not be classified as doors.
    pass


def test_window_catches_transoms():
    # The 0.15 m lower bound on window_height_m_range is intentional —
    # it catches short transom windows.
    pass


def test_passage_wide_range():
    # passage_width_m_range extends to 4.0 m to accommodate archways.
    pass


def test_density_thresholds():
    # density_ratio < open_threshold → is_open True;
    # density_ratio > closed_threshold → is_open False;
    # in between → is_open None (ambiguous "ajar").
    pass


def test_vision_and_gap_dedup():
    # When a vision-detected blob and a gap component overlap at IoU
    # >= dedup_iou_threshold, the vision result must win.
    pass
