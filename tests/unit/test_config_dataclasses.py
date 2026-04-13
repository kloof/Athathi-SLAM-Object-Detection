"""Validates the dataclass configs introduced in M0d."""
import pytest
from dataclasses import FrozenInstanceError

from cloud_slam.floorplan.config import Stage8Config, OpeningsConfig, M2Config


def test_stage8_config_is_frozen():
    c = Stage8Config()
    with pytest.raises(FrozenInstanceError):
        c.no_op_threshold_mm = 99.0


def test_openings_config_is_frozen():
    c = OpeningsConfig()
    with pytest.raises(FrozenInstanceError):
        c.grid_resolution_m = 0.99


def test_m2_config_is_frozen():
    c = M2Config()
    with pytest.raises(FrozenInstanceError):
        c.knn_max_px = 999


def test_stage8_has_8_fields():
    import dataclasses
    assert len(dataclasses.fields(Stage8Config)) == 8


def test_openings_has_15_fields():
    import dataclasses
    assert len(dataclasses.fields(OpeningsConfig)) == 15


def test_m2_has_10_fields():
    import dataclasses
    assert len(dataclasses.fields(M2Config)) == 10


def test_stage8_defaults():
    c = Stage8Config()
    assert c.no_op_threshold_mm == 10.0
    assert c.per_corner_reject_mm == 50.0
    assert c.per_corner_reject_frac_of_delta == 0.6
    assert c.min_wall_length_m == 0.15
    assert c.weight_clamp_ratio == 10.0
    assert c.max_demotions == 20


def test_openings_defaults_for_transom_window():
    # The 0.15 m window-height lower bound catches transom windows.
    c = OpeningsConfig()
    assert c.window_height_m_range == (0.15, 2.5)


def test_m2_defaults():
    c = M2Config()
    assert c.knn_max_px == 40
    assert c.endpoint_deviation_m == 0.04
    assert c.min_lines_per_wall == 5
    assert c.trimmed_mean_threshold == 10
