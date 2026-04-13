"""Frozen dataclass configs for floorplan package.

Defaults capture the magic numbers currently embedded in the floorplan
pipeline. Splitting them into dataclasses makes future per-scan tuning
explicit (e.g. M2 line-segment merge tolerances, M3 openings detection,
post-M1a corner snapping).

These are populated for downstream milestones; the existing pipeline does
not yet read from them. Adding the dataclasses now keeps M0d zero-impact
on behavior while providing the contract surface for M1a, M2, M3.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Stage8Config:
    no_op_threshold_mm: float = 10.0
    per_corner_reject_mm: float = 50.0
    per_corner_reject_frac_of_delta: float = 0.6
    min_wall_length_m: float = 0.15
    weight_clamp_ratio: float = 10.0
    adjacent_snap_reject_deg: float = 10.0
    kkt_cond_threshold: float = 1e10
    max_demotions: int = 20


@dataclass(frozen=True)
class OpeningsConfig:
    grid_resolution_m: float = 0.03
    band_perp_density_m: float = 0.10
    band_perp_vision_m: float = 0.20
    min_support_vision_cells: int = 30
    min_support_gap_cells: int = 60
    corner_reject_m: float = 0.10
    door_width_m_range: tuple = (0.4, 2.5)
    door_height_m_range: tuple = (1.5, 2.8)
    window_width_m_range: tuple = (0.2, 3.5)
    window_height_m_range: tuple = (0.15, 2.5)
    passage_width_m_range: tuple = (0.5, 4.0)
    passage_height_m_range: tuple = (1.8, 3.0)
    open_threshold: float = 0.25
    closed_threshold: float = 0.75
    dedup_iou_threshold: float = 0.3


@dataclass(frozen=True)
class M2Config:
    ceiling_band_m: float = 0.5
    knn_max_px: int = 40
    knn_depth_stddev_m: float = 0.15
    perp_distance_m: float = 0.10
    along_slack_m: float = 0.20
    endpoint_deviation_m: float = 0.04
    min_lines_per_wall: int = 5
    trimmed_mean_threshold: int = 10
    max_lines_per_wall: int = 100
    spatial_bin_m: float = 0.5
