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
    """Configuration for Stage 8 polygon-closure solver (M1a)."""
    no_op_threshold_mm: float = 10.0                 # skip Stage 8 entirely when |Δ| < this
    per_corner_reject_mm: float = 50.0               # reject solve if any corner shifts more than this (absolute floor)
    per_corner_reject_frac_of_delta: float = 0.6     # also reject if shift exceeds this fraction of |Δ|; use max(abs, frac·|Δ|)
    min_wall_length_m: float = 0.15                  # revert corner to pre-closure if solve shrinks a wall below this
    weight_clamp_ratio: float = 10.0                 # cap the max-over-min corner-weight ratio to prevent a single weak corner absorbing all of Δ
    adjacent_snap_reject_deg: float = 10.0           # reject snap if two adjacent walls' snapped directions are within this angle (degenerate)
    kkt_cond_threshold: float = 1e10                 # condition number above which the KKT system is considered ill-conditioned; falls back to SLSQP
    max_demotions: int = 20                          # upper bound on demotion cascade iterations (safety)


@dataclass(frozen=True)
class OpeningsConfig:
    """Configuration for M3 door/window/passage opening detector."""
    grid_resolution_m: float = 0.03                  # (t, z) histogram cell size
    band_perp_density_m: float = 0.10                # wall band half-width for occupancy + gap detection
    band_perp_vision_m: float = 0.20                 # wall band half-width for vision bucket majority assignment (wider)
    min_support_vision_cells: int = 30               # reject vision blobs smaller than this
    min_support_gap_cells: int = 60                  # reject gap components smaller than this
    corner_reject_m: float = 0.10                    # reject blobs whose bbox touches the wall ends within this distance (corner artifact)
    door_width_m_range: tuple = (0.4, 2.5)           # plausible door width range
    door_height_m_range: tuple = (1.5, 2.8)          # plausible door height range
    window_width_m_range: tuple = (0.2, 3.5)         # plausible window width range
    window_height_m_range: tuple = (0.15, 2.5)       # plausible window height range (lower bound 0.15 m catches transoms)
    passage_width_m_range: tuple = (0.5, 4.0)        # plausible passage width range (wide for archways)
    passage_height_m_range: tuple = (1.8, 3.0)       # plausible passage height range (tall for vaulted ceilings)
    open_threshold: float = 0.25                     # density_ratio < this → is_open True
    closed_threshold: float = 0.75                   # density_ratio > this → is_open False; between these → None (ambiguous "ajar")
    dedup_iou_threshold: float = 0.3                 # bbox IoU above which a vision blob and a gap component are merged (keep vision)


@dataclass(frozen=True)
class M2Config:
    """Configuration for M2 DeepLSD line-anchor matching."""
    ceiling_band_m: float = 0.5                      # keep only lines whose 3D reconstruction is within this distance of the detected ceiling plane
    knn_max_px: int = 40                             # reject a line endpoint if the nearest projected lidar pixel is farther than this
    knn_depth_stddev_m: float = 0.15                 # reject the endpoint if the 3 nearest pixels' depth stddev exceeds this (line crosses depth discontinuity)
    perp_distance_m: float = 0.10                    # line-to-wall perpendicular distance filter
    along_slack_m: float = 0.20                      # along-wall projection must fall within [-along_slack, length + along_slack]
    endpoint_deviation_m: float = 0.04               # line-endpoint-deviation tolerance after projection onto the wall's line
    min_lines_per_wall: int = 5                      # skip DeepLSD anchor if fewer than this; fall back to Stage 7 endpoints
    trimmed_mean_threshold: int = 10                 # use trimmed-mean extents at n >= this; use median otherwise
    max_lines_per_wall: int = 100                    # cap lines per wall (keep top-N by score) to bound memory and runtime
    spatial_bin_m: float = 0.5                       # grid cell size for line-to-wall spatial pruning
