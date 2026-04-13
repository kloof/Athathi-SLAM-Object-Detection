"""M2: validates DeepLSD line-anchor matching for wall extents.

Currently a placeholder — will be activated when M2 lands the line
matcher. Requires the DeepLSD optional dep; use the `requires_deeplsd`
skipif marker from conftest when those tests activate.
"""
import pytest

pytestmark = pytest.mark.skip(reason="M2 DeepLSD line matcher not yet implemented")


def test_knn_max_px_rejects_far_endpoints():
    # Line endpoints farther than knn_max_px from the nearest projected
    # lidar pixel must be rejected from the anchor set.
    pass


def test_perp_distance_filter():
    # Lines whose perpendicular distance to the wall exceeds
    # perp_distance_m must not contribute to the wall's extent.
    pass


def test_trimmed_mean_vs_median():
    # With n >= trimmed_mean_threshold lines contributing, the extent
    # estimator uses trimmed mean; below that threshold, it uses median.
    pass


def test_min_lines_per_wall_fallback():
    # If fewer than min_lines_per_wall lines contribute, the matcher
    # must fall back to the Stage 7 raw endpoints.
    pass
