"""M1a: validates Stage 8 polygon-closure solver applies the correct
per-corner KKT distribution and respects Manhattan snap constraints.

Currently a placeholder — will be activated when M1a lands the Stage 8
solver."""
import pytest

pytestmark = pytest.mark.skip(reason="M1a Stage 8 solver not yet implemented")


def test_closed_rectangle_is_noop():
    # Uses tests/fixtures/synthetic/walls_closed_rectangle.py — Δ = 0 so
    # the solver must return corners unchanged when |Δ| < no_op_threshold_mm.
    pass


def test_small_closure_gap_distributes_across_corners():
    # A 1-cm closure gap must be absorbed by multiple corners, not
    # dumped entirely into one (weight_clamp_ratio bounds this).
    pass


def test_large_closure_gap_triggers_rejection():
    # Gaps above per_corner_reject_mm must cause the solver to reject
    # and fall back to the un-closed polygon.
    pass


def test_min_wall_length_revert():
    # If a closure solution shrinks a wall below min_wall_length_m, the
    # solver must revert that corner to its pre-closure position.
    pass
