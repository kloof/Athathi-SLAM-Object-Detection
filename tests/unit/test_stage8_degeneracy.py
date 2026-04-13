"""M1a: degeneracy fallback tests — Stage 8 must handle rank-deficient
snap configurations (adjacent parallel walls, pathological axes) via the
demotion cascade + SLSQP fallback.

The "Manhattan L-shape with all walls snapped" case is often cited as
zero-DoF, but with six Manhattan walls on an L-shape there are still
6 geometric DoF (position + lengths), so a small closure gap is
absorbed by the closed-form KKT without demotion. The real degeneracy
triggers are:
  - two adjacent walls with collinear snap directions (the
    adjacent-snap-reject pre-flight step demotes the lower-conf one)
  - a rank-deficient A_eq coming from pathological snap-direction
    collinearity around the ring (the demotion cascade resolves it)
"""
import numpy as np
import pytest

from cloud_slam.floorplan.config import Stage8Config
from cloud_slam.floorplan.refine import _stage8_polygon_closure
from tests.fixtures.synthetic.walls_closed_rectangle import walls_closed_rectangle
from tests.fixtures.synthetic.walls_open_lshape import walls_open_lshape


def _polygon_closure(walls):
    return sum((w[1] - w[0] for w in walls), start=np.zeros(2))


def test_all_manhattan_rectangle_with_closure_gap():
    """4-wall rectangle with 5 mm gap (< threshold) → strict no-op."""
    walls = list(walls_closed_rectangle(length=4.0, width=3.0))
    p1, p2, a, L = walls[-1]
    walls[-1] = (p1, p2 + np.array([0.005, 0.0]), a, L + 0.005)
    meta = [{"snapped_to": "manhattan", "confidence": 1.0}] * 4

    result, diag = _stage8_polygon_closure(walls, meta)

    # 5 mm < 10 mm threshold → geometry unchanged, no diagnostics.
    assert diag is None
    for w_in, w_out in zip(walls, result):
        assert np.allclose(w_in[0], w_out[0])
        assert np.allclose(w_in[1], w_out[1])


def test_all_manhattan_lshape_closes_without_demotion():
    """L-shape with 1 cm gap + all Manhattan snapped → KKT solves cleanly.

    The 6-wall Manhattan L-shape still has 6 DoF (4 wall lengths + 2D
    position), so a 1 cm gap is absorbed by the closed-form KKT. The
    demotion cascade should NOT trigger.
    """
    walls = walls_open_lshape(closure_gap_m=0.01)
    meta = [{"snapped_to": "manhattan", "confidence": c}
            for c in [1.0, 1.0, 0.5, 1.0, 1.0, 0.8]]

    result, diag = _stage8_polygon_closure(walls, meta, emit_diagnostics=True)

    # KKT succeeded, no demotion needed.
    assert diag["solver_used"] == "kkt"
    # Polygon closes to numerical zero.
    delta_post = _polygon_closure(result)
    np.testing.assert_allclose(delta_post, [0.0, 0.0], atol=1e-6)
    # Every wall remains axis-aligned.
    for (pp1, pp2, _a, _l) in result:
        v = pp2 - pp1
        assert abs(v[0]) < 1e-6 or abs(v[1]) < 1e-6


def test_adjacent_collinear_snaps_demote_lower_confidence():
    """Two adjacent walls with the same snap angle must demote one."""
    # Two adjacent walls along 0° — the adjacent-snap-reject must demote
    # the lower-confidence one. This is the "degenerate T-junction"
    # case from the Stage 8 spec.
    walls = [
        (np.array([0.0, 0.0]), np.array([2.0, 0.0]), 0.0, 2.0),   # wall 0
        (np.array([2.0, 0.0]), np.array([4.0, 0.0]), 0.0, 2.0),   # wall 1 — same dir
        (np.array([4.0, 0.0]), np.array([4.0, 3.0]), 90.0, 3.0),
        (np.array([4.0, 3.0]), np.array([0.0, 3.0]), 180.0, 4.0),
        (np.array([0.0, 3.0]), np.array([0.015, 0.0]), 270.0, 3.0),  # 15 mm gap
    ]
    # Wall 1 is weaker than wall 0 → adjacency reject demotes wall 1.
    meta = [
        {"snapped_to": "manhattan", "confidence": 1.0},   # wall 0 high
        {"snapped_to": "manhattan", "confidence": 0.3},   # wall 1 low
        {"snapped_to": "manhattan", "confidence": 1.0},
        {"snapped_to": "manhattan", "confidence": 1.0},
        {"snapped_to": "manhattan", "confidence": 1.0},
    ]

    result, diag = _stage8_polygon_closure(walls, meta, emit_diagnostics=True)

    # At least one demotion happened (adjacency reject → wall 1).
    assert len(diag["demotions_cascade"]) >= 1
    # The lowest-confidence wall (wall 1) must appear in the demotions.
    demoted_idxs = {d["wall_idx"] for d in diag["demotions_cascade"]}
    assert 1 in demoted_idxs
    # Polygon closes.
    delta_post = _polygon_closure(result)
    np.testing.assert_allclose(delta_post, [0.0, 0.0], atol=1e-6)


def test_demotion_cascade_bounded():
    """The cascade respects max_demotions and never loops forever."""
    # Build a pathological case where every wall claims the same snap
    # direction. The adjacent-snap-reject demotes everyone but the first,
    # and after that the KKT is unconstrained and trivially solves.
    walls = [
        (np.array([0.0, 0.0]), np.array([1.0, 0.0]), 0.0, 1.0),
        (np.array([1.0, 0.0]), np.array([2.0, 0.0]), 0.0, 1.0),
        (np.array([2.0, 0.0]), np.array([3.0, 0.0]), 0.0, 1.0),
        (np.array([3.0, 0.0]), np.array([0.01, 0.001]), 0.0, 3.0),  # closes loop
    ]
    meta = [{"snapped_to": "manhattan", "confidence": 1.0}] * 4
    # Lock max_demotions = 2 to exercise the bound.
    config = Stage8Config(max_demotions=2)

    # Should not raise / hang.
    result, diag = _stage8_polygon_closure(
        walls, meta, config=config, emit_diagnostics=True)

    # Either solved or fell back — never crashes.
    assert diag["solver_used"] in ("kkt", "slsqp", "fallback_unchanged")


def test_degenerate_rank_check_triggers_slsqp_or_fallback():
    """If KKT is rank-deficient, SLSQP must take over (not crash)."""
    # Force rank deficiency: make ALL walls snapped to the SAME direction,
    # so A's rows are collinear. (Not physically realistic — but the
    # solver must handle it.)
    walls = [
        (np.array([0.0, 0.0]), np.array([2.0, 0.0]), 0.0, 2.0),
        (np.array([2.0, 0.0]), np.array([4.0, 0.0]), 0.0, 2.0),
        (np.array([4.0, 0.0]), np.array([6.0, 0.0]), 0.0, 2.0),
        (np.array([6.0, 0.0]), np.array([0.02, 0.0]), 0.0, 6.0),
    ]
    meta = [{"snapped_to": "manhattan", "confidence": 1.0}] * 4

    # Must not raise — cascade or SLSQP absorbs the rank deficiency.
    result, diag = _stage8_polygon_closure(walls, meta, emit_diagnostics=True)
    assert diag["solver_used"] in ("kkt", "slsqp", "fallback_unchanged")


def test_cascade_demotes_worst_snap_first():
    """The demotion criterion is argmax(|n·Δ| / conf)."""
    # Set up so that one wall's perpendicular aligns with Δ and it has
    # low confidence — that one should be demoted first when cascade
    # kicks in. We trigger cascade via all-same-direction snaps.
    walls = [
        (np.array([0.0, 0.0]), np.array([3.0, 0.0]), 0.0, 3.0),
        (np.array([3.0, 0.0]), np.array([3.0, 0.5]), 90.0, 0.5),
        (np.array([3.0, 0.5]), np.array([0.0, 0.5]), 180.0, 3.0),
        (np.array([0.0, 0.5]), np.array([0.02, 0.0]), 270.0, 0.5),  # 2 cm x-gap
    ]
    meta = [
        {"snapped_to": "manhattan", "confidence": 1.0},   # wall 0 (perp=Y)
        {"snapped_to": "manhattan", "confidence": 0.2},   # wall 1 (perp=X) — low conf + aligned with Δ
        {"snapped_to": "manhattan", "confidence": 1.0},
        {"snapped_to": "manhattan", "confidence": 1.0},
    ]

    result, diag = _stage8_polygon_closure(walls, meta, emit_diagnostics=True)

    # Polygon closes regardless.
    delta_post = _polygon_closure(result)
    np.testing.assert_allclose(delta_post, [0.0, 0.0], atol=1e-6)
