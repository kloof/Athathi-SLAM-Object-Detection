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

Coverage map of the degeneracy tests below:
  - test_full_rank_manhattan_redistributes_without_cascade:
      full-rank A (distinct Manhattan directions) → direct KKT solve,
      no cascade. Sanity check on the happy path.
  - test_adjacent_collinear_snaps_pre_demote_then_solve:
      adjacent walls snapped to the SAME direction → the pre-flight
      adjacent-snap-reject demotes the lower-conf wall BEFORE the
      rank check / KKT / SLSQP runs. The rank fallback itself is
      never reached on this input.
  - test_non_adjacent_parallel_snaps_trigger_rank_cascade:
      exercises the pre-flight rank check at refine.py:1293-1296 by
      mocking out _stage8_adjacent_snap_reject so the rank-deficient
      input reaches the KKT rank check. In the real pipeline adjacent-
      snap-reject catches this kind of ring-wrapping collinearity
      upstream, but the rank check is the ultimate safety net for any
      pathology that slips past.
"""
import numpy as np
import pytest

import cloud_slam.floorplan.refine as refine_module
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


def test_adjacent_collinear_snaps_pre_demote_then_solve():
    """All-same-direction 4-wall input is demoted by the pre-flight
    adjacent-snap-reject BEFORE the rank check / KKT / SLSQP runs.

    This is the first line of defence: when ALL walls claim the SAME
    snap direction, every adjacent pair's |angle_diff| is 0 < tol, so
    the pre-flight demotes walls to 'free' until adjacent pairs are
    distinct. The KKT solver then sees a well-conditioned system.

    Primary assertions:
      - at least one demotion was recorded (the pre-flight fired)
      - the polygon closes (solver succeeded after demotion)
      - solver_used is one of the 3 expected tags (no crash)
    """
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

    # Must not raise — pre-flight adjacent-snap-reject demotes walls
    # until the solver can close the polygon.
    result, diag = _stage8_polygon_closure(walls, meta, emit_diagnostics=True)
    assert diag["solver_used"] in ("kkt", "slsqp", "fallback_unchanged")
    # The pre-flight adjacent-snap-reject must have logged at least one
    # demotion on this all-same-direction input. Pre-flight demotions
    # carry attempt == -1 in the cascade log.
    preflight_demotions = [
        d for d in diag["demotions_cascade"] if d.get("attempt") == -1
    ]
    assert len(preflight_demotions) >= 1, (
        "adjacent-snap-reject should fire on all-0° walls"
    )


def test_full_rank_manhattan_redistributes_without_cascade():
    """4-wall Manhattan rectangle → full-rank A → direct KKT solve.

    All 4 walls are at distinct Manhattan directions (0°, 90°, 180°,
    270°), so the snap-constraint matrix A is full row rank. The
    closed-form KKT path solves in one numpy.linalg.solve call — no
    cascade, no demotion, no SLSQP.

    This is the happy-path regression check for the solver cascade:
    we verify that well-conditioned inputs are handled by the fast
    path without triggering any fallback logic.
    """
    # Set up so that one wall's perpendicular aligns with Δ and it has
    # low confidence — this would be the first demotion target IF a
    # cascade were triggered. On this full-rank input, cascade is not
    # triggered and no demotion is recorded.
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
    # Full-rank input: KKT solves directly, no demotion cascade.
    assert diag["solver_used"] == "kkt"
    assert diag["demotions_cascade"] == []


def test_non_adjacent_parallel_snaps_trigger_rank_cascade(monkeypatch):
    """Rank-deficient A that bypasses adjacent-snap-reject triggers
    the KKT pre-flight rank check at refine.py:1293-1296, which then
    routes to the SLSQP fallback (and cascade-demotes if even SLSQP
    cannot close the polygon).

    In the real pipeline, any snap configuration that produces a
    rank-deficient A is also caught upstream by the adjacent-snap-
    reject pre-flight (because ring-wrapping collinearity requires
    adjacent pairs to share a direction). The rank check in
    _stage8_solve_kkt is the ultimate safety net: if a pathology
    slipped past adjacent-snap-reject, the rank check must catch it
    and force KKT → SLSQP (and, on SLSQP failure, → cascade demote).

    To exercise this safety net we mock _stage8_adjacent_snap_reject
    to return meta unchanged, then feed a 4-wall all-0° input. Rows
    0..3 each encode a y-only constraint on a disjoint corner pair,
    wrapping the ring, so they are linearly dependent (sum = 0) and
    A has rank 3 out of 4 rows.

    Primary assertions:
      - the KKT pre-flight rank check fires on the first attempt
        (sentinel: kkt_cond is inf because the rank check returns
        (None, float('inf')) before even building the KKT matrix)
      - the solver recovers via SLSQP (or cascade + KKT/SLSQP) and
        ends with one of the 3 expected tags — never crashes
      - the closure gap is absorbed: either |Δ_post| ≈ 0 (solver
        closed it) or the walls are returned unchanged (fallback).
    """
    # Feed the "all walls snapped to 0°" pathology directly into the
    # KKT path by nop'ing out the adjacent-snap-reject pre-flight.
    def _passthrough(walls, meta, tol_deg):
        return [dict(m) for m in meta], []

    monkeypatch.setattr(
        refine_module, "_stage8_adjacent_snap_reject", _passthrough
    )

    # 4-wall ring with all walls snapped to 0° → perps all (0, 1) →
    # rows 0,1,2,3 touch only y-components on disjoint corner pairs
    # wrapping the ring → sum of rows is zero → rank(A) = 3, not 4.
    walls = [
        (np.array([0.0, 0.0]), np.array([2.0, 0.0]), 0.0, 2.0),
        (np.array([2.0, 0.0]), np.array([4.0, 0.0]), 0.0, 2.0),
        (np.array([4.0, 0.0]), np.array([6.0, 0.0]), 0.0, 2.0),
        (np.array([6.0, 0.0]), np.array([0.02, 0.0]), 0.0, 6.0),
    ]
    meta = [
        {"snapped_to": "manhattan", "confidence": 1.0},
        {"snapped_to": "manhattan", "confidence": 0.3},  # weakest — first cascade target if SLSQP fails
        {"snapped_to": "manhattan", "confidence": 1.0},
        {"snapped_to": "manhattan", "confidence": 1.0},
    ]

    result, diag = _stage8_polygon_closure(walls, meta, emit_diagnostics=True)

    # The KKT rank check fires on attempt 0: the _stage8_solve_kkt
    # rank check returns (None, float('inf')), which is recorded as
    # kkt_cond. The solver then falls through to SLSQP. kkt_cond on
    # the final diagnostic reflects the LAST KKT attempt — on a
    # successful SLSQP path with no cascade, that's still the inf
    # sentinel from attempt 0.
    assert diag["kkt_cond"] == float("inf"), (
        "rank check must return inf sentinel on attempt 0 to route KKT -> SLSQP"
    )
    # Solver must end in one of the 3 expected tags — never crash.
    # On this input SLSQP typically closes the polygon (rank-deficient
    # equality constraints are feasible), producing solver_used='slsqp'.
    # If SLSQP fails too, cascade demotes and retries; either way the
    # final tag is in this set.
    assert diag["solver_used"] in ("kkt", "slsqp", "fallback_unchanged")
    # The polygon either closes or the walls are reverted unchanged.
    if diag["solver_used"] != "fallback_unchanged":
        delta_post = _polygon_closure(result)
        np.testing.assert_allclose(delta_post, [0.0, 0.0], atol=1e-6)
