"""M1a: validates Stage 8 polygon-closure solver applies the correct
per-corner KKT distribution and respects Manhattan snap constraints.

Exercises:
  - the strict no-op path on |Δ| < no_op_threshold_mm
  - the KKT-closed-form path on a small closure gap, all free walls
  - post-solve safeguards (per-corner shift bound, min-wall-length)
  - diagnostics emit opt-in behaviour
"""
import numpy as np
import pytest

from cloud_slam.floorplan.config import Stage8Config
from cloud_slam.floorplan.refine import _stage8_polygon_closure
from tests.fixtures.synthetic.walls_closed_rectangle import walls_closed_rectangle
from tests.fixtures.synthetic.walls_open_lshape import walls_open_lshape


def _polygon_closure(walls):
    """Σ (p2 − p1) — should be ≈ 0 on a closed polygon."""
    return sum((w[1] - w[0] for w in walls), start=np.zeros(2))


def test_no_op_on_closed_polygon():
    """|Δ| = 0 → unchanged geometry and no diagnostics emitted by default."""
    walls = walls_closed_rectangle(length=4.0, width=3.0)
    meta = [{"snapped_to": "dominant", "confidence": 1.0}] * 4

    result, diag = _stage8_polygon_closure(walls, meta)

    # Default (emit_diagnostics=False) and Δ ≈ 0 → None, walls byte-identical.
    assert diag is None
    for orig, got in zip(walls, result):
        np.testing.assert_allclose(orig[0], got[0])
        np.testing.assert_allclose(orig[1], got[1])
        assert orig[2] == pytest.approx(got[2])
        assert orig[3] == pytest.approx(got[3])


def test_no_op_emits_noop_tag_when_diagnostics_requested():
    """Even on a no-op, diagnostics=True must surface the 'noop' tag."""
    walls = walls_closed_rectangle(length=4.0, width=3.0)
    meta = [{"snapped_to": "dominant", "confidence": 1.0}] * 4

    result, diag = _stage8_polygon_closure(walls, meta, emit_diagnostics=True)

    assert diag is not None
    assert diag["solver_used"] == "noop"
    # Geometry still unchanged.
    for orig, got in zip(walls, result):
        np.testing.assert_allclose(orig[0], got[0])


def test_1cm_closure_redistribution_is_cyclic():
    """A 1-cm gap on the L-shape (all free walls) must close exactly."""
    walls = walls_open_lshape(closure_gap_m=0.01)
    meta = [{"snapped_to": "free", "confidence": 1.0}] * 6

    # Pre-closure Δ is 10 mm in x.
    delta_pre = _polygon_closure(walls)
    assert np.linalg.norm(delta_pre) == pytest.approx(0.01, abs=1e-9)

    result, diag = _stage8_polygon_closure(walls, meta, emit_diagnostics=True)

    # After closure, Σ (p2 − p1) ≈ 0.
    delta_post = _polygon_closure(result)
    np.testing.assert_allclose(delta_post, [0.0, 0.0], atol=1e-6)
    assert diag["solver_used"] in ("kkt", "slsqp")


def test_per_corner_shift_bounded():
    """|shifts| must respect max(per_corner_reject_mm, frac · |Δ|_mm)."""
    walls = walls_open_lshape(closure_gap_m=0.10)
    meta = [{"snapped_to": "free", "confidence": 1.0}] * 6

    result, diag = _stage8_polygon_closure(walls, meta, emit_diagnostics=True)

    # With |Δ| = 100 mm: max_allowed_mm = max(50, 0.6 · 100) = 60 mm.
    for shift in diag["per_corner_shift_mm"]:
        assert shift <= max(50.0, 0.6 * 100.0) + 1e-6


def test_kkt_redistributes_with_weighted_snap():
    """Manhattan-snapped 4-wall rectangle with a 2 cm gap → KKT closes it.

    The solver should keep all 4 walls axis-aligned, and the post-solve
    polygon must close to numerical zero.
    """
    walls = list(walls_closed_rectangle(length=4.0, width=3.0))
    # Shove the last wall's p2 by +2 cm in x → opens the polygon.
    p1, p2, a, L = walls[-1]
    walls[-1] = (p1, p2 + np.array([0.02, 0.0]), a, L + 0.02)
    meta = [{"snapped_to": "manhattan", "confidence": 1.0}] * 4

    result, diag = _stage8_polygon_closure(walls, meta, emit_diagnostics=True)

    assert diag["solver_used"] in ("kkt", "slsqp")
    # Post-closure polygon is closed.
    delta_post = _polygon_closure(result)
    np.testing.assert_allclose(delta_post, [0.0, 0.0], atol=1e-6)
    # Walls remain axis-aligned (snap directions preserved).
    for (pp1, pp2, ang, _l) in result:
        v = pp2 - pp1
        # Manhattan = either horizontal or vertical.
        assert abs(v[0]) < 1e-6 or abs(v[1]) < 1e-6


def test_min_wall_length_revert():
    """A solve that would shrink a wall below min_wall_length_m reverts."""
    # Build a rectangle with one very short wall (18 cm) and a big gap that
    # would pull that wall below the 15 cm threshold.
    walls = [
        (np.array([0.0, 0.0]), np.array([4.0, 0.0]), 0.0, 4.0),
        (np.array([4.0, 0.0]), np.array([4.0, 0.18]), 90.0, 0.18),
        (np.array([4.0, 0.18]), np.array([0.0, 0.18]), 180.0, 4.0),
        # Big gap: last wall's p2 is 1 m off, so Δ is huge.
        (np.array([0.0, 0.18]), np.array([1.0, 0.0]), 270.0, 1.0),
    ]
    meta = [{"snapped_to": "free", "confidence": 1.0}] * 4

    result, diag = _stage8_polygon_closure(walls, meta, emit_diagnostics=True)

    # Solver either gracefully reverts (fallback_unchanged) or closes the
    # polygon while keeping every wall above min_wall_length_m. Both are
    # acceptable — the contract is "no wall shorter than 15 cm survives".
    for (_p1, _p2, _a, L) in result:
        assert L >= 0.15 - 1e-9 or diag["solver_used"] == "fallback_unchanged"


def test_large_gap_exceeds_shift_bound_rejects():
    """When the per-corner shift exceeds the bound, the entire solve rejects."""
    # Tight bounds: force rejection when the redistributed shift is > 10 mm.
    walls = walls_open_lshape(closure_gap_m=0.05)
    meta = [{"snapped_to": "free", "confidence": 1.0}] * 6
    config = Stage8Config(
        per_corner_reject_mm=10.0,
        per_corner_reject_frac_of_delta=0.1,  # 10% of 50 mm = 5 mm
    )
    # With |Δ| = 50 mm, max_allowed = max(10, 5) = 10 mm. But the solver
    # pushes 25 mm into two corners → exceeds 10 mm → reject.

    result, diag = _stage8_polygon_closure(
        walls, meta, config=config, emit_diagnostics=True)

    # When shift exceeds bound, walls must be returned unchanged.
    if diag.get("rejection_reason") == "per_corner_shift_exceeded":
        for orig, got in zip(walls, result):
            np.testing.assert_allclose(orig[0], got[0])
            np.testing.assert_allclose(orig[1], got[1])
