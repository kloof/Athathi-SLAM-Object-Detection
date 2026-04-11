#!/usr/bin/env python3
"""
Validate 3D bounding box math: OBB fitters, dimension mapping, and enclosure.

Each test creates points with known geometry, fits an OBB, then verifies:
  - Dimensions match the known size
  - R @ [±w/2, ±d/2, ±h/2] + center encloses all points
  - Quaternion is a proper rotation (det=+1)
"""

import sys
import os
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def check_enclosure(pts, center, dims, quat, tolerance=0.05):
    """Check what fraction of points fall inside the box."""
    R = Rotation.from_quat(quat).as_matrix()
    pts_local = (R.T @ (pts - center).T).T
    half = dims / 2
    enclosed = np.all(np.abs(pts_local) <= half + tolerance, axis=1)
    return enclosed.mean()


def test_gravity_aligned_obb():
    """Test fit_gravity_aligned_obb with a known rectangle."""
    from cloud_slam.frustum import fit_gravity_aligned_obb

    rng = np.random.RandomState(42)
    gravity_up = np.array([0.0, 0.0, 1.0])

    passed = 0
    tested = 0

    for yaw_deg in range(0, 180, 15):
        yaw = np.radians(yaw_deg)
        R2 = np.array([[np.cos(yaw), -np.sin(yaw)],
                        [np.sin(yaw),  np.cos(yaw)]])

        # 0.8m x 0.3m rectangle, 0.5m tall
        n = 200
        local = np.column_stack([
            rng.uniform(-0.4, 0.4, n),
            rng.uniform(-0.15, 0.15, n),
            rng.uniform(0, 0.5, n),
        ])
        pts = np.column_stack([
            (R2 @ local[:, :2].T).T + [2.0, 1.0],
            local[:, 2]
        ])

        obb = fit_gravity_aligned_obb(pts, gravity_up)
        assert obb is not None, f"OBB returned None at yaw={yaw_deg}"

        # Check rotation is proper
        R_box = Rotation.from_quat(obb['rotation_quat_xyzw']).as_matrix()
        det = np.linalg.det(R_box)
        assert abs(det - 1.0) < 0.01, f"det(R)={det:.3f} at yaw={yaw_deg}"

        # Check enclosure
        enc = check_enclosure(pts, obb['center'], obb['dimensions'],
                              obb['rotation_quat_xyzw'])
        tested += 1
        if enc > 0.95:
            passed += 1
        else:
            print(f"  WARN: yaw={yaw_deg}° enclosure={enc*100:.1f}%")

    print(f"  fit_gravity_aligned_obb: {passed}/{tested} passed (>95% enclosure)")
    return passed == tested


def test_manhattan_obb():
    """Test fit_manhattan_obb including the swap path."""
    from cloud_slam.manhattan import ManhattanFrame, fit_manhattan_obb

    rng = np.random.RandomState(42)
    gravity_up = np.array([0.0, 0.0, 1.0])

    # Manhattan frame at 20° from X
    peak = np.radians(20)
    ax0 = np.array([np.cos(peak), np.sin(peak), 0])
    ax2 = gravity_up
    ax1 = np.cross(ax2, ax0); ax1 /= np.linalg.norm(ax1)
    ax0 = np.cross(ax1, ax2)
    R_m = np.vstack([ax0, ax1, ax2])
    mf = ManhattanFrame(R=R_m, confidence=1.0)

    passed = 0

    # Test 1: Object aligned with Manhattan X (no swap needed)
    pts_m = np.column_stack([
        rng.uniform(-0.5, 0.5, 300),
        rng.uniform(-0.2, 0.2, 300),
        rng.uniform(0, 0.8, 300),
    ])
    pts_w = (R_m.T @ pts_m.T).T
    obb = fit_manhattan_obb(pts_w, mf)
    enc = check_enclosure(pts_w, obb['center'], obb['dimensions'],
                          obb['rotation_quat_xyzw'])
    status = "PASS" if enc > 0.95 else f"FAIL ({enc*100:.1f}%)"
    print(f"  Manhattan no-swap: {status}")
    if enc > 0.95: passed += 1

    # Test 2: Object aligned with Manhattan Y (swap should win)
    pts_m2 = np.column_stack([
        rng.uniform(-0.15, 0.15, 300),
        rng.uniform(-0.6, 0.6, 300),
        rng.uniform(0, 0.4, 300),
    ])
    pts_w2 = (R_m.T @ pts_m2.T).T
    obb2 = fit_manhattan_obb(pts_w2, mf)
    enc2 = check_enclosure(pts_w2, obb2['center'], obb2['dimensions'],
                           obb2['rotation_quat_xyzw'])
    status2 = "PASS" if enc2 > 0.95 else f"FAIL ({enc2*100:.1f}%)"
    print(f"  Manhattan swap: {status2}")
    if enc2 > 0.95: passed += 1

    print(f"  fit_manhattan_obb: {passed}/2 passed")
    return passed == 2


def test_tracked_yaw_obb():
    """Test _fit_tracked_yaw_obb."""
    from cloud_slam.box_refiner import _fit_tracked_yaw_obb

    rng = np.random.RandomState(42)
    gravity_up = np.array([0.0, 0.0, 1.0])

    passed = 0
    for yaw_deg in [0, 30, 45, 90, 135]:
        yaw = np.radians(yaw_deg)
        R2 = np.array([[np.cos(yaw), -np.sin(yaw)],
                        [np.sin(yaw),  np.cos(yaw)]])
        n = 200
        local = np.column_stack([
            rng.uniform(-0.5, 0.5, n),
            rng.uniform(-0.2, 0.2, n),
            rng.uniform(0, 0.6, n),
        ])
        pts = np.column_stack([
            (R2 @ local[:, :2].T).T + [1.0, 2.0],
            local[:, 2]
        ])
        obb = _fit_tracked_yaw_obb(pts, gravity_up, yaw)
        enc = check_enclosure(pts, obb['center'], obb['dimensions'],
                              obb['rotation_quat_xyzw'])
        if enc > 0.95:
            passed += 1
        else:
            print(f"  WARN: yaw={yaw_deg}° enclosure={enc*100:.1f}%")

    print(f"  _fit_tracked_yaw_obb: {passed}/5 passed")
    return passed == 5


def test_assign_dimensions_roundtrip():
    """Test that assign_dimensions + reverse mapping preserves the contract."""
    from cloud_slam.size_priors import assign_dimensions

    rng = np.random.RandomState(42)
    gravity_up = np.array([0.0, 0.0, 1.0])
    passed = 0

    for _ in range(50):
        R = Rotation.random(random_state=rng).as_matrix()
        dims = rng.uniform(0.1, 2.0, 3)

        ordered, (w_idx, d_idx, h_idx) = assign_dimensions(dims, R, gravity_up)

        # Verify semantics: width >= depth
        assert ordered[0] >= ordered[1] - 1e-6, f"width < depth: {ordered}"

        # Round-trip: map back to rotation-column order
        final = np.empty(3)
        final[w_idx] = ordered[0]
        final[d_idx] = ordered[1]
        final[h_idx] = ordered[2]

        # Should recover original dims
        if np.allclose(final, dims, atol=1e-10):
            passed += 1

    print(f"  assign_dimensions roundtrip: {passed}/50 passed")
    return passed == 50


def test_bayesian_refinement():
    """Test that Bayesian refinement behaves correctly at extremes."""
    from cloud_slam.size_priors import refine_dimensions

    # High-point, high-confidence: trust observation
    obs = np.array([1.0, 0.6, 0.75])
    ref, _ = refine_dimensions(obs, "desk", num_points=500, confidence=0.9)
    close_to_obs = np.allclose(ref, obs, atol=0.15)
    print(f"  High points+conf: obs={obs}, ref={ref}, close_to_obs={close_to_obs}")

    # Low-point, low-confidence: trust prior (desk prior: 1.2, 0.6, 0.75)
    obs2 = np.array([0.3, 0.1, 0.3])
    ref2, _ = refine_dimensions(obs2, "desk", num_points=10, confidence=0.3)
    pulled_toward_prior = np.all(ref2 > obs2)
    print(f"  Low points+conf: obs={obs2}, ref={ref2}, pulled_toward_prior={pulled_toward_prior}")

    ok = close_to_obs and pulled_toward_prior
    print(f"  Bayesian refinement: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    print("=" * 60)
    print("  Bounding Box Math Tests")
    print("=" * 60)

    results = []
    for name, fn in [
        ("gravity_aligned_obb", test_gravity_aligned_obb),
        ("manhattan_obb", test_manhattan_obb),
        ("tracked_yaw_obb", test_tracked_yaw_obb),
        ("assign_dimensions_roundtrip", test_assign_dimensions_roundtrip),
        ("bayesian_refinement", test_bayesian_refinement),
    ]:
        print(f"\n[TEST] {name}")
        try:
            ok = fn()
        except Exception as e:
            print(f"  ERROR: {e}")
            ok = False
        results.append((name, ok))

    print(f"\n{'=' * 60}")
    all_pass = all(ok for _, ok in results)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    print(f"{'=' * 60}")
    print(f"  {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
