"""Unit tests for cloud_slam.deskew.deskew_scan."""

import time
import warnings

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cloud_slam.deskew import deskew_scan
import cloud_slam.deskew as deskew_mod


@pytest.fixture(autouse=True)
def _reset_warn_state():
    """Each test sees a fresh warn-once dict."""
    deskew_mod._WARNED = {"missing_imu": False}
    yield


def _make_imu(duration=0.1, rate=250.0, gyro=(0.0, 0.0, 0.0), t0=0.0):
    n = max(int(duration * rate), 2)
    times = t0 + np.linspace(0.0, duration, n)
    gyros = np.tile(np.asarray(gyro, dtype=np.float64), (n, 1))
    return times, gyros


def test_zero_imu_returns_input_unchanged():
    xyz = np.array([[1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]])
    offsets = np.array([0.0, 0.04, 0.08])
    imu_times, imu_gyros = _make_imu(duration=0.1, gyro=(0.0, 0.0, 0.0),
                                     t0=1000.0)
    out = deskew_scan(xyz, offsets, 1000.0, imu_times, imu_gyros)
    np.testing.assert_allclose(out, xyz, atol=1e-12)


def test_constant_gyro_rotates_each_point_backward():
    # 1 rad/s around Z for 80 ms. Point at t_offset rotated by
    # -(tmax - t) rad about Z, because reference="end".
    xyz = np.array([[1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0]])
    offsets = np.array([0.0, 0.04, 0.08])
    t0 = 1000.0
    imu_times, imu_gyros = _make_imu(duration=0.1, gyro=(0.0, 0.0, 1.0),
                                     t0=t0)
    out = deskew_scan(xyz, offsets, t0, imu_times, imu_gyros, reference="end")
    tmax = offsets.max()
    for i, t in enumerate(offsets):
        dtheta = tmax - t  # forward rotation from t_pt to t_ref
        expected = Rotation.from_rotvec([0.0, 0.0, -dtheta]).as_matrix() @ xyz[i]
        np.testing.assert_allclose(out[i], expected, atol=2e-3)


def test_reference_start_anchors_at_scan_start():
    xyz = np.array([[1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0]])
    offsets = np.array([0.0, 0.08])
    t0 = 1000.0
    imu_times, imu_gyros = _make_imu(duration=0.1, gyro=(0.0, 0.0, 1.0),
                                     t0=t0)
    out = deskew_scan(xyz, offsets, t0, imu_times, imu_gyros, reference="start")
    # Point at offset 0 is exactly at t_ref → unchanged.
    np.testing.assert_allclose(out[0], xyz[0], atol=1e-6)
    # Point at offset 0.08 rotated forward by 0.08 rad about Z.
    expected = Rotation.from_rotvec([0.0, 0.0, 0.08]).as_matrix() @ xyz[1]
    np.testing.assert_allclose(out[1], expected, atol=2e-3)


def test_missing_imu_window_warns_and_returns_input():
    xyz = np.array([[1.0, 0.0, 0.0]])
    offsets = np.array([0.0, 0.08])  # need 2+ distinct offsets
    xyz = np.tile(xyz, (2, 1))
    # IMU far away in time.
    imu_times, imu_gyros = _make_imu(duration=0.1, gyro=(0.0, 0.0, 1.0),
                                     t0=500.0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = deskew_scan(xyz, offsets, 1000.0, imu_times, imu_gyros)
    np.testing.assert_allclose(out, xyz, atol=1e-12)
    assert any("imu samples missing" in str(x.message) for x in w)


def test_all_same_time_offset_no_deskew():
    xyz = np.random.default_rng(0).normal(size=(50, 3))
    offsets = np.zeros(50)
    imu_times, imu_gyros = _make_imu(duration=0.1, gyro=(0.0, 0.0, 5.0),
                                     t0=1000.0)
    out = deskew_scan(xyz, offsets, 1000.0, imu_times, imu_gyros)
    np.testing.assert_allclose(out, xyz, atol=0.0)


def test_large_scan_perf_under_50ms():
    rng = np.random.default_rng(0)
    xyz = rng.normal(size=(5000, 3))
    offsets = rng.uniform(0.0, 0.08, size=5000)
    imu_times, imu_gyros = _make_imu(duration=0.1, gyro=(0.1, 0.2, 0.3),
                                     t0=1000.0)
    # Warm-up to absorb scipy JIT cost from the first call in this process.
    deskew_scan(xyz[:10], offsets[:10], 1000.0, imu_times, imu_gyros)
    start = time.perf_counter()
    out = deskew_scan(xyz, offsets, 1000.0, imu_times, imu_gyros)
    elapsed = time.perf_counter() - start
    assert out.shape == xyz.shape
    assert elapsed < 0.05, f"deskew took {elapsed*1000:.1f} ms (budget 50 ms)"
