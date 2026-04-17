"""
LiDAR intra-scan deskew via IMU gyro integration.

Each lidar scan spans ~80 ms on the Unitree L2. During hand-held motion,
the sensor rotates meaningfully within that window, producing visible
distortion in the accumulated map. This module rotates each point back
to a single reference time (default: scan end) by integrating gyro
samples between the point's capture time and the reference.

Translation is deliberately ignored — integrating accelerometer over
80 ms is noise-dominated for hand-held scans. Rotation dominates.
"""

import warnings

import numpy as np
from scipy.spatial.transform import Rotation


_WARNED = {"missing_imu": False}


def _warn_once(key: str, message: str) -> None:
    if not _WARNED.get(key):
        _WARNED[key] = True
        warnings.warn(message, stacklevel=3)


def _angular_delta(imu_times: np.ndarray,
                   imu_gyros: np.ndarray,
                   t0: float,
                   t1: float) -> np.ndarray:
    """Integrate gyro from t0 to t1. Result is a rotation vector (rad)."""
    if t1 == t0:
        return np.zeros(3)

    sign = 1.0
    a, b = t0, t1
    if b < a:
        sign = -1.0
        a, b = b, a

    # Select IMU samples that cover [a, b] plus one on each side for edges.
    lo = int(np.searchsorted(imu_times, a, side="right")) - 1
    hi = int(np.searchsorted(imu_times, b, side="left")) + 1
    lo = max(lo, 0)
    hi = min(hi, len(imu_times))

    if hi - lo < 2:
        # Fall back to nearest-sample constant-gyro assumption.
        idx = int(np.clip(np.searchsorted(imu_times, 0.5 * (a + b)), 0,
                          len(imu_gyros) - 1))
        return sign * imu_gyros[idx] * (b - a)

    # Trapezoidal integration clamped to [a, b].
    t = imu_times[lo:hi]
    g = imu_gyros[lo:hi]
    acc = np.zeros(3)
    for k in range(len(t) - 1):
        ta, tb = t[k], t[k + 1]
        if tb <= a or ta >= b:
            continue
        sa = max(ta, a)
        sb = min(tb, b)
        if sb <= sa:
            continue
        # Linear interp of gyro at sa, sb.
        frac_a = (sa - ta) / (tb - ta)
        frac_b = (sb - ta) / (tb - ta)
        ga = g[k] * (1 - frac_a) + g[k + 1] * frac_a
        gb = g[k] * (1 - frac_b) + g[k + 1] * frac_b
        acc += 0.5 * (ga + gb) * (sb - sa)
    return sign * acc


def deskew_scan(xyz: np.ndarray,
                time_offsets: np.ndarray,
                scan_stamp: float,
                imu_times: np.ndarray,
                imu_gyros: np.ndarray,
                *,
                reference: str = "end",
                bucket_ms: float = 1.0,
                ) -> np.ndarray:
    """Undo intra-scan rotation using IMU gyro integration.

    Returns xyz rotated from each point's capture-time lidar frame back
    to the scan reference time (default: scan end). Translation is
    ignored.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    time_offsets = np.asarray(time_offsets, dtype=np.float64)

    if xyz.size == 0 or time_offsets.size == 0:
        return xyz

    if imu_times is None or len(imu_times) == 0:
        _warn_once("missing_imu",
                   "imu samples missing for deskew, returning raw scan")
        return xyz

    tmax = float(time_offsets.max())
    tmin = float(time_offsets.min())
    if tmax == tmin:
        return xyz  # co-located in time, nothing to undo

    if reference == "end":
        t_ref_offset = tmax
    elif reference == "start":
        t_ref_offset = tmin
    else:
        raise ValueError(f"reference must be 'end' or 'start', got {reference!r}")

    t_ref = scan_stamp + t_ref_offset

    imu_times = np.asarray(imu_times, dtype=np.float64)
    imu_gyros = np.asarray(imu_gyros, dtype=np.float64)

    # Check that IMU window overlaps the scan window.
    t_scan_lo = scan_stamp + tmin
    t_scan_hi = scan_stamp + tmax
    if imu_times[-1] < t_scan_lo or imu_times[0] > t_scan_hi:
        _warn_once("missing_imu",
                   "imu samples missing for deskew, returning raw scan")
        return xyz

    # Bucket points by time_offset to avoid per-point integration.
    bucket = np.round(time_offsets / (bucket_ms * 1e-3)).astype(np.int64)
    unique_buckets, inverse = np.unique(bucket, return_inverse=True)

    # Pre-compute rotation matrices per bucket.
    rot_mats = np.empty((len(unique_buckets), 3, 3))
    for k, b in enumerate(unique_buckets):
        t_pt = scan_stamp + b * bucket_ms * 1e-3
        # Rotation that takes a vector from the point's lidar body frame
        # at t_pt to the reference frame at t_ref. Using gyro in body
        # frame, the angular delta from t_pt forward to t_ref represents
        # how the body rotated; to align points captured earlier with
        # the later reference we apply the inverse of that rotation.
        dtheta = _angular_delta(imu_times, imu_gyros, t_pt, t_ref)
        rot_mats[k] = Rotation.from_rotvec(-dtheta).as_matrix()

    out = np.empty_like(xyz)
    # Apply per-bucket rotation — cheap loop, <100 iterations typical.
    for k in range(len(unique_buckets)):
        mask = inverse == k
        if not mask.any():
            continue
        out[mask] = xyz[mask] @ rot_mats[k].T
    return out
