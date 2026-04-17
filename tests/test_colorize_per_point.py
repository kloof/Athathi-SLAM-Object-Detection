"""Unit tests for cloud_slam.colorizer.colorize_cloud_per_point."""

import numpy as np
import pytest

from cloud_slam import colorizer
from cloud_slam.colorizer import colorize_cloud, colorize_cloud_per_point


def _calib(w=100, h=100):
    """Synthetic calibration: identity extrinsics, simple pinhole K."""
    K = np.array([[100.0, 0.0, w / 2],
                  [0.0, 100.0, h / 2],
                  [0.0, 0.0, 1.0]])
    dist = np.zeros(5)
    T = np.eye(4)
    # Rotate lidar→camera so +X lidar axis maps to +Z camera axis.
    # Lidar frame: x-forward, y-left, z-up.
    # Camera frame: x-right, y-down, z-forward.
    T[:3, :3] = np.array([[0.0, -1.0, 0.0],
                          [0.0, 0.0, -1.0],
                          [1.0, 0.0, 0.0]])
    return {"K": K, "dist_coeffs": dist, "T_lidar_cam": T,
            "image_size": (w, h)}


def _solid_image(color, h=100, w=100):
    """H x W x 3 uint8 BGR image filled with a single color (BGR tuple)."""
    img = np.full((h, w, 3), color, dtype=np.uint8)
    return img


def _compressed_placeholder(idx):
    """Tiny valid JPEG-ish bytes; we'll monkeypatch cv2.imdecode anyway."""
    return bytes([0xFF, 0xD8, 0xFF, 0xD9]) + bytes([idx])


def test_single_image_matches_direct_call(monkeypatch):
    calib = _calib()
    red_bgr = _solid_image((0, 0, 255))  # BGR red
    # Points in front of the camera (positive x in lidar frame).
    xyz = np.array([[1.0, 0.0, 0.0],
                    [2.0, 0.1, 0.0],
                    [1.5, -0.1, 0.1]])
    point_ts = np.array([100.0, 100.01, 100.02])
    images = [(100.01, _compressed_placeholder(0), "jpeg")]
    image_ts = np.array([100.01])

    def fake_imdecode(buf, flags):
        return red_bgr

    monkeypatch.setattr(colorizer.cv2, "imdecode", fake_imdecode)

    per_point_colors = colorize_cloud_per_point(
        xyz, point_ts, images, image_ts, calib, max_dt=0.15)
    direct_colors = colorize_cloud(xyz, red_bgr, calib)
    np.testing.assert_allclose(per_point_colors, direct_colors, atol=0.0)


def test_two_images_split_by_timestamp(monkeypatch):
    calib = _calib()
    red_bgr = _solid_image((0, 0, 255))
    blue_bgr = _solid_image((255, 0, 0))

    # 4 points split 50/50 by timestamp. All project to image center so
    # pixel sample is always (50, 50) regardless of small XY jitter.
    xyz = np.array([[1.0, 0.0, 0.0]] * 4)
    point_ts = np.array([100.00, 100.00, 100.10, 100.10])
    images = [(100.00, _compressed_placeholder(0), "jpeg"),
              (100.10, _compressed_placeholder(1), "jpeg")]
    image_ts = np.array([100.00, 100.10])

    def fake_imdecode(buf, flags):
        # Last byte identifies which image we're decoding.
        idx_byte = buf.tobytes()[-1] if hasattr(buf, "tobytes") else int(buf[-1])
        return red_bgr if idx_byte == 0 else blue_bgr

    monkeypatch.setattr(colorizer.cv2, "imdecode", fake_imdecode)

    out = colorize_cloud_per_point(xyz, point_ts, images, image_ts, calib,
                                   max_dt=0.15)
    # Red (RGB = 1,0,0) for first two, blue (RGB = 0,0,1) for last two.
    np.testing.assert_allclose(out[0], [1.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(out[1], [1.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(out[2], [0.0, 0.0, 1.0], atol=1e-9)
    np.testing.assert_allclose(out[3], [0.0, 0.0, 1.0], atol=1e-9)


def test_no_match_returns_defaults(monkeypatch):
    calib = _calib()
    xyz = np.array([[1.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    # Point timestamps 10 seconds from any image → beyond max_dt.
    point_ts = np.array([100.0, 100.01])
    images = [(200.0, _compressed_placeholder(0), "jpeg")]
    image_ts = np.array([200.0])

    called = {"n": 0}

    def fake_imdecode(buf, flags):
        called["n"] += 1
        return _solid_image((0, 255, 0))

    monkeypatch.setattr(colorizer.cv2, "imdecode", fake_imdecode)

    out = colorize_cloud_per_point(xyz, point_ts, images, image_ts, calib,
                                   max_dt=0.15,
                                   default_color=(128, 128, 128))
    expected = np.tile(np.array([128, 128, 128]) / 255.0, (2, 1))
    np.testing.assert_allclose(out, expected, atol=0.0)
    assert called["n"] == 0, "unmatched points should not trigger any decode"


def test_mixed_matched_and_unmatched(monkeypatch):
    calib = _calib()
    red_bgr = _solid_image((0, 0, 255))

    xyz = np.array([[1.0, 0.0, 0.0]] * 3)
    # First two points match image at t=100; third is far in the future.
    point_ts = np.array([100.00, 100.02, 200.00])
    images = [(100.00, _compressed_placeholder(0), "jpeg")]
    image_ts = np.array([100.00])

    monkeypatch.setattr(colorizer.cv2, "imdecode",
                        lambda buf, flags: red_bgr)

    out = colorize_cloud_per_point(xyz, point_ts, images, image_ts, calib,
                                   max_dt=0.15,
                                   default_color=(128, 128, 128))
    np.testing.assert_allclose(out[0], [1.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(out[1], [1.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(out[2], [128 / 255.0] * 3, atol=1e-9)


def test_empty_xyz_returns_empty():
    calib = _calib()
    xyz = np.zeros((0, 3))
    point_ts = np.zeros(0)
    images = [(100.0, _compressed_placeholder(0), "jpeg")]
    image_ts = np.array([100.0])
    out = colorize_cloud_per_point(xyz, point_ts, images, image_ts, calib)
    assert out.shape == (0, 3)
