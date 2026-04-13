"""M0c: validates verify_calibration IoU computation logic.

These tests exercise the pure-numpy reprojection-IoU path — they avoid
invoking Mask2Former, Open3D, or the SLAM pipeline. Test inputs are
hand-crafted binary masks and a mock `project_fn` so the arithmetic is
trivially verifiable.
"""
from __future__ import annotations

import numpy as np
import pytest

from cloud_slam.calibration import (
    verify_calibration,
    should_warn,
    warning_message,
    _classify_tier,
    _rasterize_projected_mask,
    _compute_iou,
)


# --- Pure-unit helpers ------------------------------------------------------

def test_classify_tier_boundaries():
    # None → coarse (unverified).
    assert _classify_tier(None) == "coarse"
    # Below 0.70 → coarse.
    assert _classify_tier(0.0) == "coarse"
    assert _classify_tier(0.699) == "coarse"
    # [0.70, 0.85) → good.
    assert _classify_tier(0.70) == "good"
    assert _classify_tier(0.849) == "good"
    # >= 0.85 → tight.
    assert _classify_tier(0.85) == "tight"
    assert _classify_tier(1.0) == "tight"


def test_compute_iou_perfect_overlap():
    m = np.zeros((10, 10), dtype=bool)
    m[2:8, 2:8] = True
    assert _compute_iou(m, m) == 1.0


def test_compute_iou_no_overlap():
    a = np.zeros((10, 10), dtype=bool)
    a[0:3, 0:3] = True
    b = np.zeros((10, 10), dtype=bool)
    b[7:10, 7:10] = True
    assert _compute_iou(a, b) == 0.0


def test_compute_iou_partial_overlap():
    a = np.zeros((10, 10), dtype=bool)
    a[0:5, 0:5] = True   # 25 px
    b = np.zeros((10, 10), dtype=bool)
    b[3:8, 3:8] = True   # 25 px
    # intersection = 2x2 = 4, union = 25 + 25 - 4 = 46
    assert abs(_compute_iou(a, b) - 4 / 46) < 1e-9


def test_compute_iou_shape_mismatch_returns_zero():
    a = np.zeros((10, 10), dtype=bool)
    b = np.zeros((10, 12), dtype=bool)
    assert _compute_iou(a, b) == 0.0


def test_compute_iou_empty_masks_returns_zero():
    z = np.zeros((10, 10), dtype=bool)
    assert _compute_iou(z, z) == 0.0


def test_rasterize_dilates_single_point():
    H, W = 30, 30
    pixels = np.array([[15.0, 15.0]])
    in_front = np.array([True])
    mask = _rasterize_projected_mask(pixels, in_front, (H, W))
    # 3-px dilation of a single point → 7x7 square of True.
    assert mask.sum() == 49
    assert mask[15, 15]


def test_rasterize_skips_out_of_bounds():
    H, W = 30, 30
    pixels = np.array([[-5.0, -5.0], [100.0, 100.0], [15.0, 15.0]])
    in_front = np.array([True, True, True])
    mask = _rasterize_projected_mask(pixels, in_front, (H, W))
    # Only the in-bounds point at (15, 15) contributes.
    assert mask.sum() == 49


def test_rasterize_drops_behind_camera():
    H, W = 30, 30
    pixels = np.array([[15.0, 15.0], [5.0, 5.0]])
    in_front = np.array([False, True])
    mask = _rasterize_projected_mask(pixels, in_front, (H, W))
    assert not mask[15, 15]  # first point was dropped
    assert mask[5, 5]        # second kept (and dilated)


# --- verify_calibration end-to-end (pure-python mocks) ----------------------

def test_empty_input_returns_null_dict():
    result = verify_calibration(
        images_with_poses=[],
        lidar_wall_inliers_per_frame=[],
        project_fn=lambda pts, pose: (
            np.zeros((0, 2)), np.array([], dtype=bool)),
        image_shape=(720, 1280),
    )
    assert result["reprojection_iou_mean"] is None
    assert result["reprojection_iou_min"] is None
    assert result["reprojection_iou_frames_checked"] == 0
    assert result["accuracy_tier"] == "coarse"


def test_disabled_masks_yields_null_dict():
    """All frames report wall_mask=None (e.g., Mask2Former disabled)."""
    result = verify_calibration(
        images_with_poses=[(None, np.eye(4), None) for _ in range(10)],
        lidar_wall_inliers_per_frame=[np.zeros((5, 3)) for _ in range(10)],
        project_fn=lambda pts, pose: (
            np.zeros((0, 2)), np.array([], dtype=bool)),
        image_shape=(100, 100),
        sample_every=1,
    )
    assert result["reprojection_iou_frames_checked"] == 0
    assert result["reprojection_iou_mean"] is None


def test_perfect_overlap_is_tight_tier():
    """Projected-lidar mask aligns with vision mask → high IoU, tight tier."""
    H, W = 100, 100
    wall_mask = np.zeros((H, W), dtype=bool)
    wall_mask[30:70, 30:70] = True  # 40x40 square → 1600 px

    # Project points densely across the same 40x40 area.
    def proj(pts_world, pose):
        pixels = np.array(
            [[float(u), float(v)]
             for v in range(30, 70, 2)
             for u in range(30, 70, 2)],
            dtype=float,
        )
        return pixels, np.ones(len(pixels), dtype=bool)

    result = verify_calibration(
        images_with_poses=[(None, np.eye(4), wall_mask)],
        lidar_wall_inliers_per_frame=[np.zeros((400, 3))],
        project_fn=proj,
        image_shape=(H, W),
        sample_every=1,
    )
    assert result["reprojection_iou_mean"] is not None
    # After 3-px dilation the projected mask slightly exceeds the vision
    # square; expect IoU comfortably in the "good" / "tight" band.
    assert result["reprojection_iou_mean"] > 0.7
    assert result["accuracy_tier"] in ("good", "tight")
    assert result["reprojection_iou_frames_checked"] == 1


def test_no_overlap_is_coarse_tier():
    H, W = 100, 100
    wall_mask = np.zeros((H, W), dtype=bool)
    wall_mask[10:30, 10:30] = True

    def proj(pts_world, pose):
        # Project far from the vision mask.
        pixels = np.array(
            [[float(u), float(v)]
             for v in range(60, 80, 2)
             for u in range(60, 80, 2)],
            dtype=float,
        )
        return pixels, np.ones(len(pixels), dtype=bool)

    result = verify_calibration(
        images_with_poses=[(None, np.eye(4), wall_mask)],
        lidar_wall_inliers_per_frame=[np.zeros((100, 3))],
        project_fn=proj,
        image_shape=(H, W),
        sample_every=1,
    )
    assert result["reprojection_iou_mean"] < 0.1
    assert result["accuracy_tier"] == "coarse"


def test_sample_every_subsamples_frames():
    """sample_every=5 across 20 frames processes exactly 4 (indices 0,5,10,15)."""
    H, W = 50, 50
    wall_mask = np.zeros((H, W), dtype=bool)
    wall_mask[20:30, 20:30] = True

    def proj(pts_world, pose):
        pixels = np.array(
            [[float(u), float(v)]
             for v in range(20, 30)
             for u in range(20, 30)],
            dtype=float,
        )
        return pixels, np.ones(len(pixels), dtype=bool)

    triplets = [(None, np.eye(4), wall_mask) for _ in range(20)]
    inliers = [np.zeros((10, 3)) for _ in range(20)]

    result = verify_calibration(
        images_with_poses=triplets,
        lidar_wall_inliers_per_frame=inliers,
        project_fn=proj,
        image_shape=(H, W),
        sample_every=5,
    )
    assert result["reprojection_iou_frames_checked"] == 4


def test_projection_exception_is_skipped_not_fatal():
    """A single bad frame shouldn't kill verification of the others."""
    H, W = 50, 50
    wall_mask = np.zeros((H, W), dtype=bool)
    wall_mask[20:30, 20:30] = True

    call_count = {'n': 0}

    def proj(pts_world, pose):
        call_count['n'] += 1
        if call_count['n'] == 2:
            raise RuntimeError("sim projection failure")
        pixels = np.array(
            [[float(u), float(v)]
             for v in range(20, 30)
             for u in range(20, 30)],
            dtype=float,
        )
        return pixels, np.ones(len(pixels), dtype=bool)

    triplets = [(None, np.eye(4), wall_mask) for _ in range(3)]
    inliers = [np.zeros((10, 3)) for _ in range(3)]
    result = verify_calibration(
        images_with_poses=triplets,
        lidar_wall_inliers_per_frame=inliers,
        project_fn=proj,
        image_shape=(H, W),
        sample_every=1,
    )
    # 3 triplets, 1 errored → 2 valid frames.
    assert result["reprojection_iou_frames_checked"] == 2


# --- Warning helpers --------------------------------------------------------

def test_should_warn_respects_threshold():
    # None → no warning (no signal).
    assert not should_warn({"reprojection_iou_mean": None})
    # Below 0.6 → warn.
    assert should_warn({"reprojection_iou_mean": 0.3})
    assert should_warn({"reprojection_iou_mean": 0.599})
    # At/above 0.6 → no warn.
    assert not should_warn({"reprojection_iou_mean": 0.6})
    assert not should_warn({"reprojection_iou_mean": 0.9})


def test_warning_message_mentions_tier_and_iou():
    block = {
        "reprojection_iou_mean": 0.42,
        "reprojection_iou_min": 0.11,
        "reprojection_iou_frames_checked": 12,
        "accuracy_tier": "coarse",
    }
    msg = warning_message(block)
    assert "0.420" in msg
    assert "0.110" in msg
    assert "12" in msg
    assert "coarse" in msg
    # Must describe the remediation path.
    assert "Kalibr" in msg or "calibrateHandEye" in msg
