"""M0c: per-scan calibration verification via reprojection IoU.

Cross-checks the lidar-camera extrinsic calibration on a per-scan basis by
comparing Stage-1 RANSAC wall points (projected to each camera frame) against
Mask2Former's `wall` mask for the same frame. The intersection-over-union of
the two binary masks is a direct reprojection-quality signal: a well-calibrated
sensor pair yields IoU >= 0.85 ("tight"), a coarse manual alignment
typically sits below 0.70 ("coarse").

This module does NOT fix the calibration — it only exposes its quality so
downstream consumers (JSON schema, warnings, converters) can flag scans where
M2 accuracy will be floored by extrinsic drift.

See docs/plans/roomplan-quality.md §M0c for the tier boundaries and rationale.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore
    _HAVE_CV2 = True
except ImportError:
    _HAVE_CV2 = False


# --- Tier boundaries --------------------------------------------------------
# "coarse": manual visual alignment (no target) — 30-50 mm floor at 3 m.
# "good":   checkerboard-based monocular calibration; 10-30 mm at 3 m.
# "tight":  target-based joint lidar-camera calibration (Kalibr / HandEye);
#           sub-2 cm at 3 m, required for M2 ≤20 mm wall-edge accuracy.
_TIER_GOOD = 0.70
_TIER_TIGHT = 0.85

# Default dilation radius (px) applied to the projected-lidar mask before
# IoU. Lidar is sparse (one projected point per ~cm² of wall surface at 3 m
# and 10 Hz rotation) so a single pixel under-represents the wall patch it
# was sampled from. 3 px ≈ 1.5 cm in image-space for a 1280x720 frame at
# typical FOVs — matches the sparsity and stays below Mask2Former's edge
# uncertainty (~5-8 px on wall boundaries).
_LIDAR_MASK_DILATE_PX = 3

# Warn-below threshold. Below this, `accuracy_tier` is already "coarse" —
# the warning is for user attention ("your M2 wall accuracy is capped").
_WARN_IOU_BELOW = 0.6


def _classify_tier(mean_iou: Optional[float]) -> str:
    """Map mean IoU to accuracy tier.

    None → 'coarse' (can't verify → assume unverified).
    """
    if mean_iou is None:
        return "coarse"
    if mean_iou >= _TIER_TIGHT:
        return "tight"
    if mean_iou >= _TIER_GOOD:
        return "good"
    return "coarse"


def _rasterize_projected_mask(pixels_uv: np.ndarray,
                              in_front: np.ndarray,
                              image_shape: Tuple[int, int]) -> np.ndarray:
    """Rasterize projected lidar points to a binary mask, dilated by 3 px.

    Args:
        pixels_uv:    (N, 2) float pixel coords (can contain NaN for back-of-camera).
        in_front:     (N,) bool — which rows of `pixels_uv` are valid.
        image_shape:  (H, W) integer image shape.

    Returns:
        (H, W) bool mask. True where the lidar projection (dilated) lies.
    """
    H, W = int(image_shape[0]), int(image_shape[1])
    mask = np.zeros((H, W), dtype=np.uint8)
    if pixels_uv is None or len(pixels_uv) == 0:
        return mask.astype(bool)

    in_front = np.asarray(in_front, dtype=bool)
    if in_front.ndim == 0 or in_front.size == 0:
        return mask.astype(bool)

    # Select in-front + finite pixels only.
    uv = np.asarray(pixels_uv, dtype=float)
    if uv.ndim != 2 or uv.shape[1] != 2:
        return mask.astype(bool)
    valid = in_front & np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1])
    if not np.any(valid):
        return mask.astype(bool)

    # Filter out astronomical finite values that cv2.projectPoints can produce
    # on points at glancing angles (they'd overflow int32 cast silently).
    sane = np.abs(uv[:, 0]) < 1e6
    sane &= np.abs(uv[:, 1]) < 1e6
    valid = valid & sane
    if not np.any(valid):
        return mask.astype(bool)

    u = uv[valid, 0].astype(np.int32)
    v = uv[valid, 1].astype(np.int32)
    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not np.any(in_bounds):
        return mask.astype(bool)
    mask[v[in_bounds], u[in_bounds]] = 1

    # Dilate to account for lidar sparsity. cv2 is available everywhere in
    # this codebase (colorizer, frustum, wall_segmenter); fall back to a
    # naive numpy dilation only if cv2 import failed.
    if _HAVE_CV2:
        k = 2 * _LIDAR_MASK_DILATE_PX + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        mask = cv2.dilate(mask, kernel)
    else:  # pragma: no cover — cv2 is a hard dep in this codebase
        # Square dilation via repeated neighbor-OR. Slow but correct.
        d = _LIDAR_MASK_DILATE_PX
        pad = np.pad(mask, d, mode='constant', constant_values=0)
        acc = np.zeros_like(mask)
        for dy in range(-d, d + 1):
            for dx in range(-d, d + 1):
                acc |= pad[d + dy:d + dy + H, d + dx:d + dx + W]
        mask = acc

    return mask.astype(bool)


def _compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """IoU of two (H, W) boolean masks. Returns 0.0 on empty union."""
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    if a.shape != b.shape:
        return 0.0
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def verify_calibration(
    *,
    images_with_poses: Iterable[Tuple[Optional[np.ndarray], np.ndarray, np.ndarray]],
    lidar_wall_inliers_per_frame: List[np.ndarray],
    project_fn,
    image_shape: Tuple[int, int],
    sample_every: int = 30,
) -> dict:
    """Compare projected lidar wall inliers vs Mask2Former wall mask per frame.

    For every `sample_every`-th frame in the input iterable:
      1. Take the corresponding lidar-wall inliers in world frame.
      2. Project them to the image via `project_fn(pts_world, pose)`.
      3. Rasterize to a binary mask, dilate by 3 px (lidar sparsity).
      4. Compute IoU against Mask2Former's wall mask for that frame.

    Args:
        images_with_poses:
            Iterable of (image_np_or_None, pose_4x4, wall_mask_hw) triplets.
            `image_np_or_None` is unused (accepted for signature compatibility
            and future extensions); pose is the 4x4 lidar→world transform for
            the frame; wall_mask_hw is Mask2Former's binary wall mask.
        lidar_wall_inliers_per_frame:
            Parallel list: for frame i, (N_i, 3) float ndarray of
            world-frame wall points. For the M0c pragmatic simplification,
            the same merged-wall-points array is passed for every sampled
            frame — the signal is still valid (measures whether the final
            wall estimate aligns with per-frame vision).
        project_fn:
            Callable(pts_world, pose_4x4) -> (pixels_uv, in_front_bool).
            `pixels_uv` is (N, 2) float, `in_front_bool` is (N,) bool
            selecting points ahead of the camera.
        image_shape:
            (H, W) integer tuple of the raw camera image dimensions.
        sample_every:
            Process every `sample_every`-th frame. Default 30 → ~16 samples
            on a 500-frame scan, ~100 ms of verification per scan.

    Returns:
        {
          "reprojection_iou_mean": float | None,
          "reprojection_iou_min":  float | None,
          "reprojection_iou_frames_checked": int,
          "accuracy_tier": "coarse" | "good" | "tight",
        }

        When no frames are processed (empty iterable, or Mask2Former
        disabled → wall_mask is None everywhere), all IoU fields are None,
        `frames_checked` is 0, and `accuracy_tier` is "coarse" (no signal
        → assume unverified).
    """
    if sample_every < 1:
        sample_every = 1

    # Materialize the iterable so we can index in parallel with
    # lidar_wall_inliers_per_frame (which is a list by signature).
    triplets = list(images_with_poses)
    n_frames = len(triplets)

    per_frame_iou: List[float] = []

    for frame_idx in range(0, n_frames, sample_every):
        if frame_idx >= len(lidar_wall_inliers_per_frame):
            # Silent skip: inliers list shorter than image list. Happens if
            # floorplan generation failed but we still want a partial
            # signal.
            continue

        _img, pose, wall_mask = triplets[frame_idx]
        if wall_mask is None:
            continue
        pts_world = lidar_wall_inliers_per_frame[frame_idx]
        if pts_world is None or len(pts_world) == 0:
            continue
        if pose is None or np.asarray(pose).shape != (4, 4):
            continue

        # Project: callers adapt their `project_lidar_to_camera` into this
        # (pts_world, pose) -> (pixels, in_front) shape.
        try:
            pixels_uv, in_front = project_fn(pts_world, np.asarray(pose))
        except Exception:
            # A single frame's projection failure shouldn't kill the whole
            # scan's verification — the remaining frames still give a
            # statistically meaningful signal.
            continue

        projected_mask = _rasterize_projected_mask(
            pixels_uv, in_front, image_shape)
        vision_mask = np.asarray(wall_mask, dtype=bool)

        if projected_mask.shape != vision_mask.shape:
            # Shape mismatch (e.g. upstream resized the vision mask).
            # Skip rather than silently compare mis-aligned masks.
            continue

        # If both masks are empty (no wall in view that frame), IoU is
        # undefined — exclude from the mean. This is DIFFERENT from
        # "disagreement": it's "no signal."
        if not projected_mask.any() and not vision_mask.any():
            continue

        per_frame_iou.append(_compute_iou(projected_mask, vision_mask))

    if len(per_frame_iou) == 0:
        return {
            "reprojection_iou_mean": None,
            "reprojection_iou_min": None,
            "reprojection_iou_frames_checked": 0,
            "accuracy_tier": "coarse",
        }

    mean_iou = float(np.mean(per_frame_iou))
    min_iou = float(np.min(per_frame_iou))
    return {
        "reprojection_iou_mean": round(mean_iou, 4),
        "reprojection_iou_min": round(min_iou, 4),
        "reprojection_iou_frames_checked": len(per_frame_iou),
        "accuracy_tier": _classify_tier(mean_iou),
    }


def should_warn(calibration_block: dict) -> bool:
    """True iff the IoU mean is below the warn threshold (and was measured).

    Does NOT warn when frames_checked == 0 — we can't assess calibration
    without vision masks, so silence is the correct behavior.
    """
    mean_iou = calibration_block.get("reprojection_iou_mean")
    if mean_iou is None:
        return False
    return mean_iou < _WARN_IOU_BELOW


def warning_message(calibration_block: dict) -> str:
    """Multi-line user-facing warning to print when `should_warn` is True.

    Explains the observed tier, caps M2 can expect, and the remediation path
    (target-based calibration). Kept as a pure formatter so the caller owns
    stdout/stderr choice.
    """
    mean_iou = calibration_block.get("reprojection_iou_mean")
    min_iou = calibration_block.get("reprojection_iou_min")
    tier = calibration_block.get("accuracy_tier", "coarse")
    frames = calibration_block.get("reprojection_iou_frames_checked", 0)
    lines = [
        "=" * 66,
        "  [WARN] Calibration reprojection-IoU is low — M2 accuracy capped.",
        "=" * 66,
        f"  mean IoU = {mean_iou:.3f}  (min = {min_iou:.3f}, "
        f"frames checked = {frames})",
        f"  accuracy_tier = \"{tier}\"",
        "",
        "  At this IoU, Stage-1 RANSAC walls are misaligned with the camera",
        "  by ~30-50 mm at 3 m. That floors downstream wall-edge accuracy and",
        "  breaks the M2 milestone's <=20 mm target on camera-covered walls.",
        "",
        "  Remediation:",
        "    - Recalibrate lidar-camera extrinsics with a target (Kalibr,",
        "      OpenCV checkerboard + cv2.calibrateHandEye, or similar).",
        "    - Record the method + date in calibration/extrinsics.yaml so",
        "      the JSON schema's `calibration.method` / `date` / `age_days`",
        "      reflect the new calibration.",
        "    - Re-run this pipeline; the new mean IoU should reach >= 0.85",
        "      (\"tight\" tier) before relying on sub-2 cm wall alignment.",
        "=" * 66,
    ]
    return "\n".join(lines)
