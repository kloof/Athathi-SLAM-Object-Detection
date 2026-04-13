"""
Indoor wall / window / door / glass semantic segmenter.

Wraps `facebook/mask2former-swin-large-ade-semantic` (ADE20K-150, ~216M
params). Lazy loads on first `segment()` call so importing this module
carries no cost unless it's actually used. FP16 inference on CUDA.

If the model fails to load (transformers missing, HF Hub unreachable,
CUDA OOM, etc.) the wrapper flips to a disabled state and subsequent
`segment()` calls return None — the pipeline degrades to YOLOE + D_refined
geometry only, never crashing.

Output: (H, W) uint8 mask of our 5-bucket id, one of:
    0 = other       — anything not in the four wall-like classes
    1 = wall        — ADE20K 0 (wall), 1 (building)
    2 = window      — ADE20K 8 (windowpane)
    3 = door        — ADE20K 14 (door), 58 (screen door)
    4 = glass       — ADE20K 27 (mirror), 147 (glass)

Design notes:
- ADE20K `building` (1) is merged into `wall` — some annotators use it for
  interior walls, mapping is safer than dropping.
- ADE20K `mirror` (27) is grouped with `glass` (147) because on a wall
  plane both are geometrically wall-like but optically transparent /
  reflective (lidar often sees through / reflects off them).
- Per-pixel softmax is NOT exposed in v1; argmax class is used and the
  downstream accumulator weights votes by geometry (ray angle, distance),
  not per-pixel confidence.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


BUCKET_NAMES = ('other', 'wall', 'window', 'door', 'glass')
N_BUCKETS = len(BUCKET_NAMES)

# ADE20K-150 class id → our 5-bucket id. Classes not in this table map to 0.
_ADE_TO_BUCKET = {
    0: 1,    # wall          → wall
    1: 1,    # building      → wall (interior-wall aliasing in some scenes)
    8: 2,    # windowpane    → window
    14: 3,   # door          → door
    58: 3,   # screen door   → door
    27: 4,   # mirror        → glass
    147: 4,  # glass         → glass
}


class WallSegmenter:
    """Mask2Former ADE20K wrapper with 5-bucket remap.

    Usage:
        seg = WallSegmenter()          # no model load yet
        mask = seg.segment(rgb_image)  # (H, W) uint8, or None if disabled
    """

    MODEL_ID = "facebook/mask2former-swin-large-ade-semantic"

    def __init__(self, device: Optional[str] = None, fp16: bool = True):
        self._requested_device = device
        self._fp16 = fp16
        self._device: Optional[str] = None
        self._processor = None
        self._model = None
        self._loaded = False
        self._disabled = False
        # M0a: per-call ADE20K class histogram accumulator. Lets the
        # floorplan pipeline vote on `room.category` (bed → bedroom,
        # sofa → livingroom, etc.) without re-running inference. Keys
        # are the raw ADE-150 ids (NOT the 5-bucket remap), values are
        # cumulative pixel counts across every successful segment() call.
        self._ade_class_counts: dict[int, int] = {}

    # ------------------------------------------------------------------
    # Lazy model load
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> bool:
        if self._loaded:
            return True
        if self._disabled:
            return False
        try:
            import torch
            from transformers import (
                AutoImageProcessor,
                Mask2FormerForUniversalSegmentation,
            )

            device = self._requested_device or (
                'cuda' if torch.cuda.is_available() else 'cpu')

            processor = AutoImageProcessor.from_pretrained(self.MODEL_ID)
            model = Mask2FormerForUniversalSegmentation.from_pretrained(
                self.MODEL_ID)
            model.eval()
            model.to(device)
            if self._fp16 and device == 'cuda':
                model = model.half()

            self._processor = processor
            self._model = model
            self._device = device
            self._loaded = True
            suffix = ' (fp16)' if self._fp16 and device == 'cuda' else ''
            print(f"[WallSegmenter] loaded {self.MODEL_ID} on {device}{suffix}")
            return True
        except Exception as e:
            # Any failure (ImportError, ConnectionError, CUDA OOM, ...)
            # disables the segmenter for the rest of the run.
            self._disabled = True
            print(f"[WallSegmenter] disabled — {type(e).__name__}: {e}")
            return False

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def segment(self, image_rgb: np.ndarray) -> Optional[np.ndarray]:
        """Segment an RGB image → (H, W) uint8 of 5-bucket ids.

        Returns None if the segmenter is disabled or input is invalid.
        """
        if image_rgb is None or image_rgb.size == 0:
            return None
        if not self._ensure_loaded():
            return None

        import torch

        H, W = image_rgb.shape[:2]
        assert image_rgb.ndim == 3 and image_rgb.shape[2] == 3, (
            f"expected (H,W,3) RGB, got shape {image_rgb.shape}")

        try:
            with torch.no_grad():
                inputs = self._processor(
                    images=image_rgb, return_tensors="pt")
                inputs = {k: v.to(self._device)
                          for k, v in inputs.items()}
                if self._fp16 and self._device == 'cuda':
                    inputs = {
                        k: (v.half() if v.dtype == torch.float32 else v)
                        for k, v in inputs.items()
                    }
                outputs = self._model(**inputs)
                sem_seg = (self._processor
                           .post_process_semantic_segmentation(
                               outputs, target_sizes=[(H, W)])[0])
            ade_mask = sem_seg.cpu().numpy().astype(np.int32)
        except Exception as e:
            # A transient inference failure shouldn't kill the whole pipeline.
            print(f"[WallSegmenter] segment() failed — "
                  f"{type(e).__name__}: {e}")
            return None

        # M0a: accumulate per-frame ADE20K class pixel counts for the
        # room.category vote. np.bincount over the flattened mask gives a
        # dense histogram for any id 0..max; we fold that into the
        # cumulative counter. Cheap (~microseconds per frame).
        try:
            flat = ade_mask.ravel()
            if flat.size > 0:
                counts = np.bincount(flat[flat >= 0])
                for cid in np.nonzero(counts)[0]:
                    self._ade_class_counts[int(cid)] = (
                        self._ade_class_counts.get(int(cid), 0)
                        + int(counts[cid]))
        except Exception:
            # Histogram update is best-effort; never fail segment() on it.
            pass

        return self._remap_ade(ade_mask)

    def get_ade_class_counts(self) -> dict:
        """Return cumulative ADE20K class histogram across every segment() call.

        Keys are raw ADE-150 class ids (e.g. 7=bed, 23=sofa, 71=stove);
        values are cumulative pixel counts. Empty dict if segment() has
        never run or was always disabled.
        """
        return dict(self._ade_class_counts)

    # ------------------------------------------------------------------
    # ADE20K → 5-bucket remap
    # ------------------------------------------------------------------

    @staticmethod
    def _remap_ade(ade_mask: np.ndarray) -> np.ndarray:
        """Vectorised remap of (H,W) int ADE ids → (H,W) uint8 bucket ids."""
        bucket = np.zeros_like(ade_mask, dtype=np.uint8)
        for ade_id, b in _ADE_TO_BUCKET.items():
            bucket[ade_mask == ade_id] = b
        return bucket

    # ------------------------------------------------------------------
    # Static helper: convert per-point bucket ids → RANSAC class weights
    # ------------------------------------------------------------------

    @staticmethod
    def bucket_to_weight(bucket_ids: np.ndarray) -> np.ndarray:
        """Map bucket id → Tier-2 RANSAC weight in {0.3, 0.7, 1.0}.

        wall → 1.0; window/door/glass → 0.7; other/unknown → 0.3.
        """
        w = np.full(bucket_ids.shape, 0.3, dtype=np.float32)
        w[bucket_ids == 1] = 1.0
        w[(bucket_ids == 2) | (bucket_ids == 3) | (bucket_ids == 4)] = 0.7
        return w
