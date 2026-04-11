"""
YOLOE-26 object detection wrapper with BoT-SORT tracking.

Lazy-imports ultralytics to avoid 2-3s startup penalty when detection is unused.
Auto-selects model size based on GPU availability.
"""

import numpy as np
from dataclasses import dataclass

DEFAULT_CLASSES = [
    "person", "chair", "table", "desk", "sofa", "bed",
    "shelf", "door", "monitor", "lamp", "plant",
]


@dataclass
class Detection:
    track_id: int
    class_name: str
    confidence: float
    bbox_xyxy: np.ndarray   # [x1, y1, x2, y2]
    mask: np.ndarray | None  # (H, W) binary mask or None


class YOLODetector:
    def __init__(self, model_name=None, classes=None, conf=0.3, device=None):
        self._model_name = model_name
        self._classes = classes or DEFAULT_CLASSES
        self._conf = conf
        self._device = device
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return

        import torch
        from ultralytics import YOLOE

        # Auto-select model size
        if self._model_name is None:
            if torch.cuda.is_available():
                self._model_name = "yoloe-26l-seg.pt"
            else:
                self._model_name = "yoloe-26s-seg.pt"

        self._model = YOLOE(self._model_name)
        self._model.set_classes(self._classes)

        if self._device:
            self._model.to(self._device)

        print(f"[YOLOE] Loaded {self._model_name}, classes={self._classes}")

    def detect_and_track(self, image):
        """
        Run YOLOE detection + BoT-SORT tracking on a BGR image.

        Returns list of Detection objects with persistent track IDs.
        """
        self._ensure_model()

        results = self._model.track(
            image, tracker="botsort.yaml", persist=True,
            conf=self._conf, verbose=False,
        )
        r = results[0]

        detections = []
        if r.boxes is None or r.boxes.id is None:
            return detections

        track_ids = r.boxes.id.int().cpu().tolist()
        bboxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.int().cpu().tolist()
        confs = r.boxes.conf.cpu().numpy()

        # Class name lookup
        names = r.names  # {idx: name}

        # Segmentation masks
        masks = None
        if r.masks is not None:
            masks = r.masks.data.cpu().numpy()  # (N, H, W)

        for i, (tid, bbox, cls_idx, conf) in enumerate(
            zip(track_ids, bboxes, classes, confs)
        ):
            mask = masks[i] if masks is not None else None
            detections.append(Detection(
                track_id=tid,
                class_name=names.get(cls_idx, f"class_{cls_idx}"),
                confidence=float(conf),
                bbox_xyxy=bbox,
                mask=mask,
            ))

        return detections
