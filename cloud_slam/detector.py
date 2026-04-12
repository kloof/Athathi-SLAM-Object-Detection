"""
YOLOE-26 object detection wrapper with BoT-SORT tracking.

Lazy-imports ultralytics to avoid 2-3s startup penalty when detection is unused.
Auto-selects model size based on GPU availability.
"""

import numpy as np
from dataclasses import dataclass

DEFAULT_CLASSES = [
    "chair", "table", "desk", "sofa", "bed",
    "shelf", "monitor", "lamp", "plant",
    "cabinet", "wardrobe",
]
# Intentionally NOT in the default class list:
#   "person" — transient (people move during scans), 3D boxes are junk
#   "door"   — doors now come from vision wall features (Mask2Former
#              via `--label-walls`), attached to their host wall in
#              `floorplan_metadata.json` instead of as standalone 3D
#              objects in `objects.json`. No more redundancy.
# Override with `--classes "person,door,..."` on the CLI if you
# specifically need them re-enabled for a given scan.


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

        return _cross_class_nms(detections)


def _cross_class_nms(detections, iou_threshold=0.5):
    """Suppress duplicate detections on the same object across classes."""
    if len(detections) <= 1:
        return detections

    # Sort by confidence descending
    detections.sort(key=lambda d: -d.confidence)
    keep = [True] * len(detections)

    for i in range(len(detections)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(detections)):
            if not keep[j]:
                continue
            if _bbox_iou(detections[i].bbox_xyxy, detections[j].bbox_xyxy) > iou_threshold:
                keep[j] = False

    return [d for d, k in zip(detections, keep) if k]


def _bbox_iou(a, b):
    """Compute IoU between two [x1,y1,x2,y2] bboxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-6)
