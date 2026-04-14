"""Dump a side-by-side video of YOLOE detections + BoT-SORT tracks.

Runs the same YOLOE-26 detector the main pipeline uses on every image
of an MCAP scan, draws bounding boxes + class labels + persistent track
IDs on each frame, and writes an MP4. Mirrors the
``dump_mask2former_video.py`` layout so the two videos can be watched
in parallel for the same scan.

Side-by-side layout (default):
  left panel:  original camera frame
  right panel: same frame with box overlays + track IDs + class colors
               + per-frame detection count legend

Usage:
    python3 scripts/dump_yolo_video.py <rosbag_path> <output_mp4> \
        [--stride 3] [--conf 0.3] [--mode sidebyside|annotated|mask]
        [--classes "chair,table,sofa,bed,door,person"]

Standalone: does not touch the main pipeline or its outputs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Allow running without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cloud_slam.detector import DEFAULT_CLASSES, YOLODetector
from cloud_slam.mcap_reader import read_mcap


# Stable class -> BGR color. Hash-based so new classes get consistent colors.
def _class_color(name: str) -> tuple:
    h = hash(name) & 0xFFFFFF
    b = (h & 0xFF)
    g = (h >> 8) & 0xFF
    r = (h >> 16) & 0xFF
    # Brighten so dark classes aren't invisible on dark backgrounds
    b = int(80 + b * 0.68)
    g = int(80 + g * 0.68)
    r = int(80 + r * 0.68)
    return (b, g, r)


def _decode_image(entry):
    """Handle (stamp, compressed_bytes, format_str) tuples from read_mcap."""
    if len(entry) == 3:
        _stamp, buf, _fmt = entry
    else:
        _stamp, buf = entry[:2]
    if isinstance(buf, np.ndarray):
        return buf
    arr = np.frombuffer(buf, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _draw_detection(img, det, draw_mask=False):
    """Draw one box + label on the image (in place)."""
    x1, y1, x2, y2 = det.bbox_xyxy.astype(int)
    color = _class_color(det.class_name)

    # Box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    # Label: class + track + confidence
    label = f"{det.class_name} #{det.track_id} {det.confidence:.2f}"
    (tw, th), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )
    # Label background pill
    ly = max(y1 - 6, th + 4)
    cv2.rectangle(
        img, (x1, ly - th - 4), (x1 + tw + 6, ly + 2),
        color, thickness=-1,
    )
    cv2.putText(
        img, label, (x1 + 3, ly - 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
    )

    # Optional mask overlay (dimmed — non-intrusive)
    if draw_mask and det.mask is not None:
        mask_u8 = det.mask.astype(np.uint8)
        if mask_u8.shape[:2] != img.shape[:2]:
            mask_u8 = cv2.resize(
                mask_u8, (img.shape[1], img.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        sel = mask_u8 > 0
        if sel.any():
            overlay = img[sel].astype(np.float32)
            tint = np.array(color, dtype=np.float32)
            img[sel] = np.clip(
                0.7 * overlay + 0.3 * tint, 0, 255
            ).astype(np.uint8)


def _draw_legend(img, detections):
    """Write per-frame detection count + class breakdown in top-left."""
    by_class = {}
    for d in detections:
        by_class.setdefault(d.class_name, []).append(d.track_id)
    lines = [f"YOLOE: {len(detections)} detections"]
    for cls, ids in sorted(by_class.items()):
        lines.append(f"  {cls}: {len(ids)}  (tracks: {sorted(set(ids))})")
    y = 22
    for line in lines:
        cv2.putText(img, line, (11, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (240, 240, 240), 1, cv2.LINE_AA)
        y += 18


def _compose_sidebyside(left, right,
                        left_label="original", right_label="YOLOE+BoT-SORT"):
    """Stack two equal-size frames horizontally with small labels."""
    H, W = left.shape[:2]
    out = np.zeros((H, W * 2, 3), dtype=np.uint8)
    out[:, :W] = left
    out[:, W:] = right
    out[:, W - 1:W + 1] = 80
    for x, text in [(10, left_label), (W + 10, right_label)]:
        cv2.putText(out, text, (x + 1, H - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, (x, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (240, 240, 240), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rosbag", help="Path to rosbag directory (contains *.mcap)")
    ap.add_argument("out_mp4", help="Output MP4 path")
    ap.add_argument("--stride", type=int, default=3,
                    help="Process every N-th frame (default: 3)")
    ap.add_argument("--fps", type=float, default=10.0,
                    help="Output video FPS (default: 10)")
    ap.add_argument("--conf", type=float, default=0.3,
                    help="Detection confidence threshold (default: 0.3)")
    ap.add_argument("--classes", type=str, default=None,
                    help=f"Comma-separated class list (default: pipeline default "
                         f"= {','.join(DEFAULT_CLASSES)})")
    ap.add_argument("--mode", choices=["sidebyside", "annotated", "mask"],
                    default="sidebyside",
                    help="Visualization mode (default: sidebyside):\n"
                         "  sidebyside = original + boxes side-by-side\n"
                         "  annotated  = single-panel image with boxes\n"
                         "  mask       = single-panel with boxes + segmentation tint")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="Stop after N frames (default: 0 = no limit)")
    args = ap.parse_args()

    classes = (args.classes.split(",") if args.classes else DEFAULT_CLASSES)

    print(f"[dump] reading MCAP: {args.rosbag}")
    _clouds, _imus, images = read_mcap(args.rosbag)
    print(f"[dump] {len(images)} images")

    if len(images) == 0:
        print("[dump] no images — exiting")
        return 1

    first = _decode_image(images[0])
    if first is None:
        print("[dump] first frame failed to decode")
        return 1
    H, W = first.shape[:2]
    out_W = W * 2 if args.mode == "sidebyside" else W

    print(f"[dump] loading YOLOE detector (classes={classes}, conf={args.conf})")
    detector = YOLODetector(classes=classes, conf=args.conf)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out_mp4, fourcc, args.fps, (out_W, H))
    if not writer.isOpened():
        print(f"[dump] failed to open writer for {args.out_mp4}")
        return 1
    print(f"[dump] mode={args.mode}, writing {args.out_mp4} @ "
          f"{args.fps:.1f}fps, size {out_W}x{H}")

    count = 0
    draw_mask_flag = (args.mode == "mask")
    for i, entry in enumerate(images):
        if args.stride > 1 and i % args.stride != 0:
            continue
        if args.max_frames > 0 and count >= args.max_frames:
            break
        frame_bgr = _decode_image(entry)
        if frame_bgr is None:
            continue

        try:
            dets = detector.detect_and_track(frame_bgr)
        except Exception as e:
            print(f"[dump] frame {i}: detect_and_track raised {e}")
            dets = []

        if args.mode == "sidebyside":
            right = frame_bgr.copy()
            for d in dets:
                _draw_detection(right, d, draw_mask=False)
            _draw_legend(right, dets)
            out = _compose_sidebyside(frame_bgr, right)
        else:
            out = frame_bgr.copy()
            for d in dets:
                _draw_detection(out, d, draw_mask=draw_mask_flag)
            _draw_legend(out, dets)

        writer.write(out)
        count += 1
        if count % 50 == 0:
            print(f"[dump] processed {count} frames")

    writer.release()
    print(f"[dump] done — {count} frames written to {args.out_mp4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
