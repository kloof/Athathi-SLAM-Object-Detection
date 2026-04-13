"""Dump a side-by-side video of Mask2Former per-pixel classifications.

Runs WallSegmenter (the same Mask2Former wall-typing head used by the
main pipeline) on every image from an MCAP scan, overlays the 5-bucket
color map on the original frame, and writes an MP4.

5-bucket colors:
    wall   = blue      (bucket 1)
    door   = yellow    (bucket 2)
    window = cyan      (bucket 3)
    glass  = magenta   (bucket 4)
    other  = transparent (bucket 0 — shows original pixels through)

Usage:
    python3 scripts/dump_mask2former_video.py <rosbag_path> <output_mp4> \
        [--stride 3] [--alpha 0.55]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Allow running without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cloud_slam.mcap_reader import read_mcap
from cloud_slam.wall_segmenter import WallSegmenter


# Bucket → BGR (cv2 uses BGR order).
BUCKET_COLORS = {
    0: None,                 # 'other' — transparent, keep original
    1: (200, 70, 40),        # 'wall' — blue
    2: (40, 200, 240),       # 'door' — yellow
    3: (220, 220, 40),       # 'window' — cyan
    4: (200, 60, 200),       # 'glass' — magenta
}

BUCKET_NAMES = {0: 'other', 1: 'wall', 2: 'door', 3: 'window', 4: 'glass'}


def overlay_mask(image_bgr: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    """Blend the 5-bucket color map on top of image_bgr.

    image_bgr: (H, W, 3) uint8 BGR.
    mask: (H, W) int, values in {0..4}.
    alpha: 0..1 blend weight for the color overlay.
    """
    out = image_bgr.copy()
    for bucket, bgr in BUCKET_COLORS.items():
        if bgr is None:
            continue
        sel = mask == bucket
        if not sel.any():
            continue
        # Blend only the selected pixels.
        out[sel] = np.clip(
            (1.0 - alpha) * out[sel].astype(np.float32)
            + alpha * np.array(bgr, dtype=np.float32),
            0, 255,
        ).astype(np.uint8)
    return out


def draw_legend(image: np.ndarray, counts: dict, total_px: int) -> np.ndarray:
    """Write the legend + per-bucket % in the top-left corner."""
    lines = []
    lines.append(f"Mask2Former (ADE20K -> 5 buckets, alpha={total_px / max(total_px, 1):.0%})")
    for bucket in (0, 1, 2, 3, 4):
        name = BUCKET_NAMES[bucket]
        pct = 100.0 * counts.get(bucket, 0) / max(total_px, 1)
        lines.append(f"  {name}: {pct:5.1f}%")
    h = image.shape[0]
    y = 20
    for line in lines:
        # Shadow for legibility on any background
        cv2.putText(image, line, (11, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y += 22
    return image


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rosbag", help="Path to rosbag directory (contains *.mcap)")
    ap.add_argument("out_mp4", help="Output MP4 path")
    ap.add_argument("--stride", type=int, default=1,
                    help="Process every N-th frame (default: 1 = all frames)")
    ap.add_argument("--fps", type=float, default=10.0,
                    help="Output video FPS (default: 10)")
    ap.add_argument("--alpha", type=float, default=0.55,
                    help="Overlay blend alpha 0..1 (default: 0.55)")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="Stop after N frames (default: 0 = no limit)")
    args = ap.parse_args()

    print(f"[dump] reading MCAP: {args.rosbag}")
    _clouds, _imus, images = read_mcap(args.rosbag)
    print(f"[dump] {len(images)} images")

    if len(images) == 0:
        print("[dump] no images — exiting")
        return 1

    def _decode(entry):
        """Handle (stamp, bytes, format) tuples from read_mcap."""
        if len(entry) == 3:
            _stamp, buf, _fmt = entry
        else:
            _stamp, buf = entry[:2]
        if isinstance(buf, np.ndarray):
            return buf
        arr = np.frombuffer(buf, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img

    seg = WallSegmenter()
    try:
        seg._ensure_loaded()
    except Exception as e:
        print(f"[dump] WallSegmenter failed to load: {e}")
        return 1

    first = _decode(images[0])
    if first is None:
        print("[dump] failed to decode first image")
        return 1
    H, W = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out_mp4, fourcc, args.fps, (W, H))
    if not writer.isOpened():
        print(f"[dump] failed to open writer for {args.out_mp4}")
        return 1
    print(f"[dump] writing {args.out_mp4} @ {args.fps:.1f}fps, size {W}x{H}")

    count = 0
    for i, entry in enumerate(images):
        frame_bgr = _decode(entry)
        if frame_bgr is None:
            continue
        if args.stride > 1 and i % args.stride != 0:
            continue
        if args.max_frames > 0 and count >= args.max_frames:
            break
        bucket_mask = seg.segment(frame_bgr)
        if bucket_mask is None:
            # Segmenter failed; show original with a note
            out = frame_bgr.copy()
            cv2.putText(out, "Mask2Former: FAILED", (10, H - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            out = overlay_mask(frame_bgr, bucket_mask, args.alpha)
            flat = bucket_mask.ravel()
            counts = {b: int((flat == b).sum()) for b in range(5)}
            out = draw_legend(out, counts, flat.size)

        writer.write(out)
        count += 1
        if count % 50 == 0:
            print(f"[dump] processed {count} frames")

    writer.release()
    print(f"[dump] done — {count} frames written to {args.out_mp4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
