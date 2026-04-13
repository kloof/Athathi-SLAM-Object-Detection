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


def solid_mask(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Render the bucket mask as solid colors — no transparency.

    `other` pixels are shown in dark gray so they're visually distinct
    from unclassified black borders. Classified regions get their full
    bucket color, which makes adjacent classes crisp and unambiguous.
    """
    H, W = mask.shape
    out = np.full((H, W, 3), 40, dtype=np.uint8)  # dark gray for 'other'
    for bucket, bgr in BUCKET_COLORS.items():
        if bgr is None:
            continue
        out[mask == bucket] = bgr
    return out


def contour_mode(image_bgr: np.ndarray, mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    """Draw crisp bucket boundaries as colored lines over the original.

    Keeps the underlying image fully visible (no alpha fill) but outlines
    every bucket region with a thick colored border. Best for comparing
    classification edges against physical object edges in the scene.
    """
    out = image_bgr.copy()
    for bucket, bgr in BUCKET_COLORS.items():
        if bgr is None:
            continue
        binary = (mask == bucket).astype(np.uint8)
        if binary.sum() == 0:
            continue
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, bgr, thickness, cv2.LINE_AA)
    return out


def compose_sidebyside(left: np.ndarray, right: np.ndarray,
                       left_label: str = "original",
                       right_label: str = "mask") -> np.ndarray:
    """Stack two equal-size frames horizontally with small labels."""
    H, W = left.shape[:2]
    out = np.zeros((H, W * 2, 3), dtype=np.uint8)
    out[:, :W] = left
    out[:, W:] = right
    # Divider line
    out[:, W - 1:W + 1] = 80
    # Labels
    for x, text in [(10, left_label), (W + 10, right_label)]:
        cv2.putText(out, text, (x + 1, H - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, (x, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (240, 240, 240), 1, cv2.LINE_AA)
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
                    help="Overlay blend alpha 0..1, used by --mode blend (default: 0.55)")
    ap.add_argument("--mode", choices=["sidebyside", "blend", "mask", "contour"],
                    default="sidebyside",
                    help="Visualization mode (default: sidebyside):\n"
                         "  sidebyside = original + solid-color mask side-by-side "
                         "(crispest, best when classes overlap)\n"
                         "  blend      = alpha overlay on the original image\n"
                         "  mask       = pure solid-color mask, original hidden\n"
                         "  contour    = original + crisp class boundary outlines")
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
    # Side-by-side doubles width.
    out_W = W * 2 if args.mode == "sidebyside" else W
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out_mp4, fourcc, args.fps, (out_W, H))
    if not writer.isOpened():
        print(f"[dump] failed to open writer for {args.out_mp4}")
        return 1
    print(f"[dump] mode={args.mode}, writing {args.out_mp4} @ "
          f"{args.fps:.1f}fps, size {out_W}x{H}")

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
            if args.mode == "sidebyside":
                out = compose_sidebyside(out, out, "original", "MASK FAILED")
        else:
            flat = bucket_mask.ravel()
            counts = {b: int((flat == b).sum()) for b in range(5)}
            if args.mode == "blend":
                out = overlay_mask(frame_bgr, bucket_mask, args.alpha)
                out = draw_legend(out, counts, flat.size)
            elif args.mode == "mask":
                out = solid_mask(frame_bgr, bucket_mask)
                out = draw_legend(out, counts, flat.size)
            elif args.mode == "contour":
                out = contour_mode(frame_bgr, bucket_mask, thickness=2)
                out = draw_legend(out, counts, flat.size)
            else:  # sidebyside
                mask_panel = solid_mask(frame_bgr, bucket_mask)
                mask_panel = draw_legend(mask_panel, counts, flat.size)
                out = compose_sidebyside(frame_bgr, mask_panel,
                                         "original", "Mask2Former 5-bucket")

        writer.write(out)
        count += 1
        if count % 50 == 0:
            print(f"[dump] processed {count} frames")

    writer.release()
    print(f"[dump] done — {count} frames written to {args.out_mp4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
