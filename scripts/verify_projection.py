"""Standalone calibration-verification tool.

Takes one frame from an MCAP scan, projects its nearest lidar sweep
into the camera image using the scan's extrinsic + intrinsic calibration,
and writes a PNG overlay so you can visually confirm the 3D -> 2D
projection is correct.

The lidar points are colored by height (Z in the gravity-aligned frame)
so you can sanity-check: floor points should land on actual floor pixels
in the image, ceiling points on actual ceiling pixels, wall points on
actual wall surfaces. Misaligned calibration shows as an offset between
the colored dots and the features they're supposed to hit.

This script is intentionally standalone: it does not integrate into the
main pipeline, does not modify calibration files, and does not produce
any output beyond the PNG you name. It only READS the MCAP and
calibration files.

Usage:
  python3 scripts/verify_projection.py \
      <rosbag_dir> <calibration_dir> <output.png> \
      [--frame-idx N] [--max-points 3000] [--dot-radius 2]

Example:
  python3 scripts/verify_projection.py \
      /mnt/c/.../scan_20260411_121711/rosbag \
      /mnt/c/.../scan_20260411_121711/calibration \
      /tmp/projection_check.png \
      --frame-idx 150
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Allow running without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cloud_slam.colorizer import load_calibration, match_nearest_image
from cloud_slam.mcap_reader import read_mcap
from cloud_slam.projection import project_lidar_to_camera


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


def _color_from_height(z, z_min, z_max):
    """Map Z (in lidar frame, roughly world-up) to a BGR color.

    Green = floor-ish (low Z), red = ceiling-ish (high Z),
    blue in between. Standard jet-like ramp.
    """
    if z_max <= z_min:
        t = 0.5
    else:
        t = (z - z_min) / (z_max - z_min)
    t = float(np.clip(t, 0.0, 1.0))
    # Jet-ish: green -> blue -> red
    if t < 0.5:
        g = int(255 * (1 - 2 * t))
        b = int(255 * (2 * t))
        r = 0
    else:
        g = 0
        b = int(255 * (2 - 2 * t))
        r = int(255 * (2 * t - 1))
    return (b, g, r)  # BGR


def _draw_legend(img, z_min, z_max, n_plotted, n_visible, n_total, calib_summary):
    """Draw a small caption with height ramp + projection stats."""
    lines = [
        "verify_projection.py (standalone calibration check)",
        f"lidar -> camera projection, {n_plotted}/{n_visible} visible ({n_total} total lidar pts)",
        f"dot color = Z in lidar frame, ramp {z_min:.2f}m (green) -> {z_max:.2f}m (red)",
        f"calib: {calib_summary}",
        "",
        "INTERPRETATION:",
        "  - Floor points (green) should land on actual floor pixels",
        "  - Ceiling points (red) should land on actual ceiling pixels",
        "  - Wall points (blue) should land on wall surfaces",
        "  - Misalignment = calibration error; edge-smear ~= residual mm/m",
    ]
    y = 22
    for line in lines:
        cv2.putText(img, line, (11, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (240, 240, 240), 1, cv2.LINE_AA)
        y += 18


def _calib_summary(calib) -> str:
    """Short human-readable summary of the loaded calibration dict."""
    method = calib.get("extrinsics_method", calib.get("method", "?"))
    date = calib.get("extrinsics_date", calib.get("calibration_date", "?"))
    return f"method={method}, date={date}"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("rosbag", help="Path to the scan's rosbag directory")
    ap.add_argument("calibration", help="Path to the scan's calibration directory")
    ap.add_argument("out_png", help="Output PNG path")
    ap.add_argument(
        "--frame-idx", type=int, default=-1,
        help="Image frame index to use (-1 = middle frame; default -1)",
    )
    ap.add_argument(
        "--max-points", type=int, default=3000,
        help="Cap dots drawn per frame (default 3000; uniformly subsampled)",
    )
    ap.add_argument(
        "--dot-radius", type=int, default=2,
        help="Dot radius in pixels (default 2)",
    )
    args = ap.parse_args()

    print(f"[verify] reading MCAP: {args.rosbag}")
    clouds, _imus, images = read_mcap(args.rosbag)
    if not images:
        print("[verify] no camera frames in MCAP — cannot verify projection.")
        return 1
    if not clouds:
        print("[verify] no lidar sweeps in MCAP — cannot verify projection.")
        return 1
    print(f"[verify] {len(clouds)} lidar sweeps, {len(images)} camera frames")

    # Pick the frame
    if args.frame_idx < 0 or args.frame_idx >= len(images):
        fi = len(images) // 2
    else:
        fi = args.frame_idx
    img_stamp, img_buf, *_ = images[fi]
    image = _decode_image((img_stamp, img_buf))
    if image is None:
        print(f"[verify] frame {fi} failed to decode")
        return 1
    H, W = image.shape[:2]
    print(f"[verify] using frame {fi} @ t={img_stamp:.3f}, size {W}x{H}")

    # Find nearest lidar sweep by timestamp (inline — the pipeline's helper
    # is oriented the other way: cloud -> image, not image -> cloud).
    cloud_stamps = np.asarray([float(c[0]) for c in clouds], dtype=float)
    dts = np.abs(cloud_stamps - float(img_stamp))
    idx_match = int(np.argmin(dts))
    dt = float(dts[idx_match])
    if dt > 0.5:
        print(
            f"[verify] nearest lidar sweep is {dt*1000:.0f}ms from frame {fi}; "
            "that's too far to use for projection verification"
        )
        return 1
    _cloud_stamp, xyz_lidar, _timestamps = clouds[idx_match]
    print(
        f"[verify] paired with lidar sweep {idx_match}, "
        f"dt={dt*1000:.1f}ms, {len(xyz_lidar)} points"
    )

    # Load calibration (intrinsics + extrinsics) via the main pipeline helper.
    # The helper takes two separate paths — resolve them from the calib dir.
    calib_dir = Path(args.calibration)
    intr_path = calib_dir / "intrinsics.yaml"
    ext_path = calib_dir / "extrinsics.yaml"
    if not intr_path.exists() or not ext_path.exists():
        print(f"[verify] missing intrinsics.yaml or extrinsics.yaml in {calib_dir}")
        return 1
    calib = load_calibration(str(intr_path), str(ext_path))
    # Also read the extrinsics yaml directly for the method/date summary
    import yaml as _yaml
    with open(ext_path) as _f:
        _ext_raw = _yaml.safe_load(_f)
    calib_summary = (
        f"method={_ext_raw.get('method', '?')}, "
        f"date={_ext_raw.get('calibration_date', '?')}"
    )
    print(f"[verify] {calib_summary}")

    # Project all lidar points into the camera image
    xyz_lidar = np.asarray(xyz_lidar, dtype=float)
    # Strip any extra columns (intensity, etc.)
    if xyz_lidar.ndim == 2 and xyz_lidar.shape[1] > 3:
        xyz_lidar = xyz_lidar[:, :3]
    pts_cam, pixels, in_front = project_lidar_to_camera(xyz_lidar, calib)
    # pts_cam (N,3), pixels (N,2), in_front (N,)
    in_img = (
        in_front
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < W)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < H)
        & np.isfinite(pixels[:, 0])
        & np.isfinite(pixels[:, 1])
    )
    n_visible = int(in_img.sum())
    n_total = int(len(xyz_lidar))
    if n_visible == 0:
        print("[verify] no lidar points land inside the camera frame.")
        print("        Possible causes: calibration rotation flipped, camera/lidar")
        print("        pointing orthogonally, or this frame was captured looking")
        print("        at the ceiling/floor.")
        return 1
    print(f"[verify] {n_visible}/{n_total} lidar points fall inside the image")

    # Subsample for plotting
    vis_idx = np.where(in_img)[0]
    if n_visible > args.max_points:
        rng = np.random.default_rng(0)
        vis_idx = rng.choice(vis_idx, size=args.max_points, replace=False)
    # Depth-sort so deeper points draw first (closer points overlay).
    # pts_cam[:, 2] is the Z distance along the camera axis — cleanest depth.
    cam_z = pts_cam[vis_idx, 2]
    order = np.argsort(-cam_z)  # far first
    vis_idx = vis_idx[order]

    # Color ramp bounds from the Z of visible points (in lidar frame)
    z_vals = xyz_lidar[vis_idx, 2]
    z_min = float(np.percentile(z_vals, 2))
    z_max = float(np.percentile(z_vals, 98))

    out = image.copy()
    r = max(1, int(args.dot_radius))
    for i in vis_idx:
        u = int(round(pixels[i, 0]))
        v = int(round(pixels[i, 1]))
        color = _color_from_height(xyz_lidar[i, 2], z_min, z_max)
        cv2.circle(out, (u, v), r, color, thickness=-1, lineType=cv2.LINE_AA)

    _draw_legend(
        out,
        z_min,
        z_max,
        n_plotted=len(vis_idx),
        n_visible=n_visible,
        n_total=n_total,
        calib_summary=calib_summary,
    )

    Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out_png, out)
    print(f"[verify] wrote {args.out_png}")
    print()
    print("Inspection tip:")
    print("  Open the PNG and check: do floor-colored (green) dots sit on the")
    print("  floor, ceiling-colored (red) on the ceiling, wall-colored (blue)")
    print("  on walls? Any systematic offset between dot color and the visible")
    print("  surface is the calibration residual. Edge-smear within 1-2 dot")
    print("  radii ~ a few pixels ~ roughly <3cm at 3m range = good calibration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
