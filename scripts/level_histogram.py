#!/usr/bin/env python3
"""
Sanity-check scan leveling by plotting Z (and XY) histograms.

If gravity leveling is correct, Z should show two sharp peaks:
  - floor near the lowest Z
  - ceiling ~2.4-3.0 m above the floor
and the inter-peak band should be mostly empty (just walls/objects).

If leveling is wrong, floor/ceiling points will smear across a wide Z range
because the "horizontal" planes are tilted relative to the Z axis.

Usage:
    python3 scripts/level_histogram.py /tmp/detect_test/colored_map.ply [out.png]
"""

import os
import sys
import argparse
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_peaks(counts, centers, min_sep_m=0.3):
    """Return the two tallest peaks at least min_sep_m apart."""
    order = np.argsort(counts)[::-1]
    peaks = []
    for idx in order:
        z = centers[idx]
        if all(abs(z - p[0]) >= min_sep_m for p in peaks):
            peaks.append((z, counts[idx]))
        if len(peaks) >= 2:
            break
    return peaks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ply", help="Point cloud (PLY)")
    parser.add_argument("out", nargs="?", default=None,
                        help="Output PNG (default: <ply_dir>/level_histogram.png)")
    parser.add_argument("--bin", type=float, default=0.02,
                        help="Z histogram bin size in meters (default 0.02)")
    args = parser.parse_args()

    pcd = o3d.io.read_point_cloud(args.ply)
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        print("[ERROR] Empty point cloud", file=sys.stderr)
        sys.exit(1)

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    # Ignore extreme outliers for the histogram range
    z_lo, z_hi = np.percentile(z, [0.5, 99.5])
    bins = np.arange(z_lo, z_hi + args.bin, args.bin)
    counts, edges = np.histogram(z, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    peaks = find_peaks(counts, centers)
    peak_str = ", ".join(f"{p[0]:.3f}m ({p[1]} pts)" for p in peaks) if peaks else "none"

    print(f"[INFO] {len(pts)} points")
    print(f"[INFO] X range: [{x.min():.2f}, {x.max():.2f}] span={x.max()-x.min():.2f}m")
    print(f"[INFO] Y range: [{y.min():.2f}, {y.max():.2f}] span={y.max()-y.min():.2f}m")
    print(f"[INFO] Z range: [{z.min():.2f}, {z.max():.2f}] span={z.max()-z.min():.2f}m")
    print(f"[INFO] Z peaks: {peak_str}")
    if len(peaks) == 2:
        sep = abs(peaks[0][0] - peaks[1][0])
        print(f"[INFO] Peak separation (expected ~2.4-3.0 m for a room): {sep:.3f} m")

    # Concentration: what fraction of points fall in the tallest 5 cm slice?
    max_bin = counts.max()
    max_z = centers[np.argmax(counts)]
    slab_frac = counts[(centers >= max_z - 0.025) & (centers <= max_z + 0.025)].sum() / counts.sum()
    print(f"[INFO] Tallest bin: z={max_z:.3f} m, {max_bin} pts "
          f"({100 * slab_frac:.1f}% of cloud in 5cm around it)")

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    axes[0].bar(centers, counts, width=args.bin, align='center', color='steelblue', edgecolor='none')
    for zp, cp in peaks:
        axes[0].axvline(zp, color='red', linestyle='--', alpha=0.7)
        axes[0].text(zp, cp, f" {zp:.2f}m", color='red', va='bottom', fontsize=9)
    axes[0].set_title(f"Z histogram (bin={args.bin*100:.0f} cm)  —  {os.path.basename(args.ply)}")
    axes[0].set_xlabel("Z (m)")
    axes[0].set_ylabel("points")
    axes[0].grid(alpha=0.3)

    # X histogram
    bx = np.arange(np.percentile(x, 0.5), np.percentile(x, 99.5) + args.bin, args.bin)
    axes[1].hist(x, bins=bx, color='seagreen', edgecolor='none')
    axes[1].set_title("X histogram")
    axes[1].set_xlabel("X (m)")
    axes[1].set_ylabel("points")
    axes[1].grid(alpha=0.3)

    # Y histogram
    by = np.arange(np.percentile(y, 0.5), np.percentile(y, 99.5) + args.bin, args.bin)
    axes[2].hist(y, bins=by, color='indianred', edgecolor='none')
    axes[2].set_title("Y histogram")
    axes[2].set_xlabel("Y (m)")
    axes[2].set_ylabel("points")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()

    out_path = args.out or os.path.join(os.path.dirname(os.path.abspath(args.ply)),
                                         "level_histogram.png")
    fig.savefig(out_path, dpi=120)
    print(f"[INFO] Saved {out_path}")


if __name__ == "__main__":
    main()
