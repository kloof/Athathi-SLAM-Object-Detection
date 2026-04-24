#!/usr/bin/env python3
"""Render a 2D floorplan PNG from a SpatialLM layout .txt file.

Walls draw as thick black segments, doors green, windows blue,
object bbox footprints as rotated rectangles colored by class.

Usage:
    python3 scripts/draw_floorplan.py IN.txt
    python3 scripts/draw_floorplan.py IN.txt -o OUT.png --dpi 200
    python3 scripts/draw_floorplan.py IN.txt --no-labels
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cloud_slam.spatiallm_pipeline.merge import parse_layout


def _wall_segments(walls):
    return [((w[0], w[1]), (w[3], w[4])) for w in walls]


def _nearest_wall_tangent(px, py, segs):
    """Unit tangent (tx, ty) of the wall segment nearest (px, py).

    Merged layouts reset every door/window's wall_id to wall_0, so we
    recover the true carrier by projecting onto each wall segment.
    """
    best_d2 = float("inf")
    best_t = (1.0, 0.0)
    for (ax, ay), (bx, by) in segs:
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        if L2 < 1e-12:
            continue
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
        qx, qy = ax + t * dx, ay + t * dy
        d2 = (px - qx) ** 2 + (py - qy) ** 2
        if d2 < best_d2:
            best_d2 = d2
            L = math.sqrt(L2)
            best_t = (dx / L, dy / L)
    return best_t


def _rotated_rect_xy(cx, cy, sx, sy, yaw):
    hx, hy = sx / 2.0, sy / 2.0
    local = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy), (-hx, -hy)]
    c, s = math.cos(yaw), math.sin(yaw)
    xs = [cx + c * lx - s * ly for lx, ly in local]
    ys = [cy + s * lx + c * ly for lx, ly in local]
    return xs, ys


_CLASS_COLORS = {
    "sofa":         "#e57373",
    "armchair":     "#ff8a65",
    "chair":        "#ffb74d",
    "dining_chair": "#ffb74d",
    "stool":        "#ffb74d",
    "bed":          "#ba68c8",
    "table":        "#aed581",
    "dining_table": "#aed581",
    "coffee_table": "#aed581",
    "side_table":   "#aed581",
    "desk":         "#aed581",
    "cabinet":      "#90a4ae",
    "shelf":        "#90a4ae",
    "wardrobe":     "#90a4ae",
    "bookshelf":    "#90a4ae",
    "tv":           "#455a64",
    "refrigerator": "#78909c",
    "oven":         "#78909c",
    "sink":         "#78909c",
    "toilet":       "#78909c",
    "bathtub":      "#78909c",
    "carpet":       "#f5f5dc",
    "rug":          "#f5f5dc",
    "curtain":      "#b0bec5",
    "plants":       "#81c784",
    "pillow":       "#f8bbd0",
    "mirror":       "#cfd8dc",
    "lamp":         "#fff176",
}
_DEFAULT_COLOR = "#bdbdbd"


def _color_for(cls):
    return _CLASS_COLORS.get(cls, _DEFAULT_COLOR)


def render_floorplan(
    layout_txt,
    out_png,
    *,
    dpi=200,
    show_labels=True,
    min_label_area=0.2,
    wall_lw=3.5,
    opening_lw=5.0,
    show_wall_lengths=True,
):
    layout = parse_layout(layout_txt)
    walls   = layout["walls"]
    doors   = layout["doors"]
    windows = layout["windows"]
    bboxes  = layout["bboxes"]

    if not walls and not bboxes:
        raise ValueError(f"No walls or bboxes in {layout_txt}")

    segs = _wall_segments(walls)

    xs, ys = [], []
    for (a, b) in segs:
        xs += [a[0], b[0]]; ys += [a[1], b[1]]
    for _cls, v in bboxes:
        cx, cy, _cz, yaw, sx, sy, _sz = v
        rx, ry = _rotated_rect_xy(cx, cy, sx, sy, yaw)
        xs += rx; ys += ry
    if not xs:
        xs, ys = [0.0, 1.0], [0.0, 1.0]

    pad = 0.5
    xmin, xmax = min(xs) - pad, max(xs) + pad
    ymin, ymax = min(ys) - pad, max(ys) + pad
    w_m = max(xmax - xmin, 1e-3)
    h_m = max(ymax - ymin, 1e-3)
    fig_w = 12.0
    fig_h = max(4.0, fig_w * (h_m / w_m))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    for cls, v in bboxes:
        cx, cy, _cz, yaw, sx, sy, _sz = v
        rx, ry = _rotated_rect_xy(cx, cy, sx, sy, yaw)
        ax.fill(rx, ry, color=_color_for(cls), alpha=0.55, zorder=1.0,
                edgecolor="#263238", linewidth=0.6)
        if show_labels and (sx * sy) >= min_label_area:
            ax.text(cx, cy, cls.replace("_", " "),
                    ha="center", va="center", fontsize=7,
                    color="#263238", zorder=1.5, fontweight="bold")

    for (a, b) in segs:
        ax.plot([a[0], b[0]], [a[1], b[1]],
                color="black", linewidth=wall_lw,
                solid_capstyle="round", zorder=2.0)

    if show_wall_lengths:
        for (a, b) in segs:
            dx, dy = b[0] - a[0], b[1] - a[1]
            length_m = math.hypot(dx, dy)
            if length_m < 0.2:
                continue
            mx, my = 0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])
            angle_deg = math.degrees(math.atan2(dy, dx))
            # Flip text so it reads left-to-right (never upside down).
            if angle_deg > 90:
                angle_deg -= 180
            elif angle_deg < -90:
                angle_deg += 180
            # Perpendicular offset outward (small).
            seg_len = max(length_m, 1e-6)
            nx, ny = -dy / seg_len, dx / seg_len
            off = 0.12
            ax.text(mx + off * nx, my + off * ny,
                    f"{length_m:.2f} m",
                    ha="center", va="center",
                    fontsize=7, color="black",
                    rotation=angle_deg, rotation_mode="anchor",
                    zorder=2.5,
                    bbox=dict(boxstyle="round,pad=0.15",
                              facecolor="white",
                              edgecolor="none", alpha=0.8))

    for d in doors:
        px, py, _pz, wdt, _ht = d
        tx, ty = _nearest_wall_tangent(px, py, segs)
        ax.plot([px - 0.5 * wdt * tx, px + 0.5 * wdt * tx],
                [py - 0.5 * wdt * ty, py + 0.5 * wdt * ty],
                color="#43a047", linewidth=opening_lw,
                solid_capstyle="butt", zorder=3.0)

    for win in windows:
        px, py, _pz, wdt, _ht = win
        tx, ty = _nearest_wall_tangent(px, py, segs)
        ax.plot([px - 0.5 * wdt * tx, px + 0.5 * wdt * tx],
                [py - 0.5 * wdt * ty, py + 0.5 * wdt * ty],
                color="#29b6f6", linewidth=opening_lw,
                solid_capstyle="butt", zorder=3.0)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.grid(True, color="#eeeeee", linewidth=0.5, zorder=0)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"{Path(layout_txt).name}   "
                 f"{len(walls)}w  {len(doors)}d  {len(windows)}win  "
                 f"{len(bboxes)}obj")

    handles = [
        patches.Patch(color="black",   label="wall"),
        patches.Patch(color="#43a047", label="door"),
        patches.Patch(color="#29b6f6", label="window"),
    ]
    for cls in sorted({c for c, _ in bboxes}):
        handles.append(patches.Patch(color=_color_for(cls), alpha=0.55,
                                     label=cls.replace("_", " ")))
    ax.legend(handles=handles, loc="center left",
              bbox_to_anchor=(1.01, 0.5), fontsize=7, frameon=False)

    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_png


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("layout", type=Path, help="SpatialLM layout .txt file")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Output PNG (default: <layout_stem>_floorplan.png)")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--no-labels", action="store_true",
                   help="Omit per-bbox class labels")
    p.add_argument("--min-label-area", type=float, default=0.2,
                   help="Skip labels for bboxes with XY area (m^2) below "
                        "this (default 0.2; hides pillows/curtains).")
    p.add_argument("--no-wall-lengths", action="store_true",
                   help="Omit wall-length labels on each wall segment.")
    args = p.parse_args()

    if not args.layout.is_file():
        print(f"Error: {args.layout} not found", file=sys.stderr)
        sys.exit(1)

    out = args.output or args.layout.with_name(
        f"{args.layout.stem}_floorplan.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    path = render_floorplan(
        args.layout, out,
        dpi=args.dpi,
        show_labels=not args.no_labels,
        min_label_area=args.min_label_area,
        show_wall_lengths=not args.no_wall_lengths,
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
