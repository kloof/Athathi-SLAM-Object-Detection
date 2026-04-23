"""Yaw-align the cropped cloud so walls are parallel to X/Y.

Uses the wall angles from floorplan/room_metadata.json (D_refined variant)
to find the length-weighted mode (mod 90 deg) and rotate by its negative.
SpatialLM trains on ScanNet-style axis-aligned data, so this materially
helps downstream detection/classification quality.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import open3d as o3d


def yaw_align_by_walls(
    src_ply: Path | str,
    floorplan_meta: Path | str,
    out_ply: Path | str,
    *,
    verbose: bool = True,
) -> tuple[Path, float]:
    """Rotate cloud around +Z so the mode wall angle (mod 90) -> 0.

    Returns (output-path, applied-yaw-deg).
    """
    src_ply = Path(src_ply)
    floorplan_meta = Path(floorplan_meta)
    out_ply = Path(out_ply)

    meta = json.loads(floorplan_meta.read_text())
    walls = meta["variants"]["D_refined"]["walls"]
    angles = np.asarray([w["angle_deg"] for w in walls], dtype=np.float64)
    lens = np.asarray([w["length_m"] for w in walls], dtype=np.float64)

    # walls parallel to X and walls parallel to Y are both axis-aligned -- wrap to [0,90)
    ang_mod = np.mod(angles, 90.0)
    bins = np.arange(0, 91)
    hist, _ = np.histogram(ang_mod, bins=bins, weights=lens)
    mode_deg = float(bins[np.argmax(hist)] + 0.5)
    candidates = [mode_deg, mode_deg - 90.0]
    rot_deg = min(candidates, key=lambda x: abs(x))  # shortest-path rotation
    if verbose:
        print(f"[manhattan] wall angle mode (mod 90) = {mode_deg:.1f} deg")
        print(f"[manhattan] applying yaw rotation = {-rot_deg:+.2f} deg")

    yaw = np.deg2rad(-rot_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)

    pcd = o3d.io.read_point_cloud(str(src_ply))
    pcd.rotate(R, center=(0, 0, 0))
    o3d.io.write_point_cloud(str(out_ply), pcd)

    if verbose:
        pts = np.asarray(pcd.points)
        print(f"[manhattan] post-rotate bbox range "
              f"{(pts.max(0) - pts.min(0)).round(2)}  wrote {out_ply}")
    return out_ply, -rot_deg
