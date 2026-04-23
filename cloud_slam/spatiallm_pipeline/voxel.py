"""Colored-priority voxel downsample.

A plain voxel_down_sample averages points in each voxel cell -- when a
cell contains both camera-colored points and default-gray (128,128,128)
placeholder points, the averaged color is a gray+color mud that gives
SpatialLM an ambiguous "is this a real gray surface" signal.

This downsampler splits the input by color (colored vs default-gray),
voxels each separately, and for any cell where both ended up with a
representative: the colored one wins, the gray is dropped. Result:
every cell the camera saw has a true color; only cells the camera
never saw stay gray.

Default voxel = 1 cm which gave us 8/8 dining_chair recall in
empirical testing. Bumping to 2.5 cm matches SpatialLM's grid and
thins more aggressively (useful if point count pushes past ~2 M).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d


def colored_priority_voxel(
    src_ply: Path | str,
    dst_ply: Path | str,
    *,
    voxel_m: float = 0.01,
    gray_rgb_float: float = 128.0 / 255.0,
    gray_tol: float = 1e-3,
    verbose: bool = True,
) -> Path:
    """Downsample prioritizing camera-colored points over default-gray."""
    src_ply = Path(src_ply)
    dst_ply = Path(dst_ply)
    pcd = o3d.io.read_point_cloud(str(src_ply))
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)
    n_in = len(pts)
    if verbose:
        print(f"[voxel] in: {n_in:,} pts")

    is_gray = (np.abs(cols - gray_rgb_float) < gray_tol).all(axis=1)
    colored = o3d.geometry.PointCloud()
    colored.points = o3d.utility.Vector3dVector(pts[~is_gray])
    colored.colors = o3d.utility.Vector3dVector(cols[~is_gray])
    gray = o3d.geometry.PointCloud()
    gray.points = o3d.utility.Vector3dVector(pts[is_gray])
    gray.colors = o3d.utility.Vector3dVector(cols[is_gray])
    if verbose:
        print(f"[voxel] split: {len(colored.points):,} colored  "
              f"{len(gray.points):,} gray")

    col_ds = colored.voxel_down_sample(voxel_m)
    gry_ds = gray.voxel_down_sample(voxel_m)
    if verbose:
        print(f"[voxel] after {voxel_m*100:.1f}cm voxel: "
              f"{len(col_ds.points):,} colored  {len(gry_ds.points):,} gray")

    def voxel_keys(_pcd, v):
        p = np.asarray(_pcd.points)
        idx = np.floor(p / v).astype(np.int64)
        # 21-bit-per-axis pack, safe for ~10km worlds at 1cm
        return (idx[:, 0] + (1 << 20)) * (1 << 42) + \
               (idx[:, 1] + (1 << 20)) * (1 << 21) + \
               (idx[:, 2] + (1 << 20))

    col_keys = voxel_keys(col_ds, voxel_m)
    gry_keys = voxel_keys(gry_ds, voxel_m)
    keep_gray = ~np.isin(gry_keys, col_keys)
    n_overlap = int((~keep_gray).sum())
    if verbose:
        print(f"[voxel] gray cells overlapping colored (dropped): {n_overlap:,}")
    gry_ds = gry_ds.select_by_index(np.flatnonzero(keep_gray))

    merged = col_ds + gry_ds
    if verbose:
        print(f"[voxel] merged: {len(merged.points):,}  "
              f"({len(col_ds.points):,} colored + {len(gry_ds.points):,} gray-fill)")

    o3d.io.write_point_cloud(str(dst_ply), merged)
    if verbose:
        print(f"[voxel] wrote {dst_ply}")
    return dst_ply
