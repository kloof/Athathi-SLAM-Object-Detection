"""K-NN color interpolation for default-gray points.

SpatialLM sees default-gray (128,128,128) as an ambiguous "is this real
gray or camera-missed-it?" signal that flips classifications between
runs. For each default-gray point, this module finds the K nearest
CAMERA-COLORED points within a radius and blends their RGB values
(inverse-distance weighted). Points with <2 colored neighbors in the
radius stay gray.

Default radius = 30 cm, K = 8. Cascade (multiple passes) gets close to
100% fill at the cost of a second radius traversal.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import open3d as o3d


def interpolate_gray_colors(
    src_ply: Path | str,
    dst_ply: Path | str,
    *,
    k: int = 8,
    radius_m: float = 0.30,
    gray_rgb_float: float = 128.0 / 255.0,
    gray_tol: float = 1e-3,
    also_stat_outlier: bool = True,
    stat_nb: int = 10,
    stat_std: float = 1.5,
    verbose: bool = True,
) -> Path:
    """Fill default-gray points with inverse-distance weighted RGB from
    nearby camera-colored neighbors. Optionally runs stat-outlier removal
    at the end, which is cheap on the result.
    """
    src_ply = Path(src_ply)
    dst_ply = Path(dst_ply)
    pcd = o3d.io.read_point_cloud(str(src_ply))
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)
    n = len(pts)
    if verbose:
        print(f"[interp] in: {n:,} pts")

    is_gray = (np.abs(cols - gray_rgb_float) < gray_tol).all(axis=1)
    n_gray = int(is_gray.sum())
    if verbose:
        print(f"[interp] colored: {n-n_gray:,}  gray: {n_gray:,} "
              f"({100*n_gray/n:.1f}%)")

    colored_pts = pts[~is_gray]
    colored_cols = cols[~is_gray]

    new_cols = cols.copy()
    gray_idx = np.flatnonzero(is_gray)
    n_filled = 0
    t0 = time.time()
    if len(gray_idx) > 0 and len(colored_pts) > 0:
        # Batched KNN: scipy cKDTree runs the whole query (N_gray x k) in a
        # single C call instead of a Python loop over per-point searches.
        # Unreached neighbors are returned as (inf, n_colored) — mask those
        # out before IDW-averaging.
        from scipy.spatial import cKDTree
        tree = cKDTree(colored_pts)
        gray_pts = pts[gray_idx]
        dists, idxs = tree.query(
            gray_pts, k=k, distance_upper_bound=radius_m, workers=-1)
        if k == 1:  # scipy collapses trailing dim when k=1; keep 2D contract
            dists = dists[:, None]
            idxs = idxs[:, None]
        valid = np.isfinite(dists)
        n_found = valid.sum(axis=1)
        fillable = n_found >= 2
        if fillable.any():
            # Replace OOR indices with 0 (harmless; weight will be zero too).
            safe_idxs = np.where(valid, idxs, 0)
            neighbor_cols = colored_cols[safe_idxs]           # (N_gray, k, 3)
            w = np.where(valid, 1.0 / (dists + 1e-3), 0.0)   # (N_gray, k)
            w_sum = w.sum(axis=1, keepdims=True)
            w_sum[w_sum == 0] = 1.0
            w = w / w_sum                                     # (N_gray, k)
            filled = (w[..., None] * neighbor_cols).sum(axis=1)  # (N_gray, 3)
            fi = gray_idx[fillable]
            new_cols[fi] = filled[fillable]
            n_filled = int(fillable.sum())
    if verbose:
        print(f"[interp] K={k} r={radius_m}m  filled {n_filled:,}/{n_gray:,} "
              f"({100*n_filled/max(1,n_gray):.1f}%)  "
              f"[{time.time()-t0:.1f}s]")

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts)
    out.colors = o3d.utility.Vector3dVector(new_cols)

    if also_stat_outlier:
        out, _ = out.remove_statistical_outlier(nb_neighbors=stat_nb,
                                                 std_ratio=stat_std)
        if verbose:
            print(f"[interp] stat-outlier nb={stat_nb} std={stat_std}: "
                  f"-> {len(out.points):,}")

    # shift floor to z=0 if it drifted
    out_pts = np.asarray(out.points)
    z_min = float(out_pts[:, 2].min())
    if abs(z_min) > 0.05:
        out.translate((0, 0, -z_min))
        if verbose:
            print(f"[interp] shifted floor from z={z_min:.3f} to z=0")

    o3d.io.write_point_cloud(str(dst_ply), out)
    if verbose:
        print(f"[interp] wrote {dst_ply}")
    return dst_ply
