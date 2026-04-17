"""Accuracy metrics for registered point clouds.

The goal: measure how *clean* each backend's reconstruction is. A clean
map has pencil-thin walls, a flat floor, and colored/uncolored points
co-located on the same surface. A smeared map has double-layered walls
and colored/uncolored points offset by centimeters.

Metrics computed here are all scalars written to metrics.json so
COMPARISON.md can rank backends on each axis.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d


def wall_plane_rmse(pcd: o3d.geometry.PointCloud,
                    distance_thresh: float = 0.05,
                    target_walls: int = 3,
                    max_attempts: int = 15,
                    min_inliers: int = 2000
                    ) -> dict:
    """RANSAC-fit the N largest near-vertical planes, report inlier RMSE.

    Wall RMSE is the cleanest geometric accuracy metric we have: a
    drift-free reconstruction puts every repeat-observation of a wall
    on the same plane to within lidar noise (~1-2 cm for L2). Drift
    doubles or triples this.

    We loop up to max_attempts extracting the current largest plane.
    If the plane normal is near-vertical (floor/ceiling — |n_z| > 0.5),
    we discard the inliers but don't count it. Otherwise it counts as
    a wall. Stops after target_walls are found or no more planes with
    at least min_inliers remain.
    """
    pts_all = np.asarray(pcd.points)
    if len(pts_all) < min_inliers:
        return {"wall_rmse_m": None, "wall_count": 0, "wall_inliers": []}

    remaining = o3d.geometry.PointCloud()
    remaining.points = o3d.utility.Vector3dVector(pts_all)

    wall_rmses = []
    wall_counts = []
    for _ in range(max_attempts):
        if len(remaining.points) < min_inliers:
            break
        if len(wall_rmses) >= target_walls:
            break
        try:
            plane_model, inliers = remaining.segment_plane(
                distance_threshold=distance_thresh,
                ransac_n=3, num_iterations=1000)
        except Exception:
            break
        if len(inliers) < min_inliers:
            break
        a, b, c, d = plane_model
        n = np.array([a, b, c])
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-9:
            remaining = remaining.select_by_index(inliers, invert=True)
            continue
        n /= n_norm
        # Skip floor / ceiling: their normal is near-vertical (|n_z|→1).
        # Wall normals are near-horizontal (|n_z|→0). Discard if |n_z|>0.5.
        if abs(n[2]) > 0.5:
            remaining = remaining.select_by_index(inliers, invert=True)
            continue

        pts_in = np.asarray(remaining.points)[inliers]
        dists = np.abs(pts_in @ n + d)
        wall_rmses.append(float(np.sqrt((dists ** 2).mean())))
        wall_counts.append(int(len(inliers)))
        remaining = remaining.select_by_index(inliers, invert=True)

    if not wall_rmses:
        return {"wall_rmse_m": None, "wall_count": 0, "wall_inliers": []}

    return {
        "wall_rmse_m": float(np.mean(wall_rmses)),
        "wall_rmse_individual_m": [round(r, 5) for r in wall_rmses],
        "wall_count": len(wall_rmses),
        "wall_inliers": wall_counts,
    }


def color_uncolored_coherence(pcd: o3d.geometry.PointCloud,
                              voxel_size: float = 0.03,
                              gray_tol: float = 0.01
                              ) -> dict:
    """Measure how well colored and uncolored points co-locate.

    For voxels that contain both a camera-colored point and a gray
    fallback point, report the positional spread within those voxels.
    Low spread = same surface, both layers aligned — good. High spread
    = colored layer offset from uncolored layer (classic drift smear).
    """
    pts = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    if len(pts) == 0 or len(colors) == 0:
        return {
            "mixed_voxel_count": 0,
            "color_uncolored_spread_m": None,
        }

    gray_default = np.array([128 / 255.0] * 3)
    is_gray = np.all(np.abs(colors - gray_default) < gray_tol, axis=1)
    if is_gray.all() or (~is_gray).all():
        # No mix possible
        return {
            "mixed_voxel_count": 0,
            "color_uncolored_spread_m": None,
        }

    voxel_idx = np.floor(pts / voxel_size).astype(np.int64)
    unique_keys, inverse = np.unique(voxel_idx, axis=0, return_inverse=True)

    # For each voxel count gray and non-gray populations
    n_vox = len(unique_keys)
    gray_count = np.zeros(n_vox, dtype=np.int64)
    colored_count = np.zeros(n_vox, dtype=np.int64)
    np.add.at(gray_count, inverse[is_gray], 1)
    np.add.at(colored_count, inverse[~is_gray], 1)

    mixed = (gray_count > 0) & (colored_count > 0)
    if not mixed.any():
        return {
            "mixed_voxel_count": 0,
            "color_uncolored_spread_m": None,
        }

    spreads = []
    mixed_indices = np.where(mixed)[0]
    # Sample up to 2000 mixed voxels for speed
    if len(mixed_indices) > 2000:
        rng = np.random.default_rng(42)
        mixed_indices = rng.choice(mixed_indices, 2000, replace=False)

    for vk in mixed_indices:
        voxel_mask = inverse == vk
        vox_pts = pts[voxel_mask]
        vox_gray = is_gray[voxel_mask]
        if vox_gray.any() and (~vox_gray).any():
            centroid_g = vox_pts[vox_gray].mean(axis=0)
            centroid_c = vox_pts[~vox_gray].mean(axis=0)
            spreads.append(np.linalg.norm(centroid_c - centroid_g))

    if not spreads:
        return {
            "mixed_voxel_count": 0,
            "color_uncolored_spread_m": None,
        }

    return {
        "mixed_voxel_count": int(mixed.sum()),
        "color_uncolored_spread_m": float(np.mean(spreads)),
        "color_uncolored_spread_p95_m": float(np.percentile(spreads, 95)),
    }


def trajectory_jerk(poses: list[np.ndarray]) -> dict:
    """Mean third-derivative magnitude over the trajectory.

    A clean SLAM trajectory for a hand-held scan is smooth — low jerk.
    A noisy pose sequence with frame-to-frame registration error shows
    up as spikes in jerk. Unit-free (per-sample).
    """
    if len(poses) < 4:
        return {"trajectory_jerk_mean": None, "trajectory_jerk_p95": None}
    positions = np.array([T[:3, 3] for T in poses])
    v = np.diff(positions, axis=0)
    a = np.diff(v, axis=0)
    j = np.diff(a, axis=0)
    mags = np.linalg.norm(j, axis=1)
    return {
        "trajectory_jerk_mean": float(mags.mean()),
        "trajectory_jerk_p95": float(np.percentile(mags, 95)),
        "trajectory_length_m": float(np.linalg.norm(
            np.diff(positions, axis=0), axis=1).sum()),
    }


def floor_flatness(pcd: o3d.geometry.PointCloud,
                   distance_thresh: float = 0.05
                   ) -> dict:
    """RMSE of the inliers of the largest horizontal plane (the floor).

    After post-process leveling, floor z=0. Remaining z-spread inside
    the floor inliers is the flatness — another clean-reconstruction
    indicator.
    """
    pts = np.asarray(pcd.points)
    if len(pts) < 1000:
        return {"floor_rmse_m": None, "floor_inliers": 0}
    pcd_tmp = o3d.geometry.PointCloud()
    pcd_tmp.points = o3d.utility.Vector3dVector(pts)
    try:
        plane_model, inliers = pcd_tmp.segment_plane(
            distance_threshold=distance_thresh,
            ransac_n=3, num_iterations=1000)
    except Exception:
        return {"floor_rmse_m": None, "floor_inliers": 0}
    a, b, c, d = plane_model
    n = np.array([a, b, c])
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-9 or abs(n[2]) / n_norm < np.cos(np.radians(30)):
        # Not a horizontal plane — skip
        return {"floor_rmse_m": None, "floor_inliers": 0}
    n /= n_norm
    pts_in = pts[inliers]
    dists = np.abs(pts_in @ n + d)
    return {
        "floor_rmse_m": float(np.sqrt((dists ** 2).mean())),
        "floor_inliers": int(len(inliers)),
    }


def compute_all(pcd: o3d.geometry.PointCloud,
                poses: list[np.ndarray]
                ) -> dict:
    """Run every metric, return one flat dict suitable for metrics.json."""
    out: dict = {}
    out.update(wall_plane_rmse(pcd))
    out.update(floor_flatness(pcd))
    out.update(color_uncolored_coherence(pcd))
    out.update(trajectory_jerk(poses))
    return out


def cross_section_png(pcd: o3d.geometry.PointCloud,
                      out_path: str,
                      slice_z: float = 1.0,
                      thickness: float = 0.1,
                      pixel_size: float = 0.02
                      ) -> bool:
    """Top-down slice of the map at z=slice_z, exported as PNG.

    This is the visual read of wall thickness: pencil-thin lines in a
    clean map, fuzzy bands in a smeared map. Returns True on success.
    """
    pts = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    if len(pts) == 0:
        return False

    mask = np.abs(pts[:, 2] - slice_z) < thickness * 0.5
    if mask.sum() < 100:
        return False
    slab = pts[mask]
    slab_colors = colors[mask] if len(colors) == len(pts) else None

    x0, y0 = slab[:, 0].min(), slab[:, 1].min()
    x1, y1 = slab[:, 0].max(), slab[:, 1].max()
    w = max(int(np.ceil((x1 - x0) / pixel_size)), 16)
    h = max(int(np.ceil((y1 - y0) / pixel_size)), 16)

    # Safety cap: if SLAM diverged and the map is kilometers wide, don't
    # allocate a 100 GB image. Fall back to fitting into a 4000x4000 tile.
    max_side = 4000
    if w > max_side or h > max_side:
        scale = max(w, h) / max_side
        pixel_size = pixel_size * scale
        w = max(int(np.ceil((x1 - x0) / pixel_size)), 16)
        h = max(int(np.ceil((y1 - y0) / pixel_size)), 16)

    img = np.ones((h, w, 3), dtype=np.uint8) * 255
    xs = np.clip(((slab[:, 0] - x0) / pixel_size).astype(int), 0, w - 1)
    ys = np.clip(((slab[:, 1] - y0) / pixel_size).astype(int), 0, h - 1)

    if slab_colors is not None:
        rgb = (slab_colors * 255).clip(0, 255).astype(np.uint8)
        img[h - 1 - ys, xs] = rgb
    else:
        img[h - 1 - ys, xs] = [0, 0, 0]

    try:
        import cv2
        cv2.imwrite(out_path, img[:, :, ::-1])
        return True
    except Exception:
        return False
