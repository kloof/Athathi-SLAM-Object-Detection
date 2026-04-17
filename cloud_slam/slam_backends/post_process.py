"""Shared post-processing for all SLAM backends.

Given per-frame poses from any backend, this module:
  1. Deskews each scan using IMU gyro (when available)
  2. Colorizes every point via per-point camera matching (when images+calib)
  3. Transforms each scan into the world frame
  4. Accumulates xyz / color / has-color across all frames
  5. Voxel-reduces with a color-aware rule: the mean color of a voxel is
     computed from ONLY the points in that voxel that were actually
     colorized by the camera. Gray-padded points never dilute a real color.
  6. Optionally levels the final cloud so the floor lies at z=0

The has-color mask is the key correctness guarantee: Open3D's built-in
voxel_down_sample averages all colors together, so a voxel with 4 real
colored points and 6 gray-padded points ends up muddy. Our reducer drops
the gray-padded points from the color average unless the voxel has zero
colored points, in which case it falls back to gray.
"""

from __future__ import annotations

import time

import numpy as np
import open3d as o3d

from cloud_slam.colorizer import colorize_cloud_per_point
from cloud_slam.deskew import deskew_scan
from cloud_slam.level import detect_floor_plane, level_points


DEFAULT_GRAY = np.array([128, 128, 128], dtype=np.float64) / 255.0


def _is_colorized(colors: np.ndarray,
                  default: np.ndarray = DEFAULT_GRAY,
                  tol: float = 1e-6) -> np.ndarray:
    """Points that got a real camera color (as opposed to the gray fallback)."""
    return np.any(np.abs(colors - default) > tol, axis=1)


def _voxel_reduce(xyz: np.ndarray,
                  colors: np.ndarray,
                  has_color: np.ndarray,
                  voxel_size: float,
                  default_gray: np.ndarray = DEFAULT_GRAY
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Voxel-reduce with a color-aware averaging rule.

    A voxel's output color is the mean of ONLY the colorized points it
    contains. If no colorized points land in the voxel, the output color
    is the gray default. The output position is the mean of all points
    in the voxel regardless of color status.
    """
    if len(xyz) == 0:
        return xyz, colors

    voxel_idx = np.floor(xyz / voxel_size).astype(np.int64)
    unique_keys, inverse, counts = np.unique(
        voxel_idx, axis=0, return_inverse=True, return_counts=True)
    n_voxels = len(unique_keys)

    pos_sums = np.zeros((n_voxels, 3), dtype=np.float64)
    np.add.at(pos_sums, inverse, xyz)
    pos_out = pos_sums / counts[:, None]

    color_sums = np.zeros((n_voxels, 3), dtype=np.float64)
    color_counts = np.zeros(n_voxels, dtype=np.int64)
    if has_color.any():
        np.add.at(color_sums, inverse[has_color], colors[has_color])
        np.add.at(color_counts, inverse[has_color], 1)

    color_out = np.tile(default_gray, (n_voxels, 1))
    colored_voxels = color_counts > 0
    color_out[colored_voxels] = (
        color_sums[colored_voxels] / color_counts[colored_voxels, None])

    return pos_out, color_out


def _accumulate(
    clouds: list[tuple[float, np.ndarray, np.ndarray]],
    imus: list[tuple[float, np.ndarray, np.ndarray]],
    poses: list[np.ndarray],
    images: list[tuple[float, bytes, str]] | None,
    calib: dict | None,
    deskew: bool,
    colorize: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform every frame to world, colorize, accumulate."""
    if len(poses) != len(clouds):
        raise ValueError(
            f"pose count {len(poses)} != cloud count {len(clouds)}")

    if imus:
        imu_times = np.array([t for t, _, _ in imus], dtype=np.float64)
        imu_gyros = np.array([g for _, g, _ in imus], dtype=np.float64)
    else:
        imu_times = np.array([], dtype=np.float64)
        imu_gyros = np.empty((0, 3), dtype=np.float64)

    if colorize and images and calib:
        image_timestamps = np.array(
            [t for t, _, _ in images], dtype=np.float64)
    else:
        image_timestamps = np.array([], dtype=np.float64)

    xyz_chunks, color_chunks, has_color_chunks = [], [], []
    for i, (stamp, xyz, time_offsets) in enumerate(clouds):
        if len(xyz) == 0:
            continue

        xyz_f = np.asarray(xyz, dtype=np.float64)

        if deskew and len(imu_times) > 0 and len(time_offsets) > 0:
            xyz_f = deskew_scan(xyz_f, time_offsets, stamp,
                                imu_times, imu_gyros)

        if colorize and images and calib and len(image_timestamps) > 0:
            if len(time_offsets) > 0:
                point_timestamps_abs = stamp + np.asarray(
                    time_offsets, dtype=np.float64)
            else:
                point_timestamps_abs = np.full(
                    len(xyz_f), stamp, dtype=np.float64)
            colors = colorize_cloud_per_point(
                xyz_f, point_timestamps_abs, images, image_timestamps, calib)
            has_color = _is_colorized(colors)
        else:
            colors = np.tile(DEFAULT_GRAY, (len(xyz_f), 1))
            has_color = np.zeros(len(xyz_f), dtype=bool)

        T = poses[i]
        xyz_world = xyz_f @ T[:3, :3].T + T[:3, 3]

        xyz_chunks.append(xyz_world)
        color_chunks.append(colors)
        has_color_chunks.append(has_color)

    if not xyz_chunks:
        empty = np.empty((0, 3), dtype=np.float64)
        return empty, empty, np.empty((0,), dtype=bool)

    return (np.concatenate(xyz_chunks),
            np.concatenate(color_chunks),
            np.concatenate(has_color_chunks))


def build_map(
    clouds: list[tuple[float, np.ndarray, np.ndarray]],
    imus: list[tuple[float, np.ndarray, np.ndarray]],
    poses: list[np.ndarray],
    images: list[tuple[float, bytes, str]] | None = None,
    calib: dict | None = None,
    *,
    voxel_size: float = 0.01,
    deskew: bool = True,
    colorize: bool = True,
    outlier_neighbors: int = 20,
    outlier_std_ratio: float = 2.0,
    level_to_floor: bool = True,
) -> tuple[o3d.geometry.PointCloud, dict]:
    """Shared pipeline from (clouds, poses) to a clean colored PLY.

    Returns (point_cloud, stats). stats is a JSON-serializable dict with
    timing and size info that each backend's run writer merges into
    metrics.json.
    """
    t0 = time.time()

    xyz, colors, has_color = _accumulate(
        clouds, imus, poses, images, calib, deskew=deskew, colorize=colorize)
    t_accum = time.time() - t0

    n_raw = len(xyz)
    if n_raw == 0:
        return o3d.geometry.PointCloud(), {
            "post_accumulate_s": round(t_accum, 2),
            "post_raw_points": 0,
            "post_final_points": 0,
            "colorized": False,
            "colorized_fraction": 0.0,
        }

    t1 = time.time()
    xyz_r, colors_r = _voxel_reduce(xyz, colors, has_color, voxel_size)
    t_voxel = time.time() - t1

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_r)
    pcd.colors = o3d.utility.Vector3dVector(colors_r)

    t2 = time.time()
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=outlier_neighbors, std_ratio=outlier_std_ratio)
    t_outlier = time.time() - t2

    leveled = False
    rotation_deg = 0.0
    z_shift = 0.0
    if level_to_floor and len(pcd.points) > 100:
        try:
            pts = np.asarray(pcd.points)
            result = detect_floor_plane(pts)
            if result is not None:
                normal, _ = result
                leveled_xyz, rotation_deg, z_shift = level_points(
                    pts.astype(np.float32), normal)
                pcd.points = o3d.utility.Vector3dVector(
                    leveled_xyz.astype(np.float64))
                leveled = True
        except Exception as exc:
            print(f"  [post] leveling failed: {exc}")

    stats = {
        "post_accumulate_s": round(t_accum, 2),
        "post_voxel_s": round(t_voxel, 2),
        "post_outlier_s": round(t_outlier, 2),
        "post_raw_points": int(n_raw),
        "post_final_points": int(len(pcd.points)),
        "post_voxel_size_m": voxel_size,
        "colorized": bool(has_color.any()),
        "colorized_fraction": float(has_color.mean()) if n_raw else 0.0,
        "leveled": bool(leveled),
        "level_rotation_deg": float(rotation_deg),
        "level_z_shift_m": float(z_shift),
    }

    extent = pcd.get_axis_aligned_bounding_box().get_extent()
    stats["bounding_box_m"] = [
        float(extent[0]), float(extent[1]), float(extent[2])]

    return pcd, stats


def write_trajectory_csv(poses: list[np.ndarray],
                         clouds: list[tuple[float, np.ndarray, np.ndarray]],
                         path: str) -> None:
    """Write a simple trajectory CSV: timestamp, x, y, z, qw, qx, qy, qz."""
    from scipy.spatial.transform import Rotation
    if len(poses) != len(clouds):
        raise ValueError("pose/cloud length mismatch")
    with open(path, "w") as f:
        f.write("timestamp,x,y,z,qw,qx,qy,qz\n")
        for (stamp, _, _), T in zip(clouds, poses):
            t = T[:3, 3]
            q = Rotation.from_matrix(T[:3, :3]).as_quat()  # xyzw
            f.write(f"{stamp:.6f},{t[0]:.6f},{t[1]:.6f},{t[2]:.6f},"
                    f"{q[3]:.6f},{q[0]:.6f},{q[1]:.6f},{q[2]:.6f}\n")
