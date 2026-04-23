"""
Camera-to-LiDAR color projection.

Projects camera images onto lidar point clouds using calibrated
extrinsics (lidar→camera) and intrinsics to produce colored point clouds.
"""

from typing import List, Tuple

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from cloud_slam.projection import project_lidar_to_camera


def load_calibration(intrinsics_path, extrinsics_path):
    """
    Load camera intrinsics and lidar→camera extrinsics from YAML files.

    Returns dict with keys:
        K:           (3,3) camera matrix
        dist_coeffs: (5,) distortion coefficients
        T_lidar_cam: (4,4) homogeneous transform lidar→camera
        image_size:  (width, height)
    """
    with open(intrinsics_path) as f:
        intr = yaml.safe_load(f)

    with open(extrinsics_path) as f:
        ext = yaml.safe_load(f)

    K = np.array(intr['camera_matrix']['data'], dtype=np.float64).reshape(3, 3)
    dist_coeffs = np.array(intr['distortion_coefficients']['data'], dtype=np.float64)
    image_size = (intr['image_width'], intr['image_height'])

    # Quaternion: YAML has (w,x,y,z), scipy wants (x,y,z,w)
    q = ext['rotation']
    quat = [q['x'], q['y'], q['z'], q['w']]
    # YAML convention: parent=lidar, child=camera_optical_frame — so (R, t) describe
    # the camera pose IN the lidar frame (R_lc, t_lc). projection.py uses the stored
    # matrix directly as "lidar -> camera" (p_cam = R @ p_lidar + t), so we must
    # invert here:
    #     R_cl = R_lc.T         t_cl = -R_cl @ t_lc
    # Cherry-picked from commit 6d81934 (calibration/extrinsic-refinement branch)
    # which first caught this direction flip (same bug that made the YOLOE boxes
    # oversized in the old pipeline).
    R_lc = Rotation.from_quat(quat).as_matrix()
    t_lc = np.array([ext['translation']['x'], ext['translation']['y'], ext['translation']['z']])
    R_cl = R_lc.T
    t_cl = -R_cl @ t_lc

    T_lidar_cam = np.eye(4)
    T_lidar_cam[:3, :3] = R_cl
    T_lidar_cam[:3, 3] = t_cl

    return {
        'K': K,
        'dist_coeffs': dist_coeffs,
        'T_lidar_cam': T_lidar_cam,
        'image_size': image_size,
    }


def match_nearest_image(cloud_stamp, image_timestamps, max_dt=0.15):
    """
    Find the nearest image timestamp to a cloud timestamp.
    Returns index into image_timestamps, or None if beyond max_dt.
    """
    idx = np.searchsorted(image_timestamps, cloud_stamp)
    candidates = []
    if idx > 0:
        candidates.append(idx - 1)
    if idx < len(image_timestamps):
        candidates.append(idx)

    if not candidates:
        return None

    best = min(candidates, key=lambda i: abs(image_timestamps[i] - cloud_stamp))
    if abs(image_timestamps[best] - cloud_stamp) > max_dt:
        return None
    return best


def colorize_cloud(xyz, image, calib, default_color=(128, 128, 128)):
    """
    Project lidar points into camera image and sample RGB colors.

    Args:
        xyz:     (N, 3) float64, points in lidar frame
        image:   (H, W, 3) uint8 BGR image from cv2
        calib:   dict from load_calibration()
        default_color: RGB tuple [0-255] for points outside camera FOV

    Returns:
        colors: (N, 3) float64 in [0, 1] range (Open3D convention)
    """
    N = len(xyz)
    colors = np.full((N, 3), np.array(default_color, dtype=np.float64) / 255.0)

    if N == 0:
        return colors

    H, W = image.shape[:2]

    # Project lidar points to camera image plane
    pts_cam, pixels, in_front = project_lidar_to_camera(xyz, calib)

    if not in_front.any():
        return colors

    u = pixels[in_front, 0]
    v = pixels[in_front, 1]
    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)

    if not in_bounds.any():
        return colors

    ui = u[in_bounds].astype(int)
    vi = v[in_bounds].astype(int)

    # Sample BGR, convert to RGB, scale to [0,1]
    bgr = image[vi, ui]  # (M', 3)
    rgb = bgr[:, ::-1].astype(np.float64) / 255.0

    # Map back through masks
    idx_front = np.where(in_front)[0]
    idx_visible = idx_front[in_bounds]
    colors[idx_visible] = rgb

    return colors


def _match_nearest_image_vectorized(point_timestamps_abs, image_timestamps,
                                    max_dt):
    """Vectorized version of match_nearest_image returning -1 for no match."""
    if len(image_timestamps) == 0 or len(point_timestamps_abs) == 0:
        return np.full(len(point_timestamps_abs), -1, dtype=np.int64)

    right = np.searchsorted(image_timestamps, point_timestamps_abs)
    left = right - 1
    n_imgs = len(image_timestamps)

    # Candidate distances (inf where out of range).
    dt_left = np.where(left >= 0,
                       np.abs(point_timestamps_abs
                              - image_timestamps[np.clip(left, 0, n_imgs - 1)]),
                       np.inf)
    dt_right = np.where(right < n_imgs,
                        np.abs(image_timestamps[np.clip(right, 0, n_imgs - 1)]
                               - point_timestamps_abs),
                        np.inf)

    use_left = dt_left <= dt_right
    idx = np.where(use_left, left, right).astype(np.int64)
    dt = np.where(use_left, dt_left, dt_right)
    idx[dt > max_dt] = -1
    return idx


def colorize_cloud_per_point(xyz: np.ndarray,
                             point_timestamps_abs: np.ndarray,
                             images: List[Tuple[float, bytes, str]],
                             image_timestamps: np.ndarray,
                             calib: dict,
                             max_dt: float = 0.15,
                             default_color=(128, 128, 128)) -> np.ndarray:
    """Per-point camera colorization.

    For each point, find the image nearest to its absolute timestamp,
    then project the point onto that image. Groups points by image index
    so each unique image is decoded only once per scan.
    """
    N = len(xyz)
    default_rgb = np.array(default_color, dtype=np.float64) / 255.0
    colors = np.tile(default_rgb, (N, 1))

    if N == 0:
        return colors
    if not images or len(image_timestamps) == 0:
        return colors

    target_idx = _match_nearest_image_vectorized(
        np.asarray(point_timestamps_abs, dtype=np.float64),
        np.asarray(image_timestamps, dtype=np.float64),
        max_dt,
    )

    unique_idx = np.unique(target_idx)
    for img_idx in unique_idx:
        if img_idx < 0:
            continue
        mask = target_idx == img_idx
        if not mask.any():
            continue
        _, compressed_bytes, _ = images[int(img_idx)]
        img_arr = np.frombuffer(compressed_bytes, dtype=np.uint8)
        decoded = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if decoded is None:
            continue
        sub_colors = colorize_cloud(xyz[mask], decoded, calib,
                                    default_color=default_color)
        colors[mask] = sub_colors

    return colors
