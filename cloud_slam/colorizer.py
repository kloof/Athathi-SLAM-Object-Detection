"""
Camera-to-LiDAR color projection.

Projects camera images onto lidar point clouds using calibrated
extrinsics (lidar→camera) and intrinsics to produce colored point clouds.
"""

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
    R = Rotation.from_quat(quat).as_matrix()
    t = np.array([ext['translation']['x'], ext['translation']['y'], ext['translation']['z']])

    T_lidar_cam = np.eye(4)
    T_lidar_cam[:3, :3] = R
    T_lidar_cam[:3, 3] = t

    return {
        'K': K,
        'dist_coeffs': dist_coeffs,
        'T_lidar_cam': T_lidar_cam,
        'image_size': image_size,
    }


def match_nearest_image(cloud_stamp, image_timestamps, max_dt=0.15):
    """
    Find the nearest image timestamp to a cloud timestamp.

    M4a: returns `(idx, dt)` where `dt = abs(stamp_diff)` on a hit,
    or `(None, None)` when no image is within `max_dt`. The dt value
    feeds the scan_quality.time_sync telemetry (p50/p95/p99 across
    matched frames) so downstream users can see whether the capture
    pipeline is hovering near the 150 ms hard cap.
    """
    idx = np.searchsorted(image_timestamps, cloud_stamp)
    candidates = []
    if idx > 0:
        candidates.append(idx - 1)
    if idx < len(image_timestamps):
        candidates.append(idx)

    if not candidates:
        return None, None

    best = min(candidates, key=lambda i: abs(image_timestamps[i] - cloud_stamp))
    dt = float(abs(image_timestamps[best] - cloud_stamp))
    if dt > max_dt:
        return None, None
    return best, dt


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
