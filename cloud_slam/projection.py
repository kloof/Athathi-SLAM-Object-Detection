"""
Shared LiDAR-to-camera projection utilities.

Used by both colorizer (color sampling) and frustum (3D detection).
"""

import numpy as np
import cv2


def project_lidar_to_camera(xyz, calib):
    """
    Transform lidar points to camera frame and project to pixel coordinates.

    Args:
        xyz:   (N, 3) float64, points in lidar frame
        calib: dict with K, dist_coeffs, T_lidar_cam

    Returns:
        pts_cam:  (N, 3) points in camera frame
        pixels:   (N, 2) pixel coordinates (u, v) — distorted, matching raw image space
        in_front: (N,) bool mask for points with z > 0 in camera frame
    """
    N = len(xyz)
    T = calib['T_lidar_cam']

    # Transform to camera frame
    pts_cam = (T[:3, :3] @ xyz.T + T[:3, 3:4]).T  # (N, 3)

    # Points in front of camera
    in_front = pts_cam[:, 2] > 0

    # Project to pixel coordinates (only valid for in_front points)
    pixels = np.full((N, 2), np.nan)
    if in_front.any():
        rvec, _ = cv2.Rodrigues(T[:3, :3])
        tvec = T[:3, 3]
        proj, _ = cv2.projectPoints(
            xyz[in_front].astype(np.float64), rvec, tvec,
            calib['K'], calib['dist_coeffs']
        )
        pixels[in_front] = proj.squeeze(1)

    return pts_cam, pixels, in_front


def project_world_to_image(xyz_world, T_cam_world, K, D):
    """Project world-frame points into a camera image given an arbitrary pose.

    Sibling to :func:`project_lidar_to_camera`; used by stage 8 (best-view
    picker) where the camera pose varies per frame and the caller already
    holds the ``T_cam_world`` transform (world -> camera).

    Convention matches the existing helper: ``T_cam_world`` maps
    world-frame points to camera-frame via ``p_cam = R @ p_world + t``, and
    :func:`cv2.projectPoints` is called with zero ``rvec``/``tvec`` on the
    already-transformed points + plumb-bob distortion coefficients.

    Args:
        xyz_world:   (N, 3) float64, points in the world frame.
        T_cam_world: (4, 4) homogeneous transform, world -> camera.
        K:           (3, 3) camera intrinsics matrix.
        D:           (5,) plumb-bob distortion coefficients.

    Returns:
        pts_cam:  (N, 3) points in camera frame.
        pixels:   (N, 2) pixel coordinates (u, v) in the raw image (distorted).
                  Entries where ``in_front`` is False are filled with NaN.
        in_front: (N,) bool mask for points with z > 0 in camera frame.
    """
    xyz_world = np.asarray(xyz_world, dtype=np.float64)
    if xyz_world.ndim != 2 or xyz_world.shape[1] != 3:
        raise ValueError(f"xyz_world must be (N, 3); got {xyz_world.shape}")
    T = np.asarray(T_cam_world, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"T_cam_world must be (4, 4); got {T.shape}")

    N = len(xyz_world)
    pts_cam = (T[:3, :3] @ xyz_world.T + T[:3, 3:4]).T  # (N, 3)
    in_front = pts_cam[:, 2] > 0

    pixels = np.full((N, 2), np.nan)
    if in_front.any():
        zero_rvec = np.zeros(3, dtype=np.float64)
        zero_tvec = np.zeros(3, dtype=np.float64)
        proj, _ = cv2.projectPoints(
            pts_cam[in_front].astype(np.float64),
            zero_rvec, zero_tvec,
            np.asarray(K, dtype=np.float64),
            np.asarray(D, dtype=np.float64),
        )
        pixels[in_front] = proj.squeeze(1)

    return pts_cam, pixels, in_front
