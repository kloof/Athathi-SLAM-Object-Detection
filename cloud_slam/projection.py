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
