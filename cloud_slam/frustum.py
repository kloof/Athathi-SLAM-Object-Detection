"""
Frustum-based 3D object extraction from lidar + 2D detections.

Projects lidar points into camera image, extracts points within each
2D detection (mask or bbox), filters depth outliers, and fits 3D bounding boxes.
"""

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from cloud_slam.projection import project_lidar_to_camera


def extract_frustum_points(xyz, detection, calib, padding=15):
    """
    Extract lidar points that project inside a 2D detection.

    Uses segmentation mask when available, falls back to bbox + padding.

    Args:
        xyz:       (N, 3) float64, points in lidar frame
        detection: Detection object with bbox_xyxy and optional mask
        calib:     dict from load_calibration()
        padding:   pixel padding around bbox for calibration tolerance

    Returns:
        indices: array of point indices within the frustum
    """
    if len(xyz) == 0:
        return np.array([], dtype=int)

    pts_cam, pixels, in_front = project_lidar_to_camera(xyz, calib)
    W, H = calib['image_size']

    # Start with points in front of camera and within image bounds
    u = pixels[:, 0]
    v = pixels[:, 1]
    valid = in_front & (u >= 0) & (u < W) & (v >= 0) & (v < H)

    if not valid.any():
        return np.array([], dtype=int)

    if detection.mask is not None:
        # Use segmentation mask for tighter extraction
        mask_h, mask_w = detection.mask.shape
        ui = np.clip(u[valid].astype(int), 0, mask_w - 1)
        vi = np.clip(v[valid].astype(int), 0, mask_h - 1)
        in_mask = detection.mask[vi, ui] > 0.5
        valid_indices = np.where(valid)[0]
        return valid_indices[in_mask]

    # Fall back to bbox + padding
    x1, y1, x2, y2 = detection.bbox_xyxy
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(W - 1, x2 + padding)
    y2 = min(H - 1, y2 + padding)

    in_bbox = valid & (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
    return np.where(in_bbox)[0]


def filter_depth_mad(points, scale=3.0):
    """
    Filter depth outliers using MAD (Median Absolute Deviation).

    Keeps points within median ± scale * MAD * 1.4826.
    Better than DBSCAN/histogram for sparse data (12-80 pts/frustum).

    Args:
        points: (M, 3) float64
        scale:  number of sigma-equivalents

    Returns:
        filtered: (M', 3) float64
    """
    if len(points) < 3:
        return points

    # Use the depth along the first principal axis (approximated by Z in camera frame)
    # But points are in lidar frame, so use distance from origin as depth proxy
    depths = np.linalg.norm(points, axis=1)
    median_d = np.median(depths)
    mad = np.median(np.abs(depths - median_d))

    if mad < 1e-6:
        return points  # All at same depth, no filtering needed

    threshold = scale * mad * 1.4826  # MAD → sigma conversion
    mask = np.abs(depths - median_d) < threshold
    return points[mask]


def estimate_gravity(imus, n_samples=50):
    """
    Estimate gravity direction from IMU accelerometer data.

    At rest or slow motion, accelerometer ≈ gravity vector.

    Args:
        imus: list of (timestamp, gyro_xyz, acc_xyz)
        n_samples: number of initial samples to average

    Returns:
        gravity_up: (3,) unit vector pointing up (opposite to gravity acceleration)
    """
    if not imus:
        return np.array([0.0, 0.0, 1.0])  # Default: Z-up

    n = min(n_samples, len(imus))
    acc_samples = np.array([acc for _, _, acc in imus[:n]])
    gravity = acc_samples.mean(axis=0)
    norm = np.linalg.norm(gravity)

    if norm < 5.0 or norm > 15.0:
        # Unusual magnitude — fall back to Z-up
        return np.array([0.0, 0.0, 1.0])

    gravity_up = gravity / norm

    # IMU convention check: accelerometer at rest should report the support
    # force (pointing UP). Some IMUs report gravity direction (pointing DOWN).
    # For a roughly-upright sensor, gravity_up.z should be positive.
    if gravity_up[2] < 0:
        gravity_up = -gravity_up

    return gravity_up


def fit_gravity_aligned_obb(points, gravity_up, min_points=10):
    """
    Fit a gravity-aligned oriented bounding box to a point cluster.

    Rotates points so gravity→Z, fits 2D min-rect on XY + full Z range.

    Args:
        points:     (M, 3) float64, world-frame points
        gravity_up: (3,) unit vector for "up" direction
        min_points: minimum points for OBB fitting

    Returns:
        dict with {center, dimensions, rotation_quat_xyzw, confidence, num_points}
        or None if insufficient points
    """
    if len(points) < min_points:
        if len(points) < 3:
            return None
        # Low confidence: axis-aligned bounding box
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        aabb = pcd.get_axis_aligned_bounding_box()
        center = aabb.get_center()
        extent = aabb.get_extent()
        return {
            'center': center,
            'dimensions': np.maximum(extent, 0.02),
            'rotation_quat_xyzw': np.array([0.0, 0.0, 0.0, 1.0]),
            'confidence': 'low',
            'num_points': len(points),
        }

    # Compute rotation to align gravity_up with Z axis
    z_axis = np.array([0.0, 0.0, 1.0])
    if np.allclose(gravity_up, z_axis, atol=0.01):
        R_align = np.eye(3)
    elif np.allclose(gravity_up, -z_axis, atol=0.01):
        R_align = np.diag([1.0, -1.0, -1.0])
    else:
        rot = Rotation.align_vectors([z_axis], [gravity_up])[0]
        R_align = rot.as_matrix()

    # Rotate points so gravity is along Z
    aligned = (R_align @ points.T).T

    # Fit 2D minimum bounding rectangle on XY plane
    pcd_2d = o3d.geometry.PointCloud()
    xy_points = np.column_stack([aligned[:, :2], np.zeros(len(aligned))])
    pcd_2d.points = o3d.utility.Vector3dVector(xy_points)

    try:
        obb_2d = pcd_2d.get_minimal_oriented_bounding_box(robust=True)
    except Exception:
        obb_2d = pcd_2d.get_axis_aligned_bounding_box()
        center_2d = obb_2d.get_center()[:2]
        extent_2d = obb_2d.get_extent()[:2]
        yaw = 0.0
    else:
        center_2d = np.asarray(obb_2d.center)[:2]
        extent_2d = np.asarray(obb_2d.extent)[:2]
        # Extract yaw from the OBB rotation (fix det=-1 from Open3D)
        R_obb = np.asarray(obb_2d.R).copy()
        if np.linalg.det(R_obb) < 0:
            R_obb[:, 2] *= -1
        yaw = np.arctan2(R_obb[1, 0], R_obb[0, 0])

    # Z range
    z_min = aligned[:, 2].min()
    z_max = aligned[:, 2].max()
    z_center = (z_min + z_max) / 2
    z_extent = z_max - z_min

    # Build the gravity-aligned center and dimensions
    center_aligned = np.array([center_2d[0], center_2d[1], z_center])
    dimensions = np.maximum(
        np.array([extent_2d[0], extent_2d[1], z_extent]),
        0.02,  # Minimum 2cm for planar objects
    )

    # Build rotation: yaw around gravity axis
    R_yaw = Rotation.from_euler('z', yaw).as_matrix()
    # Full rotation: undo gravity alignment, then apply yaw
    R_full = R_align.T @ R_yaw
    quat = Rotation.from_matrix(R_full).as_quat()  # (x, y, z, w)

    # Transform center back to world frame
    center_world = R_align.T @ center_aligned

    confidence = 'high' if len(points) >= 30 else 'medium'

    return {
        'center': center_world,
        'dimensions': dimensions,
        'rotation_quat_xyzw': quat,
        'confidence': confidence,
        'num_points': len(points),
    }
