"""
Manhattan World alignment for indoor 3D object detection.

Estimates the 3 dominant orthogonal directions in a scene from wall normals,
then fits OBBs constrained to those directions.
"""

import numpy as np
from dataclasses import dataclass
from scipy.spatial.transform import Rotation


@dataclass
class ManhattanFrame:
    """Three mutually-orthogonal dominant directions."""
    R: np.ndarray          # (3,3) rotation matrix: world → Manhattan frame
    confidence: float      # 0-1, how well scene fits Manhattan assumption


def estimate_manhattan_frame(walls, gravity_up):
    """
    Estimate Manhattan frame from detected wall normals + gravity.

    Args:
        walls: list of Plane objects with .normal attributes
        gravity_up: (3,) unit vector for "up"

    Returns:
        ManhattanFrame with rotation matrix and confidence score
    """
    if len(walls) < 1:
        return ManhattanFrame(R=_gravity_frame(gravity_up), confidence=0.0)

    # Project wall normals onto horizontal plane
    horiz_normals = []
    for wall in walls:
        n = wall.normal.copy()
        # Remove gravity component
        n -= (n @ gravity_up) * gravity_up
        norm = np.linalg.norm(n)
        if norm > 0.1:
            horiz_normals.append(n / norm)

    if not horiz_normals:
        return ManhattanFrame(R=_gravity_frame(gravity_up), confidence=0.0)

    horiz_normals = np.array(horiz_normals)

    # Fold to 0-180° range (normals are undirected)
    angles = np.arctan2(horiz_normals[:, 1], horiz_normals[:, 0])
    angles = angles % np.pi  # fold to [0, π)

    # Histogram of azimuth angles (1° bins)
    n_bins = 180
    hist, bin_edges = np.histogram(angles, bins=n_bins, range=(0, np.pi),
                                    weights=[w.num_inliers for w in walls[:len(horiz_normals)]])

    # Find peak
    peak_bin = np.argmax(hist)
    peak_angle = (bin_edges[peak_bin] + bin_edges[peak_bin + 1]) / 2

    # Refine: mean of normals within ±10° of peak
    tolerance = np.radians(10)
    diffs = np.minimum(np.abs(angles - peak_angle),
                       np.pi - np.abs(angles - peak_angle))
    near_peak = diffs < tolerance
    if near_peak.any():
        # Weighted mean angle (weight by wall inlier count)
        weights = np.array([walls[i].num_inliers for i in range(len(horiz_normals))])[near_peak]
        peak_angle = np.average(angles[near_peak], weights=weights)

    # Compute confidence: fraction of inliers near a Manhattan direction
    near_0 = diffs < tolerance
    near_90 = np.minimum(np.abs(angles - (peak_angle + np.pi/2) % np.pi),
                          np.pi - np.abs(angles - (peak_angle + np.pi/2) % np.pi)) < tolerance
    total_inliers = sum(w.num_inliers for w in walls[:len(horiz_normals)])
    manhattan_inliers = sum(walls[i].num_inliers for i in range(len(horiz_normals))
                           if near_0[i] or near_90[i])
    confidence = manhattan_inliers / max(total_inliers, 1)

    # Build rotation matrix: Manhattan → World
    # Axis 0: dominant wall direction (horizontal)
    ax0 = np.array([np.cos(peak_angle), np.sin(peak_angle), 0.0])
    # Axis 2: gravity up
    ax2 = gravity_up / np.linalg.norm(gravity_up)
    # Axis 1: perpendicular (right-hand rule)
    ax1 = np.cross(ax2, ax0)
    ax1 /= np.linalg.norm(ax1)
    # Recompute ax0 for exact orthogonality
    ax0 = np.cross(ax1, ax2)

    R = np.column_stack([ax0, ax1, ax2])  # columns = Manhattan axes in world
    # Ensure proper rotation (det=+1), not reflection
    if np.linalg.det(R) < 0:
        ax1 = -ax1
        R = np.column_stack([ax0, ax1, ax2])
    R_world_to_manhattan = R.T  # rows = Manhattan axes

    return ManhattanFrame(R=R_world_to_manhattan, confidence=confidence)


def _gravity_frame(gravity_up):
    """Build a rotation matrix with only gravity alignment (no wall info)."""
    z = gravity_up / np.linalg.norm(gravity_up)
    # Pick an arbitrary perpendicular x axis
    if abs(z[0]) < 0.9:
        x = np.cross(z, np.array([1, 0, 0]))
    else:
        x = np.cross(z, np.array([0, 1, 0]))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.row_stack([x, y, z])


def fit_manhattan_obb(points, manhattan_frame, min_points=10):
    """
    Fit an OBB aligned to the Manhattan frame.

    Tests only 0° and 90° rotation in the Manhattan XY plane,
    guaranteeing wall-aligned output.

    Args:
        points: (N, 3) float64, world-frame points
        manhattan_frame: ManhattanFrame from estimate_manhattan_frame()
        min_points: minimum points for OBB

    Returns:
        dict with {center, dimensions, rotation_quat_xyzw, confidence, num_points}
        or None if insufficient points
    """
    if len(points) < 3:
        return None

    R = manhattan_frame.R  # world → Manhattan

    # Rotate points into Manhattan frame
    pts_m = (R @ points.T).T  # (N, 3)

    if len(points) < min_points:
        # Low confidence: AABB in Manhattan frame
        mins = pts_m.min(axis=0)
        maxs = pts_m.max(axis=0)
        center_m = (mins + maxs) / 2
        dims = np.maximum(maxs - mins, 0.02)
        center_w = R.T @ center_m
        R_inv = R.T.copy()
        if np.linalg.det(R_inv) < 0:
            R_inv[:, 2] *= -1
        quat = Rotation.from_matrix(R_inv).as_quat()  # (x,y,z,w)
        return {
            'center': center_w,
            'dimensions': dims,
            'rotation_quat_xyzw': quat,
            'confidence': 'low',
            'num_points': len(points),
        }

    # Test 0° and 90° rotation in Manhattan XY
    best_vol = np.inf
    best_result = None

    for swap in [False, True]:
        if swap:
            # 90° rotation: swap X and Y
            pts_test = pts_m[:, [1, 0, 2]]
            pts_test[:, 0] *= -1  # maintain right-handedness
        else:
            pts_test = pts_m

        mins = pts_test.min(axis=0)
        maxs = pts_test.max(axis=0)
        dims = np.maximum(maxs - mins, 0.02)
        vol = np.prod(dims)

        if vol < best_vol:
            best_vol = vol
            center_test = (mins + maxs) / 2

            if swap:
                # Undo the swap for center
                center_m = np.array([center_test[1], -center_test[0], center_test[2]])
                # Additional 90° rotation around Z
                R_extra = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
                R_total = R.T @ R_extra.T
                # Dimensions stay in rotated frame (matches R_total)
                dims_out = dims.copy()
            else:
                center_m = center_test
                R_total = R.T
                dims_out = dims

            center_w = R.T @ center_m
            if np.linalg.det(R_total) < 0:
                R_total[:, 2] *= -1
            quat = Rotation.from_matrix(R_total).as_quat()

            best_result = {
                'center': center_w,
                'dimensions': dims_out,
                'rotation_quat_xyzw': quat,
                'confidence': 'high' if len(points) >= 30 else 'medium',
                'num_points': len(points),
            }

    return best_result
