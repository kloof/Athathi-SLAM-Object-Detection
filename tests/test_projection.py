"""Unit tests for cloud_slam.projection.project_world_to_image."""

import numpy as np

from cloud_slam.projection import project_lidar_to_camera, project_world_to_image


def _pinhole(w=1280, h=720, f=900.0):
    K = np.array([[f, 0.0, w / 2.0],
                  [0.0, f, h / 2.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    D = np.zeros(5, dtype=np.float64)
    return K, D, (w, h)


def test_point_in_front_projects_to_center():
    """A point on the camera +Z axis should project to the principal point."""
    K, D, (w, h) = _pinhole()
    T_cam_world = np.eye(4)  # camera == world
    xyz_world = np.array([[0.0, 0.0, 2.0]])  # directly in front, 2m away
    pts_cam, pixels, in_front = project_world_to_image(
        xyz_world, T_cam_world, K, D)

    assert in_front[0]
    assert np.allclose(pts_cam[0], [0.0, 0.0, 2.0])
    assert np.allclose(pixels[0], [w / 2.0, h / 2.0], atol=1e-6)


def test_point_behind_is_not_in_front():
    K, D, _ = _pinhole()
    T_cam_world = np.eye(4)
    xyz_world = np.array([[0.0, 0.0, -1.0]])
    _pts, pixels, in_front = project_world_to_image(
        xyz_world, T_cam_world, K, D)

    assert not in_front[0]
    assert np.all(np.isnan(pixels[0]))


def test_translated_world_pose():
    """When the camera is translated, points in front of the camera
    (in camera frame) still project near principal point."""
    K, D, (w, h) = _pinhole()
    # camera placed at world (1, 2, 3), looking along world +X
    # T_cam_world maps world -> camera; we want a point at world (4, 2, 3)
    # to land at the principal point (dz=3m forward along cam +Z).
    # Build T_cam_world from camera pose in the world: R_wc (cam axes in world),
    # t_wc (cam origin in world). Then T_cam_world = inv([R_wc, t_wc; 0, 1]).
    # Camera axes in world: cam-X = -world-Y, cam-Y = -world-Z, cam-Z = +world-X.
    # Columns are cam-X, cam-Y, cam-Z expressed in world coords:
    # cam-X = world-(-Y), cam-Y = world-(-Z), cam-Z = world-(+X).
    R_wc = np.array([[0.0, 0.0, 1.0],
                     [-1.0, 0.0, 0.0],
                     [0.0, -1.0, 0.0]])
    t_wc = np.array([1.0, 2.0, 3.0])
    T_wc = np.eye(4); T_wc[:3, :3] = R_wc; T_wc[:3, 3] = t_wc
    T_cw = np.linalg.inv(T_wc)

    xyz_world = np.array([[4.0, 2.0, 3.0]])  # 3m along world+X ahead of cam
    pts_cam, pixels, in_front = project_world_to_image(xyz_world, T_cw, K, D)

    assert in_front[0]
    assert np.allclose(pts_cam[0], [0.0, 0.0, 3.0], atol=1e-6)
    assert np.allclose(pixels[0], [w / 2.0, h / 2.0], atol=1e-6)


def test_matches_project_lidar_to_camera_when_pose_is_extrinsic():
    """Feeding the calibration extrinsic directly as T_cam_world must give
    the same pixels as the existing project_lidar_to_camera helper."""
    K, D, _ = _pinhole()
    # Match style of existing tests: lidar -> camera rotation.
    T_lidar_cam = np.eye(4)
    T_lidar_cam[:3, :3] = np.array([[0.0, -1.0, 0.0],
                                     [0.0, 0.0, -1.0],
                                     [1.0, 0.0, 0.0]])
    calib = {"K": K, "dist_coeffs": D, "T_lidar_cam": T_lidar_cam}

    xyz = np.array([[2.0, 0.1, 0.2], [5.0, -0.3, 0.4], [-1.0, 0.0, 0.0]])
    pts_cam_a, px_a, inf_a = project_lidar_to_camera(xyz, calib)
    pts_cam_b, px_b, inf_b = project_world_to_image(xyz, T_lidar_cam, K, D)

    assert np.array_equal(inf_a, inf_b)
    assert np.allclose(pts_cam_a, pts_cam_b, atol=1e-9)
    # NaN-safe compare
    mask = ~np.isnan(px_a).any(axis=1)
    assert np.allclose(px_a[mask], px_b[mask], atol=1e-6)
