"""M0b: validates _apply_transform_to_buffers() applies (R, t) in lockstep.

Covers the Risk #8 silent-drift hazard: if any buffer is missed at a
transform site, that buffer silently rotates relative to merged.

Quaternion convention: the codebase stores object orientations as
scipy (x, y, z, w) — see cloud_slam/box_refiner.py ('format': 'xyzw').
The helper uses scipy.spatial.transform.Rotation, matching the existing
call sites in scripts/detect_and_slam.py.
"""
import importlib.util
import os

import numpy as np
import open3d as o3d
import pytest
from scipy.spatial.transform import Rotation as SciRot


def _load_helper():
    # Script is at <repo>/scripts/detect_and_slam.py — anchor via __file__
    # so the test is runnable from any cwd (CI, local editor, etc.).
    script_path = os.path.normpath(os.path.join(
        os.path.dirname(__file__), "..", "..", "scripts", "detect_and_slam.py"
    ))
    spec = importlib.util.spec_from_file_location("detect_and_slam", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._apply_transform_to_buffers


def _make_merged(n=50):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.random.rand(n, 3))
    return pcd


def test_identity_is_noop():
    helper = _load_helper()
    merged = _make_merged()
    wall_labels = {'xyz': np.random.rand(20, 3).astype(np.float32)}
    before = np.asarray(merged.points).copy()
    before_xyz = wall_labels['xyz'].copy()

    helper(np.eye(3), np.zeros(3), merged=merged, wall_labels=wall_labels)

    np.testing.assert_allclose(np.asarray(merged.points), before)
    np.testing.assert_allclose(wall_labels['xyz'], before_xyz)


def test_all_buffers_transform_in_lockstep():
    """If a helper call is 'lockstep', every buffer must pick up the same R."""
    helper = _load_helper()
    # 90-degree rotation around Z
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)

    merged = _make_merged()
    wall_labels = {
        'xyz': np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
    }
    # Identity quaternion in scipy xyzw ordering.
    objects = [{
        'center': np.array([1.0, 0.0, 0.0]),
        'orientation': {'quaternion': np.array([0.0, 0.0, 0.0, 1.0])},
    }]
    lines = {
        'start': np.array([[1.0, 0.0, 0.0]]),
        'end':   np.array([[0.0, 1.0, 0.0]]),
    }

    before_merged = np.asarray(merged.points).copy()

    helper(R, np.zeros(3),
           merged=merged, wall_labels=wall_labels,
           objects=objects, lines=lines)

    # merged rotated in place
    np.testing.assert_allclose(
        np.asarray(merged.points), (R @ before_merged.T).T, atol=1e-9)
    # wall_labels rotated: (1,0,0) -> (0,1,0), (0,1,0) -> (-1,0,0)
    np.testing.assert_allclose(wall_labels['xyz'][0], [0.0, 1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(wall_labels['xyz'][1], [-1.0, 0.0, 0.0], atol=1e-6)
    # object center rotated same way
    np.testing.assert_allclose(
        np.asarray(objects[0]['center']), [0.0, 1.0, 0.0], atol=1e-9)
    # object quaternion: identity composed with R gives R's quaternion.
    q_expected = SciRot.from_matrix(R).as_quat()
    np.testing.assert_allclose(
        np.asarray(objects[0]['orientation']['quaternion']),
        q_expected, atol=1e-9)
    # line endpoints rotated
    np.testing.assert_allclose(lines['start'][0], [0.0, 1.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(lines['end'][0], [-1.0, 0.0, 0.0], atol=1e-9)


def test_translation_applies_to_all():
    helper = _load_helper()
    t = np.array([10.0, 20.0, 30.0])
    merged = _make_merged()
    wall_labels = {'xyz': np.array([[0.0, 0.0, 0.0]], dtype=np.float32)}
    objects = [{
        'center': np.array([0.0, 0.0, 0.0]),
        'orientation': {'quaternion': np.array([0.0, 0.0, 0.0, 1.0])},
    }]
    lines = {
        'start': np.array([[0.0, 0.0, 0.0]]),
        'end':   np.array([[0.0, 0.0, 0.0]]),
    }

    before = np.asarray(merged.points).copy()

    helper(np.eye(3), t,
           merged=merged, wall_labels=wall_labels,
           objects=objects, lines=lines)

    np.testing.assert_allclose(np.asarray(merged.points), before + t, atol=1e-9)
    np.testing.assert_allclose(wall_labels['xyz'][0], t, atol=1e-6)
    np.testing.assert_allclose(np.asarray(objects[0]['center']), t, atol=1e-9)
    # Translation must NOT modify the quaternion.
    np.testing.assert_allclose(
        np.asarray(objects[0]['orientation']['quaternion']),
        [0.0, 0.0, 0.0, 1.0], atol=1e-9)
    np.testing.assert_allclose(lines['start'][0], t, atol=1e-9)
    np.testing.assert_allclose(lines['end'][0], t, atol=1e-9)


def test_none_buffers_skipped():
    helper = _load_helper()
    # Should not raise regardless of which subset of buffers is provided.
    helper(np.eye(3), np.zeros(3))              # nothing provided
    helper(np.eye(3), None)                      # no translation
    helper(np.eye(3), np.zeros(3), merged=None)  # explicit None
    # Empty wall_labels / lines / objects iterables must also be safe.
    helper(np.eye(3), None,
           wall_labels={'xyz': np.zeros((0, 3), dtype=np.float32)},
           lines={'start': np.zeros((0, 3)), 'end': np.zeros((0, 3))},
           objects=[])
