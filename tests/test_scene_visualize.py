"""Tests for scene_visualize.build_overlay — overlay line geometry for V2 PLYs."""

import numpy as np
import pytest

from cloud_slam.detectors.scene import Bbox, Door, Scene, Wall
from cloud_slam.detectors.scene_visualize import build_overlay


def _bbox(id_: int, cls: str, x, y, z, *, yaw=0.0, sx=1.0, sy=1.0, sz=1.0) -> Bbox:
    return Bbox(id=id_, class_name=cls,
                position_x=x, position_y=y, position_z=z,
                angle_z=yaw, scale_x=sx, scale_y=sy, scale_z=sz)


def _wall(id_: int, ax, ay, bx, by, *, h=2.5, t=0.0, az=0.0, bz=0.0) -> Wall:
    return Wall(id=id_, ax=ax, ay=ay, az=az, bx=bx, by=by, bz=bz,
                height=h, thickness=t)


def test_empty_scene_empty_overlay():
    pts, cols = build_overlay(Scene())
    assert pts.shape == (0, 3)
    assert cols.shape == (0, 3)
    assert pts.dtype == np.float64
    assert cols.dtype == np.uint8


def test_single_bbox_nonempty_and_red():
    scene = Scene()
    scene.bboxes.append(_bbox(0, "chair", 1.0, 2.0, 0.5, sx=0.6, sy=0.6, sz=1.0))
    pts, cols = build_overlay(scene)
    assert pts.shape[0] > 0
    assert pts.shape[1] == 3
    # All colors red (255,0,0).
    assert np.all(cols[:, 0] == 255)
    assert np.all(cols[:, 1] == 0)
    assert np.all(cols[:, 2] == 0)
    # All points within the bbox AABB (axis-aligned since yaw=0), with small slack.
    slack = 1e-6
    assert np.all(pts[:, 0] >= 1.0 - 0.3 - slack)
    assert np.all(pts[:, 0] <= 1.0 + 0.3 + slack)
    assert np.all(pts[:, 1] >= 2.0 - 0.3 - slack)
    assert np.all(pts[:, 1] <= 2.0 + 0.3 + slack)
    assert np.all(pts[:, 2] >= 0.5 - 0.5 - slack)
    assert np.all(pts[:, 2] <= 0.5 + 0.5 + slack)


def test_single_axis_aligned_wall_rectangle_shape():
    scene = Scene()
    # Wall from (0,0,0) to (2,0,0), height 2.5, thickness 0 (drawn as zero-thick).
    scene.walls.append(_wall(0, 0.0, 0.0, 2.0, 0.0, h=2.5))
    pts, cols = build_overlay(scene)
    assert pts.shape[0] > 0
    # All blue.
    assert np.all(cols[:, 0] == 0)
    assert np.all(cols[:, 1] == 120)
    assert np.all(cols[:, 2] == 255)
    # Bounds: x in [0,2], y ≈ 0 (thickness zero), z in [0, 2.5].
    tol = 1e-6
    assert pts[:, 0].min() >= -tol
    assert pts[:, 0].max() <= 2.0 + tol
    assert pytest.approx(pts[:, 1].min(), abs=tol) == 0.0
    assert pytest.approx(pts[:, 1].max(), abs=tol) == 0.0
    assert pts[:, 2].min() >= -tol
    assert pts[:, 2].max() <= 2.5 + tol
    # Length span should be 2.0 along x, height span 2.5 along z.
    assert pytest.approx(pts[:, 0].max() - pts[:, 0].min(), abs=1e-6) == 2.0
    assert pytest.approx(pts[:, 2].max() - pts[:, 2].min(), abs=1e-6) == 2.5


def test_door_flush_to_wall():
    scene = Scene()
    scene.walls.append(_wall(0, 0.0, 0.0, 3.0, 0.0, h=2.5))
    scene.doors.append(Door(id=0, wall_id=0,
                            position_x=1.5, position_y=0.0, position_z=1.0,
                            width=0.9, height=2.1))
    pts, cols = build_overlay(scene)
    # Door points only (isolate by color = green).
    green = (cols[:, 0] == 0) & (cols[:, 1] == 255) & (cols[:, 2] == 0)
    door_pts = pts[green]
    assert door_pts.shape[0] > 0
    # All door points lie on wall plane y=0.
    assert np.max(np.abs(door_pts[:, 1])) < 1e-6
    # Door x bounds: [1.5 - 0.45, 1.5 + 0.45]
    assert pytest.approx(door_pts[:, 0].min(), abs=1e-6) == 1.05
    assert pytest.approx(door_pts[:, 0].max(), abs=1e-6) == 1.95
    # Door z bounds: [1.0 - 1.05, 1.0 + 1.05]
    assert pytest.approx(door_pts[:, 2].min(), abs=1e-6) == -0.05
    assert pytest.approx(door_pts[:, 2].max(), abs=1e-6) == 2.05


def test_confidence_filter_drops_weak_bbox():
    scene = Scene()
    scene.bboxes.append(_bbox(0, "chair", 0.0, 0.0, 0.5))
    scene.bboxes.append(_bbox(1, "chair", 5.0, 5.0, 0.5))
    meta = {
        "walls": [], "doors": [], "windows": [],
        "bboxes": [
            {"id": 0, "observations": 3, "confidence": 0.9,
             "first_pass": 0, "last_pass": 2},
            {"id": 1, "observations": 1, "confidence": 0.3,
             "first_pass": 0, "last_pass": 0},
        ],
    }
    pts_all, _ = build_overlay(scene, element_meta=meta)
    pts_filt, _ = build_overlay(scene, element_meta=meta, min_observations=2)
    # Two identical-sized bboxes → roughly half the points survive the filter.
    assert pts_filt.shape[0] > 0
    assert pts_filt.shape[0] < pts_all.shape[0]
    assert abs(pts_filt.shape[0] - pts_all.shape[0] / 2) <= pts_all.shape[0] * 0.05


def test_edge_spacing_affects_point_count():
    scene = Scene()
    scene.bboxes.append(_bbox(0, "chair", 0.0, 0.0, 0.5, sx=1.0, sy=1.0, sz=1.0))
    pts_dense, _ = build_overlay(scene, edge_spacing=0.1)
    pts_sparse, _ = build_overlay(scene, edge_spacing=0.5)
    assert pts_sparse.shape[0] < pts_dense.shape[0]
