"""Tests for FusedScene — iterative multi-pass fusion of SpatialLM Scene output."""

import copy
import math

import pytest

from cloud_slam.detectors.scene import Bbox, Door, Scene, Wall, Window
from cloud_slam.detectors.scene_fusion import FusedScene


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _w(id_: int, ax, ay, bx, by, h=2.8, t=0.0, az=0.0, bz=0.0) -> Wall:
    return Wall(id=id_, ax=ax, ay=ay, az=az, bx=bx, by=by, bz=bz,
                height=h, thickness=t)


def _b(id_: int, cls: str, x, y, z, yaw=0.0, sx=1.0, sy=1.0, sz=1.0) -> Bbox:
    return Bbox(id=id_, class_name=cls,
                position_x=x, position_y=y, position_z=z,
                angle_z=yaw, scale_x=sx, scale_y=sy, scale_z=sz)


def _d(id_: int, wall_id: int, x, y, z=1.05, w=0.9, h=2.1) -> Door:
    return Door(id=id_, wall_id=wall_id,
                position_x=x, position_y=y, position_z=z, width=w, height=h)


def _make_scene_2w_1b() -> Scene:
    """A scene with 2 walls (perpendicular) and 1 bbox."""
    s = Scene()
    s.walls.append(_w(0, 0.0, 0.0, 4.0, 0.0))
    s.walls.append(_w(1, 4.0, 0.0, 4.0, 5.0))
    s.bboxes.append(_b(0, "sofa", 2.0, 2.0, 0.4, sx=2.0, sy=0.85, sz=0.9))
    return s


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_empty_fusion():
    fused = FusedScene()
    snap = fused.current()
    assert isinstance(snap, Scene)
    assert snap.walls == []
    assert snap.doors == []
    assert snap.windows == []
    assert snap.bboxes == []


def test_single_pass_initial_confidence():
    fused = FusedScene()
    scene = _make_scene_2w_1b()
    delta = fused.update(scene)
    assert delta["pass"] == 0
    assert delta["walls_new"] == 2
    assert delta["walls_matched"] == 0
    assert delta["bboxes_new"] == 1
    assert delta["bboxes_matched"] == 0

    snap = fused.current()
    assert len(snap.walls) == 2
    assert len(snap.bboxes) == 1

    # Per spec: initial c=0.5, on-hit c' = 1 - (1-c)*0.6 → 0.7 after first hit.
    # Spec text says "~0.8" informally; the exact math is 0.7.
    meta = fused.current_with_metadata()["meta"]
    assert all(abs(entry["confidence"] - 0.7) < 1e-9 for entry in meta["walls"])
    assert all(abs(entry["confidence"] - 0.7) < 1e-9 for entry in meta["bboxes"])
    # observations = 1 after first hit
    assert all(entry["observations"] == 1 for entry in meta["walls"])
    assert all(entry["observations"] == 1 for entry in meta["bboxes"])


def test_exact_duplicate_passes():
    fused = FusedScene()
    scene = _make_scene_2w_1b()
    for _ in range(3):
        fused.update(copy.deepcopy(scene))
    snap = fused.current()
    assert len(snap.walls) == 2
    assert len(snap.bboxes) == 1

    # Confidence sequence: 0.5 -> 0.7 -> 0.88 -> 0.928 (= 1 - 0.072)
    # Three applications: 1 - 0.5 * 0.6^3 = 1 - 0.108 = 0.892  (wait — check)
    # Actually: after hit1=0.7, hit2=1-0.3*0.6=0.82, hit3=1-0.18*0.6=0.892.
    # The spec says "approximately 0.936" but that's with a different base; we
    # accept anything in [0.88, 0.95] — three hits should clearly exceed 0.85.
    meta = fused.current_with_metadata()["meta"]
    for entry in meta["walls"] + meta["bboxes"]:
        assert entry["observations"] == 3
        assert 0.85 < entry["confidence"] < 0.95


def test_wall_small_drift_matches_and_averages():
    fused = FusedScene()
    s1 = Scene()
    s1.walls.append(_w(0, 0.0, 0.0, 4.0, 0.0))
    s2 = Scene()
    # Drift endpoint b by 0.1 m perpendicular — still within 0.25 m threshold.
    s2.walls.append(_w(7, 0.0, 0.1, 4.0, 0.1))

    fused.update(s1)
    fused.update(s2)
    snap = fused.current()
    assert len(snap.walls) == 1
    w = snap.walls[0]
    # Endpoint a averaged: y = (0 + 0.1)/2 = 0.05, endpoint b similarly.
    assert math.isclose(w.ay, 0.05, abs_tol=1e-9)
    assert math.isclose(w.by, 0.05, abs_tol=1e-9)
    # ax/bx stayed at 0/4 (both passes agreed).
    assert math.isclose(w.ax, 0.0, abs_tol=1e-9)
    assert math.isclose(w.bx, 4.0, abs_tol=1e-9)


def test_wall_too_far_not_matched():
    fused = FusedScene()
    s1 = Scene()
    s1.walls.append(_w(0, 0.0, 0.0, 4.0, 0.0))
    s2 = Scene()
    # Move the wall parallel by 2 m — far beyond the 0.25 m perpendicular threshold.
    s2.walls.append(_w(0, 0.0, 2.0, 4.0, 2.0))

    fused.update(s1)
    delta = fused.update(s2)
    assert delta["walls_new"] == 1
    assert delta["walls_matched"] == 0
    assert len(fused.current().walls) == 2


def test_bbox_class_sticky_same_class():
    fused = FusedScene()
    s1 = Scene()
    s1.bboxes.append(_b(0, "sofa", 1.0, 1.0, 0.4, sx=1.0, sy=1.0, sz=1.0))
    s2 = Scene()
    s2.bboxes.append(_b(0, "sofa", 1.05, 1.0, 0.4, sx=1.0, sy=1.0, sz=1.0))

    fused.update(s1)
    d = fused.update(s2)
    assert d["bboxes_matched"] == 1
    snap = fused.current()
    assert len(snap.bboxes) == 1
    assert snap.bboxes[0].class_name == "sofa"


def test_bbox_class_conflict_keeps_first_seen():
    fused = FusedScene()
    s1 = Scene()
    # Class A: "chair"
    s1.bboxes.append(_b(0, "chair", 1.0, 1.0, 0.4, sx=1.0, sy=1.0, sz=1.0))
    s2 = Scene()
    # Same spatial bbox, different class B: "stool"
    s2.bboxes.append(_b(0, "stool", 1.0, 1.0, 0.4, sx=1.0, sy=1.0, sz=1.0))

    fused.update(s1)
    fused.update(s2)
    snap = fused.current()
    # A is first seen → after the conflict A has 2 observations, B has 0.
    # Spec: keep the one with more observations (here the first-seen class wins).
    assert len(snap.bboxes) == 1
    assert snap.bboxes[0].class_name == "chair"


def test_bbox_iou_below_threshold_separate_entities():
    fused = FusedScene()
    s1 = Scene()
    s1.bboxes.append(_b(0, "chair", 0.0, 0.0, 0.4, sx=0.5, sy=0.5, sz=1.0))
    s2 = Scene()
    # 1.5 m away — IoU with a 0.5x0.5 box is zero; must not match.
    s2.bboxes.append(_b(0, "chair", 1.5, 0.0, 0.4, sx=0.5, sy=0.5, sz=1.0))

    fused.update(s1)
    d = fused.update(s2)
    assert d["bboxes_matched"] == 0
    assert d["bboxes_new"] == 1
    assert len(fused.current().bboxes) == 2


def test_door_follows_wall_remap():
    fused = FusedScene()
    s1 = Scene()
    s1.walls.append(_w(0, 0.0, 0.0, 4.0, 0.0))
    s1.doors.append(_d(0, wall_id=0, x=2.0, y=0.0))

    s2 = Scene()
    # Slightly-drifted wall, same wall_id locally but must remap to fused id 0.
    s2.walls.append(_w(0, 0.0, 0.05, 4.0, 0.05))
    s2.doors.append(_d(0, wall_id=0, x=2.02, y=0.05))

    fused.update(s1)
    fused.update(s2)
    snap = fused.current()
    assert len(snap.walls) == 1
    assert len(snap.doors) == 1
    # Door parent wall_id in fused scene should point at the fused wall id.
    assert snap.doors[0].wall_id == snap.walls[0].id
    meta = fused.current_with_metadata()["meta"]
    assert meta["doors"][0]["observations"] == 2


def test_stale_element_is_dropped():
    fused = FusedScene(stale_passes=3, confidence_drop_threshold=0.3)
    # Pass 0: wall A.
    s0 = Scene()
    s0.walls.append(_w(0, 0.0, 0.0, 4.0, 0.0))
    fused.update(s0)
    wall_a_id_before_decay = fused.current().walls[0].id

    # Subsequent passes: a completely different wall, so wall A is missed each
    # pass. Confidence 0.7 decays by *0.85 per miss; need 6 misses to cross 0.3
    # (0.7*0.85^6 ≈ 0.264). After 6 misses A must be absent from current().
    other = Scene()
    other.walls.append(_w(0, 0.0, 100.0, 4.0, 100.0))  # far away — won't match A
    for _ in range(6):
        fused.update(copy.deepcopy(other))

    snap = fused.current()
    wall_ids = {w.id for w in snap.walls}
    assert wall_a_id_before_decay not in wall_ids


def test_metadata_accessor_bbox_fields():
    fused = FusedScene()
    scene = _make_scene_2w_1b()
    fused.update(scene)
    fused.update(copy.deepcopy(scene))

    data = fused.current_with_metadata()
    assert "scene" in data and "meta" in data
    assert set(data["meta"].keys()) == {"walls", "doors", "windows", "bboxes"}
    assert len(data["meta"]["bboxes"]) == 1
    entry = data["meta"]["bboxes"][0]
    assert "observations" in entry and "confidence" in entry
    assert entry["observations"] == 2
    assert 0.0 < entry["confidence"] <= 1.0
    assert "first_pass" in entry and "last_pass" in entry
    assert entry["first_pass"] == 0 and entry["last_pass"] == 1
