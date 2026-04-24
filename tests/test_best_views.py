"""Unit tests for stage 8 (best-view-per-bbox).

Covers the 7 scenarios listed in the design doc's Testing section.
All tests are self-contained — no real rosbag required.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from cloud_slam.spatiallm_pipeline import best_views as bv
from cloud_slam.projection import project_world_to_image


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _pinhole(w=1280, h=720, f=900.0):
    K = np.array([[f, 0.0, w / 2.0],
                  [0.0, f, h / 2.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    D = np.zeros(5, dtype=np.float64)
    return K, D, (w, h)


def _identity_cam_world():
    """Camera sits at world origin, cam axes == world axes (+Z forward)."""
    return np.eye(4)


def _camera_at(xyz, yaw_about_world_z_deg=0.0):
    """T_cam_world for a camera at world position xyz looking +X (cam +Z
    aligned with world +X)."""
    # cam-X = world-(-Y), cam-Y = world-(-Z), cam-Z = world-(+X)
    R_wc = np.array([[0.0, 0.0, 1.0],
                     [-1.0, 0.0, 0.0],
                     [0.0, -1.0, 0.0]], dtype=np.float64)
    if yaw_about_world_z_deg != 0:
        yaw = np.deg2rad(yaw_about_world_z_deg)
        c, s = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[c, -s, 0.0],
                       [s,  c, 0.0],
                       [0.0, 0.0, 1.0]])
        R_wc = Rz @ R_wc
    T_wc = np.eye(4)
    T_wc[:3, :3] = R_wc
    T_wc[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return np.linalg.inv(T_wc)


# ---------------------------------------------------------------------------
# 1. Geometry test — projected AABB matches hand-computed values
# ---------------------------------------------------------------------------

def test_geometry_projected_aabb_matches_hand_math():
    """Axis-aligned 1m cube centered 3m in front of camera projects to the
    predicted pixel AABB."""
    K, D, image_size = _pinhole(w=1280, h=720, f=900.0)
    # camera at world origin, looking along +Z (identity pose)
    T_cw = _identity_cam_world()
    # 1m cube, center (0, 0, 3), yaw 0
    corners = bv._bbox_corners_world(0.0, 0.0, 3.0, 0.0, 1.0, 1.0, 1.0)

    _pts_cam, pixels, in_front = project_world_to_image(corners, T_cw, K, D)
    assert in_front.all()
    valid = pixels[in_front]
    # Front face at Z=2.5 (closest), back face at Z=3.5.
    # x extent in camera: +/- 0.5 m; projected half-width at Z=2.5 is
    # f * 0.5 / 2.5 = 180; at Z=3.5 is f * 0.5 / 3.5 ~= 128.57. Take AABB.
    overlaps, aabb = bv._aabb_intersects_image(valid, image_size)
    assert overlaps
    W, H = image_size
    x0, y0, x1, y1 = aabb
    # Expected x-range: [W/2 - 180, W/2 + 180] (front face dominates AABB)
    assert abs(x0 - (W / 2 - 180.0)) < 1.0
    assert abs(x1 - (W / 2 + 180.0)) < 1.0
    assert abs(y0 - (H / 2 - 180.0)) < 1.0
    assert abs(y1 - (H / 2 + 180.0)) < 1.0


# ---------------------------------------------------------------------------
# 2. Occlusion — clear: empty voxel cloud -> occlusion = 1.0
# ---------------------------------------------------------------------------

def test_occlusion_clear_empty_voxel_cloud():
    T_cw = _identity_cam_world()
    corners = bv._bbox_corners_world(0.0, 0.0, 3.0, 0.0, 1.0, 1.0, 1.0)
    score = bv._occlusion_score(corners, T_cw, kdtree=None,
                                voxel_points_world=None)
    assert score == 1.0

    # Also explicit empty cloud
    from scipy.spatial import cKDTree
    empty = np.empty((0, 3))
    score2 = bv._occlusion_score(corners, T_cw,
                                 kdtree=cKDTree(empty) if len(empty) else None,
                                 voxel_points_world=empty)
    assert score2 == 1.0


# ---------------------------------------------------------------------------
# 3. Occlusion — blocked: points on every corner ray -> occlusion = 0.0
# ---------------------------------------------------------------------------

def test_occlusion_blocked_by_midpoint_occluders():
    T_cw = _identity_cam_world()
    cam_origin = np.array([0.0, 0.0, 0.0])
    corners = bv._bbox_corners_world(0.0, 0.0, 3.0, 0.0, 1.0, 1.0, 1.0)

    # Place an occluder halfway between camera and each corner.
    blockers = []
    for corner in corners:
        blockers.append(cam_origin + 0.5 * (corner - cam_origin))
    blockers = np.asarray(blockers, dtype=np.float64)

    from scipy.spatial import cKDTree
    kdtree = cKDTree(blockers)
    score = bv._occlusion_score(corners, T_cw, kdtree, blockers,
                                radius=0.05, depth_margin=0.10)
    assert score == 0.0


# ---------------------------------------------------------------------------
# 4. Crop clipping — AABB partially off-image
# ---------------------------------------------------------------------------

def test_crop_clipping_off_image():
    image_size = (1280, 720)
    # AABB with negative x0, y0 and x1>W — should clamp and still be non-empty
    aabb = (-50.0, -30.0, 400.0, 200.0)
    x0, y0, x1, y1 = bv._pad_and_clip_aabb(aabb, image_size, pad_frac=0.1)
    assert x0 == 0
    assert y0 == 0
    assert x1 > 0 and x1 <= 1280
    assert y1 > 0 and y1 <= 720
    assert x1 > x0 and y1 > y0


# ---------------------------------------------------------------------------
# 5. Crop padding — 100x100 at image center -> 120x120 after 10% pad each side
# ---------------------------------------------------------------------------

def test_crop_padding_grows_by_10_percent_each_side():
    image_size = (1280, 720)
    cx, cy = 640.0, 360.0
    aabb = (cx - 50.0, cy - 50.0, cx + 50.0, cy + 50.0)
    x0, y0, x1, y1 = bv._pad_and_clip_aabb(aabb, image_size, pad_frac=0.10)
    # width = 100 -> pad 10 each side -> final 120; allow 1 px rounding slop
    assert 118 <= (x1 - x0) <= 122
    assert 118 <= (y1 - y0) <= 122


# ---------------------------------------------------------------------------
# 6. Scoring sanity — closer + centered beats far + off-center
# ---------------------------------------------------------------------------

def test_scoring_prefers_close_centered_frame():
    K, D, image_size = _pinhole()

    # bbox centered at world (0, 0, 0), 1m cube
    bbox_center = np.array([0.0, 0.0, 0.0])
    corners = bv._bbox_corners_world(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0)

    # Frame A: camera 2m in front of box, centered
    # Looking along -world-Z toward box at origin. Use identity-oriented camera
    # (cam+Z aligns with world+Z) placed at (0, 0, -2): then box_center in cam
    # frame is (0, 0, +2) — centered and close.
    T_wc_A = np.eye(4); T_wc_A[:3, 3] = np.array([0.0, 0.0, -2.0])
    T_cw_A = np.linalg.inv(T_wc_A)

    # Frame B: same orientation, camera at (3, 2, -8) — far + off-center.
    T_wc_B = np.eye(4); T_wc_B[:3, 3] = np.array([3.0, 2.0, -8.0])
    T_cw_B = np.linalg.inv(T_wc_B)

    def score(T_cw):
        pts_cam, pixels, in_front = project_world_to_image(corners, T_cw, K, D)
        if not in_front.any():
            return 0.0
        _, aabb = bv._aabb_intersects_image(pixels[in_front], image_size)
        a = bv._area_score(aabb, image_size)
        c = bv._centering_score(aabb, image_size)
        # leave occlusion=1.0, sharpness=0.5 for both
        scores = {"area": a, "centering": c, "occlusion": 1.0, "sharpness": 0.5}
        return bv._composite(scores)

    s_A = score(T_cw_A)
    s_B = score(T_cw_B)
    assert s_A > s_B, f"close+centered={s_A} should beat far+off={s_B}"


# ---------------------------------------------------------------------------
# 7. Skip path — bbox never visible -> manifest has "never_visible"
# ---------------------------------------------------------------------------

def test_never_visible_bbox_is_skipped(tmp_path: Path):
    """Build a synthetic output_dir, run run_best_views, verify the
    manifest marks the bbox 'never_visible' and no crop is emitted."""
    out = tmp_path / "run"
    slam = out / "slam"
    slam.mkdir(parents=True)

    # Place the bbox at world (0, 0, 0). Place the trajectory pose far
    # away looking away so the bbox center never projects in front.
    # trajectory: single pose at world (100, 100, 100) with identity R.
    # After level (identity) + yaw 0, pose in world is itself. The camera
    # sees whatever is in its +Z half-space of its own frame; but we set
    # T_lidar_cam to identity so cam==lidar. Lidar is +Z up at origin, so
    # world origin is at (-100, -100, -100) in lidar frame; cam-Z of a
    # lidar-identity camera means looking along world+Z from (100,100,100),
    # so world origin lies at z = -100 < 0 -> not in front. Perfect for
    # "never visible".
    (slam / "trajectory.csv").write_text(
        "timestamp,x,y,z,qw,qx,qy,qz\n"
        "1.000000,100.000000,100.000000,100.000000,1.000000,0.000000,0.000000,0.000000\n"
        "2.000000,100.000000,100.000000,100.000000,1.000000,0.000000,0.000000,0.000000\n"
    )

    # frames_index with a single camera timestamp inside the trajectory range
    (slam / "frames_index.json").write_text(json.dumps({
        "mcap_path": str(tmp_path / "nonexistent.mcap"),
        "topic": "/camera/image_raw/compressed",
        "frames": [{"t_ns": 1_500_000_000}],  # 1.5 sec, inside [1.0, 2.0]
        "level_rotation": np.eye(3).tolist(),
        "level_z_shift_m": 0.0,
        "manhattan_yaw_deg": 0.0,
    }))

    # layout_merged.txt with ONE bbox at world origin
    (out / "layout_merged.txt").write_text(
        "bbox_0=Bbox(sofa,0.0,0.0,0.0,0.0,1.0,1.0,1.0)\n"
    )

    # Synthetic calibration with identity extrinsic + small intrinsic
    calib_dir = tmp_path / "calibration"
    calib_dir.mkdir()
    (calib_dir / "intrinsics.yaml").write_text(
        "image_width: 1280\n"
        "image_height: 720\n"
        "camera_matrix:\n"
        "  data: [900.0, 0.0, 640.0, 0.0, 900.0, 360.0, 0.0, 0.0, 1.0]\n"
        "distortion_coefficients:\n"
        "  data: [0.0, 0.0, 0.0, 0.0, 0.0]\n"
    )
    # Identity extrinsic (cam==lidar). Quaternion identity.
    (calib_dir / "extrinsics.yaml").write_text(
        "rotation:\n"
        "  w: 1.0\n"
        "  x: 0.0\n"
        "  y: 0.0\n"
        "  z: 0.0\n"
        "translation:\n"
        "  x: 0.0\n"
        "  y: 0.0\n"
        "  z: 0.0\n"
    )

    # No voxel.ply -> occlusion defaults to 1.0

    manifest = bv.run_best_views(
        output_dir=out,
        mcap_path=tmp_path / "nonexistent.mcap",
        calibration_dir=calib_dir,
        verbose=False,
    )

    assert len(manifest["entries"]) == 1
    e = manifest["entries"][0]
    assert e["bbox_id"] == 0
    assert e["class"] == "sofa"
    assert e.get("skipped") == "never_visible"
    # No JPG should have been written
    jpgs = list((out / "best_views").glob("*.jpg"))
    assert jpgs == []


# ---------------------------------------------------------------------------
# 8. Camera-inside-bbox filter — frame with camera inside must be rejected
# ---------------------------------------------------------------------------

def test_camera_inside_bbox_filter():
    """Direct unit test of _camera_inside_bbox for axis-aligned and
    yaw-rotated boxes, including the buffer."""
    # Axis-aligned 2x2x2 box at origin
    assert bv._camera_inside_bbox(
        np.array([0.0, 0.0, 0.0]), 0, 0, 0, 0.0, 2, 2, 2, 0.05) is True
    # Just outside the box on X
    assert bv._camera_inside_bbox(
        np.array([1.1, 0.0, 0.0]), 0, 0, 0, 0.0, 2, 2, 2, 0.05) is False
    # Within the buffer: 2x2x2 + 0.05 buffer -> half-extent 1.05 on every axis
    assert bv._camera_inside_bbox(
        np.array([1.04, 0.0, 0.0]), 0, 0, 0, 0.0, 2, 2, 2, 0.05) is True

    # Yaw-rotated 45 deg around Z: box diagonal now points along world +X.
    # A point at (1.0, 0.0, 0.0) in world is at local (cos45+sin45·0,
    # -sin45·0+cos45·0)·1 = ~(0.707, -0.707) -> still inside a 2x2 footprint.
    yaw = np.pi / 4
    assert bv._camera_inside_bbox(
        np.array([1.0, 0.0, 0.0]), 0, 0, 0, yaw, 2, 2, 2, 0.0) is True
    # After 45 deg rotation, the corner of the 2x2 box sits at world x ≈ sqrt(2).
    # A point at (1.5, 0.0, 0.0) should be outside (local x ≈ 1.06, just past 1).
    assert bv._camera_inside_bbox(
        np.array([1.5, 0.0, 0.0]), 0, 0, 0, yaw, 2, 2, 2, 0.0) is False


def test_scoring_rejects_camera_inside_bbox():
    """Two candidate frames for the same bbox:
       - Frame A: camera inside the bbox (should be rejected outright).
       - Frame B: camera 3 m in front of the bbox (should win).
    The previous scoring picked A because the projected AABB was huge.
    """
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        out = td / "run"
        slam = out / "slam"
        slam.mkdir(parents=True)

        # Bbox at (0,0,0), size 2x2x2.
        (out / "layout_merged.txt").write_text(
            "bbox_0=Bbox(sofa,0.0,0.0,0.0,0.0,2.0,2.0,2.0)\n"
        )

        # Trajectory with two poses:
        #   A: camera inside the bbox at (0.0, 0.0, 0.0).
        #   B: camera 3 m along world -Y, looking along world +Y toward bbox.
        # Quaternion convention: (qw, qx, qy, qz). A 90-deg rotation around
        # world X brings camera +Z (looking axis) to align with world +Y.
        # That matches our "cam looking toward world origin from world -Y".
        import math
        q_w = math.cos(math.pi / 4)
        q_x = math.sin(math.pi / 4)
        (slam / "trajectory.csv").write_text(
            "timestamp,x,y,z,qw,qx,qy,qz\n"
            f"1.000000,0.0,0.0,0.0,{q_w},{q_x},0.0,0.0\n"
            f"2.000000,0.0,-3.0,0.0,{q_w},{q_x},0.0,0.0\n"
        )

        (slam / "frames_index.json").write_text(json.dumps({
            "mcap_path": str(td / "nonexistent.mcap"),
            "topic": "/camera/image_raw/compressed",
            # Two frames: one at A (t=1.0s), one at B (t=2.0s).
            "frames": [
                {"t_ns": 1_000_000_000},
                {"t_ns": 2_000_000_000},
            ],
            "level_rotation": np.eye(3).tolist(),
            "level_z_shift_m": 0.0,
            "manhattan_yaw_deg": 0.0,
        }))

        calib_dir = td / "calibration"
        calib_dir.mkdir()
        (calib_dir / "intrinsics.yaml").write_text(
            "image_width: 1280\n"
            "image_height: 720\n"
            "camera_matrix:\n"
            "  data: [900.0, 0.0, 640.0, 0.0, 900.0, 360.0, 0.0, 0.0, 1.0]\n"
            "distortion_coefficients:\n"
            "  data: [0.0, 0.0, 0.0, 0.0, 0.0]\n"
        )
        (calib_dir / "extrinsics.yaml").write_text(
            "rotation:\n  w: 1.0\n  x: 0.0\n  y: 0.0\n  z: 0.0\n"
            "translation:\n  x: 0.0\n  y: 0.0\n  z: 0.0\n"
        )

        # Monkey-patch read_frames_by_time_ns to supply a synthetic JPEG
        # only for frame B (t_ns = 2_000_000_000). If the scorer asks for
        # frame A, the test fails because frame A should have been
        # filtered out before decode.
        asked = []
        def fake_reader(mcap_path, topic, t_ns_list, **kw):
            asked.extend(list(t_ns_list))
            # Always return a 1280x720 white JPEG for whatever was asked
            buf = np.full((720, 1280, 3), 255, dtype=np.uint8)
            ok, enc = cv2.imencode(".jpg", buf)
            assert ok
            return [(int(t), bytes(enc), "jpeg") for t in t_ns_list]

        import cloud_slam.mcap_reader as mr
        orig = mr.read_frames_by_time_ns
        mr.read_frames_by_time_ns = fake_reader
        try:
            manifest = bv.run_best_views(
                output_dir=out,
                mcap_path=td / "nonexistent.mcap",
                calibration_dir=calib_dir,
                verbose=False,
            )
        finally:
            mr.read_frames_by_time_ns = orig

        assert len(manifest["entries"]) == 1
        e = manifest["entries"][0]
        # Frame A (inside) must never have been asked for.
        assert 1_000_000_000 not in asked, (
            "Frame A (camera inside bbox) was sent to the decoder — the "
            "inside-bbox filter is not being applied."
        )
        # Either frame B wins or the entry is skipped (e.g. blank-image
        # sharpness rank), but in no case should A be the winner.
        if "frame_timestamp_ns" in e:
            assert e["frame_timestamp_ns"] != 1_000_000_000
