"""Unit tests for cloud_slam.api.parse_outputs.

Pure-Python, no network, no GPU. All fixtures live alongside this file
under ``tests/fixtures/``. Should finish in well under a second.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from cloud_slam.api.parse_outputs import (
    build_result_json,
    load_best_views_manifest,
    parse_layout_merged,
)


FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_LAYOUT = FIXTURES / "sample_layout_merged.txt"
SAMPLE_BEST_VIEWS = FIXTURES / "sample_best_views.json"
EMPTY_BEST_VIEWS = FIXTURES / "empty_best_views.json"


# ---------------------------------------------------------------------------
# parse_layout_merged
# ---------------------------------------------------------------------------

def test_parse_walls():
    out = parse_layout_merged(SAMPLE_LAYOUT)
    walls = out["walls"]
    # Fixture has 16 walls.
    assert len(walls) == 16

    w0 = walls[0]
    assert w0["id"] == "wall_0"
    # Fixture line: Wall(-0.48778..., -5.47942..., 0.12599..., 4.51222..., -5.47942..., 0.12599..., 2.88, 0.0)
    assert w0["start"] == pytest.approx(
        [-0.48778104791279775, -5.479422931572759, 0.12598943710327148]
    )
    assert w0["end"] == pytest.approx(
        [4.512218952087202, -5.479422931572759, 0.12598943710327148]
    )
    assert w0["height"] == pytest.approx(2.8800000000000003)
    assert w0["thickness"] == pytest.approx(0.0)

    # IDs are dense wall_0..wall_15.
    assert [w["id"] for w in walls] == [f"wall_{i}" for i in range(16)]
    # Every wall has 3D endpoints and scalar thickness/height.
    for w in walls:
        assert len(w["start"]) == 3
        assert len(w["end"]) == 3
        assert isinstance(w["height"], float)
        assert isinstance(w["thickness"], float)


def test_parse_doors():
    out = parse_layout_merged(SAMPLE_LAYOUT)
    doors = out["doors"]
    assert len(doors) == 2

    d0 = doors[0]
    assert d0["id"] == "door_0"
    # door_0=Door(wall_0,-0.48778...,-4.20442...,1.22599...,0.92,2.18)
    assert d0["wall"] == "wall_0"
    assert d0["center"] == pytest.approx(
        [-0.48778104791279775, -4.204422931572759, 1.2259894371032716]
    )
    assert d0["width"] == pytest.approx(0.9199999999999999)
    assert d0["height"] == pytest.approx(2.18)

    # Second door also references wall_0 in the fixture.
    assert doors[1]["id"] == "door_1"
    assert doors[1]["wall"] == "wall_0"


def test_parse_windows():
    out = parse_layout_merged(SAMPLE_LAYOUT)
    windows = out["windows"]
    assert len(windows) == 3

    w0 = windows[0]
    assert w0["id"] == "window_0"
    # window_0=Window(wall_0,4.51222...,-2.25442...,1.77599...,2.68,1.3)
    assert w0["wall"] == "wall_0"
    assert w0["center"] == pytest.approx(
        [4.512218952087202, -2.2544229315727584, 1.7759894371032714]
    )
    assert w0["width"] == pytest.approx(2.68)
    assert w0["height"] == pytest.approx(1.3)

    # Shape parity with doors.
    assert set(w0.keys()) == {"id", "wall", "center", "width", "height"}


def test_parse_furniture():
    out = parse_layout_merged(SAMPLE_LAYOUT)
    furniture = out["furniture"]
    # Fixture has 26 bboxes (bbox_0..bbox_25).
    assert len(furniture) == 26

    b0 = furniture[0]
    assert b0["id"] == "bbox_0"
    assert b0["class"] == "painting"
    # bbox_0=Bbox(painting, 0.28722, -5.47942, 1.25099, -3.1416, 1.3125, 0.09375, 2.234375)
    assert b0["center"] == pytest.approx(
        [0.28721895208720216, -5.479422931572759, 1.2509894371032715]
    )
    assert b0["size"] == pytest.approx([1.3125, 0.09375, 2.234375])
    assert b0["yaw"] == pytest.approx(-3.1416)

    # Sofa lives at bbox_22.
    sofa = next(b for b in furniture if b["class"] == "sofa")
    assert sofa["id"] == "bbox_22"
    assert sofa["size"] == pytest.approx([4.78125, 2.703125, 0.765625])

    # IDs are dense bbox_0..bbox_25.
    assert [b["id"] for b in furniture] == [f"bbox_{i}" for i in range(26)]


# ---------------------------------------------------------------------------
# build_result_json (full envelope)
# ---------------------------------------------------------------------------

def _stage_output_dir(tmp_path: Path) -> Path:
    """Copy layout + best_views fixtures into a fresh ``output_dir`` that
    looks like a real pipeline artifact directory."""
    out = tmp_path / "out"
    (out / "best_views").mkdir(parents=True)
    (out / "slam").mkdir()
    (out / "layout_merged.txt").write_bytes(SAMPLE_LAYOUT.read_bytes())
    (out / "best_views" / "best_views.json").write_bytes(
        SAMPLE_BEST_VIEWS.read_bytes()
    )
    (out / "slam" / "metrics.json").write_text(
        json.dumps({"num_frames": 948, "backend_runtime_s": 32.83})
    )
    return out


def test_build_result_envelope_roundtrip(tmp_path):
    out_dir = _stage_output_dir(tmp_path)
    submitted = datetime(2026, 4, 24, 14, 3, 22)
    finished = submitted + timedelta(seconds=651)

    env = build_result_json(
        out_dir,
        job_id="j_2026-04-24_ab12cd34",
        submitted_at=submitted,
        finished_at=finished,
    )

    # Top-level keys match the spec's Result JSON schema.
    assert set(env.keys()) == {
        "job_id",
        "status",
        "submitted_at",
        "finished_at",
        "metrics",
        "floorplan",
        "furniture",
        "best_images",
        "artifacts",
    }
    assert env["job_id"] == "j_2026-04-24_ab12cd34"
    assert env["status"] == "done"
    assert env["submitted_at"] == "2026-04-24T14:03:22Z"
    assert env["finished_at"].endswith("Z")

    # Floorplan sub-tree.
    assert set(env["floorplan"].keys()) == {"walls", "doors", "windows"}
    assert len(env["floorplan"]["walls"]) == 16
    assert len(env["floorplan"]["doors"]) == 2
    assert len(env["floorplan"]["windows"]) == 3

    # Furniture is a flat list (not nested in floorplan).
    assert isinstance(env["furniture"], list)
    assert len(env["furniture"]) == 26

    # Metrics: slam payload preserved as-is; total_duration_s from dt delta.
    assert env["metrics"]["slam"]["num_frames"] == 948
    assert env["metrics"]["total_duration_s"] == 651

    # Artifacts use relative paths; endpoint layer rewrites to absolute URLs.
    assert env["artifacts"]["scene_with_boxes_ply"] == "artifacts/scene_with_boxes.ply"
    assert env["artifacts"]["result_json"] == "result.json"
    for v in env["artifacts"].values():
        assert not v.startswith("http"), "artifact paths must be relative"

    # best_images pulled through with relative_image_path (not url).
    assert len(env["best_images"]) > 0
    first = env["best_images"][0]
    assert set(first.keys()) == {
        "bbox_id",
        "class",
        "frame_timestamp_ns",
        "camera_distance_m",
        "pixel_aabb",
        "relative_image_path",
    }
    assert first["bbox_id"].startswith("bbox_")
    assert first["relative_image_path"].startswith("best_views/")
    # visible_fraction is explicitly NOT in the schema.
    assert "visible_fraction" not in first


def test_empty_best_views_is_noop(tmp_path):
    """Bag with no camera frames (Stage 8 produced empty manifest):
    load_best_views_manifest returns []; envelope is otherwise complete."""
    # Direct function call on the empty fixture.
    assert load_best_views_manifest(EMPTY_BEST_VIEWS) == []
    # Missing file also returns [].
    assert load_best_views_manifest(tmp_path / "does_not_exist.json") == []

    out_dir = _stage_output_dir(tmp_path)
    # Overwrite manifest with the empty fixture.
    (out_dir / "best_views" / "best_views.json").write_bytes(
        EMPTY_BEST_VIEWS.read_bytes()
    )

    submitted = datetime(2026, 4, 24, 14, 0, 0)
    finished = submitted + timedelta(seconds=60)
    env = build_result_json(
        out_dir,
        job_id="j_2026-04-24_deadbeef",
        submitted_at=submitted,
        finished_at=finished,
    )

    assert env["best_images"] == []
    # All other fields still populated.
    assert len(env["floorplan"]["walls"]) == 16
    assert len(env["furniture"]) == 26
    assert env["metrics"]["total_duration_s"] == 60
    assert env["status"] == "done"
