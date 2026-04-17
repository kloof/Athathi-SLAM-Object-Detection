"""Tests for the SpatialLM structured-language Scene parser."""

import math
import pytest

from cloud_slam.detectors.scene import Scene, Wall, Door, Window, Bbox


# Full bedroom scan output (verbatim from room_output.txt on 2026-03-31):
# 11 walls, 5 doors, 2 windows, 13 bboxes including bed, 2 chairs, desk,
# computer, 2 bookcases, 6 curtains.
ROOM_OUTPUT = """wall_0=Wall(-0.2818035125732421,-3.268224000930786,0.08139145374298096,3.193196487426758,-3.268224000930786,0.08139145374298096,2.9600000000000004,0.0)
wall_1=Wall(-0.2818035125732421,-3.268224000930786,0.08139145374298096,-1.0818035125732424,-2.268224000930786,0.08139145374298096,2.9600000000000004,0.0)
wall_2=Wall(3.193196487426758,-3.268224000930786,0.08139145374298096,2.5931964874267575,0.556775999069214,0.08139145374298096,2.9600000000000004,0.0)
wall_3=Wall(-1.0818035125732424,-2.268224000930786,0.08139145374298096,-3.506803512573242,-0.6432240009307861,0.08139145374298096,2.9600000000000004,0.0)
wall_4=Wall(-3.556803512573242,-0.6432240009307861,0.08139145374298096,-3.506803512573242,-0.6432240009307861,0.08139145374298096,2.9600000000000004,0.0)
wall_5=Wall(-3.556803512573242,-0.6432240009307861,0.08139145374298096,-1.3568035125732423,2.9067759990692137,0.08139145374298096,2.9600000000000004,0.0)
wall_6=Wall(2.5931964874267575,0.556775999069214,0.08139145374298096,5.6681964874267585,1.4067759990692137,0.08139145374298096,2.9600000000000004,0.0)
wall_7=Wall(5.6681964874267585,1.4067759990692137,0.08139145374298096,5.6681964874267585,3.731775999069214,0.08139145374298096,2.9600000000000004,0.0)
wall_8=Wall(-1.3568035125732423,2.9067759990692137,0.08139145374298096,2.618196487426758,3.1567759990692137,0.08139145374298096,2.9600000000000004,0.0)
wall_9=Wall(2.618196487426758,3.1567759990692137,0.08139145374298096,2.618196487426758,3.731775999069214,0.08139145374298096,2.9600000000000004,0.0)
wall_10=Wall(2.618196487426758,3.731775999069214,0.08139145374298096,5.6681964874267585,3.731775999069214,0.08139145374298096,2.9600000000000004,0.0)
door_0=Door(wall_6,1.9681964874267575,0.8817759990692142,1.131391453742981,0.9199999999999999,2.12)
door_1=Door(wall_6,3.143196487426758,1.181775999069214,1.131391453742981,0.9199999999999999,2.12)
door_2=Door(wall_7,5.6681964874267585,2.5317759990692137,1.131391453742981,0.9199999999999999,2.12)
door_3=Door(wall_8,1.3431964874267575,2.8317759990692135,1.131391453742981,0.9199999999999999,2.12)
door_4=Door(wall_10,3.143196487426758,3.731775999069214,1.131391453742981,0.9199999999999999,2.12)
window_0=Window(wall_0,1.4681964874267575,-3.268224000930786,1.656391453742981,3.4200000000000004,1.8)
window_1=Window(wall_3,-2.4568035125732424,-1.193224000930786,1.656391453742981,1.64,1.8)
bbox_0=Bbox(curtain,1.4931964874267578,-3.2182240009307863,1.556391453742981,-2.9747025,2.3125,0.09375,2.921875)
bbox_1=Bbox(curtain,-0.0068035125732421875,-2.818224000930786,1.556391453742981,-0.7854000000000001,2.3125,0.09375,2.921875)
bbox_2=Bbox(curtain,-0.3818035125732422,-2.243224000930786,1.556391453742981,-0.7854000000000001,2.3125,0.09375,2.921875)
bbox_3=Bbox(curtain,-2.681803512573242,-1.193224000930786,1.556391453742981,-0.7854000000000001,2.3125,0.09375,2.921875)
bbox_4=Bbox(curtain,1.1181964874267578,-0.918224000930786,1.306391453742981,-0.7854000000000001,2.3125,0.09375,2.46875)
bbox_5=Bbox(bed,-2.306803512573242,-0.39322400093078613,0.581391453742981,-2.8568925000000003,2.359375,2.390625,1.0)
bbox_6=Bbox(chair,0.0931964874267579,0.2817759990692137,0.581391453742981,-2.5623675,0.765625,0.875,1.015625)
bbox_7=Bbox(desk,0.6431964874267582,1.1067759990692139,0.606391453742981,-0.7854000000000001,1.140625,0.5,1.03125)
bbox_8=Bbox(chair,-1.1068035125732423,1.4567759990692135,0.581391453742981,-0.7854000000000001,0.765625,0.875,1.015625)
bbox_9=Bbox(computer,-1.7068035125732421,1.6567759990692137,1.006391453742981,-0.7854000000000001,0.703125,0.5,0.453125)
bbox_10=Bbox(bookcase,-0.23180351257324228,1.931775999069214,1.1063914537429809,-0.7854000000000001,0.796875,0.421875,2.046875)
bbox_11=Bbox(bookcase,-0.9568035125732424,2.1317759990692142,1.1063914537429809,-2.5623675,0.796875,0.421875,2.046875)
bbox_12=Bbox(curtain,-2.531803512573242,2.4067759990692137,1.556391453742981,-2.5623675,2.3125,0.09375,2.921875)"""


# Minimal living-room fixture — 3 walls, 1 door, 2 bboxes, 1 window.
# Values kept simple so it's easy to debug.
MINI_LIVING_ROOM = """wall_0=Wall(0.0,0.0,0.0,4.0,0.0,0.0,2.8,0.0)
wall_1=Wall(4.0,0.0,0.0,4.0,5.0,0.0,2.8,0.0)
wall_2=Wall(4.0,5.0,0.0,0.0,5.0,0.0,2.8,0.0)
door_0=Door(wall_0,2.0,0.0,1.05,0.9,2.1)
window_0=Window(wall_1,4.0,2.5,1.5,1.2,1.4)
bbox_0=Bbox(sofa,2.0,2.0,0.4,0.0,2.0,0.85,0.9)
bbox_1=Bbox(coffee_table,2.0,3.0,0.25,0.0,0.9,0.5,0.45)"""


# ---------- fixtures ----------

@pytest.fixture
def room_scene() -> Scene:
    return Scene.from_language_string(ROOM_OUTPUT)


@pytest.fixture
def living_room_scene() -> Scene:
    return Scene.from_language_string(MINI_LIVING_ROOM)


# ---------- helpers ----------

TOL = 1e-6


def _close(a: float, b: float, tol: float = TOL) -> bool:
    return math.isclose(a, b, rel_tol=0, abs_tol=tol)


def _walls_equal(w1: Wall, w2: Wall) -> bool:
    return all([
        w1.id == w2.id,
        _close(w1.ax, w2.ax), _close(w1.ay, w2.ay), _close(w1.az, w2.az),
        _close(w1.bx, w2.bx), _close(w1.by, w2.by), _close(w1.bz, w2.bz),
        _close(w1.height, w2.height), _close(w1.thickness, w2.thickness),
    ])


def _fixtures_equal(f1, f2) -> bool:
    return all([
        f1.id == f2.id, f1.wall_id == f2.wall_id,
        _close(f1.position_x, f2.position_x),
        _close(f1.position_y, f2.position_y),
        _close(f1.position_z, f2.position_z),
        _close(f1.width, f2.width), _close(f1.height, f2.height),
    ])


def _bboxes_equal(b1: Bbox, b2: Bbox) -> bool:
    return all([
        b1.id == b2.id, b1.class_name == b2.class_name,
        _close(b1.position_x, b2.position_x),
        _close(b1.position_y, b2.position_y),
        _close(b1.position_z, b2.position_z),
        _close(b1.angle_z, b2.angle_z),
        _close(b1.scale_x, b2.scale_x),
        _close(b1.scale_y, b2.scale_y),
        _close(b1.scale_z, b2.scale_z),
    ])


def _scenes_equal(s1: Scene, s2: Scene) -> bool:
    if (len(s1.walls), len(s1.doors), len(s1.windows), len(s1.bboxes)) != \
       (len(s2.walls), len(s2.doors), len(s2.windows), len(s2.bboxes)):
        return False
    for w1, w2 in zip(s1.walls, s2.walls):
        if not _walls_equal(w1, w2):
            return False
    for d1, d2 in zip(s1.doors, s2.doors):
        if not _fixtures_equal(d1, d2):
            return False
    for w1, w2 in zip(s1.windows, s2.windows):
        if not _fixtures_equal(w1, w2):
            return False
    for b1, b2 in zip(s1.bboxes, s2.bboxes):
        if not _bboxes_equal(b1, b2):
            return False
    return True


# ---------- tests ----------

def test_parses_bedroom_counts(room_scene: Scene):
    assert len(room_scene.walls) == 11
    assert len(room_scene.doors) == 5
    assert len(room_scene.windows) == 2
    assert len(room_scene.bboxes) == 13


def test_parses_bedroom_classes(room_scene: Scene):
    classes = [b.class_name for b in room_scene.bboxes]
    assert "bed" in classes
    assert classes.count("chair") == 2
    assert "desk" in classes
    assert "computer" in classes
    assert classes.count("bookcase") == 2


def test_bedroom_summary(room_scene: Scene):
    assert room_scene.summary() == "11 walls, 5 doors, 2 windows, 13 bboxes"


def test_parses_living_room_counts_and_class(living_room_scene: Scene):
    assert len(living_room_scene.walls) == 3
    assert len(living_room_scene.doors) == 1
    assert len(living_room_scene.windows) == 1
    assert len(living_room_scene.bboxes) == 2
    classes = {b.class_name for b in living_room_scene.bboxes}
    assert "sofa" in classes
    # and ensure underscore-containing class name parsed correctly here too
    assert "coffee_table" in classes


def test_roundtrip_language_string(room_scene: Scene):
    """Serialize -> parse must yield a semantically equal Scene."""
    s = room_scene.to_language_string()
    reparsed = Scene.from_language_string(s)
    assert _scenes_equal(room_scene, reparsed)


def test_roundtrip_json(room_scene: Scene):
    """to_json -> from_json must yield a semantically equal Scene."""
    data = room_scene.to_json()
    # validate that to_json returns a plain dict with the expected schema keys
    assert set(data.keys()) == {"walls", "doors", "windows", "bboxes"}
    rebuilt = Scene.from_json(data)
    assert _scenes_equal(room_scene, rebuilt)


def test_json_schema_shape(living_room_scene: Scene):
    data = living_room_scene.to_json()
    w0 = data["walls"][0]
    assert set(w0.keys()) == {"id", "a", "b", "height", "thickness"}
    assert len(w0["a"]) == 3 and len(w0["b"]) == 3
    d0 = data["doors"][0]
    assert set(d0.keys()) == {"id", "wall_id", "position", "width", "height"}
    assert len(d0["position"]) == 3
    b0 = data["bboxes"][0]
    assert set(b0.keys()) == {"id", "class", "center", "yaw", "dimensions"}
    assert len(b0["center"]) == 3 and len(b0["dimensions"]) == 3


def test_malformed_lines_are_skipped():
    mixed = (
        "wall_0=Wall(0,0,0,1,0,0,2.5,0.0)\n"
        "THIS IS GARBAGE AND SHOULD BE IGNORED\n"
        "\n"
        "wall_1=Wall(corrupt,data,here)\n"  # too few params → IndexError inside ctor
        "bbox_0=Bbox(chair,1.0,1.0,0.4,0.0,0.5,0.5,0.85)\n"
        "not_even_close_to_valid\n"
        "door_0=Door(wall_99,0,0,0,1,2)\n"  # references a non-existent wall → skipped
    )
    scene = Scene.from_language_string(mixed)
    assert len(scene.walls) == 1
    assert scene.walls[0].id == 0
    assert len(scene.bboxes) == 1
    assert scene.bboxes[0].class_name == "chair"
    assert len(scene.doors) == 0
    assert len(scene.windows) == 0


def test_bbox_underscore_class_parses():
    s = "bbox_0=Bbox(coffee_table,1.0,2.0,0.3,-1.5708,0.9,0.5,0.45)"
    scene = Scene.from_language_string(s)
    assert len(scene.bboxes) == 1
    b = scene.bboxes[0]
    assert b.class_name == "coffee_table"
    assert _close(b.position_x, 1.0)
    assert _close(b.angle_z, -1.5708)
    assert _close(b.scale_x, 0.9)


def test_empty_input():
    assert Scene.from_language_string("").summary() == "0 walls, 0 doors, 0 windows, 0 bboxes"
    assert Scene.from_language_string("\n\n\n").summary() == "0 walls, 0 doors, 0 windows, 0 bboxes"
