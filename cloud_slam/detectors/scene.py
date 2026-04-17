# Vendored from ManyCore SpatialLM (https://github.com/manycore-research/SpatialLM) —
# layout parser logic reused under Llama3.2 license for parsing SpatialLM1.1 output strings.
#
# The main cloud_slam pipeline runs in a torch 2.11 env where the `spatiallm`
# package cannot be imported (it needs torch 2.4). This module re-implements
# the minimal subset of spatiallm.layout needed to read/write the structured
# language output: entity dataclasses + Scene (mirrors their `Layout`).
"""Parser for SpatialLM1.1 structured-language scene output."""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any


@dataclass
class Wall:
    """A wall segment between two endpoints, with vertical height + thickness."""
    id: int
    ax: float
    ay: float
    az: float
    bx: float
    by: float
    bz: float
    height: float
    thickness: float
    entity_label: str = "wall"

    def __post_init__(self):
        self.id = int(self.id)
        self.ax = float(self.ax)
        self.ay = float(self.ay)
        self.az = float(self.az)
        self.bx = float(self.bx)
        self.by = float(self.by)
        self.bz = float(self.bz)
        self.height = float(self.height)
        self.thickness = float(self.thickness)

    def to_language_string(self) -> str:
        return (
            f"wall_{self.id}=Wall("
            f"{self.ax},{self.ay},{self.az},"
            f"{self.bx},{self.by},{self.bz},"
            f"{self.height},{self.thickness})"
        )


@dataclass
class Door:
    """A door, attached to a wall by id, centered at (x,y,z) with width+height."""
    id: int
    wall_id: int
    position_x: float
    position_y: float
    position_z: float
    width: float
    height: float
    entity_label: str = "door"

    def __post_init__(self):
        self.id = int(self.id)
        self.wall_id = int(self.wall_id)
        self.position_x = float(self.position_x)
        self.position_y = float(self.position_y)
        self.position_z = float(self.position_z)
        self.width = float(self.width)
        self.height = float(self.height)

    def to_language_string(self) -> str:
        return (
            f"{self.entity_label}_{self.id}={self.entity_label.capitalize()}("
            f"wall_{self.wall_id},"
            f"{self.position_x},{self.position_y},{self.position_z},"
            f"{self.width},{self.height})"
        )


@dataclass
class Window(Door):
    """A window, same structure as a Door (attached to a wall)."""
    entity_label: str = "window"


@dataclass
class Bbox:
    """Axis-aligned-in-yaw oriented box for a detected object."""
    id: int
    class_name: str
    position_x: float
    position_y: float
    position_z: float
    angle_z: float
    scale_x: float
    scale_y: float
    scale_z: float
    entity_label: str = "bbox"

    def __post_init__(self):
        self.id = int(self.id)
        self.class_name = str(self.class_name)
        self.position_x = float(self.position_x)
        self.position_y = float(self.position_y)
        self.position_z = float(self.position_z)
        self.angle_z = float(self.angle_z)
        self.scale_x = abs(float(self.scale_x))
        self.scale_y = abs(float(self.scale_y))
        self.scale_z = abs(float(self.scale_z))

    def to_language_string(self) -> str:
        return (
            f"bbox_{self.id}=Bbox("
            f"{self.class_name},"
            f"{self.position_x},{self.position_y},{self.position_z},"
            f"{self.angle_z},"
            f"{self.scale_x},{self.scale_y},{self.scale_z})"
        )


@dataclass
class Scene:
    """A parsed SpatialLM scene: walls, doors, windows, bounding boxes."""
    walls: List[Wall] = field(default_factory=list)
    doors: List[Door] = field(default_factory=list)
    windows: List[Window] = field(default_factory=list)
    bboxes: List[Bbox] = field(default_factory=list)

    # ----- Parsing -----

    @classmethod
    def from_language_string(cls, s: str) -> "Scene":
        """Parse SpatialLM structured-language output. Skips malformed lines."""
        scene = cls()
        if not s:
            return scene
        existing_walls: List[int] = []
        for line in s.lstrip("\n").split("\n"):
            line = line.strip()
            if not line or "=" not in line or "(" not in line or ")" not in line:
                continue
            try:
                label = line.split("=")[0]
                entity_id = int(label.split("_")[1])
                entity_label = label.split("_")[0]
                start = line.find("(")
                end = line.find(")")
                params = line[start + 1:end].split(",")

                if entity_label == "wall":
                    scene.walls.append(Wall(
                        id=entity_id,
                        ax=params[0], ay=params[1], az=params[2],
                        bx=params[3], by=params[4], bz=params[5],
                        height=params[6], thickness=params[7],
                    ))
                    existing_walls.append(entity_id)
                elif entity_label == "door":
                    wall_id = int(params[0].split("_")[1])
                    if wall_id not in existing_walls:
                        continue
                    scene.doors.append(Door(
                        id=entity_id, wall_id=wall_id,
                        position_x=params[1], position_y=params[2], position_z=params[3],
                        width=params[4], height=params[5],
                    ))
                elif entity_label == "window":
                    wall_id = int(params[0].split("_")[1])
                    if wall_id not in existing_walls:
                        continue
                    scene.windows.append(Window(
                        id=entity_id, wall_id=wall_id,
                        position_x=params[1], position_y=params[2], position_z=params[3],
                        width=params[4], height=params[5],
                    ))
                elif entity_label == "bbox":
                    scene.bboxes.append(Bbox(
                        id=entity_id,
                        class_name=params[0],
                        position_x=params[1], position_y=params[2], position_z=params[3],
                        angle_z=params[4],
                        scale_x=params[5], scale_y=params[6], scale_z=params[7],
                    ))
            except (IndexError, ValueError):
                # Malformed line: skip and continue.
                continue
        return scene

    # ----- Serialization -----

    def to_language_string(self) -> str:
        """Serialize back to SpatialLM's structured-language format."""
        parts: List[str] = []
        for e in self.walls + self.doors + self.windows + self.bboxes:
            parts.append(e.to_language_string())
        return "\n".join(parts)

    def to_json(self) -> Dict[str, Any]:
        """Produce a JSON-serializable dict representation of the scene."""
        return {
            "walls": [
                {
                    "id": w.id,
                    "a": [w.ax, w.ay, w.az],
                    "b": [w.bx, w.by, w.bz],
                    "height": w.height,
                    "thickness": w.thickness,
                }
                for w in self.walls
            ],
            "doors": [
                {
                    "id": d.id,
                    "wall_id": d.wall_id,
                    "position": [d.position_x, d.position_y, d.position_z],
                    "width": d.width,
                    "height": d.height,
                }
                for d in self.doors
            ],
            "windows": [
                {
                    "id": w.id,
                    "wall_id": w.wall_id,
                    "position": [w.position_x, w.position_y, w.position_z],
                    "width": w.width,
                    "height": w.height,
                }
                for w in self.windows
            ],
            "bboxes": [
                {
                    "id": b.id,
                    "class": b.class_name,
                    "center": [b.position_x, b.position_y, b.position_z],
                    "yaw": b.angle_z,
                    "dimensions": [b.scale_x, b.scale_y, b.scale_z],
                }
                for b in self.bboxes
            ],
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "Scene":
        """Inverse of to_json — reconstruct a Scene from its dict form."""
        scene = cls()
        for w in data.get("walls", []):
            a, b = w["a"], w["b"]
            scene.walls.append(Wall(
                id=w["id"],
                ax=a[0], ay=a[1], az=a[2],
                bx=b[0], by=b[1], bz=b[2],
                height=w["height"], thickness=w["thickness"],
            ))
        for d in data.get("doors", []):
            p = d["position"]
            scene.doors.append(Door(
                id=d["id"], wall_id=d["wall_id"],
                position_x=p[0], position_y=p[1], position_z=p[2],
                width=d["width"], height=d["height"],
            ))
        for w in data.get("windows", []):
            p = w["position"]
            scene.windows.append(Window(
                id=w["id"], wall_id=w["wall_id"],
                position_x=p[0], position_y=p[1], position_z=p[2],
                width=w["width"], height=w["height"],
            ))
        for b in data.get("bboxes", []):
            c, dims = b["center"], b["dimensions"]
            scene.bboxes.append(Bbox(
                id=b["id"], class_name=b["class"],
                position_x=c[0], position_y=c[1], position_z=c[2],
                angle_z=b["yaw"],
                scale_x=dims[0], scale_y=dims[1], scale_z=dims[2],
            ))
        return scene

    # ----- Geometry -----

    def apply_transform(self, R, t) -> None:
        """Rotate + translate every element in-place. R is 3x3, t is length-3.

        Rotates wall endpoints, door/window positions, and bbox centers.
        Adjusts bbox yaw by the Z-component of the rotation (small fine-
        leveling rotations around non-Z axes leave yaw effectively unchanged;
        a pure Z rotation changes yaw by its angle).
        """
        import math
        R = [[float(R[i][j]) for j in range(3)] for i in range(3)]
        tx, ty, tz = float(t[0]), float(t[1]), float(t[2])

        def _rt(x, y, z):
            nx = R[0][0]*x + R[0][1]*y + R[0][2]*z + tx
            ny = R[1][0]*x + R[1][1]*y + R[1][2]*z + ty
            nz = R[2][0]*x + R[2][1]*y + R[2][2]*z + tz
            return nx, ny, nz

        for w in self.walls:
            w.ax, w.ay, w.az = _rt(w.ax, w.ay, w.az)
            w.bx, w.by, w.bz = _rt(w.bx, w.by, w.bz)
        for d in self.doors:
            d.position_x, d.position_y, d.position_z = _rt(
                d.position_x, d.position_y, d.position_z)
        for w in self.windows:
            w.position_x, w.position_y, w.position_z = _rt(
                w.position_x, w.position_y, w.position_z)
        yaw_delta = math.atan2(R[1][0], R[0][0])
        for b in self.bboxes:
            b.position_x, b.position_y, b.position_z = _rt(
                b.position_x, b.position_y, b.position_z)
            b.angle_z = b.angle_z + yaw_delta

    # ----- Misc -----

    def summary(self) -> str:
        """Short human-readable one-liner describing scene contents."""
        return (
            f"{len(self.walls)} walls, {len(self.doors)} doors, "
            f"{len(self.windows)} windows, {len(self.bboxes)} bboxes"
        )
