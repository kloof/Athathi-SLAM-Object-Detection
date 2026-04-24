"""Parse pipeline artifacts into the Modal API `result.json` envelope.

Pure-Python, no network, no GPU, no heavy deps. Designed to run in the
lightweight ASGI / runner container *after* the main pipeline has
already emitted ``layout_merged.txt``, ``best_views/best_views.json``,
and ``slam/metrics.json`` into ``output_dir``.

The artifact paths produced here are **relative** (e.g.
``"artifacts/scene_with_boxes.ply"``). The endpoint layer rewrites
them to absolute URLs at serve time.

Regexes are imported from ``cloud_slam.spatiallm_pipeline.merge`` so
there is exactly one definition of the layout-text grammar in the
codebase.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from cloud_slam.spatiallm_pipeline.merge import (
    _WALL_RX,
    _DOOR_RX,
    _WIN_RX,
    _BBOX_RX,
)


# ---------------------------------------------------------------------------
# layout_merged.txt -> floorplan + furniture
# ---------------------------------------------------------------------------

def parse_layout_merged(path: Path) -> dict:
    """Parse a ``layout_merged.txt`` into ``{walls, doors, windows, furniture}``.

    Shapes match the ``result.json`` sub-trees in the spec:

    - Wall: ``{"id", "start", "end", "height", "thickness"}``
    - Door / Window: ``{"id", "wall", "center", "width", "height"}``
    - Furniture bbox: ``{"id", "class", "center", "size", "yaw"}``

    The wall reference on doors / windows is re-extracted from the raw
    line via a lightweight second pass — the shared regex captures the
    numeric payload only.
    """
    path = Path(path)
    walls: list[dict] = []
    doors: list[dict] = []
    windows: list[dict] = []
    furniture: list[dict] = []

    wall_idx = door_idx = win_idx = bbox_idx = 0

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("wall_"):
            m = _WALL_RX.match(line)
            if not m:
                continue
            vals = [float(x) for x in m.group(1).split(",")]
            # Wall(x1,y1,z1, x2,y2,z2, height, thickness)
            if len(vals) != 8:
                continue
            walls.append({
                "id": f"wall_{wall_idx}",
                "start": vals[0:3],
                "end": vals[3:6],
                "height": vals[6],
                "thickness": vals[7],
            })
            wall_idx += 1

        elif line.startswith("door_"):
            m = _DOOR_RX.match(line)
            if not m:
                continue
            wall_ref = _extract_wall_ref(line)
            vals = [float(x) for x in m.group(1).split(",")]
            # Door(wall_M, x,y,z, width, height)
            if len(vals) != 5 or wall_ref is None:
                continue
            doors.append({
                "id": f"door_{door_idx}",
                "wall": wall_ref,
                "center": vals[0:3],
                "width": vals[3],
                "height": vals[4],
            })
            door_idx += 1

        elif line.startswith("window_"):
            m = _WIN_RX.match(line)
            if not m:
                continue
            wall_ref = _extract_wall_ref(line)
            vals = [float(x) for x in m.group(1).split(",")]
            if len(vals) != 5 or wall_ref is None:
                continue
            windows.append({
                "id": f"window_{win_idx}",
                "wall": wall_ref,
                "center": vals[0:3],
                "width": vals[3],
                "height": vals[4],
            })
            win_idx += 1

        elif line.startswith("bbox_"):
            m = _BBOX_RX.match(line)
            if not m:
                continue
            cls = m.group(1)
            vals = [float(x) for x in m.group(2).split(",")]
            # Bbox(class, cx,cy,cz, yaw, w,h,d)
            if len(vals) != 7:
                continue
            furniture.append({
                "id": f"bbox_{bbox_idx}",
                "class": cls,
                "center": vals[0:3],
                "size": vals[4:7],
                "yaw": vals[3],
            })
            bbox_idx += 1

    return {
        "walls": walls,
        "doors": doors,
        "windows": windows,
        "furniture": furniture,
    }


def _extract_wall_ref(line: str) -> str | None:
    """Pull the ``wall_M`` reference out of a Door/Window line.

    The shared ``_DOOR_RX`` / ``_WIN_RX`` only capture the numeric tail,
    so we do a minimal substring scan here instead of introducing a
    second regex.
    """
    lparen = line.find("(")
    comma = line.find(",", lparen + 1)
    if lparen < 0 or comma < 0:
        return None
    ref = line[lparen + 1 : comma].strip()
    return ref if ref.startswith("wall_") else None


# ---------------------------------------------------------------------------
# best_views.json -> best_images[]
# ---------------------------------------------------------------------------

def load_best_views_manifest(path: Path) -> list[dict]:
    """Reshape ``best_views.json`` entries into the API ``best_images`` list.

    Internal fields (``bbox_3d``, ``scores``, ``crop_aabb``) are dropped.
    Entries without an ``image_path`` (score-floor skipped) are omitted
    entirely — they have no servable asset. ``bbox_id`` is normalized to
    the ``bbox_N`` string form used elsewhere in the envelope.

    If the manifest file is missing, unreadable, or its ``entries`` list
    is empty, returns ``[]``. The endpoint layer converts
    ``relative_image_path`` -> absolute ``url`` at serve time.
    """
    path = Path(path)
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return []

    entries = raw.get("entries") or []
    out: list[dict] = []
    for e in entries:
        image_path = e.get("image_path")
        if not image_path:
            # skipped entry (score_floor or crop_empty) -- no servable asset
            continue

        bbox_id_raw = e.get("bbox_id")
        if isinstance(bbox_id_raw, int):
            bbox_id = f"bbox_{bbox_id_raw}"
        else:
            bbox_id = str(bbox_id_raw)

        out.append({
            "bbox_id": bbox_id,
            "class": e.get("class"),
            "frame_timestamp_ns": e.get("frame_timestamp_ns"),
            "camera_distance_m": e.get("camera_distance_m"),
            "pixel_aabb": e.get("pixel_aabb"),
            "relative_image_path": image_path,
        })
    return out


# ---------------------------------------------------------------------------
# slam/metrics.json
# ---------------------------------------------------------------------------

def load_slam_metrics(slam_dir: Path) -> dict:
    """Return ``slam/metrics.json`` contents as-is, or ``{}`` if missing."""
    metrics_path = Path(slam_dir) / "metrics.json"
    if not metrics_path.is_file():
        return {}
    try:
        return json.loads(metrics_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


# ---------------------------------------------------------------------------
# envelope
# ---------------------------------------------------------------------------

_ARTIFACT_FILES = {
    "colored_map_ply": "artifacts/slam/colored_map.ply",
    "scene_with_boxes_ply": "artifacts/scene_with_boxes.ply",
    "layout_merged_txt": "artifacts/layout_merged.txt",
    "result_json": "result.json",
}


def build_result_json(
    output_dir: Path,
    *,
    job_id: str,
    submitted_at: datetime,
    finished_at: datetime,
) -> dict:
    """Stitch layout + best_views + slam metrics into the API envelope.

    Artifact paths are **relative** to the job directory on the volume;
    the endpoint layer rewrites them to absolute URLs when serving.
    ``metrics.total_duration_s = (finished_at - submitted_at).seconds``.
    """
    output_dir = Path(output_dir)

    floorplan_src = parse_layout_merged(output_dir / "layout_merged.txt")
    furniture = floorplan_src.pop("furniture")
    floorplan = floorplan_src  # walls, doors, windows

    best_views_path = output_dir / "best_views" / "best_views.json"
    best_images = load_best_views_manifest(best_views_path)

    slam_metrics = load_slam_metrics(output_dir / "slam")

    total_duration_s = (finished_at - submitted_at).seconds

    envelope: dict[str, Any] = {
        "job_id": job_id,
        "status": "done",
        "submitted_at": _iso_z(submitted_at),
        "finished_at": _iso_z(finished_at),
        "metrics": {
            "slam": slam_metrics,
            "total_duration_s": total_duration_s,
        },
        "floorplan": floorplan,
        "furniture": furniture,
        "best_images": best_images,
        "artifacts": dict(_ARTIFACT_FILES),
    }
    return envelope


def _iso_z(dt: datetime) -> str:
    """Serialize a datetime as ``YYYY-MM-DDTHH:MM:SSZ`` (spec shape)."""
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
