"""Merge two SpatialLM layout .txt files.

Strategy (empirically best on TEST_SCAN):
  - Walls/doors/windows come from Qwen-0.5B (better structural recall)
  - Objects (bboxes) come from Llama-1B (better class specificity,
    e.g. dining_table that Qwen missed)

Also provides a same-class proximity deduplicator for bboxes.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence


_WALL_RX = re.compile(r"wall_\d+=Wall\(([-0-9.,]+)\)")
_DOOR_RX = re.compile(r"door_\d+=Door\(wall_\d+,([-0-9.,]+)\)")
_WIN_RX = re.compile(r"window_\d+=Window\(wall_\d+,([-0-9.,]+)\)")
_BBOX_RX = re.compile(r"bbox_\d+=Bbox\(([A-Za-z_\-]+),([-0-9.,]+)\)")


def parse_layout(layout_txt: Path | str) -> dict:
    """Return a dict with `walls`, `doors`, `windows`, `bboxes` lists of tuples."""
    text = Path(layout_txt).read_text().strip().splitlines()
    walls, doors, windows, bboxes = [], [], [], []
    for line in text:
        if line.startswith("wall_"):
            m = _WALL_RX.match(line)
            if m:
                walls.append([float(x) for x in m.group(1).split(",")])
        elif line.startswith("door_"):
            m = _DOOR_RX.match(line)
            if m:
                doors.append([float(x) for x in m.group(1).split(",")])
        elif line.startswith("window_"):
            m = _WIN_RX.match(line)
            if m:
                windows.append([float(x) for x in m.group(1).split(",")])
        elif line.startswith("bbox_"):
            m = _BBOX_RX.match(line)
            if m:
                cls = m.group(1)
                v = [float(x) for x in m.group(2).split(",")]
                bboxes.append((cls, v))
    return dict(walls=walls, doors=doors, windows=windows, bboxes=bboxes)


def dedup_bboxes(bboxes: Sequence[tuple[str, list[float]]],
                 radius_m: float = 0.20
                 ) -> list[tuple[str, list[float]]]:
    """Keep first of any same-class cluster whose centers are within radius."""
    kept: list[tuple[str, list[float]]] = []
    for cls, v in bboxes:
        cx, cy, cz = v[0], v[1], v[2]
        dup = False
        for kcls, kv in kept:
            if kcls != cls:
                continue
            if (cx-kv[0])**2 + (cy-kv[1])**2 + (cz-kv[2])**2 < radius_m**2:
                dup = True
                break
        if not dup:
            kept.append((cls, v))
    return kept


def merge_layouts(
    structure_layout: Path | str,
    objects_layout: Path | str,
    out_txt: Path | str,
    *,
    dedup_radius_m: float = 0.20,
    verbose: bool = True,
) -> Path:
    """Combine walls/doors/windows from `structure_layout` and bboxes
    from `objects_layout`, dedup bboxes, write unified layout file."""
    s = parse_layout(structure_layout)
    o = parse_layout(objects_layout)
    bboxes = dedup_bboxes(o["bboxes"], radius_m=dedup_radius_m)
    if verbose:
        print(f"[merge] struct: {len(s['walls'])}w {len(s['doors'])}d "
              f"{len(s['windows'])}win  | objects: {len(o['bboxes'])} raw "
              f"-> {len(bboxes)} deduped")

    out_lines = []
    for i, w in enumerate(s["walls"]):
        out_lines.append(f"wall_{i}=Wall({','.join(str(x) for x in w)})")
    for i, d in enumerate(s["doors"]):
        out_lines.append(f"door_{i}=Door(wall_0,{','.join(str(x) for x in d)})")
    for i, d in enumerate(s["windows"]):
        out_lines.append(f"window_{i}=Window(wall_0,{','.join(str(x) for x in d)})")
    for i, (cls, v) in enumerate(bboxes):
        out_lines.append(f"bbox_{i}=Bbox({cls},{','.join(str(x) for x in v)})")

    out_txt = Path(out_txt)
    out_txt.write_text("\n".join(out_lines) + "\n")
    if verbose:
        print(f"[merge] wrote {out_txt}")
    return out_txt
