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

import numpy as np


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


def consensus_bboxes(
    layouts: Sequence[Path | str],
    *,
    radius_m: float = 0.30,
    min_votes: int = 2,
    verbose: bool = True,
) -> list[tuple[str, list[float]]]:
    """Merge bboxes across N layout files, keep those seen in >=min_votes.

    Within a single layout, duplicate same-class boxes are first collapsed.
    Cross-layout, boxes of the same class within `radius_m` (center-to-center)
    are treated as the same object and get a vote each. Any object whose vote
    count reaches `min_votes` survives, using the median center + median scale
    across its contributing observations.
    """
    # Build one deduped set per layout (vote count == number of layouts agreeing).
    per_layout = []
    for p in layouts:
        parsed = parse_layout(p)
        per_layout.append(dedup_bboxes(parsed["bboxes"], radius_m=0.15))

    # Cluster across layouts. Each cluster = same class + overlapping centers.
    # Use single-pass greedy: for each (cls, box) pick existing cluster if
    # center within radius_m and NOT already populated by same layout.
    clusters: list[dict] = []  # each: {cls, centers:[], scales:[], layouts:set}

    for layout_idx, bboxes in enumerate(per_layout):
        for cls, v in bboxes:
            cx, cy, cz = v[0], v[1], v[2]
            picked = None
            best_d2 = float("inf")
            for c in clusters:
                if c["cls"] != cls:
                    continue
                if layout_idx in c["layouts"]:
                    continue
                # distance to cluster centroid
                mc = np.mean(c["centers"], axis=0)
                d2 = (cx - mc[0])**2 + (cy - mc[1])**2 + (cz - mc[2])**2
                if d2 < radius_m * radius_m and d2 < best_d2:
                    picked = c
                    best_d2 = d2
            if picked is None:
                clusters.append({
                    "cls": cls,
                    "centers": [[cx, cy, cz]],
                    "scales": [v[3:]],   # [yaw, sx, sy, sz]
                    "layouts": {layout_idx},
                })
            else:
                picked["centers"].append([cx, cy, cz])
                picked["scales"].append(v[3:])
                picked["layouts"].add(layout_idx)

    survivors: list[tuple[str, list[float]]] = []
    for c in clusters:
        votes = len(c["layouts"])
        if votes < min_votes:
            continue
        median_center = np.median(np.array(c["centers"]), axis=0)
        median_scale = np.median(np.array(c["scales"]), axis=0)
        v = [float(median_center[0]), float(median_center[1]), float(median_center[2]),
             *[float(x) for x in median_scale]]
        survivors.append((c["cls"], v))

    if verbose:
        total_raw = sum(len(bs) for bs in per_layout)
        print(f"[consensus] {len(layouts)} passes, {total_raw} raw bboxes -> "
              f"{len(clusters)} clusters -> {len(survivors)} kept "
              f"(>={min_votes} votes, r={radius_m}m)")
    return survivors


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


def merge_layouts_consensus(
    structure_layout: Path | str,
    objects_layouts: Sequence[Path | str],
    out_txt: Path | str,
    *,
    consensus_radius_m: float = 0.30,
    min_votes: int = 2,
    verbose: bool = True,
) -> Path:
    """Same output shape as merge_layouts, but bboxes come from a consensus
    over N Llama passes (kept only when seen in >= min_votes passes)."""
    s = parse_layout(structure_layout)
    bboxes = consensus_bboxes(
        objects_layouts,
        radius_m=consensus_radius_m,
        min_votes=min_votes,
        verbose=verbose,
    )
    if verbose:
        print(f"[merge] struct: {len(s['walls'])}w {len(s['doors'])}d "
              f"{len(s['windows'])}win  | consensus bboxes: {len(bboxes)}")

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
