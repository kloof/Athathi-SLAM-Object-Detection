"""Merge two SpatialLM layout .txt files.

Strategy (empirically best on TEST_SCAN):
  - Walls/doors/windows come from Qwen-0.5B (better structural recall)
  - Objects (bboxes) come from Llama-1B (better class specificity,
    e.g. dining_table that Qwen missed)

Also provides a same-class proximity deduplicator for bboxes.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Sequence

import numpy as np


_WALL_RX = re.compile(r"wall_\d+=Wall\(([-0-9.,]+)\)")
_DOOR_RX = re.compile(r"door_\d+=Door\(wall_\d+,([-0-9.,]+)\)")
_WIN_RX = re.compile(r"window_\d+=Window\(wall_\d+,([-0-9.,]+)\)")
_BBOX_RX = re.compile(r"bbox_\d+=Bbox\(([A-Za-z_\-]+),([-0-9.,]+)\)")


# Class alias groups for consensus clustering.
#
# SpatialLM flips between near-synonym class names for the same physical
# object across seeds — e.g. 8 dining chairs labeled "chair" by seed 0 and
# "dining_chair" by seed 1 fail to consensus because our clustering requires
# exact class match. Each entry lists members from MOST SPECIFIC to LEAST
# SPECIFIC; when a cluster receives votes from multiple members, it keeps
# the most-specific name. Groups are deliberately conservative — only labels
# SpatialLM actually confuses for the same visual object are grouped.
_ALIAS_GROUPS: dict[str, list[str]] = {
    # seating with a back
    "chairs":       ["dining_chair", "bar_chair", "chair"],
    # small tables next to a sofa/bed
    "side_tables":  ["nightstand", "side_table"],
    # All cabinet/storage types in one group — SpatialLM flips between
    # cupboard/sideboard/cabinet for the same physical object. Spatial
    # clustering (0.3 m radius) keeps genuinely different cabinets apart.
    "cabinets":     ["tv_cabinet", "wardrobe", "bookcase",
                     "shoe_cabinet", "entrance_cabinet",
                     "decorative_cabinet", "bathroom_cabinet",
                     "washing_cabinet", "wall_cabinet", "wine_cabinet",
                     "sideboard", "cupboard", "cabinet"],
}


def _alias_info(cls_name: str) -> tuple[str, int]:
    """Return (group_key, specificity_rank). Smaller rank = more specific.

    Classes not in any alias group are their own singleton group.
    """
    for key, members in _ALIAS_GROUPS.items():
        if cls_name in members:
            return key, members.index(cls_name)
    return cls_name, 0


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


def load_element_confidences(layout_txt: Path | str) -> dict[str, float]:
    """Read the sidecar .conf.json; keys are "wall_N"/"door_N"/"window_N"/
    "bbox_N", values are mean token log-probs (negative; closer to 0 = more
    confident). Empty dict if missing.
    """
    layout_txt = Path(layout_txt)
    conf_path = layout_txt.with_suffix("").parent / (
        layout_txt.with_suffix("").name + ".conf.json"
    )
    if not conf_path.is_file():
        return {}
    try:
        raw = json.loads(conf_path.read_text())
        return {str(k): float(v) for k, v in raw.items()}
    except Exception:
        return {}


def load_bbox_confidences(layout_txt: Path | str) -> dict[int, float]:
    """Bbox-only slice of load_element_confidences, keyed by bbox id (int).

    Also accepts legacy sidecars whose keys were bare bbox ids.
    """
    raw = load_element_confidences(layout_txt)
    out: dict[int, float] = {}
    for k, v in raw.items():
        if k.startswith("bbox_"):
            try:
                out[int(k.split("_", 1)[1])] = float(v)
            except Exception:
                continue
        elif k.isdigit():
            # legacy schema: keys were bare bbox ids
            out[int(k)] = float(v)
    return out


def load_opening_confidences(layout_txt: Path | str, kind: str) -> dict[int, float]:
    """Like load_bbox_confidences but for doors or windows. kind in
    {'door','window','wall'}."""
    raw = load_element_confidences(layout_txt)
    prefix = f"{kind}_"
    out: dict[int, float] = {}
    for k, v in raw.items():
        if k.startswith(prefix):
            try:
                out[int(k.split("_", 1)[1])] = float(v)
            except Exception:
                continue
    return out


def filter_bboxes_by_confidence(
    layout_txt: Path | str,
    min_logprob: float,
    *,
    verbose: bool = True,
) -> list[tuple[str, list[float]]]:
    """Return layout bboxes whose sidecar confidence >= min_logprob.

    Bboxes without a sidecar entry are kept unchanged (fail-open — so this
    degrades gracefully on older layout files without confidences).
    """
    parsed = parse_layout(layout_txt)
    confs = load_bbox_confidences(layout_txt)
    if not confs:
        return parsed["bboxes"]

    kept = []
    dropped_count = 0
    for bbox_id, (cls, v) in enumerate(parsed["bboxes"]):
        c = confs.get(bbox_id)
        if c is None or c >= min_logprob:
            kept.append((cls, v))
        else:
            dropped_count += 1
    if verbose and dropped_count:
        print(f"[conf-filter] {Path(layout_txt).name}: dropped "
              f"{dropped_count}/{len(parsed['bboxes'])} bboxes below "
              f"min_logprob={min_logprob}")
    return kept


def filter_openings_by_confidence(
    layout_txt: Path | str,
    kind: str,
    min_logprob: float,
    *,
    verbose: bool = True,
) -> list[list[float]]:
    """Return doors or windows whose sidecar confidence >= min_logprob.

    kind is 'door' or 'window'. Fail-open when sidecar is missing.
    """
    assert kind in ("door", "window"), kind
    parsed = parse_layout(layout_txt)
    key_in = f"{kind}s"  # 'doors' / 'windows' in parse_layout dict
    items = parsed[key_in]
    confs = load_opening_confidences(layout_txt, kind)
    if not confs:
        return items

    kept = []
    dropped_count = 0
    for idx, v in enumerate(items):
        c = confs.get(idx)
        if c is None or c >= min_logprob:
            kept.append(v)
        else:
            dropped_count += 1
    if verbose and dropped_count:
        print(f"[conf-filter] {Path(layout_txt).name}: dropped "
              f"{dropped_count}/{len(items)} {kind}s below "
              f"min_logprob={min_logprob}")
    return kept


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


def _bbox_footprint(cx, cy, yaw, sx, sy):
    """Return shapely Polygon for the 2D footprint of a rotated bbox."""
    from shapely.geometry import Polygon
    hx, hy = 0.5 * sx, 0.5 * sy
    local = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    world = local @ R.T + np.array([cx, cy])
    return Polygon(world)


def merge_overlapping_bboxes(
    bboxes: Sequence[tuple[str, list[float]]],
    *,
    iou_thresh: float = 0.15,
    yaw_tol_deg: float = 15.0,
    size_ratio_tol: float = 2.5,
    verbose: bool = True,
) -> list[tuple[str, list[float]]]:
    """Collapse same-class bboxes whose 2D footprints overlap significantly.

    Two bboxes merge when ALL hold:
      - same class
      - 2D rotated-rect IoU  >= iou_thresh
      - |yaw diff| mod pi    <= yaw_tol_deg  (handles 180° symmetry)
      - per-axis size ratio  <= size_ratio_tol (prevents merging a small
        nightstand that happens to overlap a big sofa)

    Merged bbox takes median center and median extents; yaw is the circular
    mean of the contributors (restricted to [-pi/2, pi/2) to stay symmetric).
    """
    n = len(bboxes)
    if n < 2:
        return list(bboxes)

    # precompute footprints + extract scale arrays
    polys = []
    for cls, v in bboxes:
        cx, cy, _cz, yaw, sx, sy, _sz = v
        polys.append(_bbox_footprint(cx, cy, yaw, sx, sy))

    # union-find over pairs meeting the merge predicate
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    yaw_tol = math.radians(yaw_tol_deg)

    for i in range(n):
        cls_i, v_i = bboxes[i]
        yaw_i = v_i[3]; sx_i, sy_i = v_i[4], v_i[5]
        for j in range(i + 1, n):
            cls_j, v_j = bboxes[j]
            if cls_i != cls_j:
                continue
            yaw_j = v_j[3]; sx_j, sy_j = v_j[4], v_j[5]
            dyaw = abs(yaw_i - yaw_j) % math.pi
            dyaw = min(dyaw, math.pi - dyaw)
            if dyaw > yaw_tol:
                continue
            # size sanity — if one is >N× the other along any axis, not the same object
            if max(sx_i, sx_j) > size_ratio_tol * max(1e-6, min(sx_i, sx_j)):
                continue
            if max(sy_i, sy_j) > size_ratio_tol * max(1e-6, min(sy_i, sy_j)):
                continue
            if not polys[i].intersects(polys[j]):
                continue
            inter = polys[i].intersection(polys[j]).area
            if inter <= 0:
                continue
            union_area = polys[i].union(polys[j]).area
            iou = inter / union_area if union_area > 0 else 0.0
            if iou >= iou_thresh:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    merged: list[tuple[str, list[float]]] = []
    for members in groups.values():
        if len(members) == 1:
            merged.append(bboxes[members[0]])
            continue
        cls = bboxes[members[0]][0]
        arr = np.array([bboxes[i][1] for i in members])  # (k, 7)
        center = np.median(arr[:, :3], axis=0)
        # circular mean of yaws folded into [-pi/2, pi/2) (180°-symmetric rects)
        yaws = ((arr[:, 3] + math.pi / 2) % math.pi) - math.pi / 2
        mean_yaw = float(math.atan2(np.mean(np.sin(2 * yaws)),
                                     np.mean(np.cos(2 * yaws))) / 2.0)
        scale = np.median(arr[:, 4:], axis=0)
        merged.append((cls, [float(center[0]), float(center[1]), float(center[2]),
                              mean_yaw,
                              float(scale[0]), float(scale[1]), float(scale[2])]))

    if verbose and len(merged) < n:
        print(f"[merge-overlap] {n} -> {len(merged)} "
              f"(iou>={iou_thresh}, yaw_tol={yaw_tol_deg}°, "
              f"size_ratio<={size_ratio_tol})")
    return merged


def consensus_bboxes(
    layouts: Sequence[Path | str],
    *,
    radius_m: float = 0.30,
    min_votes: int = 2,
    overlap_iou_thresh: float = 0.15,
    overlap_yaw_tol_deg: float = 15.0,
    min_bbox_confidence: float | None = None,
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
        if min_bbox_confidence is not None:
            raw = filter_bboxes_by_confidence(p, min_bbox_confidence,
                                              verbose=verbose)
        else:
            raw = parse_layout(p)["bboxes"]
        per_layout.append(dedup_bboxes(raw, radius_m=0.15))

    # Cluster across layouts. Each cluster = same class + overlapping centers.
    # Use single-pass greedy: for each (cls, box) pick existing cluster if
    # center within radius_m and NOT already populated by same layout.
    clusters: list[dict] = []  # each: {cls, centers:[], scales:[], layouts:set}

    for layout_idx, bboxes in enumerate(per_layout):
        for cls, v in bboxes:
            cx, cy, cz = v[0], v[1], v[2]
            group, rank = _alias_info(cls)
            picked = None
            best_d2 = float("inf")
            for c in clusters:
                if c["group"] != group:
                    continue
                if layout_idx in c["layouts"]:
                    continue
                mc = np.mean(c["centers"], axis=0)
                d2 = (cx - mc[0])**2 + (cy - mc[1])**2 + (cz - mc[2])**2
                if d2 < radius_m * radius_m and d2 < best_d2:
                    picked = c
                    best_d2 = d2
            if picked is None:
                clusters.append({
                    "group": group,
                    "cls": cls,
                    "rank": rank,
                    "centers": [[cx, cy, cz]],
                    "scales": [v[3:]],   # [yaw, sx, sy, sz]
                    "layouts": {layout_idx},
                })
            else:
                picked["centers"].append([cx, cy, cz])
                picked["scales"].append(v[3:])
                picked["layouts"].add(layout_idx)
                # Keep the MOST SPECIFIC name that any seed emitted for
                # this cluster (lower rank wins).
                if rank < picked["rank"]:
                    picked["cls"] = cls
                    picked["rank"] = rank

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

    # Post-consensus overlap merge: catches the "long sofa split into two
    # overlapping halves" case that center-distance clustering misses.
    if overlap_iou_thresh > 0 and len(survivors) > 1:
        survivors = merge_overlapping_bboxes(
            survivors,
            iou_thresh=overlap_iou_thresh,
            yaw_tol_deg=overlap_yaw_tol_deg,
            verbose=verbose,
        )
    return survivors


def merge_layouts(
    structure_layout: Path | str,
    objects_layout: Path | str,
    out_txt: Path | str,
    *,
    dedup_radius_m: float = 0.20,
    min_bbox_confidence: float | None = None,
    min_door_confidence: float | None = None,
    min_window_confidence: float | None = None,
    verbose: bool = True,
) -> Path:
    """Combine walls/doors/windows from `structure_layout` and bboxes
    from `objects_layout`, dedup bboxes, write unified layout file.

    Any of the `min_*_confidence` kwargs, when set, drops matching elements
    whose sidecar confidence (mean token log-prob) is below the threshold.
    """
    s = parse_layout(structure_layout)
    # Structure-side confidence filtering (Qwen hallucinates doors/windows)
    if min_door_confidence is not None:
        s_doors = filter_openings_by_confidence(
            structure_layout, "door", min_door_confidence, verbose=verbose)
    else:
        s_doors = s["doors"]
    if min_window_confidence is not None:
        s_windows = filter_openings_by_confidence(
            structure_layout, "window", min_window_confidence, verbose=verbose)
    else:
        s_windows = s["windows"]
    s = dict(s, doors=s_doors, windows=s_windows)

    if min_bbox_confidence is not None:
        raw_bboxes = filter_bboxes_by_confidence(
            objects_layout, min_bbox_confidence, verbose=verbose)
    else:
        raw_bboxes = parse_layout(objects_layout)["bboxes"]
    bboxes = dedup_bboxes(raw_bboxes, radius_m=dedup_radius_m)
    if verbose:
        print(f"[merge] struct: {len(s['walls'])}w {len(s['doors'])}d "
              f"{len(s['windows'])}win  | objects: {len(raw_bboxes)} raw "
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


def _normalize_wall_endpoints(w: list[float]) -> tuple[tuple, tuple]:
    """Return endpoints sorted lexicographically so direction doesn't matter."""
    a = (round(w[0], 3), round(w[1], 3), round(w[2], 3))
    b = (round(w[3], 3), round(w[4], 3), round(w[5], 3))
    return (a, b) if a <= b else (b, a)


def consensus_walls(
    layouts: Sequence[Path | str],
    *,
    endpoint_tol_m: float = 0.40,
    min_votes: int = 2,
    verbose: bool = True,
) -> list[list[float]]:
    """Consensus vote over wall segments across N SpatialLM layouts.

    Two walls are treated as the same segment when BOTH sorted endpoints
    are within endpoint_tol_m. Clusters with >=min_votes survive. The
    surviving wall takes the median of contributing endpoints + median
    height + thickness.
    """
    per_layout: list[list[list[float]]] = [parse_layout(p)["walls"] for p in layouts]

    clusters: list[dict] = []  # {norm_a, norm_b, walls:[full], layouts:set}
    for layout_idx, walls in enumerate(per_layout):
        for w in walls:
            a, b = _normalize_wall_endpoints(w)
            picked = None
            best_d = float("inf")
            for c in clusters:
                if layout_idx in c["layouts"]:
                    continue
                # distance = max of the two endpoint distances, after matching
                # (a1<->a2, b1<->b2) vs (a1<->b2, b1<->a2).
                d_aa = math.dist(a, c["norm_a"])
                d_bb = math.dist(b, c["norm_b"])
                d1 = max(d_aa, d_bb)
                d_ab = math.dist(a, c["norm_b"])
                d_ba = math.dist(b, c["norm_a"])
                d2 = max(d_ab, d_ba)
                d = min(d1, d2)
                if d < endpoint_tol_m and d < best_d:
                    picked = c
                    best_d = d
            if picked is None:
                clusters.append({
                    "norm_a": a, "norm_b": b,
                    "walls": [w],
                    "layouts": {layout_idx},
                })
            else:
                picked["walls"].append(w)
                picked["layouts"].add(layout_idx)

    out = []
    for c in clusters:
        if len(c["layouts"]) < min_votes:
            continue
        arr = np.array(c["walls"])  # (k, 8)
        # Re-orient each contributing wall so endpoint A sorts first.
        reoriented = []
        for w in c["walls"]:
            a, b = _normalize_wall_endpoints(w)
            reoriented.append(list(a) + list(b) + [w[6], w[7]])
        arr2 = np.array(reoriented)  # (k, 8) — [ax,ay,az,bx,by,bz,height,thickness]
        med = np.median(arr2, axis=0)
        out.append([float(x) for x in med])

    if verbose:
        print(f"[consensus-walls] {len(layouts)} passes, "
              f"{sum(len(x) for x in per_layout)} raw -> "
              f"{len(clusters)} clusters -> {len(out)} kept "
              f"(>={min_votes} votes, tol={endpoint_tol_m}m)")
    return out


def consensus_openings(
    layouts: Sequence[Path | str],
    kind: str,
    *,
    pos_tol_m: float = 0.35,
    min_votes: int = 2,
    verbose: bool = True,
) -> list[list[float]]:
    """Same voting logic, but for doors or windows (kind in {'doors','windows'}).

    Clusters by (position_x, position_y, position_z) within pos_tol_m.
    Returns list of [pos_x, pos_y, pos_z, width, height] entries.
    """
    assert kind in ("doors", "windows")
    per_layout = [parse_layout(p)[kind] for p in layouts]

    clusters: list[dict] = []
    for layout_idx, openings in enumerate(per_layout):
        for v in openings:
            px, py, pz = v[0], v[1], v[2]
            picked = None
            best_d2 = float("inf")
            for c in clusters:
                if layout_idx in c["layouts"]:
                    continue
                mc = np.mean(c["centers"], axis=0)
                d2 = (px - mc[0])**2 + (py - mc[1])**2 + (pz - mc[2])**2
                if d2 < pos_tol_m * pos_tol_m and d2 < best_d2:
                    picked = c
                    best_d2 = d2
            if picked is None:
                clusters.append({
                    "centers": [[px, py, pz]],
                    "vs": [v],
                    "layouts": {layout_idx},
                })
            else:
                picked["centers"].append([px, py, pz])
                picked["vs"].append(v)
                picked["layouts"].add(layout_idx)

    out = []
    for c in clusters:
        if len(c["layouts"]) < min_votes:
            continue
        arr = np.array(c["vs"])  # (k, 5) — [px, py, pz, width, height]
        med = np.median(arr, axis=0)
        out.append([float(x) for x in med])

    if verbose:
        print(f"[consensus-{kind[:-1]}] {len(layouts)} passes, "
              f"{sum(len(x) for x in per_layout)} raw -> "
              f"{len(clusters)} clusters -> {len(out)} kept "
              f"(>={min_votes} votes, tol={pos_tol_m}m)")
    return out


def merge_layouts_full_consensus(
    structure_layouts: Sequence[Path | str],
    objects_layouts: Sequence[Path | str],
    out_txt: Path | str,
    *,
    struct_min_votes: int = 2,
    object_min_votes: int = 2,
    object_consensus_radius_m: float = 0.30,
    verbose: bool = True,
) -> Path:
    """Consensus over BOTH Qwen structure and Llama objects.

    Qwen runs N times, walls/doors/windows each voted; surviving elements
    take the median of their contributors. Llama runs N times, bboxes
    voted via consensus_bboxes.
    """
    walls   = consensus_walls(structure_layouts, min_votes=struct_min_votes,
                              verbose=verbose)
    doors   = consensus_openings(structure_layouts, "doors",
                                 min_votes=struct_min_votes, verbose=verbose)
    windows = consensus_openings(structure_layouts, "windows",
                                 min_votes=struct_min_votes, verbose=verbose)
    bboxes = consensus_bboxes(objects_layouts,
                              radius_m=object_consensus_radius_m,
                              min_votes=object_min_votes,
                              verbose=verbose)
    if verbose:
        print(f"[merge] struct: {len(walls)}w {len(doors)}d {len(windows)}win  "
              f"| consensus bboxes: {len(bboxes)}")

    out_lines = []
    for i, w in enumerate(walls):
        out_lines.append(f"wall_{i}=Wall({','.join(str(x) for x in w)})")
    for i, d in enumerate(doors):
        out_lines.append(f"door_{i}=Door(wall_0,{','.join(str(x) for x in d)})")
    for i, d in enumerate(windows):
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
    min_bbox_confidence: float | None = None,
    min_door_confidence: float | None = None,
    min_window_confidence: float | None = None,
    verbose: bool = True,
) -> Path:
    """Same output shape as merge_layouts, but bboxes come from a consensus
    over N Llama passes (kept only when seen in >= min_votes passes).

    `min_bbox_confidence` applies a per-bbox confidence threshold on each
    pass before voting (uses the .conf.json sidecar). Similarly,
    `min_door_confidence` / `min_window_confidence` filter Qwen's
    structural openings by their sidecar confidence.
    """
    s = parse_layout(structure_layout)
    if min_door_confidence is not None:
        s_doors = filter_openings_by_confidence(
            structure_layout, "door", min_door_confidence, verbose=verbose)
    else:
        s_doors = s["doors"]
    if min_window_confidence is not None:
        s_windows = filter_openings_by_confidence(
            structure_layout, "window", min_window_confidence, verbose=verbose)
    else:
        s_windows = s["windows"]
    s = dict(s, doors=s_doors, windows=s_windows)

    bboxes = consensus_bboxes(
        objects_layouts,
        radius_m=consensus_radius_m,
        min_votes=min_votes,
        min_bbox_confidence=min_bbox_confidence,
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
