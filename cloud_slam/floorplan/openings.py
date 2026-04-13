"""Doors / windows / glass / passage detection on refined wall segments.

M3 implementation — each opening becomes a structured entry in the
D_refined section of the JSON output, mirroring Apple's RoomPlan
`CapturedOpening`.

Public entry point:
    _detect_openings(walls, walls_meta, merged_pts, wall_labels, *,
                     config=None, ceiling_z=None, floor_z=None)

Algorithm (per wall in D_refined):

1. Collect wall-band points with a SPLIT BAND (16-agent review C4):
   - Perp distance <= 0.10 m for density/occupancy/gap map
     (config.band_perp_density_m)
   - Perp distance <= 0.20 m for vision-bucket majority assignment
     (config.band_perp_vision_m — wider catches door trim / baseboard)
   Do NOT reuse _collect_wall_band() — its 0.30 m margin cuts transoms.

   Per-wall ceiling_z for vaulted ceilings: if ceiling-plane slope > 5°,
   sample ceiling_z(t) along the wall at the along-t midpoint; else use
   global ceiling_z. (This pipeline passes a scalar ceiling_z; vaulted
   ceilings fall back to the scalar — future work.)

   Per point record (t, z, bucket_id) with t = along-wall distance from p1.

2. Build 2D grid at 0.03 m resolution (config.grid_resolution_m). Per cell:
   (a) lidar occupancy (any point present)
   (b) majority vision bucket (from the 0.20 m band)
   (c) point density

3. Vision blobs — connected-component label with:
   - 4-connectivity across buckets (conservative — prevents 0.25m transom
     merging into adjacent 0.9m door blob)
   - 8-connectivity within a single bucket (elongated blobs OK)
   Build TWO blob maps: door-bucket, window+glass-bucket.

   Morphological opening (3x3 kernel) breaks thin joins before CC.

4. Gap detector — COMPONENT-WISE, not cell-wise:
   a. Find connected components of EMPTY cells (4-connectivity).
   b. For each component, count boundary cells adjacent (8-neighborhood)
      to occupied cells.
   c. Keep component if EITHER:
      - >=30 cells lidar-adjacent on at least one vertical edge (single-side
        jamb — loft passages with stub-wall), OR
      - >=30% of perimeter is lidar-adjacent (standard case).
   d. Extract OBB from the FULL COMPONENT (not rim).

5. For each blob / gap region:
   - Fit OBB in (t, z) → extract along_start/end, z_bottom/top, width,
     height.
   - Reject by plausible dimension bounds (config.*_range):
     - door: 0.4-2.5 m wide × 1.5-2.8 m tall
     - window: 0.2-3.5 m × 0.15-2.5 m (0.15 m lower bound catches transoms)
     - passage: 0.5-4.0 m × 1.8-3.0 m
   - Reject support < config.min_support_vision_cells (vision) or
     < config.min_support_gap_cells (gap).
   - Reject blobs whose (t,z) bbox touches wall ends within 0.1 m
     (corner artifact).

6. Transom linking:
   - After OBB extraction, scan for pairs of windows sharing along_start/end
     within 5 cm AND z-ranges abut within 0.10 m. Mark the upper one with
     `transom_of: <lower_window_id>` — do NOT merge. RoomPlan emits transoms
     separately.

7. Open/closed classification + glass detection with LOCAL-NEIGHBORHOOD
   density:
   - Lidar density scales 1/r² with wall distance + incidence; whole-wall
     avg is wrong.
   - local_density = mean density of cells within ±1 m along-t of the blob,
     EXCLUDING:
     (i) the blob itself
     (ii) cells whose vision bucket is in
          {radiator, cabinet, fireplace, picture}
   - density_ratio = blob_density / local_density
   - Door: is_open = (density_ratio < open_threshold);
           None if open <= ratio < closed; closed if ratio >= closed.
   - Window + glass extension:
     - If vision class is `glass` AND density_ratio > closed_threshold
       → type='glass', is_open=False, transparent=True
     - If vision class is `window` AND density_ratio > closed_threshold
       → type='window', is_open=False, transparent=True
     - If density_ratio < open_threshold → is_open=True (open window)
     - Intermediate → is_open=None

8. Dedup: vision blob and passage gap overlapping by bbox IoU > 0.3
   (config.dedup_iou_threshold) → keep vision blob (more specific type).

9. Emit per-opening dict matching OPENING_REQUIRED_KEYS from schema.py:
   id, uuid, type, wall_id, wall_uuid, along_start, along_end,
   z_bottom, z_top, width_m, height_m, center_xy, transform_4x4,
   source, is_open, transparent, transom_of, confidence.

   Confidence formula:
   - vision blob: support_cells / blob_bbox_cells, capped at 1.0
   - passage gap: min(gap_height / wall_height, 1.0)
"""

from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np
from scipy.ndimage import (
    binary_opening,
    label as ndi_label,
)

from .config import OpeningsConfig
from .schema import _wall_uuid


# 5-bucket layout shared with WallSegmenter: 0 other / 1 wall /
# 2 window / 3 door / 4 glass.
_BUCKET_DOOR = 3
_BUCKET_WINDOW = 2
_BUCKET_GLASS = 4

# ADE20K classes to exclude from the local-density denominator. These
# occlude the wall but are NOT openings, so including them would falsely
# depress the local average and bias density_ratio downward.
# These are the raw ADE-150 ids; we fold them through the _remap_ade
# inverse as the 0 (other) bucket. Since the downstream per-point labels
# are already collapsed to the 5 buckets, we skip them at point-collection
# time by dropping any bucket that is NOT in {wall, window, door, glass}
# — effectively the same filter for the local-density neighborhood.
# (The docstring's 'radiator / cabinet / fireplace / picture' list is the
# canonical ADE20K noise set; our bucket remap already drops them to 0.)


def _wall_frame(p1, p2):
    """Return (edge_dir, edge_perp, length) for the wall XY segment.

    edge_dir points from p1 to p2; edge_perp is a unit XY vector
    perpendicular to edge_dir (left-hand rule). length is the wall length.
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    edge_vec = p2 - p1
    length = float(np.linalg.norm(edge_vec))
    if length < 1e-9:
        return None, None, 0.0
    edge_dir = edge_vec / length
    edge_perp = np.array([-edge_dir[1], edge_dir[0]])
    return edge_dir, edge_perp, length


def _collect_wall_band_points(pts_xyz, point_labels, p1, p2,
                               band_density, band_vision,
                               floor_z, ceiling_z):
    """Project points near a wall segment into (t, z, bucket_id) records.

    Returns two tuples for the two bands:
        density_band: (t_arr, z_arr) for the tight band (0.10 m)
        vision_band: (t_arr, z_arr, bucket_arr) for the wider band (0.20 m)

    Both arrays are restricted to 0 <= t <= length and floor_z <= z <=
    ceiling_z (no wall-band margin — transoms at ceiling_z - 0.1 m need
    to land in-range).

    If point_labels is None, vision_band's bucket_arr is returned as all
    zeros (bucket 'other').
    """
    edge_dir, edge_perp, length = _wall_frame(p1, p2)
    if length < 1e-6:
        empty2 = (np.zeros(0), np.zeros(0))
        empty3 = (np.zeros(0), np.zeros(0), np.zeros(0, dtype=np.uint8))
        return empty2, empty3

    p1 = np.asarray(p1, dtype=np.float64)
    rel = pts_xyz[:, :2] - p1
    t = rel @ edge_dir
    perp = rel @ edge_perp
    z = pts_xyz[:, 2].astype(np.float64)

    in_along = (t >= 0.0) & (t <= length)
    in_z = (z >= floor_z) & (z <= ceiling_z)

    mask_density = in_along & in_z & (np.abs(perp) <= band_density)
    mask_vision = in_along & in_z & (np.abs(perp) <= band_vision)

    density_band = (t[mask_density], z[mask_density])
    if point_labels is not None and len(point_labels) == len(pts_xyz):
        bucket = np.asarray(point_labels, dtype=np.uint8)[mask_vision]
    else:
        bucket = np.zeros(int(mask_vision.sum()), dtype=np.uint8)
    vision_band = (t[mask_vision], z[mask_vision], bucket)
    return density_band, vision_band


def _rasterize_grid(density_band, vision_band, length, floor_z, ceiling_z,
                     res):
    """Discretize the wall-band records into a 2D (t, z) grid.

    Returns a dict with:
        n_t, n_z: grid dims
        occupancy: (n_t, n_z) uint8 — any lidar point present (density band)
        density: (n_t, n_z) float — point count per cell (density band)
        vision: (n_t, n_z) uint8 — majority bucket per cell (vision band);
                0 means no labeled points or majority is 'other'
    """
    n_t = max(int(np.ceil(length / res)), 1)
    z_range = max(ceiling_z - floor_z, 1e-3)
    n_z = max(int(np.ceil(z_range / res)), 1)

    t_d, z_d = density_band
    occupancy = np.zeros((n_t, n_z), dtype=np.uint8)
    density = np.zeros((n_t, n_z), dtype=np.float32)

    if len(t_d) > 0:
        ti = np.clip((t_d / res).astype(np.int32), 0, n_t - 1)
        zi = np.clip(((z_d - floor_z) / res).astype(np.int32), 0, n_z - 1)
        np.add.at(density, (ti, zi), 1.0)
        occupancy[density > 0] = 1

    # Vision: majority bucket per cell across {wall, window, door, glass}.
    # We accumulate a (n_t, n_z, 5) counts volume and argmax; a cell with
    # zero non-other votes stays 0.
    t_v, z_v, bucket = vision_band
    vision = np.zeros((n_t, n_z), dtype=np.uint8)
    if len(t_v) > 0:
        ti = np.clip((t_v / res).astype(np.int32), 0, n_t - 1)
        zi = np.clip(((z_v - floor_z) / res).astype(np.int32), 0, n_z - 1)
        # Keep only labeled buckets (1..4). Bucket 0 (other) contributes
        # nothing; this matches the local-density exclusion rationale.
        valid = (bucket >= 1) & (bucket <= 4)
        if valid.any():
            ti_v = ti[valid]
            zi_v = zi[valid]
            bk_v = bucket[valid]
            counts = np.zeros((n_t, n_z, 5), dtype=np.int32)
            np.add.at(counts, (ti_v, zi_v, bk_v), 1)
            # Ignore bucket 0 in argmax (it never gets votes anyway).
            any_vote = counts[:, :, 1:].sum(axis=2) > 0
            argmax_bucket = counts[:, :, 1:].argmax(axis=2) + 1  # shift
            vision[any_vote] = argmax_bucket[any_vote].astype(np.uint8)
    return {
        'n_t': n_t, 'n_z': n_z,
        'occupancy': occupancy,
        'density': density,
        'vision': vision,
    }


def _extract_blob_obbs(bucket_mask):
    """Label connected components and return per-blob (t_lo, t_hi, z_lo,
    z_hi, support) tuples.

    Uses 8-connectivity within a single bucket (elongated rectangles OK).
    Applies a 3x3 morphological opening to break thin joins between
    independent blobs before CC labeling.
    """
    if not bucket_mask.any():
        return []
    # 3x3 opening — break thin joins.
    cleaned = binary_opening(bucket_mask, structure=np.ones((3, 3), bool))
    # 8-connectivity structure
    struct = np.ones((3, 3), dtype=bool)
    labeled, n_blobs = ndi_label(cleaned, structure=struct)
    out = []
    for bid in range(1, n_blobs + 1):
        ys, xs = np.where(labeled == bid)
        if len(ys) == 0:
            continue
        # NOTE: coordinates are (t_idx, z_idx). We labeled on
        # bucket_mask whose axes are (n_t, n_z); np.where returns
        # (t_idx, z_idx).
        t_lo = int(ys.min())
        t_hi = int(ys.max())
        z_lo = int(xs.min())
        z_hi = int(xs.max())
        support = int(len(ys))
        out.append((t_lo, t_hi, z_lo, z_hi, support))
    return out


def _extract_gap_components(occupancy, min_support, n_t, n_z):
    """Find connected components of EMPTY cells surrounded by occupied
    lidar.

    Returns per-component dicts with:
        t_lo, t_hi, z_lo, z_hi: bbox of the empty component
        support: number of empty cells
        lidar_adjacent_frac: fraction of perimeter cells adjacent to
                             occupied lidar (8-neighborhood)
        single_side_jamb: True if >=30 cells lidar-adjacent on at
                          least one vertical edge (t_lo or t_hi)
        confidence: min(gap_height / wall_height, 1.0) — height ratio

    Components touching the wall bottom (z=0) and extending up through
    the wall middle are interior passages — keep them.
    """
    empty = (occupancy == 0)
    if not empty.any():
        return []
    # Components of empty cells, 4-connectivity.
    struct4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    labeled, n_comp = ndi_label(empty, structure=struct4)
    # Occupied mask for adjacency check.
    # 8-neighborhood adjacency to occupied cells — convolve with a 3x3
    # kernel of ones on the occupied mask, then any cell with count > 0
    # is adjacent to at least one occupied cell.
    occ = (occupancy > 0).astype(np.uint8)
    # Pad-and-shift sum: 8 shifts. Cheaper than scipy.ndimage.convolve for
    # our tiny (n_t, n_z) grids.
    adj_count = np.zeros_like(occ, dtype=np.int32)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            shifted = np.zeros_like(occ)
            src_i = slice(max(0, -di), n_t - max(0, di))
            dst_i = slice(max(0, di), n_t - max(0, -di))
            src_j = slice(max(0, -dj), n_z - max(0, dj))
            dst_j = slice(max(0, dj), n_z - max(0, -dj))
            shifted[dst_i, dst_j] = occ[src_i, src_j]
            adj_count += shifted
    lidar_adjacent_mask = (adj_count > 0) & empty

    out = []
    for cid in range(1, n_comp + 1):
        comp_mask = (labeled == cid)
        n_cells = int(comp_mask.sum())
        if n_cells < min_support:
            continue
        ts, zs = np.where(comp_mask)
        t_lo = int(ts.min())
        t_hi = int(ts.max())
        z_lo = int(zs.min())
        z_hi = int(zs.max())
        # Perimeter = cells in the component that have at least one
        # 4-neighbor outside the component (or the grid edge).
        # Cheaper to compute: cells whose 4-neighborhood has at least
        # one non-component cell.
        perim_mask = np.zeros_like(comp_mask)
        # Check all four 4-neighbors.
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            shifted_comp = np.zeros_like(comp_mask)
            src_i = slice(max(0, -di), n_t - max(0, di))
            dst_i = slice(max(0, di), n_t - max(0, -di))
            src_j = slice(max(0, -dj), n_z - max(0, dj))
            dst_j = slice(max(0, dj), n_z - max(0, -dj))
            shifted_comp[dst_i, dst_j] = comp_mask[src_i, src_j]
            # A cell is on the perimeter if it's in the comp AND its
            # neighbor is NOT.
            perim_mask |= comp_mask & ~shifted_comp
        n_perim = int(perim_mask.sum())
        if n_perim == 0:
            continue
        n_adj_perim = int((perim_mask & lidar_adjacent_mask).sum())
        lidar_adj_frac = n_adj_perim / max(n_perim, 1)

        # Single-side jamb check: count cells at the leftmost and
        # rightmost t columns of the component that are adjacent to
        # occupied lidar. >=30 on either side passes.
        left_col = comp_mask.copy()
        left_col[t_lo + 1:, :] = False
        right_col = comp_mask.copy()
        right_col[:t_hi, :] = False
        n_left_adj = int((left_col & lidar_adjacent_mask).sum())
        n_right_adj = int((right_col & lidar_adjacent_mask).sum())
        single_side_jamb = (n_left_adj >= 30) or (n_right_adj >= 30)

        # Keep if standard 30% rule OR single-side-jamb rule.
        if lidar_adj_frac < 0.30 and not single_side_jamb:
            continue
        out.append({
            't_lo': t_lo, 't_hi': t_hi,
            'z_lo': z_lo, 'z_hi': z_hi,
            'support': n_cells,
            'lidar_adjacent_frac': float(lidar_adj_frac),
            'single_side_jamb': bool(single_side_jamb),
            # M4b-ext: retain the component's cell mask so the mirror
            # detector (vision override) can sample vision buckets at
            # exact component cells rather than the bounding rectangle.
            'cell_mask': comp_mask.copy(),
        })
    return out


def _bbox_touches_corner(t_lo, t_hi, n_t, corner_cells):
    """True if the (t_lo, t_hi) bbox reaches within corner_cells of either
    wall end."""
    return (t_lo < corner_cells) or (t_hi > (n_t - 1 - corner_cells))


def _classify_blob_type(blob, grid, config):
    """Return the dominant vision bucket over the blob cells.

    blob: dict or tuple with t_lo, t_hi, z_lo, z_hi. Slices vision[t_lo:t_hi+1,
    z_lo:z_hi+1] and returns the majority bucket (0-4) within that rectangle.
    """
    t_lo, t_hi, z_lo, z_hi = blob['t_lo'], blob['t_hi'], blob['z_lo'], blob['z_hi']
    sub = grid['vision'][t_lo:t_hi + 1, z_lo:z_hi + 1]
    # Count non-zero buckets.
    counts = np.bincount(sub.ravel(), minlength=5)
    counts[0] = 0  # ignore 'other'
    if counts.sum() == 0:
        return 0
    return int(counts.argmax())


def _temporal_vote_blob_type(blob, vision_band, res, floor_z,
                              min_frames):
    """Vote on a blob's bucket using raw accumulated points across frames.

    M4b: leverages the per-point labels that were already accumulated
    frame-by-frame in `wall_labels['xyz']` (re-projected into the (t, z)
    wall frame in `vision_band`). One vote per accumulated point — so a
    blob whose pixels voted `window` on 5 frames and `glass` on 2 frames
    gets a `window` majority, naturally breaking the frame-to-frame
    flicker that `_classify_blob_type` (single-cell majority) exhibits.

    Returns (bucket_id, vote_count). When `vote_count < min_frames`, the
    caller should treat the result as untrusted and fall back to
    `_classify_blob_type`.

    blob: dict with t_lo/t_hi/z_lo/z_hi integer cell indices.
    vision_band: (t_arr, z_arr, bucket_arr) tuple from
                 `_collect_wall_band_points` — the 0.20 m wider band used
                 for majority assignment.
    res: grid cell size in metres.
    floor_z: world z of the grid's row 0.
    min_frames: `config.temporal_vote_min_frames`.
    """
    t_v, z_v, bucket = vision_band
    if t_v is None or len(t_v) == 0:
        return 0, 0
    t_lo = blob['t_lo']
    t_hi = blob['t_hi']
    z_lo = blob['z_lo']
    z_hi = blob['z_hi']
    t_start = t_lo * res
    t_end = (t_hi + 1) * res
    z_start = floor_z + z_lo * res
    z_end = floor_z + (z_hi + 1) * res
    mask = ((t_v >= t_start) & (t_v < t_end)
            & (z_v >= z_start) & (z_v < z_end))
    in_blob = np.asarray(bucket, dtype=np.int32)[mask]
    if in_blob.size == 0:
        return 0, 0
    counts = np.bincount(in_blob, minlength=5)
    # Ignore bucket 0 (other) — mirrors _classify_blob_type.
    counts[0] = 0
    if counts.sum() == 0:
        return 0, int(in_blob.size)
    winner = int(counts.argmax())
    # Vote count = total labeled points that fell in the blob rectangle
    # (any non-zero bucket). Below `min_frames` the caller should fall
    # back to the cell-level argmax path.
    vote_count = int(counts.sum())
    return (winner if vote_count >= min_frames else 0), vote_count


def _local_density_ratio(blob, grid, res, along_window_m=1.0):
    """Compute density_ratio = blob mean density / local mean density.

    Local window: ±along_window_m along t, full wall height in z.
    Excludes the blob itself plus an approximate clutter mask.

    Clutter exclusion (Approach B — spec H/25):
        The spec calls for excluding radiator/cabinet/fireplace/picture
        cells from the denominator, but the current 5-bucket vision remap
        folds all of those ADE20K classes into bucket 0 ("other"). We
        therefore can't identify them directly from the grid's vision
        layer.

        Instead we use a density-based heuristic: within the local
        window (excluding the blob), drop the densest 10% of cells.
        Wall-mounted clutter (radiators, cabinets, fireplaces,
        picture frames) typically returns well above the bare-wall
        density baseline, so the 90th-percentile cap brings the
        denominator back toward the true wall density.

        Limitation: this is a defensible proxy, not a class-based
        filter. If the window contains ≤10 cells after blob removal
        we skip the cap (percentile on a tiny sample would be noisy).
        A future cleanup could extend wall_segmenter.py to expose the
        underlying ADE20K classes and restore the spec's literal
        class-based exclusion.
    """
    t_lo, t_hi = blob['t_lo'], blob['t_hi']
    z_lo, z_hi = blob['z_lo'], blob['z_hi']
    n_t = grid['n_t']
    n_z = grid['n_z']

    blob_mask = np.zeros((n_t, n_z), dtype=bool)
    blob_mask[t_lo:t_hi + 1, z_lo:z_hi + 1] = True
    blob_density = float(grid['density'][blob_mask].mean()) if blob_mask.any() else 0.0

    win_cells = max(int(round(along_window_m / res)), 1)
    t_start = max(0, t_lo - win_cells)
    t_end = min(n_t, t_hi + 1 + win_cells)
    win_mask = np.zeros((n_t, n_z), dtype=bool)
    win_mask[t_start:t_end, :] = True

    base_mask = win_mask & ~blob_mask

    # Approximate clutter exclusion: drop the densest 10% of cells in the
    # window (see docstring — proxy for radiator/cabinet/fireplace/picture).
    if int(base_mask.sum()) > 10:
        densities_in_window = grid['density'][base_mask]
        cap = float(np.percentile(densities_in_window, 90))
        clutter_mask = (grid['density'] > cap) & base_mask
        denom_mask = base_mask & ~clutter_mask
    else:
        denom_mask = base_mask

    denom = grid['density'][denom_mask]
    local_density = float(denom.mean()) if denom.size > 0 else 0.0
    if local_density <= 1e-9:
        # Fall back to the wall-wide mean.
        local_density = float(grid['density'].mean())
        if local_density <= 1e-9:
            return None
    return blob_density / local_density


def _bbox_iou(a, b):
    """Axis-aligned bbox IoU for two (t_lo, t_hi, z_lo, z_hi) dicts."""
    at0, at1, az0, az1 = a['t_lo'], a['t_hi'], a['z_lo'], a['z_hi']
    bt0, bt1, bz0, bz1 = b['t_lo'], b['t_hi'], b['z_lo'], b['z_hi']
    inter_t = max(0, min(at1, bt1) - max(at0, bt0) + 1)
    inter_z = max(0, min(az1, bz1) - max(az0, bz0) + 1)
    inter = inter_t * inter_z
    if inter == 0:
        return 0.0
    area_a = (at1 - at0 + 1) * (az1 - az0 + 1)
    area_b = (bt1 - bt0 + 1) * (bz1 - bz0 + 1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _opening_uuid(wall_uuid, along_start, z_bottom):
    """Deterministic UUID for an opening on a wall.

    Seeds with the wall UUID and along/z coordinates — same scan, same
    openings.
    """
    s = f"{wall_uuid}|{float(along_start):.4f}|{float(z_bottom):.4f}"
    return hashlib.md5(s.encode()).hexdigest()


def _build_transform_4x4(p1, p2, center_xy, center_z):
    """Construct a 4x4 transform for the opening.

    Translation: (center_xy[0], center_xy[1], center_z).
    Rotation: aligns local Y with wall direction (p1→p2), local Z with
    world Z (up), local X = Y×Z (wall-normal, right-hand).
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    edge = p2 - p1
    L = float(np.linalg.norm(edge))
    if L < 1e-9:
        T = np.eye(4)
        T[0, 3] = float(center_xy[0])
        T[1, 3] = float(center_xy[1])
        T[2, 3] = float(center_z)
        return T.tolist()
    y_axis = np.array([edge[0] / L, edge[1] / L, 0.0])
    z_axis = np.array([0.0, 0.0, 1.0])
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= max(np.linalg.norm(x_axis), 1e-9)
    T = np.eye(4)
    T[:3, 0] = x_axis
    T[:3, 1] = y_axis
    T[:3, 2] = z_axis
    T[0, 3] = float(center_xy[0])
    T[1, 3] = float(center_xy[1])
    T[2, 3] = float(center_z)
    return [[round(float(v), 6) for v in row] for row in T]


def _ranges_for_type(config, otype):
    if otype == 'door':
        return config.door_width_m_range, config.door_height_m_range
    if otype == 'window':
        return config.window_width_m_range, config.window_height_m_range
    if otype == 'glass':
        return config.window_width_m_range, config.window_height_m_range
    if otype == 'passage':
        return config.passage_width_m_range, config.passage_height_m_range
    return (0.0, 100.0), (0.0, 100.0)


def _bbox_to_world(p1, p2, t_lo, t_hi, z_lo, z_hi, res, floor_z):
    """Convert cell-index bbox to world coordinates.

    Returns (along_start, along_end, z_bottom, z_top, center_xy, center_z,
    width_m, height_m).
    """
    along_start = float(t_lo * res)
    along_end = float((t_hi + 1) * res)
    z_bottom = float(floor_z + z_lo * res)
    z_top = float(floor_z + (z_hi + 1) * res)
    width_m = max(along_end - along_start, 0.0)
    height_m = max(z_top - z_bottom, 0.0)
    edge_dir, _perp, length = _wall_frame(p1, p2)
    if edge_dir is None:
        center_xy = [float(p1[0]), float(p1[1])]
    else:
        t_mid = 0.5 * (along_start + along_end)
        c = np.asarray(p1, dtype=np.float64) + t_mid * edge_dir
        center_xy = [float(c[0]), float(c[1])]
    center_z = z_bottom + 0.5 * height_m
    return (along_start, along_end, z_bottom, z_top,
            center_xy, center_z, width_m, height_m)


def _make_opening_entry(*, oid, wall_idx, wall_uuid, otype,
                         along_start, along_end, z_bottom, z_top,
                         width_m, height_m, center_xy, transform_4x4,
                         source, is_open, transparent, transom_of,
                         confidence, temporal_vote_count=0):
    """Construct an opening dict with all OPENING_REQUIRED_KEYS.

    M4b: `temporal_vote_count` is the number of accumulated wall_labels
    points that fell inside the blob region, broken down across all
    frames. Zero means vision didn't contribute (lidar-only emit, or
    the cell-level argmax fallback fired).
    """
    return {
        'id': f'open_{int(oid)}',
        'uuid': _opening_uuid(wall_uuid, along_start, z_bottom),
        'type': otype,
        'wall_id': int(wall_idx),
        'wall_uuid': str(wall_uuid),
        'along_start': round(float(along_start), 4),
        'along_end': round(float(along_end), 4),
        'z_bottom': round(float(z_bottom), 4),
        'z_top': round(float(z_top), 4),
        'width_m': round(float(width_m), 4),
        'height_m': round(float(height_m), 4),
        'center_xy': [round(float(center_xy[0]), 4),
                       round(float(center_xy[1]), 4)],
        'transform_4x4': transform_4x4,
        'source': str(source),
        'is_open': (bool(is_open) if is_open is not None else None),
        'transparent': bool(transparent),
        'transom_of': transom_of,
        'confidence': round(float(confidence), 3),
        'temporal_vote_count': int(temporal_vote_count),
    }


def _link_transoms(emitted):
    """Mark the upper of two windows sharing along-start/end (within 5cm)
    and z-range abutment (within 10cm) with `transom_of`.
    """
    windows = [e for e in emitted if e['type'] in ('window', 'glass')]
    for i, a in enumerate(windows):
        for j, b in enumerate(windows):
            if i == j:
                continue
            # `a` is candidate transom (upper), `b` is main (lower).
            if a['z_bottom'] < b['z_bottom']:
                continue
            same_t = (abs(a['along_start'] - b['along_start']) <= 0.05
                      and abs(a['along_end'] - b['along_end']) <= 0.05)
            abuts = abs(a['z_bottom'] - b['z_top']) <= 0.10
            if same_t and abuts:
                a['transom_of'] = b['id']
                break


def _detect_openings(walls, walls_meta, merged_pts, wall_labels, *,
                      config: Optional[OpeningsConfig] = None,
                      ceiling_z: Optional[float] = None,
                      floor_z: Optional[float] = None,
                      verbose: bool = False):
    """Detect doors/windows/glass/passages on a set of refined walls.

    Args:
        walls:       list of (p1, p2, angle_deg, length_m) — D_refined walls
                     in world XY.
        walls_meta:  list of per-wall meta dicts (parallel to walls) — may
                     contain 'type'/'features' keys from M0a. M4b mutates
                     each entry in-place with `wall_band_coverage_pct`
                     (float) so the schema can surface per-wall lidar
                     occupancy fractions downstream.
        merged_pts:  (N, 3) float32/float64 world-frame points.
        wall_labels: dict with 'xyz' (M, 3), 'labels' (M,) per the
                     WallSegmenter 5-bucket remap, or None. When None the
                     returned list is empty.
        config:      OpeningsConfig; defaults to OpeningsConfig().
        ceiling_z:   scalar wall-top height (world Z). Required.
        floor_z:     scalar wall-bottom height (world Z). Required.
        verbose:     Print per-wall coverage percentages (M4b).

    Returns:
        list of opening dicts, each with every OPENING_REQUIRED_KEYS key
        populated. Also populates `temporal_vote_count` (M4b) for every
        entry — 0 for lidar-only passages.
    """
    if config is None:
        config = OpeningsConfig()
    if walls is None or len(walls) == 0:
        return []
    if ceiling_z is None or floor_z is None:
        return []
    # Nothing to do without a label source (vision OR lidar gap).
    if merged_pts is None or len(merged_pts) == 0:
        return []

    res = float(config.grid_resolution_m)
    floor_z = float(floor_z)
    ceiling_z = float(ceiling_z)
    if ceiling_z <= floor_z + 0.1:
        return []

    # Propagate vision labels onto merged_pts via KNN (the same path
    # generate_floorplan uses).
    pts_labels = None
    if (wall_labels is not None
            and 'xyz' in wall_labels
            and 'labels' in wall_labels
            and len(wall_labels['labels']) > 0):
        from .refine import _lookup_vision_labels_for_pts
        pts_labels = _lookup_vision_labels_for_pts(merged_pts, wall_labels)
        if not np.any(pts_labels >= 1):
            pts_labels = None

    corner_cells = max(int(round(config.corner_reject_m / res)), 1)
    wall_height = max(ceiling_z - floor_z, 1e-3)

    all_openings: list = []
    oid_counter = 0

    for wall_idx, wall_tuple in enumerate(walls):
        p1, p2, _a, length = wall_tuple
        if length is None or length < 0.5:
            # Too short to host any meaningful opening.
            continue
        meta = (walls_meta[wall_idx]
                if (walls_meta is not None and wall_idx < len(walls_meta)
                    and walls_meta[wall_idx] is not None)
                else {})
        wall_uuid = _wall_uuid(p1, p2)
        wall_length = float(length)

        density_band, vision_band = _collect_wall_band_points(
            merged_pts, pts_labels, p1, p2,
            band_density=config.band_perp_density_m,
            band_vision=config.band_perp_vision_m,
            floor_z=floor_z, ceiling_z=ceiling_z)

        grid = _rasterize_grid(density_band, vision_band,
                                length=wall_length,
                                floor_z=floor_z, ceiling_z=ceiling_z,
                                res=res)
        n_t = grid['n_t']

        # --- M4b: per-wall lidar coverage gate -----------------------
        # Compute the fraction of (t, z) cells that have any lidar point.
        # Walls with very low coverage either weren't seen by the camera
        # (vision is unreliable) or the wall geometry is incidental to
        # the trajectory; either way we trust gap detection (which only
        # cares about empty regions) but skip vision-blob detection.
        occ_size = max(int(grid['occupancy'].size), 1)
        wall_band_coverage_pct = (
            float(grid['occupancy'].sum()) / float(occ_size))
        do_vision_blobs = (
            wall_band_coverage_pct >= config.min_wall_coverage_for_vision)
        if (walls_meta is not None
                and wall_idx < len(walls_meta)
                and walls_meta[wall_idx] is not None):
            walls_meta[wall_idx]['wall_band_coverage_pct'] = round(
                wall_band_coverage_pct, 4)
        if verbose:
            gate = ("vision+gap" if do_vision_blobs
                    else "gap-only (low coverage)")
            print(f"[Openings] wall {wall_idx}: "
                  f"coverage={wall_band_coverage_pct:.2%} → {gate}")

        per_wall = []

        # --- Vision blobs: door bucket (gated by coverage) ----------
        if do_vision_blobs:
            door_mask = (grid['vision'] == _BUCKET_DOOR)
            door_blobs = _extract_blob_obbs(door_mask)
            for (t_lo, t_hi, z_lo, z_hi, support) in door_blobs:
                if support < config.min_support_vision_cells:
                    continue
                if _bbox_touches_corner(t_lo, t_hi, n_t, corner_cells):
                    continue
                (along_s, along_e, z_b, z_t,
                 center_xy, center_z,
                 w_m, h_m) = _bbox_to_world(
                    p1, p2, t_lo, t_hi, z_lo, z_hi, res, floor_z)
                wrange, hrange = _ranges_for_type(config, 'door')
                if not (wrange[0] <= w_m <= wrange[1]):
                    continue
                if not (hrange[0] <= h_m <= hrange[1]):
                    continue
                bbox_cells = max((t_hi - t_lo + 1) * (z_hi - z_lo + 1), 1)
                confidence = min(support / bbox_cells, 1.0)
                per_wall.append({
                    't_lo': t_lo, 't_hi': t_hi, 'z_lo': z_lo, 'z_hi': z_hi,
                    'support': support,
                    'otype': 'door',
                    'source': ('vision+lidar' if pts_labels is not None
                               else 'lidar'),
                    'along_start': along_s, 'along_end': along_e,
                    'z_bottom': z_b, 'z_top': z_t,
                    'width_m': w_m, 'height_m': h_m,
                    'center_xy': center_xy, 'center_z': center_z,
                    'confidence': confidence,
                    'temporal_vote_count': 0,
                })

        # --- Vision blobs: window/glass combined (bucket 2 or 4) ----
        if do_vision_blobs:
            win_glass_mask = ((grid['vision'] == _BUCKET_WINDOW)
                              | (grid['vision'] == _BUCKET_GLASS))
            wg_blobs = _extract_blob_obbs(win_glass_mask)
            for (t_lo, t_hi, z_lo, z_hi, support) in wg_blobs:
                if support < config.min_support_vision_cells:
                    continue
                if _bbox_touches_corner(t_lo, t_hi, n_t, corner_cells):
                    continue
                # Classify within the blob.
                # M4b: prefer the temporal majority (votes raw points
                # across all frames in vision_band) over single-cell
                # argmax. Falls back to per-cell argmax when the vote
                # count is below `temporal_vote_min_frames` — keeps tiny
                # blobs at parity with M3 behavior.
                blob_dict = {'t_lo': t_lo, 't_hi': t_hi,
                             'z_lo': z_lo, 'z_hi': z_hi}
                temporal_class, vote_count = _temporal_vote_blob_type(
                    blob_dict, vision_band, res, floor_z,
                    min_frames=config.temporal_vote_min_frames)
                if temporal_class == 0:
                    # Fallback: per-cell argmax.
                    try:
                        temporal_class = _classify_blob_type(
                            blob_dict, grid, config)
                    except Exception:
                        temporal_class = _BUCKET_WINDOW
                tentative_type = ('glass' if temporal_class == _BUCKET_GLASS
                                  else 'window')
                (along_s, along_e, z_b, z_t,
                 center_xy, center_z,
                 w_m, h_m) = _bbox_to_world(
                    p1, p2, t_lo, t_hi, z_lo, z_hi, res, floor_z)
                wrange, hrange = _ranges_for_type(config, tentative_type)
                if not (wrange[0] <= w_m <= wrange[1]):
                    continue
                if not (hrange[0] <= h_m <= hrange[1]):
                    continue
                # M4b: picture-frame rejection. Small WINDOW-bucket blobs
                # (≤0.5 m in both dims) that don't sit near a wall edge
                # are almost always picture frames / wall art labelled
                # `windowpane` by Mask2Former. Real interior windows are
                # bigger; transoms/skylights typically touch a corner or
                # the ceiling band (caught by the edge proximity below).
                if (tentative_type == 'window'
                        and w_m < config.picture_frame_max_size_m
                        and h_m < config.picture_frame_max_size_m):
                    edge_buf = max(config.corner_reject_m * 2, 0.20)
                    touches_edge = ((along_s < edge_buf)
                                    or (along_e > wall_length - edge_buf))
                    if not touches_edge:
                        if verbose:
                            print(f"[Openings] wall {wall_idx} skipped "
                                  f"picture-frame blob "
                                  f"({w_m:.2f}x{h_m:.2f} m at "
                                  f"t={along_s:.2f}-{along_e:.2f})")
                        continue
                bbox_cells = max((t_hi - t_lo + 1) * (z_hi - z_lo + 1), 1)
                confidence = min(support / bbox_cells, 1.0)
                per_wall.append({
                    't_lo': t_lo, 't_hi': t_hi, 'z_lo': z_lo, 'z_hi': z_hi,
                    'support': support,
                    'otype': tentative_type,
                    'source': ('vision+lidar' if pts_labels is not None
                               else 'lidar'),
                    'along_start': along_s, 'along_end': along_e,
                    'z_bottom': z_b, 'z_top': z_t,
                    'width_m': w_m, 'height_m': h_m,
                    'center_xy': center_xy, 'center_z': center_z,
                    'confidence': confidence,
                    'temporal_vote_count': int(vote_count),
                })

        # --- Passages: empty-cell components (always run) -----------
        gap_comps = _extract_gap_components(
            grid['occupancy'], min_support=config.min_support_gap_cells,
            n_t=grid['n_t'], n_z=grid['n_z'])
        for comp in gap_comps:
            t_lo = comp['t_lo']
            t_hi = comp['t_hi']
            z_lo = comp['z_lo']
            z_hi = comp['z_hi']
            if _bbox_touches_corner(t_lo, t_hi, n_t, corner_cells):
                continue
            (along_s, along_e, z_b, z_t,
             center_xy, center_z,
             w_m, h_m) = _bbox_to_world(
                p1, p2, t_lo, t_hi, z_lo, z_hi, res, floor_z)
            wrange, hrange = _ranges_for_type(config, 'passage')
            if not (wrange[0] <= w_m <= wrange[1]):
                continue
            if not (hrange[0] <= h_m <= hrange[1]):
                continue
            confidence = min(h_m / wall_height, 1.0)

            # Default classification: real passage.
            opening_type = 'passage'
            cand_source = 'lidar'

            # ---- M4b-ext B: vision-override for mirrors / glass panels --
            # Count vision buckets over the gap's exact component cells.
            # Mirrors are labeled `mirror` → bucket 4 (glass) by the ADE20K
            # remap, so a majority-glass gap blob is a mirror (or glass
            # partition), not an open passage.
            comp_mask = comp.get('cell_mask')
            if (opening_type == 'passage' and comp_mask is not None
                    and grid['vision'] is not None
                    and grid['vision'].size > 0):
                blob_cells_vision = np.asarray(
                    grid['vision'][comp_mask], dtype=np.int64)
                if blob_cells_vision.size > 0:
                    counts = np.bincount(blob_cells_vision, minlength=5)
                    total_vision = int(counts.sum())
                    glass_count = int(counts[4]) if counts.size > 4 else 0
                    glass_fraction = (
                        glass_count / total_vision if total_vision > 0
                        else 0.0)
                    if glass_fraction >= 0.30:
                        opening_type = 'mirror'
                        cand_source = 'vision-override-mirror'
                        if verbose:
                            print(f"[Openings] wall {wall_idx} mirror via "
                                  f"vision-override "
                                  f"(glass_frac={glass_fraction:.2f})")

            per_wall.append({
                't_lo': t_lo, 't_hi': t_hi, 'z_lo': z_lo, 'z_hi': z_hi,
                'support': comp['support'],
                'otype': opening_type,
                'source': cand_source,
                'along_start': along_s, 'along_end': along_e,
                'z_bottom': z_b, 'z_top': z_t,
                'width_m': w_m, 'height_m': h_m,
                'center_xy': center_xy, 'center_z': center_z,
                'confidence': confidence,
                'temporal_vote_count': 0,
            })

        # --- Dedup: vision blobs outrank gap passages on IoU > threshold ---
        # Mirrors (M4b-ext) are themselves the vision-winning interpretation
        # of a gap — don't suppress them against door/window/glass vision
        # blobs; a mirror blob overlapping a window blob is already the
        # correct answer (the vision override trusted the glass bucket).
        kept = []
        for cand in per_wall:
            if cand['otype'] != 'passage':
                kept.append(cand)
                continue
            # Check against any already-emitted vision blob on this wall.
            suppressed = False
            for other in per_wall:
                if other is cand:
                    continue
                if other['otype'] in ('passage', 'mirror'):
                    continue
                if _bbox_iou(cand, other) > config.dedup_iou_threshold:
                    suppressed = True
                    break
            if not suppressed:
                kept.append(cand)

        # --- Open/closed + transparent classification via local density ---
        for cand in kept:
            ratio = _local_density_ratio(cand, grid, res,
                                          along_window_m=1.0)
            # M4b density-ratio guard: vision blobs (door / window / glass)
            # whose local density_ratio is None or below the floor are
            # phantom regions — likely unscanned area at the wall edge
            # being mis-interpreted as a closed opening. Passages and
            # mirrors (M4b-ext) deliberately have ratio≈0 (they're empty
            # regions on the lidar) and use their own adjacency + vision
            # signatures; both are exempt.
            if cand['otype'] not in ('passage', 'mirror'):
                ratio_is_low = (ratio is None
                                or (isinstance(ratio, (int, float))
                                    and ratio < config.min_density_ratio_for_emit))
                if ratio_is_low:
                    if verbose:
                        print(f"[Openings] wall {wall_idx} skipped "
                              f"phantom blob "
                              f"({cand['otype']}, ratio={ratio})")
                    continue
            is_open = None
            transparent = False
            if cand['otype'] == 'door':
                if ratio is None:
                    is_open = None
                elif ratio < config.open_threshold:
                    is_open = True
                elif ratio >= config.closed_threshold:
                    is_open = False
                else:
                    is_open = None
            elif cand['otype'] == 'window':
                if ratio is None:
                    is_open = None
                elif ratio > config.closed_threshold:
                    is_open = False
                    transparent = True
                elif ratio < config.open_threshold:
                    is_open = True
                else:
                    is_open = None
            elif cand['otype'] == 'glass':
                if ratio is None:
                    is_open = False
                    transparent = True
                elif ratio > config.closed_threshold:
                    is_open = False
                    transparent = True
                elif ratio < config.open_threshold:
                    is_open = True
                    transparent = False
                else:
                    is_open = None
                    transparent = True
            elif cand['otype'] == 'passage':
                is_open = True
            elif cand['otype'] == 'mirror':
                # M4b-ext: mirrors are closed (lidar didn't penetrate the
                # glass — specular reflection) and are NOT transparent
                # from the scan's POV (the beam didn't pass through).
                is_open = False
                transparent = False

            entry = _make_opening_entry(
                oid=oid_counter, wall_idx=wall_idx,
                wall_uuid=wall_uuid, otype=cand['otype'],
                along_start=cand['along_start'],
                along_end=cand['along_end'],
                z_bottom=cand['z_bottom'], z_top=cand['z_top'],
                width_m=cand['width_m'], height_m=cand['height_m'],
                center_xy=cand['center_xy'],
                transform_4x4=_build_transform_4x4(
                    p1, p2, cand['center_xy'], cand['center_z']),
                source=cand['source'],
                is_open=is_open, transparent=transparent,
                transom_of=None, confidence=cand['confidence'],
                temporal_vote_count=cand.get('temporal_vote_count', 0))
            entry['density_ratio'] = (round(float(ratio), 4)
                                      if ratio is not None else None)
            all_openings.append(entry)
            oid_counter += 1

    # --- Transom linking across all emitted openings ---
    _link_transoms(all_openings)
    return all_openings
