"""Floorplan JSON schema constants + wall/variant serialization helpers.

Extracted from the pre-split `generate_floorplan` orchestrator so the
metadata-construction loop (per-variant + per-wall entries) lives in one
place. M0a expanded the schema with per-wall `id`/`uuid`/`p1`/`p2`/
`thickness_m`/`next_wall_id`/`prev_wall_id` and top-level
`schema_version`/`units`/`world_up`/`handedness`/`origin`/`calibration`/
`room` keys so downstream converters (USDZ/IFC/Revit/RoomPlan) and the
M3 openings module have a stable contract.

Back-compat: the M0a additions are a strict *superset* — every pre-M0a
key still exists at the same path with the same value.
"""

import hashlib
import json
from collections import Counter

import numpy as np


# Bumped by M0a when the JSON schema grew (per-wall UUIDs, cyclic
# id links, calibration block, room category). Kept as a module constant
# so importers can assert compatibility without reading from the file.
# Bumped again by M3 when `D_refined.openings` (doors / windows / glass /
# passages) joined each variant's per-wall list — mirrors Apple RoomPlan's
# `CapturedOpening` contract.
# Bumped again by M4a with the new top-level `scan_quality` block (vision
# frame-quality %, time-sync p50/p95/p99, per-wall camera coverage),
# `secondary_ceiling_features` (dropped soffits / coffers / trays) and
# per-wall `frames_seen_count` + `curved` fields.
SCHEMA_VERSION = "2.2"


# M3 opening schema keys (documented here so M3 populates them consistently).
# Doors / windows / passages emitted by the M3 openings detector must
# populate every key in this set on every opening entry — downstream
# converters rely on the contract being total, not best-effort.
OPENING_REQUIRED_KEYS = frozenset({
    "id", "uuid", "type", "wall_id", "wall_uuid",
    "along_start", "along_end", "z_bottom", "z_top",
    "width_m", "height_m", "center_xy", "transform_4x4",
    "source", "is_open", "transparent", "transom_of", "confidence",
})

# Controlled vocabulary for the `type` field on opening entries.
OPENING_TYPES = frozenset({"door", "window", "glass", "passage"})


# Per-wall default thickness. Real thickness would need multi-room scans
# (inside + outside surface). The 0.1 m default mirrors IFC wall-layer
# conventions and is recorded at the root as `thickness_source`.
_DEFAULT_WALL_THICKNESS_M = 0.1


# --- ADE20K → room-category vote table (M0a room classifier) ---
#
# Signature object classes in ADE20K-150 that disambiguate interior spaces.
# Majority vote across the whole scan; the winner's share must strictly
# exceed `_ROOM_MIN_CONFIDENCE` or we emit "unknown". Ignores all
# background clutter (floor, wall, cabinet etc.) — only the signatures
# listed here contribute. Returns "unknown" when the winner's share does
# not exceed this fraction (strictly greater than is required).
_ADE_ROOM_SIGNATURES = {
    # bedroom
    7:   "bedroom",      # bed
    # livingroom
    23:  "livingroom",   # sofa
    30:  "livingroom",   # armchair (ADE20K "armchair")
    # kitchen
    50:  "kitchen",      # refrigerator
    71:  "kitchen",      # stove
    118: "kitchen",      # oven
    124: "kitchen",      # microwave
    # bathroom
    37:  "bathroom",     # bathtub
    65:  "bathroom",     # toilet
    # diningroom
    15:  "diningroom",   # table / dining_table
}

# Returns "unknown" when the winner's share does not exceed this
# fraction (strictly greater than is required).
_ROOM_MIN_CONFIDENCE = 0.4


def _wall_uuid(p1, p2) -> str:
    """Deterministic MD5-hex UUID for a wall's geometry.

    Rounding to 4 decimals (0.1 mm) before hashing ensures two runs on
    the same scan produce the same UUIDs (point-cloud float noise from
    ICP/voxel-downsample is far below that threshold). Full 32-char hex.
    """
    s = (f"{float(p1[0]):.4f},{float(p1[1]):.4f},"
         f"{float(p2[0]):.4f},{float(p2[1]):.4f}")
    return hashlib.md5(s.encode()).hexdigest()


def _vote_room_category(ade_class_counts) -> tuple[str, float, str]:
    """Aggregate ADE20K class histogram → (category, confidence, source).

    ade_class_counts: dict[int, int] — ADE20K class id → total pixel count
                      across the scan. Only the signature classes in
                      `_ADE_ROOM_SIGNATURES` are considered.

    Returns ("unknown", 0.0, "ade20k_vote") when no signature class
    strictly exceeds the `_ROOM_MIN_CONFIDENCE` fraction (i.e. equality at
    the threshold is treated as "unknown"); returns
    ("unknown", 0.0, "unavailable") when ade_class_counts is None/empty.
    """
    if not ade_class_counts:
        return "unknown", 0.0, "unavailable"

    category_votes = Counter()
    for ade_id, count in ade_class_counts.items():
        cat = _ADE_ROOM_SIGNATURES.get(int(ade_id))
        if cat is not None:
            category_votes[cat] += int(count)

    total = sum(category_votes.values())
    if total == 0:
        return "unknown", 0.0, "ade20k_vote"

    winner, winner_votes = category_votes.most_common(1)[0]
    confidence = winner_votes / total
    if confidence <= _ROOM_MIN_CONFIDENCE:
        return "unknown", float(round(confidence, 3)), "ade20k_vote"
    return winner, float(round(confidence, 3)), "ade20k_vote"


def _build_calibration_block(calibration_info) -> dict:
    """Build the `calibration` root block.

    calibration_info: optional dict with keys
        method (str), calibration_date (str 'YYYY-MM-DD'), age_days (int),
        reprojection_iou_mean (float | None),
        reprojection_iou_min (float | None),
        reprojection_iou_frames_checked (int),
        accuracy_tier (str: "coarse" | "good" | "tight")
        — any subset. Everything missing falls back to sensible defaults
        (null IoU, "coarse" tier = "not yet verified").

    M0c populates the reprojection-IoU fields via
    `cloud_slam.calibration.verify_calibration` in scripts/detect_and_slam.py;
    when that path doesn't run (e.g. Mask2Former disabled), the fields
    remain null and `accuracy_tier` stays "coarse".
    """
    info = calibration_info or {}
    return {
        "method": info.get("method", "manual_visual_alignment"),
        "date": info.get("calibration_date"),
        "age_days": info.get("age_days"),
        # M0c: reprojection-IoU verification. Null when Mask2Former didn't
        # run or accumulated no wall masks — consumers read this as
        # "not yet verified".
        "reprojection_iou_mean": info.get("reprojection_iou_mean"),
        "reprojection_iou_min": info.get("reprojection_iou_min"),
        "reprojection_iou_frames_checked": int(
            info.get("reprojection_iou_frames_checked", 0)),
        # Hard-coded from cloud_slam/colorizer.py::match_nearest_image(max_dt=0.15).
        # If the colorizer's threshold changes, update both places.
        "time_sync_max_dt_ms": 150,
        # Default "coarse" — flips to "good" / "tight" when M0c measures
        # mean IoU above 0.70 / 0.85.
        "accuracy_tier": info.get("accuracy_tier", "coarse"),
    }


def _build_scan_quality_block(vision_health, time_sync_dts,
                                time_sync_dropped, wall_frames_seen,
                                n_walls_d_refined):
    """Assemble the M4a `scan_quality` metadata block.

    Arguments are all optional — the block is emitted with sensible
    zero-ish defaults when vision / timing data isn't available, so
    consumers can unconditionally read it.

    vision_health: dict from `WallSegmenter.get_vision_health()` or
        None when Mask2Former wasn't run. Expected keys:
        `frame_quality_pct`, `frames_seen`, `total_pixels_seen`.
    time_sync_dts: list of per-frame lidar↔image stamp gaps (seconds),
        from `icp_imu_pipeline.run`'s stats. Empty or None when no
        color matching ran.
    time_sync_dropped: int count of frames where match_nearest_image
        returned None (beyond max_dt). Defaults to 0 when missing.
    wall_frames_seen: dict {wall_id: int frames_seen}, from
        `_compute_per_wall_frame_coverage`. Empty when no per-frame
        XYZ was accumulated.
    n_walls_d_refined: number of walls in the D_refined variant.
        Used to count "low coverage" walls as a fraction of the
        total, and to median-aggregate over. Zero-safe.
    """
    # ---- Vision sub-block ----
    if vision_health is not None:
        fq_pct = float(vision_health.get("frame_quality_pct", 0.0) or 0.0)
        frames_seen_vision = int(vision_health.get("frames_seen", 0))
        # "frames_low_quality_count" — we don't track per-frame
        # fractions (the counters are cumulative), so expose 0 for
        # now. Downstream consumers treat this as a placeholder;
        # a future milestone can add per-frame gating.
        vision_block = {
            "frame_quality_pct": round(fq_pct, 2),
            "frames_seen": frames_seen_vision,
            "frames_low_quality_count": 0,
        }
    else:
        vision_block = {
            "frame_quality_pct": None,
            "frames_seen": 0,
            "frames_low_quality_count": 0,
        }

    # ---- Time-sync sub-block ----
    dts_ms = None
    if time_sync_dts:
        dts_ms = np.asarray(time_sync_dts, dtype=float) * 1000.0
    if dts_ms is not None and dts_ms.size > 0:
        time_sync_block = {
            "p50_ms": round(float(np.percentile(dts_ms, 50)), 2),
            "p95_ms": round(float(np.percentile(dts_ms, 95)), 2),
            "p99_ms": round(float(np.percentile(dts_ms, 99)), 2),
            "frames_dropped": int(time_sync_dropped or 0),
        }
    else:
        time_sync_block = {
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "frames_dropped": int(time_sync_dropped or 0),
        }

    # ---- Per-wall coverage sub-block ----
    if wall_frames_seen:
        counts = np.asarray(
            [int(v) for v in wall_frames_seen.values()], dtype=int)
        min_fs = int(counts.min())
        med_fs = int(np.median(counts))
        # A wall is "low coverage" if < 10 frames saw it at close range.
        # The threshold matches the M4a plan spec.
        n_low = int((counts < 10).sum())
    else:
        min_fs = 0
        med_fs = 0
        # Without any per-frame coverage data, treat every wall as "low
        # coverage" so the schema reflects the missing telemetry signal.
        n_low = int(n_walls_d_refined or 0)

    walls_block = {
        "n_walls_with_low_coverage": n_low,
        "min_frames_seen": min_fs,
        "median_frames_seen": med_fs,
    }

    return {
        "vision": vision_block,
        "time_sync": time_sync_block,
        "walls_camera_coverage": walls_block,
    }


def _build_variant_wall_entry(idx, wall_tuple, key, walls_d_meta_clean,
                               n_walls, wall_frames_seen=None):
    """Build a single wall entry dict for the variants.<key>.walls list.

    M0a expansion: every wall entry (regardless of variant) now carries
    `id`, `uuid`, `p1`, `p2`, `thickness_m`, `next_wall_id`, `prev_wall_id`
    alongside the pre-existing `length_m`/`angle_deg`. The D_refined variant
    also keeps its `snapped_to`/`residual_m`/`confidence`/`type`/`features`
    fields. Back-compat: the pre-M0a keys are unchanged in name and value.
    """
    p1, p2, a, l = wall_tuple
    entry = {
        'id': int(idx),
        'uuid': _wall_uuid(p1, p2),
        'p1': [round(float(p1[0]), 4), round(float(p1[1]), 4)],
        'p2': [round(float(p2[0]), 4), round(float(p2[1]), 4)],
        'thickness_m': _DEFAULT_WALL_THICKNESS_M,
        'next_wall_id': int((idx + 1) % n_walls) if n_walls > 0 else 0,
        'prev_wall_id': int((idx - 1) % n_walls) if n_walls > 0 else 0,
        'length_m': round(float(l), 3),
        'angle_deg': round(float(a), 1),
    }

    # D_refined: add per-wall snap kind + residual + confidence
    if (key == 'D_refined' and idx < len(walls_d_meta_clean)
            and walls_d_meta_clean[idx] is not None):
        m = walls_d_meta_clean[idx]
        entry['snapped_to'] = m.get('snapped_to', 'free')
        entry['residual_m'] = m.get('residual_m', 0.0)
        entry['confidence'] = m.get('confidence', 0.0)
        # Stage 7 audit: data-extent length from RANSAC inliers
        # (compare to `length_m` to see how much Stage 7 trimmed
        # the wall to match the point cloud).
        entry['length_m_data_extent'] = m.get(
            'length_m_data_extent', 0.0)
        # Vision Tier 1: per-wall type — only present when wall_labels
        # was supplied and classification produced a result.
        if 'type' in m:
            entry['type'] = m['type']
        if 'features' in m:
            entry['features'] = m['features']
        # M4a: curved-wall flag (residual_m / length_m > 5%).
        # Always emitted — defaults to False when the refiner couldn't
        # compute one (fallback paths).
        entry['curved'] = bool(m.get('curved', False))
    elif key == 'D_refined':
        # Fell back to A_natural — meta is None. Populate the same set of
        # numeric keys with neutral values so downstream consumers can
        # iterate D_refined entries without checking for missing keys.
        # `type` / `features` remain absent (fallback has no vision info).
        entry['snapped_to'] = 'fallback_a'
        entry['residual_m'] = 0.0
        entry['confidence'] = 0.0
        entry['length_m_data_extent'] = entry['length_m']
        entry['curved'] = False

    # M4a: per-wall frames_seen_count — how many camera frames actually
    # observed this wall at close range. Surfaced on EVERY variant's
    # walls so downstream consumers don't need to special-case D_refined.
    # Defaults to 0 when no per-frame data was available.
    if wall_frames_seen is not None:
        entry['frames_seen_count'] = int(wall_frames_seen.get(idx, 0))
    else:
        entry['frames_seen_count'] = 0
    return entry


def build_floorplan_metadata(*, n_raw, pts, floor_z, ceiling_z, h, n_removed,
                              corner_coords_real, variants, walls_d_meta_clean,
                              vision_stats, elapsed,
                              calibration_info=None,
                              ade_class_counts=None,
                              stage8_diagnostics=None,
                              openings=None,
                              vision_health=None,
                              time_sync_dts=None,
                              time_sync_dropped=0,
                              wall_frames_seen=None,
                              secondary_ceiling_features=None) -> dict:
    """Assemble the floorplan metadata dict (pre-serialization).

    M0a additions (all strict superset — pre-existing keys unchanged):
        schema_version, units, angle_units, world_up, handedness, origin,
        thickness_source, calibration (block), room (block).

    calibration_info: optional dict from the scan's extrinsics.yaml
        (method / calibration_date / age_days). Missing values fall back
        to `_build_calibration_block` defaults.

    ade_class_counts: optional dict[int, int] of ADE20K class id → total
        pixel count across the scan. Drives the `room.category` vote.
        When None/empty (segmenter disabled), emits
        `{"category": "unknown", "category_confidence": 0.0,
          "category_source": "unavailable"}`.

    stage8_diagnostics: optional dict produced by
        `cloud_slam.floorplan.refine._stage8_polygon_closure` when
        emit_diagnostics=True. Keys: `solver_used`, `closure_gap_mm`,
        `per_corner_shift_mm`, `per_corner_weight`, `demotions_cascade`,
        `kkt_cond`. Surfaced at the metadata root under `stage8` for
        downstream inspection. None (default) keeps the pre-M1a JSON
        layout unchanged — the regression path.

    openings: optional list of opening dicts (M3) — doors / windows /
        glass / passages detected on D_refined walls. Emitted as
        `variants.D_refined.openings` (array). When None, the key is
        still emitted as `[]` (total contract: consumers can
        unconditionally iterate `D_refined.openings`).

    M4a kwargs (all optional; surface as the `scan_quality` block +
    `secondary_ceiling_features` at metadata root):

    vision_health: dict from `WallSegmenter.get_vision_health()` —
        `frame_quality_pct` / `frames_seen` / `total_pixels_seen`.
        None when Mask2Former didn't run → emits nulls.
    time_sync_dts: list of per-frame lidar↔image stamp gaps in
        seconds (from `icp_imu_pipeline.run` stats). Empty/None →
        emits nulls.
    time_sync_dropped: int count of frames where match_nearest_image
        returned None (beyond max_dt).
    wall_frames_seen: dict {wall_id: int frames_seen} from
        `_compute_per_wall_frame_coverage`. None → emits 0 for every
        wall's `frames_seen_count`.
    secondary_ceiling_features: list of dicts describing dropped
        soffits / coffers / trays / HVAC bulkheads (any horizontal
        plane between floor+1m and the picked ceiling). Emitted as
        `meta['secondary_ceiling_features']`. None → emits `[]`.
    """
    room_category, room_confidence, room_source = _vote_room_category(
        ade_class_counts)

    meta = {
        # --- M0a root metadata ---
        'schema_version': SCHEMA_VERSION,
        'units': 'm',
        'angle_units': 'deg',
        # Z-up right-handed frame — the floorplan post-process runs after
        # leveling in scripts/detect_and_slam.py, so all wall/point
        # coordinates are already in the canonical gravity-aligned frame.
        'world_up': [0, 0, 1],
        'handedness': 'right',
        'origin': 'first_lidar_frame',
        'thickness_source': 'default_0.1m',
        'calibration': _build_calibration_block(calibration_info),
        'room': {
            'category': room_category,
            'category_confidence': room_confidence,
            'category_source': room_source,
        },
        # --- pre-M0a keys (unchanged in name and value) ---
        'n_points_raw': int(n_raw),
        'n_points_processed': int(len(pts)),
        'floor_z': round(float(floor_z), 3),
        'ceiling_z': round(float(ceiling_z), 3),
        'room_height': round(float(h), 3),
        'n_outlier_clusters_removed': int(n_removed),
        'n_corners_detected':
            int(len(corner_coords_real)) if corner_coords_real is not None else 0,
        'variants': {},
        'processing_time_s': round(elapsed, 1),
    }
    # Top-level vision diagnostics — present only when wall_labels was used.
    if vision_stats is not None:
        meta['vision_model'] = 'mask2former-swin-large-ade-semantic'
        meta['vision_wall_point_count'] = int(
            vision_stats['wall_point_count'])
        meta['vision_wall_blob_count'] = int(
            vision_stats['wall_blob_count'])
    # Stage 8 (M1a) diagnostics — only when emit_diagnostics was set AND
    # Stage 8 actually computed a diagnostics dict. Missing / None keeps
    # the JSON layout identical to the pre-M1a baseline (the regression
    # path — used when --no-stage8 disables the solver).
    if stage8_diagnostics is not None:
        meta['stage8'] = dict(stage8_diagnostics)

    # M4a: dropped-ceiling features (soffits/coffers/trays/HVAC bulkheads).
    # Always emitted as a list so consumers can unconditionally iterate.
    meta['secondary_ceiling_features'] = (
        list(secondary_ceiling_features) if secondary_ceiling_features
        else [])

    # M4a: scan_quality block. Always emitted — the helper fills in
    # nulls/zeros for missing signals so consumers see a total contract.
    n_d_walls = 0
    if 'D_refined' in variants:
        _walls_d, _poly_d, _label_d = variants['D_refined']
        n_d_walls = len(_walls_d)
    meta['scan_quality'] = _build_scan_quality_block(
        vision_health=vision_health,
        time_sync_dts=time_sync_dts,
        time_sync_dropped=time_sync_dropped,
        wall_frames_seen=wall_frames_seen,
        n_walls_d_refined=n_d_walls)

    for key, (walls, poly, label) in variants.items():
        n_walls = len(walls)
        wall_entries = []
        for idx, wall_tuple in enumerate(walls):
            entry = _build_variant_wall_entry(
                idx, wall_tuple, key, walls_d_meta_clean, n_walls,
                wall_frames_seen=wall_frames_seen)
            wall_entries.append(entry)
        meta['variants'][key] = {
            'label': label,
            'area_m2': round(float(poly.area), 2),
            'n_walls': int(n_walls),
            'walls': wall_entries,
        }
        # M3: openings are emitted only on D_refined (the only variant
        # with per-wall type/vision provenance). Empty list when M3
        # didn't run or nothing was detected — the key is always
        # present so downstream iteration can be unconditional.
        if key == 'D_refined':
            meta['variants'][key]['openings'] = (
                list(openings) if openings else [])
    return meta


def write_floorplan_metadata(meta, output_dir, name):
    """Write the floorplan metadata dict to `{output_dir}/{name}_metadata.json`."""
    with open(f'{output_dir}/{name}_metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)
