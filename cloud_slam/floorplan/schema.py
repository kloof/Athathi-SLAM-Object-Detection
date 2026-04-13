"""Floorplan JSON schema constants + wall/variant serialization helpers.

Extracted from the pre-split `generate_floorplan` orchestrator so the
metadata-construction loop (per-variant + per-wall entries) lives in one
place. M0a will lean on `SCHEMA_VERSION` when expanding the schema with
the wall / opening payload. Zero behavior change relative to the pre-split
code path.
"""

import json


# Bumped by M0a when the JSON schema grows (dominant_directions, per-wall
# UUIDs, openings list). Kept as a module constant so importers can assert
# compatibility without reading from the file.
SCHEMA_VERSION = "2.0"


def _build_variant_wall_entry(idx, wall_tuple, key, walls_d_meta_clean):
    """Build a single wall entry dict for the variants.<key>.walls list.

    Mirrors the per-variant loop that was inlined in `generate_floorplan`
    before the package split. Kept as a helper so the orchestrator stays
    thin and schema changes land in one place (see M0a).
    """
    _, _, a, l = wall_tuple
    entry = {'length_m': round(float(l), 3),
             'angle_deg': round(float(a), 1)}

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
    elif key == 'D_refined':
        # Fell back to A_natural — meta is None
        entry['snapped_to'] = 'fallback_a'
    return entry


def build_floorplan_metadata(*, n_raw, pts, floor_z, ceiling_z, h, n_removed,
                              corner_coords_real, variants, walls_d_meta_clean,
                              vision_stats, elapsed):
    """Assemble the floorplan metadata dict (pre-serialization).

    Extracted verbatim from `generate_floorplan`'s bottom block. The
    returned dict matches the JSON produced by the pre-split code byte-
    for-byte for every existing code path.
    """
    meta = {
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
    for key, (walls, poly, label) in variants.items():
        wall_entries = []
        for idx, wall_tuple in enumerate(walls):
            entry = _build_variant_wall_entry(
                idx, wall_tuple, key, walls_d_meta_clean)
            wall_entries.append(entry)
        meta['variants'][key] = {
            'label': label,
            'area_m2': round(float(poly.area), 2),
            'n_walls': int(len(walls)),
            'walls': wall_entries,
        }
    return meta


def write_floorplan_metadata(meta, output_dir, name):
    """Write the floorplan metadata dict to `{output_dir}/{name}_metadata.json`."""
    with open(f'{output_dir}/{name}_metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)
