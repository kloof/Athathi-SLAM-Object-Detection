"""Crop a KISS-ICP colored PLY to inside the detected room polygon.

Uses cloud_slam.floorplan.generate_floorplan (RANSAC-refined walls, D_refined
variant) to get the room footprint + floor/ceiling Z, then:
  - expands the polygon by a small wall-tolerance buffer (1.5% of the short
    bbox dim, clamped to 8-15 cm) so wall-surface points aren't shaved off
  - keeps points within [floor_z - z_margin, ceiling_z + z_margin]
  - keeps both colored (camera-seen) and uncolored (default-gray) LiDAR
    returns -- uncolored is dropped LATER in the voxel step if desired.

Produces <out_dir>/colored_map_cropped.ply  + <out_dir>/floorplan/* PNG/JSON.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from shapely.geometry import Point as ShapelyPoint
from shapely.prepared import prep


def crop_to_floorplan(
    src_ply: Path | str,
    out_dir: Path | str,
    *,
    z_margin_m: float = 0.10,
    buffer_frac: float = 0.015,
    buffer_min_m: float = 0.08,
    buffer_max_m: float = 0.15,
    verbose: bool = True,
) -> Path:
    """Detect the room footprint on the input cloud and crop to it.

    Returns the path to the cropped PLY. Also writes floorplan PNGs and
    room_metadata.json under <out_dir>/floorplan/.
    """
    from cloud_slam.floorplan import generate_floorplan

    src_ply = Path(src_ply)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp_dir = out_dir / "floorplan"
    fp_dir.mkdir(exist_ok=True)

    if verbose:
        print(f"[crop] load {src_ply}")
    pcd = o3d.io.read_point_cloud(str(src_ply))
    pts = np.asarray(pcd.points)
    if verbose:
        print(f"  {pts.shape[0]:,} pts, bbox range "
              f"{(pts.max(0) - pts.min(0)).round(2)}")

    # --- floorplan detection on full cloud ---
    if verbose:
        print("[crop] running floorplan detection (D_refined)")
    variants, meta = generate_floorplan(pcd, str(fp_dir), name="room",
                                          verbose=verbose)
    preferred = ("D_refined", "C_snapped", "B_corners", "A_natural")
    room_poly = None
    chosen = None
    for k in preferred:
        if k not in variants:
            continue
        _walls, poly, _label = variants[k]
        if poly is not None and poly.area > 1.0:
            room_poly = poly
            chosen = k
            break
    if room_poly is None:
        raise RuntimeError("no usable room polygon from floorplan detection")

    floor_z = meta["floor_z"]
    ceiling_z = meta["ceiling_z"]
    if verbose:
        print(f"[crop] variant={chosen}  area={room_poly.area:.2f}m^2  "
              f"floor_z={floor_z:.3f} ceiling_z={ceiling_z:.3f}")

    # --- buffer polygon so wall-surface pts aren't shaved ---
    minx, miny, maxx, maxy = room_poly.bounds
    bbox_short = float(min(maxx - minx, maxy - miny))
    buffer_m = min(buffer_max_m, max(buffer_min_m, bbox_short * buffer_frac))
    if verbose:
        print(f"[crop] polygon buffer {buffer_m*100:.1f}cm "
              f"({100*buffer_frac:.1f}% of short dim {bbox_short:.2f}m, "
              f"clamped to [{buffer_min_m*100:.0f},{buffer_max_m*100:.0f}]cm)")
    room_poly = room_poly.buffer(buffer_m, join_style=2, mitre_limit=5.0)

    # --- Z band ---
    z_ok = (pts[:, 2] >= floor_z - z_margin_m) & (pts[:, 2] <= ceiling_z + z_margin_m)

    # --- polygon containment (vectorized with bbox prefilter) ---
    prepared = prep(room_poly)
    minx, miny, maxx, maxy = room_poly.bounds
    xy = pts[:, :2]
    bbox_ok = (xy[:, 0] >= minx) & (xy[:, 0] <= maxx) & \
              (xy[:, 1] >= miny) & (xy[:, 1] <= maxy)

    t0 = time.time()
    candidate = np.flatnonzero(z_ok & bbox_ok)
    poly_ok = np.zeros(len(pts), dtype=bool)
    for i in candidate:
        if prepared.contains(ShapelyPoint(xy[i, 0], xy[i, 1])):
            poly_ok[i] = True
    if verbose:
        print(f"[crop] polygon point-in {time.time()-t0:.1f}s  "
              f"kept {int(poly_ok.sum()):,} / {int(bbox_ok.sum()):,} candidates")

    final_mask = z_ok & poly_ok
    if verbose:
        print(f"[crop] final kept: {int(final_mask.sum()):,} / {pts.shape[0]:,} "
              f"({100 * final_mask.mean():.1f}%)")

    cropped = pcd.select_by_index(np.flatnonzero(final_mask))
    dst = out_dir / "colored_map_cropped.ply"
    o3d.io.write_point_cloud(str(dst), cropped)
    if verbose:
        print(f"[crop] wrote {dst}")

    # write the decisions we made so downstream steps can reuse
    (out_dir / "crop_info.json").write_text(json.dumps({
        "variant": chosen,
        "room_area_m2": round(room_poly.area, 2),
        "floor_z": round(floor_z, 3),
        "ceiling_z": round(ceiling_z, 3),
        "z_margin_m": z_margin_m,
        "buffer_m": round(buffer_m, 3),
        "n_in": int(pts.shape[0]),
        "n_out": int(final_mask.sum()),
    }, indent=2))
    return dst
