"""Embed wall segments + bbox wireframes + door/window crosses as
colored points into the scene PLY, so any viewer (CloudCompare,
MeshLab, Open3D) shows the SpatialLM layout alongside the cloud.

Class colors are chosen to be visually distinct; unknown classes
fall back to magenta.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d


CLASS_COLOR = {
    "sofa": (255, 140, 0),
    "chair": (0, 200, 0),
    "dining_chair": (0, 255, 0),
    "dining_table": (184, 134, 11),
    "coffee_table": (139, 69, 19),
    "side_table": (160, 82, 45),
    "desk": (210, 105, 30),
    "bed": (218, 165, 32),
    "tv": (186, 85, 211),
    "tv_cabinet": (20, 20, 20),
    "curtain": (0, 255, 255),
    "mirror": (255, 255, 255),
    "painting": (255, 192, 203),
    "plants": (34, 139, 34),
    "floor-standing_lamp": (255, 255, 0),
    "chandelier": (255, 255, 100),
    "wall_decoration": (255, 105, 180),
    "cupboard": (0, 128, 255),
    "shelf": (139, 0, 139),
    "bar": (255, 215, 0),
    "carpet": (128, 128, 128),
}
_FALLBACK = (255, 0, 255)
_WALL = (0, 0, 0)
_DOOR = (255, 0, 0)
_WINDOW = (0, 0, 255)

_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),   # bottom
    (4, 5), (5, 6), (6, 7), (7, 4),   # top
    (0, 4), (1, 5), (2, 6), (3, 7),   # verticals
]


def _seg(a, b, spacing=0.02):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = np.linalg.norm(b - a)
    n = max(2, int(np.ceil(d / spacing)) + 1)
    t = np.linspace(0, 1, n)[:, None]
    return a[None, :] * (1 - t) + b[None, :] * t


def _corners(cx, cy, cz, yaw, sx, sy, sz):
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    sign = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ], float)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return (R @ (sign * np.array([hx, hy, hz])).T).T + np.array([cx, cy, cz])


def _cross(p, half=0.10, spacing=0.02):
    x, y, z = p[:3]
    return np.concatenate([
        _seg([x - half, y, z], [x + half, y, z], spacing),
        _seg([x, y - half, z], [x, y + half, z], spacing),
        _seg([x, y, z - half], [x, y, z + half], spacing),
    ], axis=0)


def embed_bboxes_in_ply(
    cloud_ply: Path | str,
    layout_txt: Path | str,
    out_ply: Path | str,
    *,
    edge_spacing_m: float = 0.02,
    verbose: bool = True,
) -> Path:
    from cloud_slam.spatiallm_pipeline.merge import parse_layout

    cloud_ply = Path(cloud_ply)
    layout_txt = Path(layout_txt)
    out_ply = Path(out_ply)

    layout = parse_layout(layout_txt)
    if verbose:
        print(f"[embed] {len(layout['walls'])} walls  "
              f"{len(layout['doors'])} doors  "
              f"{len(layout['windows'])} windows  "
              f"{len(layout['bboxes'])} bboxes")

    P_extra, C_extra = [], []

    # bboxes -> 12 edges each
    for cls, v in layout["bboxes"]:
        corners = _corners(*v)
        color = np.asarray(CLASS_COLOR.get(cls, _FALLBACK)) / 255.0
        for i, j in _EDGES:
            s = _seg(corners[i], corners[j], edge_spacing_m)
            P_extra.append(s)
            C_extra.append(np.tile(color, (len(s), 1)))

    # walls -> single floor-line per wall
    for w in layout["walls"]:
        s = _seg(w[:3], w[3:6], edge_spacing_m)
        P_extra.append(s)
        C_extra.append(np.tile(np.asarray(_WALL) / 255.0, (len(s), 1)))

    # doors / windows -> crosses at anchor
    for d in layout["doors"]:
        p = _cross(d[:3], spacing=edge_spacing_m)
        P_extra.append(p)
        C_extra.append(np.tile(np.asarray(_DOOR) / 255.0, (len(p), 1)))
    for w in layout["windows"]:
        p = _cross(w[:3], spacing=edge_spacing_m)
        P_extra.append(p)
        C_extra.append(np.tile(np.asarray(_WINDOW) / 255.0, (len(p), 1)))

    if not P_extra:
        P_extra = [np.zeros((0, 3))]
        C_extra = [np.zeros((0, 3))]
    P_extra = np.concatenate(P_extra, axis=0)
    C_extra = np.concatenate(C_extra, axis=0)
    if verbose:
        print(f"[embed] added {len(P_extra):,} edge points")

    pcd = o3d.io.read_point_cloud(str(cloud_ply))
    op = np.asarray(pcd.points)
    oc = np.asarray(pcd.colors)
    all_pts = np.concatenate([op, P_extra], axis=0)
    all_cols = np.concatenate([oc, C_extra], axis=0)

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(all_pts)
    out.colors = o3d.utility.Vector3dVector(all_cols)
    o3d.io.write_point_cloud(str(out_ply), out)
    if verbose:
        import os
        print(f"[embed] wrote {out_ply}  "
              f"({os.path.getsize(out_ply)/(1024*1024):.1f} MB, "
              f"{len(all_pts):,} pts)")
    return out_ply
