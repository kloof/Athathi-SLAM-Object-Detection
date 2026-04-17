"""Overlay-geometry builder for the V2 SpatialLM scene.

Turns a ``Scene`` (walls/doors/windows/bboxes) into sampled line points +
per-point RGB colors suitable for concatenation onto a colored point cloud
for MeshLab inspection. Only draws wireframes — no surfaces.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from cloud_slam.detectors.scene import Bbox, Door, Scene, Wall, Window


_Z = np.array([0.0, 0.0, 1.0])


def _sample_line(a: np.ndarray, b: np.ndarray, spacing: float) -> np.ndarray:
    """Sample points along segment a→b at approximately ``spacing`` meters apart."""
    dist = float(np.linalg.norm(b - a))
    n = max(int(dist / max(spacing, 1e-6)), 2)
    t = np.linspace(0.0, 1.0, n).reshape(-1, 1)
    return a.reshape(1, 3) + t * (b - a).reshape(1, 3)


def _bbox_edges(box: Bbox) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return the 12 edges (a,b pairs) of an oriented box, yaw-only rotation."""
    R = Rotation.from_rotvec([0.0, 0.0, box.angle_z]).as_matrix()
    hx, hy, hz = 0.5 * box.scale_x, 0.5 * box.scale_y, 0.5 * box.scale_z
    local = np.array([
        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
        [-hx, -hy,  hz], [hx, -hy,  hz], [hx, hy,  hz], [-hx, hy,  hz],
    ])
    center = np.array([box.position_x, box.position_y, box.position_z])
    corners = (R @ local.T).T + center
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),   # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),   # top
        (0, 4), (1, 5), (2, 6), (3, 7),   # verticals
    ]
    return [(corners[i], corners[j]) for i, j in edges]


def _wall_edges(wall: Wall) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return the 4 edges (a,b pairs) of a wall's vertical rectangle."""
    a = np.array([wall.ax, wall.ay, wall.az])
    b = np.array([wall.bx, wall.by, wall.bz])
    up = _Z * wall.height
    return [(a, b), (a + up, b + up), (a, a + up), (b, b + up)]


def _find_wall(walls: List[Wall], wall_id: int) -> Optional[Wall]:
    for w in walls:
        if w.id == wall_id:
            return w
    return None


def _fixture_edges(
    fixture, walls: List[Wall]
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return 4 edges of a door/window rectangle flush to its parent wall.

    The rectangle lies in the wall plane, centered at fixture.position, with
    width along the wall's xy direction and height vertical.
    """
    parent = _find_wall(walls, fixture.wall_id)
    if parent is None:
        return []
    d = np.array([parent.bx - parent.ax, parent.by - parent.ay, 0.0])
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return []
    d = d / n
    pos = np.array([fixture.position_x, fixture.position_y, fixture.position_z])
    hw = 0.5 * fixture.width * d
    hh = 0.5 * fixture.height * _Z
    c_bl = pos - hw - hh
    c_br = pos + hw - hh
    c_tl = pos - hw + hh
    c_tr = pos + hw + hh
    return [(c_bl, c_br), (c_tl, c_tr), (c_bl, c_tl), (c_br, c_tr)]


def _meta_index(entries: Optional[List[Dict]]) -> Dict[int, Dict]:
    """Build an id→meta dict from an element_meta list (missing → empty)."""
    if not entries:
        return {}
    return {int(e["id"]): e for e in entries}


def _passes_filter(
    meta: Dict,
    min_confidence: Optional[float],
    min_observations: Optional[int],
) -> bool:
    """Per-element filter: missing meta means we keep the element by default."""
    if not meta:
        return True
    if min_confidence is not None and meta.get("confidence", 0.0) < min_confidence:
        return False
    if min_observations is not None and meta.get("observations", 0) < min_observations:
        return False
    return True


def build_overlay(
    scene: Scene,
    *,
    bbox_color: Tuple[int, int, int] = (255, 0, 0),
    wall_color: Tuple[int, int, int] = (0, 120, 255),
    door_color: Tuple[int, int, int] = (0, 255, 0),
    window_color: Tuple[int, int, int] = (255, 255, 0),
    edge_spacing: float = 0.02,
    element_meta: Optional[Dict[str, List[Dict]]] = None,
    min_confidence: Optional[float] = None,
    min_observations: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (points_Nx3_float64, colors_Nx3_uint8) of all overlay line points.

    If ``element_meta`` is provided along with ``min_confidence`` or
    ``min_observations``, elements that don't meet the threshold are skipped.
    Elements with no metadata are kept unconditionally.
    """
    meta = element_meta or {}
    wall_meta = _meta_index(meta.get("walls"))
    door_meta = _meta_index(meta.get("doors"))
    win_meta = _meta_index(meta.get("windows"))
    bbox_meta = _meta_index(meta.get("bboxes"))

    pts_chunks: List[np.ndarray] = []
    col_chunks: List[np.ndarray] = []

    def _add(edges: List[Tuple[np.ndarray, np.ndarray]],
             color: Tuple[int, int, int]) -> None:
        for a, b in edges:
            pts = _sample_line(a, b, edge_spacing)
            pts_chunks.append(pts)
            col_chunks.append(
                np.tile(np.asarray(color, dtype=np.uint8), (len(pts), 1)))

    for w in scene.walls:
        if not _passes_filter(wall_meta.get(w.id, {}),
                              min_confidence, min_observations):
            continue
        _add(_wall_edges(w), wall_color)

    for d in scene.doors:
        if not _passes_filter(door_meta.get(d.id, {}),
                              min_confidence, min_observations):
            continue
        _add(_fixture_edges(d, scene.walls), door_color)

    for wi in scene.windows:
        if not _passes_filter(win_meta.get(wi.id, {}),
                              min_confidence, min_observations):
            continue
        _add(_fixture_edges(wi, scene.walls), window_color)

    for b in scene.bboxes:
        if not _passes_filter(bbox_meta.get(b.id, {}),
                              min_confidence, min_observations):
            continue
        _add(_bbox_edges(b), bbox_color)

    if not pts_chunks:
        return (np.zeros((0, 3), dtype=np.float64),
                np.zeros((0, 3), dtype=np.uint8))
    return (np.concatenate(pts_chunks).astype(np.float64),
            np.concatenate(col_chunks).astype(np.uint8))
