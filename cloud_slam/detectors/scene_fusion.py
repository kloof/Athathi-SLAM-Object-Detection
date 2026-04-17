"""Iterative fusion of successive SpatialLM Scene passes into one coherent scene.

As a scan grows, SpatialLM is invoked periodically and each call yields a full
``Scene``. FusedScene matches elements across passes (walls by geometry, bboxes
by IoU+class, doors/windows by parent wall + position), accumulates running
averages, and tracks a per-element confidence that climbs on repeat observation
and decays when an element is missed.
"""

from dataclasses import dataclass, field
from math import atan2, cos, degrees, hypot, sin
from typing import Any, Dict, List, Optional, Tuple

from cloud_slam.detectors.scene import Bbox, Door, Scene, Wall, Window

FusionDelta = Dict[str, int]

# Confidence-model constants (see module docstring).
_INIT_CONF = 0.5
_HIT_ALPHA = 0.6
_MISS_FACTOR = 0.85


@dataclass
class _Tracked:
    """Tracked Scene element with running-average best estimate + confidence."""
    fused_id: int
    entity: Any
    num_observations: int
    first_pass_seen: int
    last_pass_seen: int
    confidence: float
    yaw_sin_sum: float = 0.0  # used only by bboxes (circular-mean yaw)
    yaw_cos_sum: float = 0.0


# ---- geometry helpers ----

def _seg_xy(w: Wall) -> Tuple[float, float, float]:
    """(dx, dy, length) of a wall in XY."""
    dx, dy = w.bx - w.ax, w.by - w.ay
    return dx, dy, hypot(dx, dy)


def _angle_deg(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Smallest angle in degrees between two 2D directions (anti-parallel=parallel)."""
    if hypot(*a) < 1e-9 or hypot(*b) < 1e-9:
        return 180.0
    ang = degrees(abs(atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1])))
    return 180.0 - ang if ang > 90.0 else ang


def _perp_dist(px: float, py: float, w: Wall) -> float:
    """Perpendicular distance from point to the infinite line of wall w."""
    dx, dy, L = _seg_xy(w)
    if L < 1e-9:
        return hypot(px - w.ax, py - w.ay)
    return abs((px - w.ax) * dy - (py - w.ay) * dx) / L


def _overlap_ratio(wa: Wall, wb: Wall) -> float:
    """Overlap ratio along the mean direction, normalized by shorter length."""
    ax, ay, la = _seg_xy(wa)
    bx, by, lb = _seg_xy(wb)
    if la < 1e-9 or lb < 1e-9:
        return 0.0
    ux, uy = ax / la, ay / la
    if ux * bx + uy * by < 0:
        bx, by = -bx, -by
    mx, my = ux + bx / lb, uy + by / lb
    mn = hypot(mx, my)
    if mn < 1e-9:
        return 0.0
    ux, uy = mx / mn, my / mn
    ox, oy = 0.5 * (wa.ax + wa.bx), 0.5 * (wa.ay + wa.by)
    _p = lambda x, y: (x - ox) * ux + (y - oy) * uy
    a0, a1 = sorted((_p(wa.ax, wa.ay), _p(wa.bx, wa.by)))
    b0, b1 = sorted((_p(wb.ax, wb.ay), _p(wb.bx, wb.by)))
    return max(0.0, min(a1, b1) - max(a0, b0)) / min(la, lb)


def _aabb_iou(a: Bbox, b: Bbox) -> float:
    """3D IoU of two bboxes, treated as axis-aligned in yaw (per spec)."""
    def box(e: Bbox):
        hx, hy, hz = 0.5 * e.scale_x, 0.5 * e.scale_y, 0.5 * e.scale_z
        return (e.position_x - hx, e.position_y - hy, e.position_z - hz,
                e.position_x + hx, e.position_y + hy, e.position_z + hz)
    ax0, ay0, az0, ax1, ay1, az1 = box(a)
    bx0, by0, bz0, bx1, by1, bz1 = box(b)
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    iz = max(0.0, min(az1, bz1) - max(az0, bz0))
    inter = ix * iy * iz
    if inter <= 0.0:
        return 0.0
    va = (ax1 - ax0) * (ay1 - ay0) * (az1 - az0)
    vb = (bx1 - bx0) * (by1 - by0) * (bz1 - bz0)
    union = va + vb - inter
    return inter / union if union > 1e-12 else 0.0


def _rmean(prev: float, n: int, new: float) -> float:
    """Observation-count-weighted running mean."""
    return (prev * n + new) / (n + 1)


def _bump(c: float) -> float:
    """On-hit confidence bump: c' = 1 - (1-c) * _HIT_ALPHA."""
    return 1.0 - (1.0 - c) * _HIT_ALPHA


@dataclass
class FusedScene:
    """Iteratively fuses successive SpatialLM Scene passes into one coherent Scene."""

    wall_match_dist: float = 0.25
    wall_match_angle_deg: float = 10.0
    wall_overlap_ratio: float = 0.30
    bbox_iou_threshold: float = 0.25
    door_match_dist: float = 0.40
    confidence_drop_threshold: float = 0.3
    stale_passes: int = 3

    _walls: List[_Tracked] = field(default_factory=list, init=False)
    _doors: List[_Tracked] = field(default_factory=list, init=False)
    _windows: List[_Tracked] = field(default_factory=list, init=False)
    _bboxes: List[_Tracked] = field(default_factory=list, init=False)
    _pass_idx: int = field(default=-1, init=False)
    _next_wid: int = field(default=0, init=False)
    _next_fid: int = field(default=0, init=False)
    _next_bid: int = field(default=0, init=False)

    # ---- public API ----

    def update(self, scene: Scene) -> FusionDelta:
        """Integrate a new Scene into the fused state; return a counts delta."""
        self._pass_idx += 1
        p = self._pass_idx

        pre_w = {t.fused_id for t in self._walls}
        pre_d = {t.fused_id for t in self._doors}
        pre_win = {t.fused_id for t in self._windows}
        pre_b = {t.fused_id for t in self._bboxes}
        hit_w, hit_d, hit_win, hit_b = set(), set(), set(), set()
        walls_new = bboxes_new = bboxes_matched = 0

        # Walls (build id-remap so doors/windows resolve to fused parent wall ids).
        wall_id_remap: Dict[int, int] = {}
        for w in scene.walls:
            idx = self._match_wall(w)
            if idx is not None:
                fid = self._walls[idx].fused_id
                self._fuse_wall(idx, w, p)
                hit_w.add(fid)
                wall_id_remap[w.id] = fid
            else:
                wall_id_remap[w.id] = self._add_wall(w, p)
                walls_new += 1

        for d in scene.doors:
            wid = wall_id_remap.get(d.wall_id, d.wall_id)
            idx = self._match_fixture(self._doors, d, wid)
            if idx is not None:
                hit_d.add(self._doors[idx].fused_id)
                self._fuse_fixture(self._doors, idx, d, wid, p)
            else:
                self._add_fixture(self._doors, d, wid, p, Door)

        for w in scene.windows:
            wid = wall_id_remap.get(w.wall_id, w.wall_id)
            idx = self._match_fixture(self._windows, w, wid)
            if idx is not None:
                hit_win.add(self._windows[idx].fused_id)
                self._fuse_fixture(self._windows, idx, w, wid, p)
            else:
                self._add_fixture(self._windows, w, wid, p, Window)

        for b in scene.bboxes:
            idx = self._match_bbox(b)
            if idx is not None:
                hit_b.add(self._bboxes[idx].fused_id)
                self._fuse_bbox(idx, b, p)
                bboxes_matched += 1
            else:
                self._add_bbox(b, p)
                bboxes_new += 1

        # Miss-decay: only elements that existed before this pass and weren't hit.
        for lst, pre, hit in ((self._walls, pre_w, hit_w),
                              (self._doors, pre_d, hit_d),
                              (self._windows, pre_win, hit_win),
                              (self._bboxes, pre_b, hit_b)):
            for t in lst:
                if t.fused_id in pre and t.fused_id not in hit:
                    t.confidence *= _MISS_FACTOR

        dropped = self._drop_stale()
        return {"pass": p, "walls_matched": len(hit_w), "walls_new": walls_new,
                "bboxes_matched": bboxes_matched, "bboxes_new": bboxes_new,
                "dropped": dropped}

    def current(self) -> Scene:
        """Snapshot Scene with below-threshold elements filtered out (no mutation)."""
        th = self.confidence_drop_threshold
        s = Scene()
        s.walls = [t.entity for t in self._walls if t.confidence >= th]
        s.doors = [t.entity for t in self._doors if t.confidence >= th]
        s.windows = [t.entity for t in self._windows if t.confidence >= th]
        s.bboxes = [t.entity for t in self._bboxes if t.confidence >= th]
        return s

    def current_with_metadata(self) -> Dict[str, Any]:
        """Snapshot plus per-element observation/confidence/pass-range metadata."""
        th = self.confidence_drop_threshold
        _m = lambda t: {"id": t.fused_id, "observations": t.num_observations,
                        "confidence": t.confidence,
                        "first_pass": t.first_pass_seen, "last_pass": t.last_pass_seen}
        return {
            "scene": self.current().to_json(),
            "meta": {
                "walls":   [_m(t) for t in self._walls   if t.confidence >= th],
                "doors":   [_m(t) for t in self._doors   if t.confidence >= th],
                "windows": [_m(t) for t in self._windows if t.confidence >= th],
                "bboxes":  [_m(t) for t in self._bboxes  if t.confidence >= th],
            },
        }

    # ---- wall ops ----

    def _match_wall(self, w_new: Wall) -> Optional[int]:
        d_new = (w_new.bx - w_new.ax, w_new.by - w_new.ay)
        best_idx, best_dist = None, float("inf")
        for i, t in enumerate(self._walls):
            we: Wall = t.entity
            d_e = (we.bx - we.ax, we.by - we.ay)
            if _angle_deg(d_new, d_e) > self.wall_match_angle_deg:
                continue
            mx, my = 0.5 * (we.ax + we.bx), 0.5 * (we.ay + we.by)
            pd = _perp_dist(mx, my, w_new)
            if pd > self.wall_match_dist:
                continue
            if _overlap_ratio(we, w_new) < self.wall_overlap_ratio:
                continue
            if pd < best_dist:
                best_idx, best_dist = i, pd
        return best_idx

    def _add_wall(self, w: Wall, p: int) -> int:
        fid = self._next_wid
        self._next_wid += 1
        self._walls.append(_Tracked(
            fused_id=fid,
            entity=Wall(id=fid, ax=w.ax, ay=w.ay, az=w.az, bx=w.bx, by=w.by, bz=w.bz,
                        height=w.height, thickness=w.thickness),
            num_observations=1, first_pass_seen=p, last_pass_seen=p,
            confidence=_bump(_INIT_CONF),
        ))
        return fid

    def _fuse_wall(self, idx: int, w_new: Wall, p: int) -> None:
        t = self._walls[idx]
        n, cur = t.num_observations, t.entity
        # Endpoint pairing that minimizes total distance (handle direction flip).
        d_aa = (hypot(cur.ax - w_new.ax, cur.ay - w_new.ay)
                + hypot(cur.bx - w_new.bx, cur.by - w_new.by))
        d_ab = (hypot(cur.ax - w_new.bx, cur.ay - w_new.by)
                + hypot(cur.bx - w_new.ax, cur.by - w_new.ay))
        if d_ab < d_aa:
            w_new = Wall(id=w_new.id,
                         ax=w_new.bx, ay=w_new.by, az=w_new.bz,
                         bx=w_new.ax, by=w_new.ay, bz=w_new.az,
                         height=w_new.height, thickness=w_new.thickness)
        t.entity = Wall(
            id=t.fused_id,
            ax=_rmean(cur.ax, n, w_new.ax), ay=_rmean(cur.ay, n, w_new.ay),
            az=_rmean(cur.az, n, w_new.az),
            bx=_rmean(cur.bx, n, w_new.bx), by=_rmean(cur.by, n, w_new.by),
            bz=_rmean(cur.bz, n, w_new.bz),
            height=_rmean(cur.height, n, w_new.height),
            thickness=_rmean(cur.thickness, n, w_new.thickness),
        )
        t.num_observations += 1
        t.last_pass_seen = p
        t.confidence = _bump(t.confidence)

    # ---- fixture (door/window) ops ----

    def _match_fixture(self, tracked: List[_Tracked], f_new, wall_id: int) -> Optional[int]:
        best_idx, best_dist = None, float("inf")
        for i, t in enumerate(tracked):
            fx = t.entity
            if fx.wall_id != wall_id:
                continue
            d = hypot(fx.position_x - f_new.position_x, fx.position_y - f_new.position_y)
            if d <= self.door_match_dist and d < best_dist:
                best_idx, best_dist = i, d
        return best_idx

    def _add_fixture(self, tracked: List[_Tracked], f_new, wall_id: int, p: int, cls) -> None:
        fid = self._next_fid
        self._next_fid += 1
        tracked.append(_Tracked(
            fused_id=fid,
            entity=cls(id=fid, wall_id=wall_id,
                       position_x=f_new.position_x, position_y=f_new.position_y,
                       position_z=f_new.position_z,
                       width=f_new.width, height=f_new.height),
            num_observations=1, first_pass_seen=p, last_pass_seen=p,
            confidence=_bump(_INIT_CONF),
        ))

    def _fuse_fixture(self, tracked: List[_Tracked], idx: int, f_new,
                      wall_id: int, p: int) -> None:
        t = tracked[idx]
        fx, n, cls = t.entity, t.num_observations, type(t.entity)
        t.entity = cls(
            id=t.fused_id, wall_id=wall_id,
            position_x=_rmean(fx.position_x, n, f_new.position_x),
            position_y=_rmean(fx.position_y, n, f_new.position_y),
            position_z=_rmean(fx.position_z, n, f_new.position_z),
            width=_rmean(fx.width, n, f_new.width),
            height=_rmean(fx.height, n, f_new.height),
        )
        t.num_observations += 1
        t.last_pass_seen = p
        t.confidence = _bump(t.confidence)

    # ---- bbox ops ----

    def _match_bbox(self, b_new: Bbox) -> Optional[int]:
        # Prefer same-class match above IoU threshold; otherwise allow class-conflict
        # fallback (spec: if classes disagree but IoU still strong, the existing
        # class wins by observation count).
        best_idx, best_iou = None, self.bbox_iou_threshold
        for i, t in enumerate(self._bboxes):
            if t.entity.class_name != b_new.class_name:
                continue
            iou = _aabb_iou(t.entity, b_new)
            if iou > best_iou:
                best_idx, best_iou = i, iou
        if best_idx is not None:
            return best_idx
        best_iou = self.bbox_iou_threshold
        for i, t in enumerate(self._bboxes):
            if t.entity.class_name == b_new.class_name:
                continue
            iou = _aabb_iou(t.entity, b_new)
            if iou > best_iou:
                best_idx, best_iou = i, iou
        return best_idx

    def _add_bbox(self, b: Bbox, p: int) -> None:
        fid = self._next_bid
        self._next_bid += 1
        self._bboxes.append(_Tracked(
            fused_id=fid,
            entity=Bbox(id=fid, class_name=b.class_name,
                        position_x=b.position_x, position_y=b.position_y,
                        position_z=b.position_z, angle_z=b.angle_z,
                        scale_x=b.scale_x, scale_y=b.scale_y, scale_z=b.scale_z),
            num_observations=1, first_pass_seen=p, last_pass_seen=p,
            confidence=_bump(_INIT_CONF),
            yaw_sin_sum=sin(b.angle_z), yaw_cos_sum=cos(b.angle_z),
        ))

    def _fuse_bbox(self, idx: int, b_new: Bbox, p: int) -> None:
        t = self._bboxes[idx]
        cur, n = t.entity, t.num_observations
        # Class stays sticky to the incumbent: it already has n≥1 observations,
        # so on any disagreement it wins the "more observations" tie-break.
        t.yaw_sin_sum += sin(b_new.angle_z)
        t.yaw_cos_sum += cos(b_new.angle_z)
        t.entity = Bbox(
            id=t.fused_id, class_name=cur.class_name,
            position_x=_rmean(cur.position_x, n, b_new.position_x),
            position_y=_rmean(cur.position_y, n, b_new.position_y),
            position_z=_rmean(cur.position_z, n, b_new.position_z),
            angle_z=atan2(t.yaw_sin_sum, t.yaw_cos_sum),
            scale_x=_rmean(cur.scale_x, n, b_new.scale_x),
            scale_y=_rmean(cur.scale_y, n, b_new.scale_y),
            scale_z=_rmean(cur.scale_z, n, b_new.scale_z),
        )
        t.num_observations += 1
        t.last_pass_seen = p
        t.confidence = _bump(t.confidence)

    # ---- stale drop ----

    def _drop_stale(self) -> int:
        th_pass = self._pass_idx - self.stale_passes
        th_conf = self.confidence_drop_threshold
        keep = lambda t: not (t.last_pass_seen <= th_pass and t.confidence < th_conf)
        before = len(self._walls) + len(self._doors) + len(self._windows) + len(self._bboxes)
        self._walls = [t for t in self._walls if keep(t)]
        self._doors = [t for t in self._doors if keep(t)]
        self._windows = [t for t in self._windows if keep(t)]
        self._bboxes = [t for t in self._bboxes if keep(t)]
        return before - len(self._walls) - len(self._doors) - len(self._windows) - len(self._bboxes)
