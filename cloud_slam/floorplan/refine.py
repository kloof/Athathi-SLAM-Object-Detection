"""Wall refinement pipeline (variant D_refined).

Research-grade refinement of the simplified ceiling polygon:
  Stage 1: per-edge wall-band RANSAC + Huber LO-refine
  Stage 2: learn dominant directions (weighted histogram, architectural prior)
  Stage 3: tolerance-gated tiered snap
  Stage 4: merge collinear neighbours
  Stage 5: vertex translation (constrained LSQ, fallback to re-intersection)
  Stage 6: gap-split segments
  Stage 7: length-constrained corner adjustment

Grounded in:
  - ZInD (80% Manhattan, 15% diagonal, 5% rare) — architectural prior strengths
  - Cloud2BIM arXiv:2503.11498 — 3° collinear-merge tolerance
  - Chum et al. 2003 LO-RANSAC — local Huber refinement step
  - Structure-preserving Simplification arXiv:2408.06814 — vertex translation

Extracted verbatim from the pre-split monolithic floorplan module so the
algorithms are unchanged. Imported by `cloud_slam.floorplan.__init__`.
"""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from shapely.geometry import Polygon as ShapelyPolygon


# Fixed snap priors and their pull strengths (from ZInD frequencies).
# Angles in radians, on [0, π) (edges are undirected).
_MANHATTAN_ANGLES = np.array([0.0, np.pi / 2])        # 0°, 90°
_DIAGONAL_ANGLES = np.array([np.pi / 4, 3 * np.pi / 4])  # 45°, 135°
_HEX_ANGLES = np.array([np.pi / 6, np.pi / 3,
                        2 * np.pi / 3, 5 * np.pi / 6])    # 30, 60, 120, 150°


def _wrap_angle_pi(a):
    """Wrap a scalar or array of angles to [0, π)."""
    return np.mod(a, np.pi)


def _angle_circ_diff(a, b):
    """Smallest absolute difference between two angles on [0, π)."""
    d = np.abs(_wrap_angle_pi(a) - _wrap_angle_pi(b))
    return np.minimum(d, np.pi - d)


def _fit_line_huber(points_xy):
    """Robust line fit via cv2.fitLine(DIST_HUBER). Returns (direction, point)."""
    if len(points_xy) < 2:
        return None, None
    pts = points_xy.astype(np.float32).reshape(-1, 1, 2)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
    direction = np.array([vx, vy], dtype=np.float64)
    direction /= max(np.linalg.norm(direction), 1e-9)
    return direction, np.array([x0, y0], dtype=np.float64)


def _point_line_residuals(points_xy, direction, point_on_line):
    """Perpendicular distance from each 2D point to the given line."""
    perp = np.array([-direction[1], direction[0]])
    return (points_xy - point_on_line) @ perp


def _adaptive_ransac_threshold(residuals):
    """MAD-based threshold: 2.5 * 1.4826 * MAD, clamped to [0.02, 0.06]."""
    mad = np.median(np.abs(residuals - np.median(residuals)))
    sigma = 1.4826 * mad
    return float(np.clip(2.5 * sigma, 0.02, 0.06))


# ----- Stage 1: per-edge wall-band RANSAC + Huber LO-refine -----

def _stage1_per_edge_ransac(walls_in, wall_band_xy, edge_band=0.25,
                             min_inliers=20, max_iter=200, verbose=False):
    """For each polygon edge, fit a robust line to nearby wall-band points.

    Args:
        walls_in: list of (p1, p2, angle_deg, length_m) from `extract_walls`.
        wall_band_xy: (N, 2) XY coords of wall-band points (projected cloud
                      at mid-height — avoids floor/ceiling overlap).
        edge_band: perpendicular search distance around each edge (m).
        min_inliers: minimum inliers to accept a refined fit.

    Returns:
        list of dicts with keys:
          'p1', 'p2'        — refined endpoints along the fitted line
          'direction'       — unit vector along the wall
          'point_on_line'   — any point on the fitted line
          'inliers_xy'      — (K, 2) inlier points
          'residual'        — mean |perpendicular distance| of inliers
          'length'          — length of the refined segment
          'angle_deg'       — angle on [0, 180)
          'confidence'      — float in [0, 1] (roughly inliers / edge-extent)
          'fallback'        — True if original segment is used (no good fit)
    """
    refined = []

    for p1, p2, angle_deg, length in walls_in:
        edge_vec = np.asarray(p2) - np.asarray(p1)
        edge_len = float(np.linalg.norm(edge_vec))
        if edge_len < 1e-6:
            refined.append({
                'p1': np.asarray(p1, dtype=float),
                'p2': np.asarray(p2, dtype=float),
                'direction': np.array([1.0, 0.0]),
                'point_on_line': np.asarray(p1, dtype=float),
                'inliers_xy': np.zeros((0, 2)),
                'residual': 0.0,
                'length': edge_len,
                'angle_deg': angle_deg,
                'confidence': 0.0,
                'fallback': True,
            })
            continue
        edge_dir = edge_vec / edge_len
        edge_perp = np.array([-edge_dir[1], edge_dir[0]])
        mid = (np.asarray(p1) + np.asarray(p2)) / 2

        # Gather wall-band points within edge_band of the infinite edge line
        # AND roughly within the edge's along-extent (with slack).
        rel = wall_band_xy - mid
        perp_dist = np.abs(rel @ edge_perp)
        along_dist = np.abs(rel @ edge_dir)
        slack = 0.5  # extra half-metre at each end to capture wall extension
        mask = (perp_dist < edge_band) & (along_dist < edge_len / 2 + slack)
        candidates = wall_band_xy[mask]

        if len(candidates) < min_inliers:
            # Too few points — keep original edge as fallback
            refined.append({
                'p1': np.asarray(p1, dtype=float),
                'p2': np.asarray(p2, dtype=float),
                'direction': edge_dir,
                'point_on_line': mid,
                'inliers_xy': candidates,
                'residual': 0.0,
                'length': edge_len,
                'angle_deg': angle_deg,
                'confidence': float(len(candidates)) / max(int(edge_len / 0.05), 1),
                'fallback': True,
            })
            continue

        # Initial Huber fit on candidates
        direction, point_on_line = _fit_line_huber(candidates)
        if direction is None:
            refined.append({
                'p1': np.asarray(p1, dtype=float),
                'p2': np.asarray(p2, dtype=float),
                'direction': edge_dir,
                'point_on_line': mid,
                'inliers_xy': candidates,
                'residual': 0.0,
                'length': edge_len,
                'angle_deg': angle_deg,
                'confidence': 0.0,
                'fallback': True,
            })
            continue

        # LO-RANSAC: iterate to tighten the inlier set using MAD threshold
        inliers = candidates
        for _ in range(3):
            residuals = _point_line_residuals(inliers, direction, point_on_line)
            thresh = _adaptive_ransac_threshold(residuals)
            inlier_mask = np.abs(residuals) < thresh
            if inlier_mask.sum() < min_inliers:
                break
            inliers = inliers[inlier_mask]
            new_dir, new_pt = _fit_line_huber(inliers)
            if new_dir is None:
                break
            direction, point_on_line = new_dir, new_pt

        # Project inliers onto fitted line to find endpoints
        if len(inliers) < min_inliers:
            # Keep original; mark low-confidence
            refined.append({
                'p1': np.asarray(p1, dtype=float),
                'p2': np.asarray(p2, dtype=float),
                'direction': edge_dir,
                'point_on_line': mid,
                'inliers_xy': candidates,
                'residual': 0.0,
                'length': edge_len,
                'angle_deg': angle_deg,
                'confidence': float(len(candidates)) / max(int(edge_len / 0.05), 1),
                'fallback': True,
            })
            continue

        t = (inliers - point_on_line) @ direction
        t_min, t_max = float(t.min()), float(t.max())
        new_p1 = point_on_line + direction * t_min
        new_p2 = point_on_line + direction * t_max
        new_length = float(np.linalg.norm(new_p2 - new_p1))
        # Angle on [0, 180)
        new_angle = float(np.degrees(np.arctan2(direction[1], direction[0])) % 180)

        residuals_final = _point_line_residuals(inliers, direction, point_on_line)
        residual = float(np.mean(np.abs(residuals_final)))
        # Confidence: how full is the expected edge length?
        # (inliers span at least 60% of original edge → high confidence)
        confidence = min(1.0, new_length / max(edge_len * 0.6, 0.1))
        confidence *= min(1.0, len(inliers) / max(edge_len / 0.03, 20))

        refined.append({
            'p1': new_p1,
            'p2': new_p2,
            'direction': direction,
            'point_on_line': point_on_line,
            'inliers_xy': inliers,
            'residual': residual,
            'length': new_length,
            'angle_deg': new_angle,
            'confidence': float(confidence),
            'fallback': False,
        })

    if verbose:
        n_good = sum(1 for r in refined if not r['fallback'])
        print(f"  [Stage 1] {n_good}/{len(refined)} edges refined "
              f"(mean residual={np.mean([r['residual'] for r in refined if not r['fallback']]) if n_good else 0:.4f} m)")
    return refined


# ----- Stage 2: learn dominant directions -----

def _stage2_dominant_directions(refined_edges, k_max=3, bandwidth_deg=2.0,
                                 min_separation_deg=12.0, min_edge_length=0.3,
                                 prior_weights=None, verbose=False):
    """Length-weighted circular histogram on [0, π) with NMS.

    Architectural prior (from ZInD frequencies) seeds the histogram with
    pseudo-counts at 0°/45°/90°/135° with strengths {1.0, 0.3, 1.0, 0.3}
    relative to total perimeter length.
    """
    if prior_weights is None:
        prior_weights = {0.0: 1.0, 45.0: 0.3, 90.0: 1.0, 135.0: 0.3}

    # Collect voting edges (length > min threshold)
    votes = [(r['angle_deg'], r['length']) for r in refined_edges
             if r['length'] >= min_edge_length and not r['fallback']]
    total_len = sum(l for _, l in votes) + 1e-9

    # 1° bin histogram on [0, 180)
    hist = np.zeros(180)
    for angle_deg, length in votes:
        bin_idx = int(angle_deg) % 180
        hist[bin_idx] += length

    # Seed with architectural prior (pseudo-counts scaled to total length)
    for ang_deg, strength in prior_weights.items():
        bin_idx = int(ang_deg) % 180
        hist[bin_idx] += total_len * strength * 0.10

    # Gaussian smooth on the circular domain
    # (pad with copies to simulate wrapping, then trim)
    sigma = bandwidth_deg
    padded = np.concatenate([hist[-10:], hist, hist[:10]])
    smoothed = gaussian_filter1d(padded, sigma=sigma, mode='constant')[10:190]

    # NMS: find peaks above 10% of max, separated by >= min_separation_deg
    peaks = []
    order = np.argsort(-smoothed)
    min_height = 0.10 * smoothed.max()
    for idx in order:
        if smoothed[idx] < min_height:
            break
        ang = float(idx)
        if all(min(abs(ang - p), 180 - abs(ang - p)) >= min_separation_deg
               for p in peaks):
            peaks.append(ang)
        if len(peaks) >= k_max:
            break

    peaks_sorted = sorted(peaks)
    if verbose:
        print(f"  [Stage 2] dominant directions (deg): "
              f"{[round(p, 1) for p in peaks_sorted]}")
    return np.array(peaks_sorted)


# ----- Stage 3: tolerance-gated tiered snap -----

def _stage3_tiered_snap(refined_edges, dominant_deg, tol_dominant=7.0,
                        tol_diagonal=4.0, tol_hex=3.0, verbose=False):
    """Snap each refined edge angle to the closest in-tolerance direction.

    Priority order: dominant peaks (Stage 2) > architectural priors > no snap.
    Returns list of dicts (same shape as refined_edges) with 'angle_deg' and
    new 'snapped_to' key in {"dominant", "manhattan", "diagonal", "hex", "free"}.
    """
    out = []
    for e in refined_edges:
        ang = e['angle_deg']
        if e['fallback']:
            # Cannot trust the angle — leave untouched, mark free
            out.append({**e, 'snapped_to': 'free'})
            continue

        # Try dominant peaks first (broadest tolerance)
        best_target = None
        best_kind = None
        best_diff = np.inf
        for p in dominant_deg:
            d = min(abs(ang - p), 180 - abs(ang - p))
            if d < tol_dominant and d < best_diff:
                best_diff, best_target, best_kind = d, p, 'dominant'

        if best_target is None:
            # Tier 2: architectural priors (only if not already a dominant peak)
            def _not_already_dominant(p):
                return not any(
                    min(abs(p - dp), 180 - abs(p - dp)) < 1e-3
                    for dp in dominant_deg)

            for p in np.degrees(_MANHATTAN_ANGLES):
                if not _not_already_dominant(p):
                    continue
                d = min(abs(ang - p), 180 - abs(ang - p))
                if d < tol_dominant and d < best_diff:  # same tol as Tier 1
                    best_diff, best_target, best_kind = d, p, 'manhattan'
            for p in np.degrees(_DIAGONAL_ANGLES):
                if not _not_already_dominant(p):
                    continue
                d = min(abs(ang - p), 180 - abs(ang - p))
                if d < tol_diagonal and d < best_diff:
                    best_diff, best_target, best_kind = d, p, 'diagonal'
            for p in np.degrees(_HEX_ANGLES):
                if not _not_already_dominant(p):
                    continue
                d = min(abs(ang - p), 180 - abs(ang - p))
                if d < tol_hex and d < best_diff:
                    best_diff, best_target, best_kind = d, p, 'hex'

        if best_target is None:
            out.append({**e, 'snapped_to': 'free'})
            continue

        # Rebuild direction from snapped angle, keep point_on_line the same
        new_dir = np.array([np.cos(np.radians(best_target)),
                            np.sin(np.radians(best_target))])
        # Recompute endpoints: project original inliers onto the snapped line
        pol = e['point_on_line']
        if len(e['inliers_xy']) > 0:
            t = (e['inliers_xy'] - pol) @ new_dir
            t_min, t_max = float(t.min()), float(t.max())
        else:
            # Fallback: use old p1/p2 projected
            t1 = float((e['p1'] - pol) @ new_dir)
            t2 = float((e['p2'] - pol) @ new_dir)
            t_min, t_max = min(t1, t2), max(t1, t2)
        new_p1 = pol + new_dir * t_min
        new_p2 = pol + new_dir * t_max
        out.append({**e,
                    'direction': new_dir,
                    'p1': new_p1, 'p2': new_p2,
                    'angle_deg': float(best_target),
                    'length': float(np.linalg.norm(new_p2 - new_p1)),
                    'snapped_to': best_kind})

    if verbose:
        counts = {}
        for e in out:
            counts[e['snapped_to']] = counts.get(e['snapped_to'], 0) + 1
        print(f"  [Stage 3] snap tallies: {counts}")
    return out


# ----- Stage 4: merge collinear neighbours -----

def _stage4_merge_collinear(edges, angle_tol_deg=3.0, gap_max=1.2,
                             verbose=False):
    """Walk polygon; merge consecutive edges that are collinear continuations.

    Criteria: angle diff < angle_tol_deg AND perpendicular offset within the
    noise level AND along-line gap < gap_max. Refits the merged edge by
    Huber on the union of inliers.
    """
    if len(edges) < 2:
        return edges

    merged = []
    i = 0
    n = len(edges)
    while i < n:
        current = edges[i]
        # Try to extend by consuming the next edge if collinear
        j = (i + 1) % n
        while j != i:
            nxt = edges[j]
            # Angle check
            dang = min(abs(current['angle_deg'] - nxt['angle_deg']),
                       180 - abs(current['angle_deg'] - nxt['angle_deg']))
            if dang > angle_tol_deg:
                break
            # Perpendicular offset: midpoint of nxt onto current line
            mid_next = (nxt['p1'] + nxt['p2']) / 2
            perp = np.array([-current['direction'][1], current['direction'][0]])
            off = abs((mid_next - current['point_on_line']) @ perp)
            sigma_est = max(current['residual'], 0.02)
            if off > max(sigma_est * 3, 0.10):
                break
            # Along-line gap between projected endpoints
            t_cur_max = float((current['p2'] - current['point_on_line'])
                              @ current['direction'])
            t_next_min = float((nxt['p1'] - current['point_on_line'])
                               @ current['direction'])
            gap = t_next_min - t_cur_max
            if gap > gap_max:
                break
            # All checks passed — merge
            if nxt['fallback'] or current['fallback']:
                break
            combined_inliers = np.vstack([current['inliers_xy'],
                                          nxt['inliers_xy']])
            direction, pol = _fit_line_huber(combined_inliers)
            if direction is None:
                break
            t = (combined_inliers - pol) @ direction
            new_p1 = pol + direction * float(t.min())
            new_p2 = pol + direction * float(t.max())
            new_angle = float(np.degrees(np.arctan2(direction[1],
                                                     direction[0])) % 180)
            # Replace current with merged
            current = {
                'p1': new_p1, 'p2': new_p2,
                'direction': direction,
                'point_on_line': pol,
                'inliers_xy': combined_inliers,
                'residual': float(np.mean(np.abs(
                    _point_line_residuals(combined_inliers, direction, pol)))),
                'length': float(np.linalg.norm(new_p2 - new_p1)),
                'angle_deg': new_angle,
                'confidence': min(current.get('confidence', 0) + nxt.get('confidence', 0), 1.0),
                'fallback': False,
                'snapped_to': current.get('snapped_to', 'free'),
            }
            # Consume the merged edge and don't wrap around indefinitely
            i = j
            j = (j + 1) % n
            if j == i:
                break
        merged.append(current)
        i += 1

    if verbose:
        print(f"  [Stage 4] merged {len(edges)} → {len(merged)} edges")
    return merged


# ----- Stage 5: vertex translation (constrained LSQ + fallback) -----

def _stage5_vertex_translation(edges, verbose=False):
    """Replace each polygon vertex with the intersection of neighbour lines.

    Fallback to pairwise intersection when lines are near-parallel (denom
    below threshold).
    """
    if len(edges) < 3:
        return edges

    new_corners = []
    n = len(edges)
    for i in range(n):
        e1 = edges[i]
        e2 = edges[(i + 1) % n]
        d1, p1 = e1['direction'], e1['point_on_line']
        d2, p2 = e2['direction'], e2['point_on_line']
        denom = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(denom) < 1e-6:
            # Near-parallel; use the midpoint of e1.p2 and e2.p1
            new_corners.append((e1['p2'] + e2['p1']) / 2)
            continue
        dp = p2 - p1
        t = (dp[0] * d2[1] - dp[1] * d2[0]) / denom
        new_corners.append(p1 + t * d1)

    # Rebuild edges with new corners, keeping direction + snapped_to
    out = []
    for i in range(n):
        p1 = new_corners[i - 1] if i > 0 else new_corners[-1]
        p2 = new_corners[i]
        length = float(np.linalg.norm(p2 - p1))
        if length < 1e-6:
            continue
        dir_new = (p2 - p1) / length
        angle_new = float(np.degrees(np.arctan2(dir_new[1], dir_new[0])) % 180)
        src = edges[i]
        out.append({
            'p1': p1, 'p2': p2,
            'direction': dir_new,
            'point_on_line': (p1 + p2) / 2,
            # Stash the pre-Stage-5 RANSAC-fitted direction and anchor so
            # Stage 7 can project inliers in the frame they were fitted in.
            # Stage 5's `direction`/`point_on_line` above are derived from
            # the line intersection, NOT from the data — using them for
            # inlier projection introduces a subtle coordinate-frame bias.
            'ransac_direction': np.asarray(src['direction']).copy(),
            'ransac_point_on_line': np.asarray(src['point_on_line']).copy(),
            'inliers_xy': src['inliers_xy'],
            'residual': src['residual'],
            'length': length,
            'angle_deg': angle_new,
            'confidence': src.get('confidence', 0.0),
            'fallback': src.get('fallback', False),
            'snapped_to': src.get('snapped_to', 'free'),
        })

    if verbose:
        print(f"  [Stage 5] vertex translation complete ({len(out)} edges)")
    return out


# ----- Stage 7: length-constrained corner adjustment -----

def _stage7_length_constrain(edges, slack=0.30, verbose=False):
    """Pull/push each shared corner so neither adjacent wall over- or
    under-extends beyond its data support.

    Stage 5 computes corners as line-line intersections of the snapped wall
    directions, which can place corners far outside the physical wall
    extent (over-extension). Stage 7 uses each wall's Stage-1 RANSAC
    inliers to compute a data extent [t_min, t_max] and clamps the corner's
    projection onto each wall to that extent (± slack).

    Projections use `ransac_direction` and `ransac_point_on_line` (stashed
    by Stage 5) because Stage 5 overwrites `direction`/`point_on_line` with
    intersection-derived values that are geometrically valid but in a
    different frame from the RANSAC fit.

    Safeguards (from 4-agent review):
      - Adaptive slack: `min(slack, max(0.05, 0.1 * wall_length))` — tight
        for short walls to avoid over-trimming into near-zero length.
      - Zero-length guard: skip the corner move if it would collapse a
        wall below 0.10 m.
      - Per-corner sign check: skip the move if it would flip a wall's
        p1→p2 orientation relative to its direction (bad geometry).
      - Final polygon validity check (Shapely): if the post-Stage-7 polygon
        self-intersects or invalidates, revert ALL moves for that polygon
        and leave Stage 5's output unchanged (honest: better no refinement
        than silently-corrupted geometry).
    """
    n = len(edges)
    if n < 3:
        return edges, {'n_moved': 0, 'reverted': False}

    # Snapshot for potential revert
    snapshot = [{
        'p1': np.asarray(e['p1']).copy(),
        'p2': np.asarray(e['p2']).copy(),
        'length': float(e['length']),
        'angle_deg': float(e['angle_deg']),
    } for e in edges]

    n_moved = 0
    shifts = []

    for i in range(n):
        e_curr = edges[i]
        e_next = edges[(i + 1) % n]
        if e_curr.get('fallback', False) or e_next.get('fallback', False):
            continue

        # Use pre-Stage-5 RANSAC frame (stashed by Stage 5)
        dir_i = np.asarray(e_curr.get('ransac_direction',
                                       e_curr['direction']))
        pol_i = np.asarray(e_curr.get('ransac_point_on_line',
                                       e_curr['point_on_line']))
        dir_j = np.asarray(e_next.get('ransac_direction',
                                       e_next['direction']))
        pol_j = np.asarray(e_next.get('ransac_point_on_line',
                                       e_next['point_on_line']))

        in_i = e_curr.get('inliers_xy', np.zeros((0, 2)))
        in_j = e_next.get('inliers_xy', np.zeros((0, 2)))
        if len(in_i) < 10 or len(in_j) < 10:
            continue

        # C0 is the shared corner between wall i and wall i+1
        C0 = np.asarray(e_curr['p2'])

        # Current along-wall parameters of C0
        t_i = float((C0 - pol_i) @ dir_i)
        t_j = float((C0 - pol_j) @ dir_j)

        # Data extent from inliers (in the RANSAC frame)
        t_i_data = (in_i - pol_i) @ dir_i
        t_j_data = (in_j - pol_j) @ dir_j
        t_i_min, t_i_max = float(t_i_data.min()), float(t_i_data.max())
        t_j_min, t_j_max = float(t_j_data.min()), float(t_j_data.max())

        # Adaptive slack: tight for short walls, capped at `slack` for long
        len_i = float(np.linalg.norm(e_curr['p2'] - e_curr['p1']))
        len_j = float(np.linalg.norm(e_next['p2'] - e_next['p1']))
        slack_eff = min(slack, max(0.05, 0.10 * max(len_i, len_j)))

        t_i_cl = float(np.clip(t_i, t_i_min - slack_eff, t_i_max + slack_eff))
        t_j_cl = float(np.clip(t_j, t_j_min - slack_eff, t_j_max + slack_eff))

        # No change needed if both clamps are no-ops
        if abs(t_i_cl - t_i) < 1e-6 and abs(t_j_cl - t_j) < 1e-6:
            continue

        # Two clamped candidate corner positions (one per wall's line)
        cand_i = pol_i + dir_i * t_i_cl
        cand_j = pol_j + dir_j * t_j_cl
        C_new = 0.5 * (cand_i + cand_j)

        # Safeguard: zero-length guard
        new_len_curr = float(np.linalg.norm(C_new - e_curr['p1']))
        new_len_next = float(np.linalg.norm(e_next['p2'] - C_new))
        if new_len_curr < 0.10 or new_len_next < 0.10:
            continue

        # Safeguard: per-corner sign check — ensure wall orientation
        # (p1 → p2 along direction) is preserved after the move.
        old_t_p1_curr = float((np.asarray(e_curr['p1']) - pol_i) @ dir_i)
        new_t_p2_curr = float((C_new - pol_i) @ dir_i)
        if new_t_p2_curr <= old_t_p1_curr:
            continue  # flipping the wall — abort this corner
        old_t_p2_next = float((np.asarray(e_next['p2']) - pol_j) @ dir_j)
        new_t_p1_next = float((C_new - pol_j) @ dir_j)
        if new_t_p1_next >= old_t_p2_next:
            continue

        # Commit the corner move
        shift = float(np.linalg.norm(C_new - C0))
        e_curr['p2'] = C_new
        e_next['p1'] = C_new
        e_curr['length'] = new_len_curr
        e_next['length'] = new_len_next
        n_moved += 1
        shifts.append(shift)

    # Final polygon validity check. If Shapely can't build a valid polygon,
    # revert ALL moves and keep Stage 5's output — better unchanged than
    # silently-corrupted.
    reverted = False
    if n_moved > 0:
        try:
            corners = [np.asarray(e['p1']) for e in edges]
            poly_check = ShapelyPolygon(corners)
            if (not poly_check.is_valid) or poly_check.area < 1e-3:
                # Revert
                for e, snap in zip(edges, snapshot):
                    e['p1'] = snap['p1']
                    e['p2'] = snap['p2']
                    e['length'] = snap['length']
                    e['angle_deg'] = snap['angle_deg']
                reverted = True
                if verbose:
                    print(f"  [Stage 7] polygon invalid after {n_moved} "
                          "corner moves — reverted all moves")
        except Exception as exc:
            # Any Shapely exception → revert defensively
            for e, snap in zip(edges, snapshot):
                e['p1'] = snap['p1']
                e['p2'] = snap['p2']
                e['length'] = snap['length']
                e['angle_deg'] = snap['angle_deg']
            reverted = True
            if verbose:
                print(f"  [Stage 7] Shapely exception ({exc}) — reverted")

    stats = {'n_moved': n_moved if not reverted else 0,
             'reverted': reverted}
    if verbose and not reverted:
        if n_moved > 0:
            max_shift = max(shifts)
            mean_shift = sum(shifts) / len(shifts)
            print(f"  [Stage 7] length-constrained {n_moved}/{n} corners "
                  f"(max shift={max_shift:.3f}m, "
                  f"mean shift={mean_shift:.3f}m)")
        else:
            print(f"  [Stage 7] no corners needed length-constraining")
    return edges, stats


# ----- Stage 6: gap-split segments -----

def _stage6_gap_split(edges, gap_thresh=0.3, min_segment=0.2, verbose=False):
    """For each edge, project inliers onto the line; split at gaps > gap_thresh.

    Each resulting sub-segment is emitted as a separate wall. Does not modify
    edges with no inliers or those marked fallback.
    """
    out = []
    for e in edges:
        if e.get('fallback', True) or len(e.get('inliers_xy', [])) < 5:
            out.append(e)
            continue
        direction = e['direction']
        pol = e['point_on_line']
        t = (e['inliers_xy'] - pol) @ direction
        order = np.argsort(t)
        t_sorted = t[order]
        # Find gaps
        diffs = np.diff(t_sorted)
        split_idx = np.where(diffs > gap_thresh)[0]
        if len(split_idx) == 0:
            out.append(e)
            continue
        # Split into runs
        starts = [0] + list(split_idx + 1)
        ends = list(split_idx + 1) + [len(t_sorted)]
        for s, ep in zip(starts, ends):
            run = t_sorted[s:ep]
            if len(run) < 5:
                continue
            seg_len = float(run[-1] - run[0])
            if seg_len < min_segment:
                continue
            p1 = pol + direction * float(run[0])
            p2 = pol + direction * float(run[-1])
            out.append({**e,
                        'p1': p1, 'p2': p2,
                        'length': seg_len,
                        'inliers_xy': e['inliers_xy'][order[s:ep]]})
    if verbose:
        print(f"  [Stage 6] gap-split: {len(edges)} → {len(out)} segments")
    return out


# ----- Top-level orchestration for variant D_refined -----

def _collect_wall_band(pts_3d, floor_z, ceiling_z, margin=0.3,
                        return_mask=False):
    """Return (M, 2) XY coords of points in the wall-band Z slice.

    Band: Z in [floor_z + margin, ceiling_z - margin] — the middle of the
    wall, avoiding floor/ceiling clutter.

    The `floor_z`/`ceiling_z` args may be sign-flipped by the caller for
    display purposes (when leveling-inversion was detected). We detect
    this by checking whether they lie within the actual pts Z range; if
    not, we fall back to percentile-derived bounds. This makes the
    wall-band extraction robust regardless of how the caller reports
    floor/ceiling heights.

    If `return_mask` is True, additionally returns the boolean mask into
    `pts_3d` — useful when a parallel per-point array (e.g. vision class
    labels) must be sliced in lockstep.
    """
    z = pts_3d[:, 2]
    if z.size < 100:
        xy = pts_3d[:, :2] if z.size else np.zeros((0, 2))
        if return_mask:
            mask = np.ones(z.size, dtype=bool) if z.size else np.zeros(0, dtype=bool)
            return xy, mask
        return xy

    z_pts_lo = float(np.percentile(z, 2))
    z_pts_hi = float(np.percentile(z, 98))

    lo_provided = min(float(floor_z), float(ceiling_z))
    hi_provided = max(float(floor_z), float(ceiling_z))

    # Check if the provided Z range overlaps with the actual cloud Z range.
    # If there's no overlap, the caller likely sign-flipped the values for
    # display — use percentile bounds instead.
    overlap_lo = max(lo_provided, z_pts_lo)
    overlap_hi = min(hi_provided, z_pts_hi)
    if overlap_hi - overlap_lo < 0.5:
        # No meaningful overlap — fall back to pts percentiles.
        lo_raw, hi_raw = z_pts_lo, z_pts_hi
    else:
        lo_raw, hi_raw = lo_provided, hi_provided

    m = margin
    if hi_raw - lo_raw < 2 * margin + 0.2:
        # Short room — shrink the margin
        m = max(0.10, (hi_raw - lo_raw) / 4)

    lo, hi = lo_raw + m, hi_raw - m
    mask = (z > lo) & (z < hi)
    if return_mask:
        return pts_3d[mask][:, :2], mask
    return pts_3d[mask][:, :2]


# ----- Vision helpers (Tier 1 per-wall type, Tier 3 count sanity) -----

_VISION_BUCKET_NAMES = ('other', 'wall', 'window', 'door', 'glass')


def _classify_wall_from_endpoints(p1, p2, pts_3d, point_labels,
                                   band_width=0.30, length_slack=0.30,
                                   min_support=10,
                                   special_threshold=0.45,
                                   wall_threshold=0.35,
                                   return_features=False):
    """Classify a wall segment by its *dominant* vision bucket, plus
    optional secondary features.

    Labels mean "this wall IS a ___" (not "has a ___"). Rationale:
    labeling a 5 m interior wall as 'door' just because 0.9 m of it is
    a doorway is misleading — the wall as a whole is still a wall, with
    a door in it. The door is already represented in `objects.json` via
    the 3D YOLOE detector, so we don't need to re-encode it as a wall
    type.

    A wall is tagged 'door' / 'window' / 'glass' only when that class
    is the clear majority (≥45 % of labeled band points). Walls with
    plain wall dominant (≥35 %) get 'wall'. Everything else is
    'unknown'.

    When `return_features=True`, also returns a list of secondary
    features seen at ≥10 % in the band — e.g. `('wall', ['door'])` for
    a regular wall with a door embedded.

    Args:
        p1, p2:            wall endpoints in XY (2,).
        pts_3d:            (N, 3) point cloud (same frame as refine_walls).
        point_labels:      (N,) uint8 bucket ids aligned with pts_3d.
        band_width:        perpendicular tolerance (m).
        length_slack:      extra half-metre at each end.
        min_support:       minimum non-'other' labeled points required.
        special_threshold: fraction to label wall as window/door/glass
                           (default 0.45 — clear majority).
        wall_threshold:    fraction to label as plain 'wall'. Lower than
                           `special_threshold` because walls can have
                           features (doors/windows) eating into the
                           majority.
        return_features:   if True, return (primary_label, feature_list).

    Returns:
        primary_label (str) — one of
          {'wall', 'window', 'door', 'glass', 'unknown'}.
        Or, when return_features=True,
          (primary_label, features) where features is a sorted list of
          strings drawn from {'wall','window','door','glass'} that each
          occupy ≥10 % of the band.
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    edge_vec = p2 - p1
    edge_len = float(np.linalg.norm(edge_vec))
    if edge_len < 1e-6:
        return ('unknown', []) if return_features else 'unknown'
    edge_dir = edge_vec / edge_len
    edge_perp = np.array([-edge_dir[1], edge_dir[0]])
    mid = (p1 + p2) / 2

    rel = pts_3d[:, :2] - mid
    perp_dist = np.abs(rel @ edge_perp)
    along_dist = np.abs(rel @ edge_dir)
    mask = (perp_dist < band_width) & (along_dist < edge_len / 2 + length_slack)
    if not mask.any():
        return ('unknown', []) if return_features else 'unknown'

    labels = np.asarray(point_labels)[mask]
    non_other = labels >= 1
    if non_other.sum() < min_support:
        return ('unknown', []) if return_features else 'unknown'

    labels = labels[non_other]
    counts = np.bincount(labels, minlength=len(_VISION_BUCKET_NAMES))
    total = int(counts.sum())
    if total == 0:
        return ('unknown', []) if return_features else 'unknown'

    wall_frac = counts[1] / total
    window_frac = counts[2] / total
    door_frac = counts[3] / total
    glass_frac = counts[4] / total

    # Primary label: clear majority wins. Specials outrank 'wall' only
    # when the special class is dominant.
    if glass_frac >= special_threshold:
        primary = 'glass'
    elif door_frac >= special_threshold:
        primary = 'door'
    elif window_frac >= special_threshold:
        primary = 'window'
    elif wall_frac >= wall_threshold:
        primary = 'wall'
    else:
        primary = 'unknown'

    if not return_features:
        return primary

    # Secondary features: anything ≥10 % and distinct from the primary.
    feature_threshold = 0.10
    feature_fracs = {
        'wall': wall_frac, 'window': window_frac,
        'door': door_frac, 'glass': glass_frac,
    }
    features = sorted(
        name for name, f in feature_fracs.items()
        if f >= feature_threshold and name != primary
    )
    return primary, features


def _compute_vision_wall_stats(pts_3d, point_labels,
                                grid_resolution=0.05, min_cluster_px=50):
    """Tier-3 sanity diagnostics on wall-class vision labels.

    Counts total wall-class points and how many connected XY blobs they
    form on a coarse occupancy grid. This is a log-only diagnostic; we
    do NOT override D_refined's wall count.

    Returns a dict:
        {'wall_point_count': int, 'wall_blob_count': int}
    """
    from scipy.ndimage import label as ndi_label

    if point_labels is None or len(point_labels) != len(pts_3d):
        return {'wall_point_count': 0, 'wall_blob_count': 0}

    wall_mask = np.asarray(point_labels) == 1
    n_pts = int(wall_mask.sum())
    if n_pts < min_cluster_px:
        return {'wall_point_count': n_pts, 'wall_blob_count': 0}

    xy = pts_3d[wall_mask, :2]
    x_min, y_min = xy.min(axis=0) - grid_resolution
    x_max, y_max = xy.max(axis=0) + grid_resolution
    W = max(int(np.ceil((x_max - x_min) / grid_resolution)), 1)
    H = max(int(np.ceil((y_max - y_min) / grid_resolution)), 1)

    grid = np.zeros((H, W), dtype=np.uint8)
    ui = np.clip(((xy[:, 0] - x_min) / grid_resolution).astype(np.int32), 0, W - 1)
    vi = np.clip(((xy[:, 1] - y_min) / grid_resolution).astype(np.int32), 0, H - 1)
    grid[vi, ui] = 1

    _, n_blobs = ndi_label(grid)
    return {'wall_point_count': n_pts, 'wall_blob_count': int(n_blobs)}


def _lookup_vision_labels_for_pts(pts, wall_labels_buffer, max_distance=0.5):
    """For each point in `pts`, look up the nearest vision-labeled bucket.

    `wall_labels_buffer` is the dict returned by detect_pipeline.run():
        {'xyz': (M, 3) float32, 'labels': (M,) uint8}.
    Returns a (len(pts),) uint8 array of bucket ids; points with no
    labeled neighbor within `max_distance` get bucket 0 (other).
    """
    from scipy.spatial import cKDTree

    xyz_labeled = wall_labels_buffer.get('xyz')
    lbl = wall_labels_buffer.get('labels')
    if xyz_labeled is None or lbl is None or len(lbl) == 0:
        return np.zeros(len(pts), dtype=np.uint8)

    tree = cKDTree(xyz_labeled)
    d, idx = tree.query(pts, k=1, distance_upper_bound=max_distance)
    out = np.zeros(len(pts), dtype=np.uint8)
    valid = idx < len(lbl)
    out[valid] = lbl[idx[valid]]
    return out


def refine_walls(walls_in, pts_3d, floor_z, ceiling_z, *,
                 point_labels=None,
                 edge_band=0.25, min_inliers=20, k_dominant=3,
                 tol_dominant=7.0, tol_diagonal=4.0, tol_hex=3.0,
                 merge_angle_tol=3.0, gap_thresh=0.3, verbose=False):
    """Run the full 6-stage wall refinement pipeline.

    Returns (refined_walls, meta_per_wall) where refined_walls is a list
    of (p1, p2, angle_deg, length) tuples compatible with `extract_walls`,
    and meta_per_wall is a list of dicts with 'snapped_to', 'residual',
    'confidence' for each output wall.

    If the refinement fails (e.g., not enough wall-band points), returns
    (walls_in, [None, ...]) as a safe fallback.

    Optional `point_labels` is a (N,) uint8 array of bucket ids aligned
    with `pts_3d` (0=other, 1=wall, 2=window, 3=door, 4=glass). When
    provided, the wall-band points are pre-filtered to wall-like classes
    (1..4) before Stage 1 RANSAC — Agent-4's "least invasive" Tier 2
    recommendation. Results in cleaner inlier sets in cluttered corners
    without touching the Stage 1-7 algorithms themselves.
    """
    if floor_z is None or ceiling_z is None:
        return walls_in, [None] * len(walls_in)

    # Normalize floor/ceiling ordering (after our sign-flip convention,
    # they can be in either order; pick the pair that spans the room).
    z_lo = min(float(floor_z), float(ceiling_z))
    z_hi = max(float(floor_z), float(ceiling_z))

    # Tier 2 — class-filter wall-band when vision labels are available.
    if point_labels is not None and len(point_labels) == len(pts_3d):
        wall_band_xy, wall_band_mask = _collect_wall_band(
            pts_3d, z_lo, z_hi, return_mask=True)
        wall_band_labels = np.asarray(point_labels)[wall_band_mask]
        keep = wall_band_labels >= 1  # drop 'other' (clutter/furniture)
        n_before = len(wall_band_xy)
        wall_band_xy = wall_band_xy[keep]
        if verbose:
            n_after = len(wall_band_xy)
            print(f"[D_refined][vision] wall-band filter: "
                  f"{n_before} → {n_after} pts "
                  f"({100.0 * n_after / max(n_before, 1):.0f}% kept)")
    else:
        wall_band_xy = _collect_wall_band(pts_3d, z_lo, z_hi)

    if verbose:
        print(f"[D_refined] wall-band: {len(wall_band_xy)} points "
              f"from Z in [{z_lo + 0.3:.2f}, {z_hi - 0.3:.2f}]")

    if len(wall_band_xy) < 100:
        if verbose:
            print("[D_refined] too few wall-band points — falling back to A")
        return walls_in, [None] * len(walls_in)

    # Stage 1
    refined = _stage1_per_edge_ransac(
        walls_in, wall_band_xy, edge_band=edge_band,
        min_inliers=min_inliers, verbose=verbose)

    # Stage 2
    dominant = _stage2_dominant_directions(
        refined, k_max=k_dominant, verbose=verbose)

    # Stage 3
    snapped = _stage3_tiered_snap(
        refined, dominant, tol_dominant=tol_dominant,
        tol_diagonal=tol_diagonal, tol_hex=tol_hex, verbose=verbose)

    # Stage 4
    merged = _stage4_merge_collinear(
        snapped, angle_tol_deg=merge_angle_tol, verbose=verbose)

    # Stage 5
    translated = _stage5_vertex_translation(merged, verbose=verbose)

    # Stage 7: length-constrain corners to data-supported extent
    # (per-wall: pull corners inward if they over-extend beyond the RANSAC
    # inlier extent; push outward if the wall has data reaching beyond).
    # Uses the `ransac_direction` / `ransac_point_on_line` stashed by
    # Stage 5. Safe by construction: reverts on Shapely invalidation.
    constrained, stage7_stats = _stage7_length_constrain(
        translated, slack=0.30, verbose=verbose)

    # Stage 6 (gap-split) is intentionally NOT applied to the polygon
    # topology — it would produce disconnected segments that break Shapely
    # polygon construction. Gap-split is useful as diagnostic info (where
    # are the doors?) but not for the returned outline. Keeping the code
    # for potential future use on a separate per-wall list.
    if verbose:
        _ = _stage6_gap_split(constrained, gap_thresh=gap_thresh,
                              verbose=verbose)

    # Convert to (p1, p2, angle, length) tuples + meta
    out_walls = []
    out_meta = []
    for e in constrained:
        if e['length'] < 0.10:
            continue
        # Compute data-extent length for audit (Stage-1 inlier span
        # projected onto the pre-Stage-5 RANSAC direction).
        length_data = 0.0
        try:
            inliers = e.get('inliers_xy', np.zeros((0, 2)))
            if len(inliers) >= 10:
                rdir = np.asarray(e.get('ransac_direction', e['direction']))
                rpol = np.asarray(e.get('ransac_point_on_line',
                                         e['point_on_line']))
                t = (inliers - rpol) @ rdir
                length_data = float(t.max() - t.min())
        except Exception:
            length_data = 0.0

        out_walls.append((np.asarray(e['p1']), np.asarray(e['p2']),
                          float(e['angle_deg']), float(e['length'])))
        out_meta.append({
            'snapped_to': e.get('snapped_to', 'free'),
            'residual_m': round(float(e.get('residual', 0.0)), 4),
            'confidence': round(float(e.get('confidence', 0.0)), 3),
            'length_m_data_extent': round(length_data, 3),
        })
    return out_walls, out_meta
