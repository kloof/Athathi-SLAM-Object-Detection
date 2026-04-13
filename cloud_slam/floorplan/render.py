"""Floor-plan PNG export functions.

Pipeline / floorplan / overlay / refined / comparison / corners renderers
extracted verbatim from the pre-split monolithic floorplan module.

Imported by the package orchestrator (`cloud_slam.floorplan.__init__`).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import cv2
import numpy as np


def _export_pipeline_pngs(walls, room_poly, pts, binary, clean, g8, xe, ye,
                          real_coords, snapped_coords, output_dir, name,
                          floor_z, ceiling_z):
    """Generate the pipeline / floorplan / overlay PNGs."""
    h = ceiling_z - floor_z
    ext = [xe[0], xe[-1], ye[0], ye[-1]]

    # 01: Pipeline steps
    fig, axes = plt.subplots(2, 3, figsize=(21, 14), dpi=150)
    axes[0, 0].imshow(g8.T, origin='lower', cmap='hot', extent=ext)
    axes[0, 0].set_title('Ceiling Density')
    axes[0, 1].imshow(binary.T, origin='lower', cmap='gray', extent=ext)
    axes[0, 1].set_title('After Morphology')
    axes[0, 2].imshow(clean.T, origin='lower', cmap='gray', extent=ext)
    axes[0, 2].set_title('Outliers Removed')

    rc = np.vstack([real_coords, real_coords[0:1]])
    axes[1, 0].imshow(clean.T, origin='lower', cmap='gray', extent=ext)
    axes[1, 0].plot(rc[:, 0], rc[:, 1], 'r-', linewidth=2)
    axes[1, 0].plot(real_coords[:, 0], real_coords[:, 1], 'ro', markersize=5)
    axes[1, 0].set_title(f'Simplified ({len(real_coords)} vertices)')

    sc = np.vstack([snapped_coords, snapped_coords[0:1]])
    axes[1, 1].set_facecolor('white')
    axes[1, 1].plot(sc[:, 0], sc[:, 1], 'b-', linewidth=2)
    axes[1, 1].plot(snapped_coords[:, 0], snapped_coords[:, 1], 'bo', markersize=5)
    axes[1, 1].plot(rc[:, 0], rc[:, 1], 'r--', linewidth=1, alpha=0.5)
    axes[1, 1].set_title(f'Angle Snapped ({len(snapped_coords)} corners)')

    axes[1, 2].set_facecolor('white')
    if hasattr(room_poly, 'exterior'):
        rx, ry = room_poly.exterior.xy
        axes[1, 2].fill(rx, ry, color='#E8F5E9', alpha=0.5)
        axes[1, 2].plot(rx, ry, 'k-', linewidth=2.5)
    axes[1, 2].set_title(f'Final: {room_poly.area:.1f} m2, {len(walls)} walls')

    for ax in axes.flat:
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
    fig.suptitle('Ceiling Trace Pipeline', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{name}_pipeline.png', bbox_inches='tight')
    plt.close()

    # 02: Floor plan
    fig, ax = plt.subplots(figsize=(14, 14), dpi=200)
    ax.set_facecolor('#FAFAFA')
    if hasattr(room_poly, 'exterior'):
        rx, ry = room_poly.exterior.xy
        ax.fill(rx, ry, color='#E8F5E9', alpha=0.5)
    for p1, p2, a, l in walls:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=3.5,
                solid_capstyle='round')
        if l > 0.3:
            mid = (p1 + p2) / 2
            d = p2 - p1
            nm = np.linalg.norm(d)
            perp = np.array([-d[1], d[0]]) / nm * 0.18
            ax.text(mid[0] + perp[0], mid[1] + perp[1], f'{l:.2f}m',
                    ha='center', fontsize=7, color='#444',
                    rotation=a if a <= 90 else a - 180)
    ax.text(room_poly.centroid.x, room_poly.centroid.y,
            f"Area: {room_poly.area:.1f} m2\nCeiling: {ceiling_z:.2f}m\n"
            f"Height: {h:.2f}m",
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.4'))
    ax.grid(True, alpha=0.06, color='#4488CC')
    ax.set_axisbelow(True)
    ap = np.array([w[0] for w in walls] + [w[1] for w in walls])
    sx, sy = ap[:, 0].min() - 0.3, ap[:, 1].min() - 0.8
    ax.plot([sx, sx + 1], [sy, sy], 'k-', linewidth=3)
    ax.plot([sx, sx], [sy - 0.08, sy + 0.08], 'k-', linewidth=2)
    ax.plot([sx + 1, sx + 1], [sy - 0.08, sy + 0.08], 'k-', linewidth=2)
    ax.text(sx + 0.5, sy + 0.15, '1 m', ha='center', fontsize=9, fontweight='bold')
    ax.set_title(f'Floor Plan - {len(walls)} walls', fontsize=14, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{name}_floorplan.png', bbox_inches='tight')
    plt.close()

    # 03: Overlay on raw point cloud
    res_v = 0.02
    xmn, ymn = pts[:, :2].min(0) - 0.5
    xmx, ymx = pts[:, :2].max(0) + 0.5
    nxv, nyv = int((xmx - xmn) / res_v), int((ymx - ymn) / res_v)
    gv, xev, yev = np.histogram2d(pts[:, 0], pts[:, 1], bins=[nxv, nyv],
                                   range=[[xmn, xmx], [ymn, ymx]])
    p93 = np.percentile(gv[gv > 0], 93) if np.any(gv > 0) else 1
    g8v = (np.clip(gv, 0, p93) / p93 * 255).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(14, 14), dpi=200)
    ax.imshow(g8v.T, origin='lower', cmap='gray_r',
              extent=[xev[0], xev[-1], yev[0], yev[-1]], alpha=0.3)
    if hasattr(room_poly, 'exterior'):
        rx, ry = room_poly.exterior.xy
        ax.fill(rx, ry, color='#4CAF50', alpha=0.2)
    for p1, p2, a, l in walls:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=2.5)
    ax.set_title('Floor Plan on Raw Point Cloud', fontsize=14, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{name}_overlay.png', bbox_inches='tight')
    plt.close()


def _export_refined_png(walls_d, poly_d, walls_d_meta, pts,
                         output_dir, name, floor_z, ceiling_z,
                         openings=None):
    """Dedicated refined-polygon PNG with per-wall snap-kind coloring.

    M3: overlays structured openings on top of their hosting wall:
        door    → wall line with ~1 m gap + two perpendicular ticks
                  pointing inward (⊥⊥)
        window  → wall line with gap + single perpendicular tick (⊤)
        glass   → wall line with gap + dashed perpendicular tick (to
                  distinguish from window)
        passage → wall line with plain gap (no overlay)
    """
    h = ceiling_z - floor_z

    res_v = 0.02
    xmn, ymn = pts[:, :2].min(0) - 0.5
    xmx, ymx = pts[:, :2].max(0) + 0.5
    nxv, nyv = int((xmx - xmn) / res_v), int((ymx - ymn) / res_v)
    gv, xev, yev = np.histogram2d(pts[:, 0], pts[:, 1], bins=[nxv, nyv],
                                   range=[[xmn, xmx], [ymn, ymx]])
    p93 = np.percentile(gv[gv > 0], 93) if np.any(gv > 0) else 1
    g8v = (np.clip(gv, 0, p93) / p93 * 255).astype(np.uint8)
    extv = [xev[0], xev[-1], yev[0], yev[-1]]

    # Per-snap-kind color (used when no vision type is available OR
    # the wall's type is 'wall'/'unknown').
    kind_color = {
        'dominant': '#2E7D32',    # green: learned-dominant snap
        'manhattan': '#1565C0',   # blue: 0°/90° snap
        'diagonal': '#FF6F00',    # orange: 45°/135° snap
        'hex': '#8E24AA',         # purple: 30°/60° snap
        'free': '#D84315',        # red: preserved at fitted angle
        'fallback_a': '#616161',  # grey: fell back to variant A
    }
    # Per-type color — overrides kind_color for window/door/glass. A
    # wall typed as 'wall' or 'unknown' falls through to kind_color so
    # the snap provenance stays visible.
    type_color = {
        'window': '#00BFFF',   # cyan
        'door':   '#FF7F00',   # orange (distinct from diagonal orange)
        'glass':  '#ADD8E6',   # light cyan
    }

    fig, ax = plt.subplots(figsize=(14, 14), dpi=200)
    ax.imshow(g8v.T, origin='lower', cmap='gray_r', extent=extv, alpha=0.25)
    ax.set_facecolor('#FAFAFA')
    if hasattr(poly_d, 'exterior'):
        rx, ry = poly_d.exterior.xy
        ax.fill(rx, ry, color='#E8F5E9', alpha=0.4)

    legend_kinds = set()      # snap-kind legend entries (green/blue/...)
    legend_types = set()      # type legend entries (cyan/orange/...)
    type_handles = []         # keep matplotlib Line2D refs for the 2nd legend
    from matplotlib.lines import Line2D
    # Track which feature-class markers we drew so the features legend can
    # aggregate them (dashed-line markers drawn on walls that carry a
    # feature but are not themselves typed as that class).
    feature_markers_seen = set()
    for idx, (p1, p2, a, l) in enumerate(walls_d):
        kind = 'free'
        wtype = None
        wfeatures = []
        if idx < len(walls_d_meta) and walls_d_meta[idx] is not None:
            kind = walls_d_meta[idx].get('snapped_to', 'free')
            wtype = walls_d_meta[idx].get('type')
            wfeatures = walls_d_meta[idx].get('features', []) or []

        # Primary line color: type color wins for window/door/glass;
        # otherwise fall through to the snap-kind color so the kind is
        # still visible.
        if wtype in type_color:
            color = type_color[wtype]
            # Track for the types legend (not the kinds legend).
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '-',
                    color=color, linewidth=4.0, solid_capstyle='round')
            if wtype not in legend_types:
                type_handles.append(
                    Line2D([0], [0], color=color, linewidth=4.0,
                           label=wtype))
                legend_types.add(wtype)
        else:
            color = kind_color.get(kind, '#000000')
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '-',
                    color=color, linewidth=3.5, solid_capstyle='round',
                    label=kind if kind not in legend_kinds else None)
            legend_kinds.add(kind)

        # Secondary-feature overlay: for each feature present on this
        # wall, draw a short dashed overlay in the feature's color
        # centered on the wall midpoint (25 % of wall length, capped at
        # 1.0 m). Also list features in the length-label text.
        if wfeatures and l > 0.3:
            d = p2 - p1
            nm = np.linalg.norm(d)
            udir = d / nm if nm > 1e-6 else np.array([1.0, 0.0])
            mid = (p1 + p2) / 2
            dash_len = min(1.0, 0.25 * l)
            # Stack features: slight perpendicular offset per feature so
            # multiple features on one wall don't overlap.
            perp = np.array([-udir[1], udir[0]])
            for j, feat in enumerate(wfeatures):
                if feat not in type_color:
                    continue
                offset = (j - (len(wfeatures) - 1) / 2.0) * 0.08
                a_pt = mid - udir * (dash_len / 2) + perp * offset
                b_pt = mid + udir * (dash_len / 2) + perp * offset
                ax.plot([a_pt[0], b_pt[0]], [a_pt[1], b_pt[1]],
                        linestyle='--', color=type_color[feat],
                        linewidth=2.8, solid_capstyle='round')
                feature_markers_seen.add(feat)

        if l > 0.3:
            mid = (p1 + p2) / 2
            d = p2 - p1
            nm = np.linalg.norm(d)
            perp = np.array([-d[1], d[0]]) / nm * 0.18
            feature_str = ''
            if wfeatures:
                feature_str = ' [' + ','.join(wfeatures) + ']'
            ax.text(mid[0] + perp[0], mid[1] + perp[1],
                    f'{l:.2f}m{feature_str}',
                    ha='center', fontsize=7, color='#333',
                    rotation=a if a <= 90 else a - 180)

    # --- M3: Opening overlays (doors, windows, glass, passages) ---
    opening_legend_handles = []
    if openings:
        opening_colors = {
            'door':    '#FF4500',  # red-orange, RoomPlan-style
            'window':  '#1E88E5',  # blue
            'glass':   '#80DEEA',  # cyan (matches type_color 'glass')
            'passage': '#000000',  # black gap
        }
        seen_types = set()
        wall_lookup = {i: w for i, w in enumerate(walls_d)}
        for op in openings:
            wid = int(op.get('wall_id', -1))
            wall = wall_lookup.get(wid)
            if wall is None:
                continue
            p1, p2, _a, _l = wall
            p1 = np.asarray(p1, dtype=float)
            p2 = np.asarray(p2, dtype=float)
            edge = p2 - p1
            L = float(np.linalg.norm(edge))
            if L < 1e-6:
                continue
            udir = edge / L
            perp = np.array([-udir[1], udir[0]])
            a_start = float(op['along_start'])
            a_end = float(op['along_end'])
            otype = op['type']
            color = opening_colors.get(otype, '#FF00FF')
            # Cut a gap in the wall: draw a white overlay stub between
            # the endpoints at the opening width.
            gap_a = p1 + a_start * udir
            gap_b = p1 + a_end * udir
            ax.plot([gap_a[0], gap_b[0]], [gap_a[1], gap_b[1]],
                    '-', color='white', linewidth=5.0,
                    solid_capstyle='butt', zorder=3)
            # Overlay the opening-specific glyph.
            tick_len = 0.25  # visible on the plan (meters)
            mid = 0.5 * (gap_a + gap_b)
            if otype == 'door':
                # Two perpendicular ticks pointing inward.
                inner = perp * tick_len
                for anchor in (gap_a, gap_b):
                    tip = anchor + inner
                    ax.plot([anchor[0], tip[0]], [anchor[1], tip[1]],
                            '-', color=color, linewidth=2.0, zorder=4)
            elif otype == 'window':
                tip = mid + perp * tick_len
                ax.plot([mid[0], tip[0]], [mid[1], tip[1]],
                        '-', color=color, linewidth=2.2, zorder=4)
            elif otype == 'glass':
                tip = mid + perp * tick_len
                ax.plot([mid[0], tip[0]], [mid[1], tip[1]],
                        '--', color=color, linewidth=2.2, zorder=4)
            # passage: plain gap only (handled by the white cut above).

            # Redraw the opening span in the opening color, thinner, so
            # the viewer can still see where the opening sits on the wall.
            if otype != 'passage':
                ax.plot([gap_a[0], gap_b[0]], [gap_a[1], gap_b[1]],
                        '-', color=color, linewidth=2.0,
                        alpha=0.9, zorder=4)

            if otype not in seen_types:
                seen_types.add(otype)
                ls = '--' if otype == 'glass' else '-'
                opening_legend_handles.append(
                    Line2D([0], [0], color=color, linewidth=2.5,
                           linestyle=ls, label=otype))

    ax.text(poly_d.centroid.x, poly_d.centroid.y,
            f"RANSAC-refined\nArea: {poly_d.area:.1f} m2\n"
            f"Ceiling: {ceiling_z:.2f}m\nHeight: {h:.2f}m\n"
            f"Walls: {len(walls_d)}",
            ha='center', va='center', fontsize=11, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.4'))
    ax.grid(True, alpha=0.06, color='#4488CC')
    ax.set_axisbelow(True)
    ax.set_title(f'Refined Floor Plan — {len(walls_d)} walls',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    # Primary legend: snap kinds (angle source)
    first_legend = ax.legend(loc='upper right', fontsize=9, framealpha=0.9,
                              title='Wall angle source')
    # Secondary legend: vision types + features.
    # Primary-type walls get a solid line; feature overlays get a dashed
    # line in the same color — combined here so the user sees the full
    # palette.
    combined_type_handles = list(type_handles)
    for feat in sorted(feature_markers_seen):
        if feat in legend_types:
            continue  # already in solid-line legend
        combined_type_handles.append(
            Line2D([0], [0], color=type_color[feat], linewidth=2.8,
                   linestyle='--', label=f'{feat} (feature)'))
    if combined_type_handles:
        ax.add_artist(first_legend)
        second_legend = ax.legend(
            handles=combined_type_handles, loc='lower right',
            fontsize=9, framealpha=0.9, title='Vision type')
    else:
        second_legend = None
    # M3: third legend for opening symbols (door/window/glass/passage).
    if opening_legend_handles:
        if second_legend is not None:
            ax.add_artist(second_legend)
        else:
            ax.add_artist(first_legend)
        ax.legend(handles=opening_legend_handles, loc='center right',
                  fontsize=9, framealpha=0.9, title='Openings')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{name}_refined.png', bbox_inches='tight')
    plt.close()


def _export_comparison_png(variants, pts, output_dir, name,
                           corner_coords_real):
    """Variant side-by-side over the raw point cloud (supports 3 or 4)."""
    res_v = 0.02
    xmn, ymn = pts[:, :2].min(0) - 0.5
    xmx, ymx = pts[:, :2].max(0) + 0.5
    nxv, nyv = int((xmx - xmn) / res_v), int((ymx - ymn) / res_v)
    gv, xev, yev = np.histogram2d(pts[:, 0], pts[:, 1], bins=[nxv, nyv],
                                   range=[[xmn, xmx], [ymn, ymx]])
    p93 = np.percentile(gv[gv > 0], 93) if np.any(gv > 0) else 1
    g8v = (np.clip(gv, 0, p93) / p93 * 255).astype(np.uint8)
    extv = [xev[0], xev[-1], yev[0], yev[-1]]

    n_variants = len(variants)
    colors = ['lime', 'cyan', 'orange', 'magenta'][:n_variants]
    fig, axes = plt.subplots(1, n_variants, figsize=(8 * n_variants, 8), dpi=200)
    if n_variants == 1:
        axes = [axes]
    for idx, (key, (walls, poly, label)) in enumerate(variants.items()):
        ax = axes[idx]
        ax.imshow(g8v.T, origin='lower', cmap='gray_r', extent=extv, alpha=0.3)
        if hasattr(poly, 'exterior'):
            rx, ry = poly.exterior.xy
            ax.fill(rx, ry, color=colors[idx], alpha=0.15)
        for p1, p2, a, l in walls:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=2.5,
                    solid_capstyle='round')
            if l > 0.3:
                mid = (p1 + p2) / 2
                d = p2 - p1
                nm = np.linalg.norm(d)
                perp = np.array([-d[1], d[0]]) / nm * 0.15
                ax.text(mid[0] + perp[0], mid[1] + perp[1], f'{l:.2f}m',
                        ha='center', fontsize=6, color='yellow',
                        fontweight='bold',
                        rotation=a if a <= 90 else a - 180)
        if key == 'B_corners' and corner_coords_real is not None:
            ax.scatter(corner_coords_real[:, 0], corner_coords_real[:, 1],
                       s=80, c='red', marker='o', zorder=5,
                       edgecolors='white', linewidth=1.5)
        ax.set_title(f'{label}\n{poly.area:.1f} m2, {len(walls)} walls',
                     fontsize=11, fontweight='bold')
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
    fig.suptitle(f'{n_variants} Variants on Point Cloud', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{name}_comparison.png', bbox_inches='tight')
    plt.close()


def _export_corners_png(clean, g8, xe, ye, poly_b, corner_coords_real,
                        output_dir, name):
    """Corner-detection detail PNG."""
    ext_ceil = [xe[0], xe[-1], ye[0], ye[-1]]
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), dpi=150)
    axes[0].imshow(g8.T, origin='lower', cmap='hot', extent=ext_ceil)
    axes[0].set_title('Ceiling Density')
    edges_img = cv2.Canny(clean, 50, 150)
    axes[1].imshow(edges_img.T, origin='lower', cmap='gray', extent=ext_ceil)
    if corner_coords_real is not None:
        axes[1].scatter(corner_coords_real[:, 0], corner_coords_real[:, 1],
                        s=100, c='red', marker='o', zorder=5,
                        edgecolors='white', linewidth=2)
    n_corners = len(corner_coords_real) if corner_coords_real is not None else 0
    axes[1].set_title(f'Edges + Corners ({n_corners})')
    axes[2].set_facecolor('white')
    if hasattr(poly_b, 'exterior'):
        rx, ry = poly_b.exterior.xy
        axes[2].fill(rx, ry, color='#E8F5E9', alpha=0.5)
        axes[2].plot(rx, ry, 'k-', linewidth=2.5)
    if corner_coords_real is not None:
        axes[2].scatter(corner_coords_real[:, 0], corner_coords_real[:, 1],
                        s=80, c='red', marker='o', zorder=5)
    axes[2].set_title(f'Corner-Based Polygon\n{poly_b.area:.1f} m2')
    for ax in axes:
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
    fig.suptitle('Corner Detection Pipeline', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{name}_corners.png', bbox_inches='tight')
    plt.close()
