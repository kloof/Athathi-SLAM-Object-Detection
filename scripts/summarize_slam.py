#!/usr/bin/env python3
"""Read every metrics.json under a directory and emit COMPARISON.md.

Ranks backends on each measurable axis and produces one at-a-glance
markdown table so the winner is obvious per metric. Lower-is-better on
every quality metric here (RMSE, spread, jerk) — runtime is also lower-
is-better but on a different scale, reported separately.
"""

import argparse
import json
from pathlib import Path


METRICS_TO_RANK = [
    ("wall_rmse_m", "Wall RMSE (m) ↓", "lower_better"),
    ("floor_rmse_m", "Floor RMSE (m) ↓", "lower_better"),
    ("color_uncolored_spread_m", "Color/uncolored spread (m) ↓", "lower_better"),
    ("trajectory_jerk_mean", "Trajectory jerk mean ↓", "lower_better"),
    ("trajectory_jerk_p95", "Trajectory jerk p95 ↓", "lower_better"),
    ("backend_runtime_s", "Runtime (s) ↓", "lower_better"),
]

SECONDARY_FIELDS = [
    ("num_frames", "Frames"),
    ("post_final_points", "Final points"),
    ("colorized_fraction", "Colorized fraction"),
    ("wall_count", "Walls fit"),
    ("floor_inliers", "Floor inliers"),
    ("mixed_voxel_count", "Mixed color/gray voxels"),
    ("trajectory_length_m", "Trajectory length (m)"),
    ("bounding_box_m", "Bounding box (m)"),
]


def _fmt(v, digits: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, (int, bool)):
        return str(v)
    if isinstance(v, float):
        if digits <= 2:
            return f"{v:.{digits}f}"
        return f"{v:.{digits}f}"
    if isinstance(v, list):
        return str([round(x, 3) if isinstance(x, float) else x for x in v])
    return str(v)


def _rank(values: list, direction: str) -> list:
    """Return 1-indexed rank per value (ties share rank; None → last)."""
    n = len(values)
    paired = [(i, v) for i, v in enumerate(values)]
    nones = [p for p in paired if p[1] is None]
    have = [p for p in paired if p[1] is not None]
    have.sort(key=lambda p: p[1], reverse=(direction == "higher_better"))
    rank = [None] * n
    r = 1
    for i, (idx, _) in enumerate(have):
        if i > 0 and have[i][1] == have[i - 1][1]:
            rank[idx] = rank[have[i - 1][0]]
        else:
            rank[idx] = r
        r += 1
    for idx, _ in nones:
        rank[idx] = "—"
    return rank


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("compare_root", help="Directory containing per-backend subdirs")
    ap.add_argument("--out", default=None, help="Output markdown path "
                    "(default: <compare_root>/COMPARISON.md)")
    args = ap.parse_args()

    root = Path(args.compare_root).resolve()
    out_path = Path(args.out) if args.out else root / "COMPARISON.md"

    entries = []
    for metrics_path in sorted(root.rglob("metrics.json")):
        try:
            m = json.loads(metrics_path.read_text())
        except Exception as exc:
            print(f"[warn] could not parse {metrics_path}: {exc}")
            continue
        entries.append((metrics_path.parent.name, m))

    if not entries:
        print(f"No metrics.json under {root}")
        return 1

    lines: list[str] = []
    lines.append(f"# SLAM backend comparison\n")
    lines.append(
        f"Auto-generated from `{root}`.\n\n"
        f"Lower is better on every metric here. "
        f"Rank column shows best=1; ties share ranks; — means the metric "
        f"couldn't be computed.\n")
    backend_names = [name for name, _ in entries]

    # Primary ranked metrics table
    lines.append("## Ranked quality metrics\n")
    header = ["Metric"] + backend_names + ["Winner"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for key, label, direction in METRICS_TO_RANK:
        vals = [m.get(key) for _, m in entries]
        ranks = _rank(vals, direction)
        value_cells = []
        for v, r in zip(vals, ranks):
            if v is None:
                value_cells.append("—")
            elif isinstance(v, float):
                value_cells.append(f"{v:.5f} (#{r})" if direction != "lower_better"
                                   else f"{v:.5f} (#{r})")
            else:
                value_cells.append(f"{v} (#{r})")
        winner_rank_1 = [backend_names[i]
                         for i, r in enumerate(ranks) if r == 1]
        winner_cell = winner_rank_1[0] if winner_rank_1 else "—"
        lines.append(f"| {label} | " + " | ".join(value_cells)
                     + f" | {winner_cell} |")

    # Secondary context table
    lines.append("\n## Context / scale\n")
    header2 = ["Field"] + backend_names
    lines.append("| " + " | ".join(header2) + " |")
    lines.append("|" + "|".join(["---"] * len(header2)) + "|")
    for key, label in SECONDARY_FIELDS:
        cells = [label]
        for _, m in entries:
            cells.append(_fmt(m.get(key)))
        lines.append("| " + " | ".join(cells) + " |")

    # Points of interest: per-backend output paths
    lines.append("\n## Per-backend output\n")
    for name, _ in entries:
        sub = root / name
        lines.append(f"### {name}\n")
        lines.append(f"- `{sub}/colored_map.ply`\n")
        lines.append(f"- `{sub}/trajectory.csv`\n")
        lines.append(f"- `{sub}/floor_plus_1m_slice.png`\n")
        lines.append(f"- `{sub}/metrics.json`\n")

    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
