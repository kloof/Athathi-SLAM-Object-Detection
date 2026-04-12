#!/usr/bin/env python3
"""
Compare two objects.json outputs from detect_and_slam.

Usage:
    python3 scripts/compare_objects.py reference.json candidate.json
"""

import json
import sys
import numpy as np


def load(path):
    with open(path) as f:
        return json.load(f)


def compare(ref, cand, label_a="reference", label_b="candidate"):
    print(f"\n=== Comparing {label_a} vs {label_b} ===")
    print(f"Objects: {len(ref['objects'])} vs {len(cand['objects'])}")

    ref_by_tid = {o['track_id']: o for o in ref['objects']}
    cand_by_tid = {o['track_id']: o for o in cand['objects']}

    ref_tids = set(ref_by_tid)
    cand_tids = set(cand_by_tid)
    common = sorted(ref_tids & cand_tids)
    only_ref = sorted(ref_tids - cand_tids)
    only_cand = sorted(cand_tids - ref_tids)

    if only_ref:
        print(f"  missing in {label_b}: track_ids {only_ref}")
    if only_cand:
        print(f"  extra in {label_b}:   track_ids {only_cand}")

    # Stats comparison
    ref_s = ref.get('stats', {})
    cand_s = cand.get('stats', {})
    stat_keys = sorted(set(ref_s) | set(cand_s))
    print("\nStats:")
    for k in stat_keys:
        rv = ref_s.get(k, '—')
        cv = cand_s.get(k, '—')
        flag = '' if rv == cv else ' *'
        print(f"  {k:30s}: {rv}  |  {cv}{flag}")

    # Per-object comparison
    print(f"\nPer-object diffs (track_ids in common, n={len(common)}):")
    max_center_delta = 0.0
    max_dim_delta = 0.0
    for tid in common:
        a = ref_by_tid[tid]
        b = cand_by_tid[tid]
        a_c = np.array(a['center'])
        b_c = np.array(b['center'])
        a_d = np.array(a['dimensions'])
        b_d = np.array(b['dimensions'])
        center_delta = float(np.linalg.norm(b_c - a_c))
        dim_delta = float(np.max(np.abs(b_d - a_d)))
        max_center_delta = max(max_center_delta, center_delta)
        max_dim_delta = max(max_dim_delta, dim_delta)
        class_match = '=' if a['class'] == b['class'] else '≠'
        print(f"  [{tid:3d}] {a['class']:12s} {class_match} {b['class']:12s} "
              f"  Δcenter={center_delta:.4f}m  Δmax_dim={dim_delta:.4f}m  "
              f"ref.center=({a_c[0]:+.2f},{a_c[1]:+.2f},{a_c[2]:+.2f})  "
              f"cand.center=({b_c[0]:+.2f},{b_c[1]:+.2f},{b_c[2]:+.2f})")

    print(f"\nMax center delta: {max_center_delta:.4f} m")
    print(f"Max dim delta:    {max_dim_delta:.4f} m")

    return max_center_delta, max_dim_delta


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: compare_objects.py reference.json candidate.json", file=sys.stderr)
        sys.exit(1)
    ref = load(sys.argv[1])
    cand = load(sys.argv[2])
    compare(ref, cand, sys.argv[1], sys.argv[2])
