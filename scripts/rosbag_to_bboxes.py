#!/usr/bin/env python3
"""End-to-end: rosbag -> KISS-ICP SLAM -> colored PLY with SpatialLM boxes.

Pipeline stages (each produces an artifact in <output>/):
  0. SLAM          : KISS-ICP on LiDAR, colorize with Brio, level by gravity
                     -> colored_map.ply, trajectory.csv, metrics.json
  1. Crop          : RANSAC floorplan, polygon crop with wall-tolerance buffer
                     -> colored_map_cropped.ply + floorplan/*.{png,json}
  2. Manhattan     : rotate so detected walls are parallel to X/Y
                     -> colored_map_cropped_manhattan.ply
  3. Voxel         : colored-priority 1cm voxel downsample
                     -> voxel.ply
  4. Interpolate   : K-NN color propagation for default-gray points
                     -> spatiallm_input.ply  <-- the cloud fed to SpatialLM
  5. Infer (x2)    : SpatialLM 1.1 Qwen-0.5B (structure) + Llama-1B (objects)
                     -> layout_qwen.txt, layout_llama.txt
  6. Merge         : Qwen walls/doors/windows + dedup'd Llama bboxes
                     -> layout_merged.txt
  7. Embed         : wireframe bboxes as edge points in the cloud
                     -> scene_with_boxes.ply  <-- final deliverable

Example:
  python3 scripts/rosbag_to_bboxes.py \
    /mnt/c/Users/klof/Desktop/SLAM_test/charuco_calib/TEST_SCAN/rosbag \
    /tmp/spatiallm_run

Uses the repo-vendored calibration/ by default; override with --calibration.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent


def _stage(msg):
    print(f"\n{'='*72}\n== {msg}\n{'='*72}")


def run_slam(rosbag, out_dir, calib_dir):
    """Stage 0 — delegate to compare_slam.py --backend kiss_icp."""
    _stage(f"[0/7] SLAM (KISS-ICP)")
    slam_out = Path(out_dir) / "slam"
    slam_out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(REPO / "scripts" / "compare_slam.py"),
        str(rosbag), str(slam_out), str(calib_dir),
        "--backend", "kiss_icp",
    ]
    subprocess.check_call(cmd, env={**os.environ, "PYTHONPATH": str(REPO)})
    return slam_out / "colored_map.ply"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Rosbag -> SpatialLM-labeled colored PLY")
    parser.add_argument("rosbag", type=Path,
                         help="MCAP rosbag directory or file")
    parser.add_argument("output", type=Path,
                         help="Output directory")
    default_calib = str(REPO / "calibration")
    parser.add_argument("--calibration", type=Path, default=default_calib,
                         help=f"Calibration dir (default vendored: {default_calib})")
    parser.add_argument("--voxel-m", type=float, default=0.025,
                         help="Voxel size for colored-priority downsample "
                              "(default 2.5 cm — matches SpatialLM's internal "
                              "grid_size so we don't double-downsample and "
                              "lose small objects to bin collisions)")
    parser.add_argument("--interp-k", type=int, default=8)
    parser.add_argument("--interp-radius-m", type=float, default=0.30)
    parser.add_argument("--temperature", type=float, default=0.3,
                         help="SpatialLM sampling temperature (default 0.3, tuned)")
    parser.add_argument("--top-k", type=int, default=3,
                         help="SpatialLM sampling top_k (default 3, tuned)")
    parser.add_argument("--skip-slam", action="store_true",
                         help="If <output>/slam/colored_map.ply already exists, reuse it")
    parser.add_argument("--skip-llama", action="store_true",
                         help="Use only Qwen (faster, slightly worse object recall)")
    parser.add_argument("--llama-passes", type=int, default=1,
                         help="N sequential Llama passes (sharing one model "
                              "load). When N>1, bboxes are taken as the "
                              "consensus across passes. Default 1.")
    parser.add_argument("--llama-min-votes", type=int, default=2,
                         help="Min consensus votes to keep a bbox when "
                              "--llama-passes>1. Default 2.")
    parser.add_argument("--llama-rep-penalty", type=float, default=1.15,
                         help="Llama repetition_penalty (default 1.15, "
                              "suppresses window-duplication loop). 1.0 disables.")
    parser.add_argument("--llama-seed-base", type=int, default=0,
                         help="First seed for Llama passes. Seeds used: "
                              "[base, base+1, ..., base+passes-1]. Default 0.")
    args = parser.parse_args(argv)

    # Defer heavy imports until the CLI has parsed args
    sys.path.insert(0, str(REPO))
    from cloud_slam.spatiallm_pipeline.crop import crop_to_floorplan
    from cloud_slam.spatiallm_pipeline.manhattan import yaw_align_by_walls
    from cloud_slam.spatiallm_pipeline.voxel import colored_priority_voxel
    from cloud_slam.spatiallm_pipeline.interpolate import interpolate_gray_colors
    from cloud_slam.spatiallm_pipeline.infer import (
        run_spatiallm, run_spatiallm_multi_seed, MODEL_QWEN, MODEL_LLAMA,
    )
    from cloud_slam.spatiallm_pipeline.merge import (
        merge_layouts, merge_layouts_consensus,
    )
    from cloud_slam.spatiallm_pipeline.embed import embed_bboxes_in_ply

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    # 0. SLAM
    slam_ply = output / "slam" / "colored_map.ply"
    if args.skip_slam and slam_ply.is_file():
        _stage(f"[0/7] SLAM - reusing {slam_ply}")
    else:
        slam_ply = run_slam(args.rosbag, output, args.calibration)

    # 1. crop
    _stage("[1/7] Crop to floorplan")
    cropped_ply = crop_to_floorplan(slam_ply, output)

    # 2. Manhattan
    _stage("[2/7] Manhattan yaw-align")
    floorplan_meta = output / "floorplan" / "room_metadata.json"
    manhattan_ply, _yaw = yaw_align_by_walls(
        cropped_ply, floorplan_meta, output / "colored_map_manhattan.ply"
    )

    # 3. voxel
    _stage("[3/7] Colored-priority voxel")
    voxel_ply = colored_priority_voxel(
        manhattan_ply, output / "voxel.ply", voxel_m=args.voxel_m
    )

    # 4. interpolate
    _stage("[4/7] K-NN gray->color interpolation")
    sl_input_ply = interpolate_gray_colors(
        voxel_ply, output / "spatiallm_input.ply",
        k=args.interp_k, radius_m=args.interp_radius_m,
    )

    # 5. infer
    _stage(f"[5/7] SpatialLM inference (Qwen; temp={args.temperature} topk={args.top_k})")
    layout_qwen = run_spatiallm(
        sl_input_ply, output / "layout_qwen.txt",
        model=MODEL_QWEN, temperature=args.temperature, top_k=args.top_k,
        repetition_penalty=args.llama_rep_penalty,
    )

    llama_layouts: list = []
    if args.skip_llama:
        llama_layouts = [layout_qwen]
    elif args.llama_passes <= 1:
        _stage(f"[5/7] SpatialLM inference (Llama; same sampling)")
        llama_layouts = [run_spatiallm(
            sl_input_ply, output / "layout_llama.txt",
            model=MODEL_LLAMA, temperature=args.temperature, top_k=args.top_k,
            repetition_penalty=args.llama_rep_penalty,
            seed=args.llama_seed_base,
        )]
    else:
        _stage(f"[5/7] SpatialLM inference (Llama x{args.llama_passes} "
               f"passes, shared load; rep_pen={args.llama_rep_penalty})")
        seeds = list(range(args.llama_seed_base,
                           args.llama_seed_base + args.llama_passes))
        llama_layouts = run_spatiallm_multi_seed(
            sl_input_ply, output, seeds,
            output_stem="layout_llama",
            model=MODEL_LLAMA, temperature=args.temperature, top_k=args.top_k,
            repetition_penalty=args.llama_rep_penalty,
        )

    # 6. merge
    _stage("[6/7] Merge layouts")
    if len(llama_layouts) > 1:
        layout_merged = merge_layouts_consensus(
            layout_qwen, llama_layouts, output / "layout_merged.txt",
            min_votes=args.llama_min_votes,
        )
    else:
        layout_merged = merge_layouts(
            layout_qwen, llama_layouts[0], output / "layout_merged.txt"
        )

    # 7. embed
    _stage("[7/7] Embed wireframe boxes in PLY")
    final_ply = embed_bboxes_in_ply(
        sl_input_ply, layout_merged, output / "scene_with_boxes.ply"
    )

    print(f"\n{'='*72}")
    print(f"DONE. Final deliverables in {output}:")
    for p in ["slam/colored_map.ply", "spatiallm_input.ply",
              "layout_qwen.txt", "layout_llama.txt", "layout_merged.txt",
              "scene_with_boxes.ply"]:
        pp = output / p
        if pp.exists():
            sz = pp.stat().st_size / (1024 * 1024)
            print(f"  {p:35s}  {sz:8.1f} MB")
    print(f"\nFinal viewable PLY: {output}/scene_with_boxes.ply")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
