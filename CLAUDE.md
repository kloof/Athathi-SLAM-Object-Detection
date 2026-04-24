# Cloud SLAM ICP — Project Context

## What this is
End-to-end pipeline: indoor rosbag → KISS-ICP LiDAR SLAM with camera-projected
color → SpatialLM 1.1 scene parse → 3D point cloud with wireframe bounding
boxes + 2D floorplan.

**Hardware**: Unitree L2 lidar (360°, ~2000 pts/scan) + Logitech Brio camera
(1280x720, 30 fps) + IMU. Tested on RTX 4070 Ti (12 GB) under WSL2.

## Production entry point

```bash
python3 scripts/rosbag_to_bboxes.py \
  /path/to/rosbag /path/to/output
# uses vendored calibration/ by default; override with --calibration
```

Tuned defaults (empirically validated): beam=4 single-shot, voxel 2.5 cm,
temp 0.3, top_k 3. If `--beam-size 1` is passed, falls back to 3 Llama
passes with rep_penalty 1.20 and ≥2 consensus votes.

Optional 2D floorplan PNG (separate step):

```bash
python3 scripts/draw_floorplan.py <output>/layout_merged.txt
```

## Pipeline stages (emitted artifacts)

```
0. SLAM           KISS-ICP + colorize + gravity-level
                    -> slam/colored_map.ply, trajectory.csv, metrics.json
1. Crop           RANSAC floorplan + polygon crop
                    -> colored_map_cropped.ply, floorplan/*.{png,json}
2. Manhattan      yaw-align walls to X/Y
                    -> colored_map_manhattan.ply
3. Voxel          colored-priority 2.5 cm downsample
                    -> voxel.ply
4. Interpolate    KNN color propagation for gray points
                    -> spatiallm_input.ply
5. Infer          Qwen-0.5B (structure) + Llama-1B (objects)
                    -> layout_qwen.txt, layout_llama.txt
6. Merge          walls/doors/windows (Qwen) + dedup'd bboxes (Llama)
                    -> layout_merged.txt
7. Embed          wireframe bboxes as edge points in the cloud
                    -> scene_with_boxes.ply
```

## Repository layout

### Live modules (`cloud_slam/`)
- `spatiallm_pipeline/` — package implementing stages 1–7
  - `crop.py`, `manhattan.py`, `voxel.py`, `interpolate.py`,
    `infer.py`, `merge.py`, `embed.py`
- `slam_backends/` — stage 0
  - `kiss_icp_backend.py` (production), `baseline.py` (Open3D ICP+IMU fallback)
  - `post_process.py`, `metrics.py`, `base.py`
- `pipelines/icp_imu_pipeline.py` — ICP+IMU engine used by `baseline` backend
- `floorplan.py` — RANSAC wall/floor/ceiling detection + refinement
- `room_structure.py`, `frustum.py`, `projection.py` — floorplan helpers
- `colorizer.py`, `deskew.py`, `level.py` — LiDAR post-processing
- `mcap_reader.py` — rosbag I/O

### Scripts (`scripts/`)
- `rosbag_to_bboxes.py` — production end-to-end entry
- `compare_slam.py` — SLAM-only runner (any backend), used as a subprocess
- `draw_floorplan.py` — render `layout_merged.txt` as a 2D PNG

### Tests (`tests/`)
- `test_deskew.py`, `test_colorize_per_point.py` (11 tests, all pass)

### Third-party (not in git)
- `third_party/SpatialLM/` — cloned locally via `third_party/setup_spatiallm.sh`
- `third_party/setup_spatiallm.sh` — sets up `~/spatiallm_env` venv

## Calibration

Vendored under `calibration/` (intrinsics.yaml + extrinsics.yaml). This is
the authoritative source — do **not** pass scan-bundled calibration paths
(scans carry stale auto-copied YAMLs).

## Coordinate frames
- L2 IMU reports proper acceleration with gravity pointing UP (body frame).
- Raw SLAM world is Y-up (L2 mount tilt). Stage 0 `post_process` applies
  gravity-leveling so the emitted `colored_map.ply` is Z-up.
- `frame_poses.json` (when emitted) is already camera pose —
  `T_cam_in_lidar` is composed in.

## Test data

Canonical rosbag + expected outputs live at:
`/mnt/c/Users/klof/Desktop/SLAM_test/charuco_calib/TEST_SCAN/`
(1 sectional sofa, 5–6 tables, 8 dining chairs, several mirrors + doors).

Output inspection convention: copy artifacts to the Windows-side scan dir
so the user can open them in CloudCompare.
