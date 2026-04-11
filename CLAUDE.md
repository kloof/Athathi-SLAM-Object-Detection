# Cloud SLAM ICP — Project Context

## What this is
LiDAR SLAM + camera color projection + YOLOE-26L 3D object detection with RoomPlan-style refinement.

**Hardware**: Unitree L2 lidar (360°, ~2000 pts/frame) + Logitech Brio camera (1280x720, 10fps) + IMU.  
**Test data**: `/mnt/c/Users/klof/Desktop/SLAM_test/scan_20260411_121711/`

## Current branch: `feature/yoloe-3d-detection`

## Architecture
```
MCAP → ICP+IMU SLAM → per-frame YOLOE detect+track (BoT-SORT) → frustum extraction → 
point accumulation per object → room structure detection (RANSAC) → Manhattan alignment → 
size priors + Bayesian refinement → floor/wall snapping → output PLY + JSON
```

## CRITICAL BUG (current priority)
**Bounding boxes are massively oversized** — spanning the entire room instead of individual objects.

Root cause: accumulated point clouds per object contain 500-8000 points including walls, floor, ceiling.
The frustum extraction captures too much background, and nothing removes it before OBB fitting.

### The fix needed (in `cloud_slam/box_refiner.py`):
1. **Remove structural surface points** (floor/wall/ceiling) from each object's accumulated cloud BEFORE DBSCAN and OBB fitting. Use the already-detected room planes from `room_structure.py`.
2. **Tighten DBSCAN eps** from 0.10 to 0.05m
3. **Hard-clamp OBB dimensions** to class size prior maximums

### Key data:
- Object [79] bed: 8494 pts accumulated, dims 2.55x2.21x1.03 (should be ~2.0x1.5x0.55)
- Object [12] shelf: 5715 pts, dims 1.91x1.04x1.04 (should be ~0.8x0.4x1.5)
- Object [19] chair: 589 pts, dims 0.63x0.55x0.74 (close to correct)

## Test command
```bash
python3 scripts/detect_and_slam.py "/mnt/c/Users/klof/Desktop/SLAM_test/scan_20260411_121711/rosbag" /tmp/detect_test "/mnt/c/Users/klof/Desktop/SLAM_test/scan_20260411_121711/calibration"
```
Then copy outputs: `cp /tmp/detect_test/*.ply /tmp/detect_test/*.json "/mnt/c/Users/klof/Desktop/SLAM_test/scan_20260411_121711/"`

## Key files
- `cloud_slam/box_refiner.py` — **FIX HERE** — orchestrates refinement pipeline
- `cloud_slam/room_structure.py` — RANSAC floor/wall/ceiling detection (working)
- `cloud_slam/manhattan.py` — Manhattan frame + wall-aligned OBB (working)
- `cloud_slam/size_priors.py` — per-class dimension priors (working)
- `cloud_slam/frustum.py` — frustum extraction + gravity estimation
- `cloud_slam/tracker_3d.py` — point accumulation + Kalman tracking
- `cloud_slam/detector.py` — YOLOE-26L wrapper
- `cloud_slam/pipelines/detect_pipeline.py` — combines SLAM + detection
- `scripts/detect_and_slam.py` — CLI entry point (does gravity leveling + room axis alignment)

## Gravity note
The Unitree L2 IMU reports gravity as `[3.95, 9.35, -0.04]` — the sensor is tilted ~70° from vertical. The `estimate_gravity()` function handles this with a sign check. The output script levels the scan (gravity→Z) and aligns walls to axes.
