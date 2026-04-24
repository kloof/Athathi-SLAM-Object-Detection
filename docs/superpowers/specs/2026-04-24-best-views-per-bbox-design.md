# Stage 8 — Best-View-Per-Bbox

**Date**: 2026-04-24
**Branch**: `stage-8-best-views`
**Status**: Design approved, spec under review

## Motivation

After the SpatialLM stage emits 3D bounding boxes (`layout_merged.txt`), a
reviewer has no cheap way to see what each box actually is. The 3D point
cloud shows a wireframe cube; the JPEG of the object lives only inside the
rosbag. This stage picks the clearest camera frame for each bbox and saves
a cropped image so the reviewer can verify the semantic label at a glance.

## Goals

- For every bbox in `layout_merged.txt`, produce one tightly-cropped JPG
  of the best camera frame that shows it.
- Emit a `best_views.json` manifest with per-bbox scores, source frame
  timestamp, and pixel geometry — so scoring weights can be tuned without
  re-running stages 0–7.
- Integrate as stage 8 inside `scripts/rosbag_to_bboxes.py` but keep the
  stage re-runnable standalone against an existing `<output>` directory.

## Non-goals

- Top-K frames per bbox (pick one best view; extend later if useful).
- HTML contact sheet / thumbnail browser (can be a separate script).
- Detection or re-classification — we trust SpatialLM's class label.
- Modifying any stage 0–7 behavior beyond emitting one new index file.

## Architecture

New module `cloud_slam/spatiallm_pipeline/best_views.py`, called as
stage 8 from `scripts/rosbag_to_bboxes.py`. Stage 0 gains a tiny emitter
for `slam/frames_index.json`.

### Image threading strategy

**Approach 2: frames-index + on-demand decode.**

Stage 0 writes `slam/frames_index.json` — a list of every camera frame's
timestamp and byte offset into the mcap file:

```json
{
  "mcap_path": "/path/to/scan.mcap",
  "topic": "/camera/image_raw/compressed",
  "frames": [
    {"t": 1713976420.001, "mcap_offset": 1234567},
    ...
  ]
}
```

Stage 8 opens the mcap once, seeks to the winning frames, decodes only
those (one per bbox). No heavy image buffer is threaded through stages
1–7; RAM and disk cost are both near zero.

Rationale: matches the "each stage emits artifacts" pattern already used
by CLAUDE.md; keeps stage 8 re-runnable standalone, which is critical
while tuning scoring weights.

## Inputs

| Artifact | Source | Purpose |
|---|---|---|
| `layout_merged.txt` | stage 6 | 3D bboxes (world Z-up) |
| `trajectory.csv` | stage 0 | per-scan LiDAR pose (world Z-up, gravity-leveled) |
| `slam/frames_index.json` | **new**, stage 0 | camera frame timestamps + mcap offsets |
| `calibration/intrinsics.yaml` | vendored | K (3×3), D (plumb_bob 5-vec) |
| `calibration/extrinsics.yaml` | vendored | `T_lidar_cam` (already loaded by `colorizer.py`) |
| `voxel.ply` | stage 3 | occlusion point cloud (2.5 cm) |
| mcap path | `metrics.json` / CLI arg | re-opened to decode winning frames |

## Processing pipeline

### Step 1 — Load and interpolate poses

Read `trajectory.csv` (`timestamp, x, y, z, qw, qx, qy, qz`). For each
camera frame timestamp `t`, compute `T_world_lidar(t)` by SLERP on
rotation and LERP on translation between the two bracketing scan poses.
Frames outside the trajectory range are dropped.

Compose: `T_world_cam(t) = T_world_lidar(t) · T_lidar_cam` — this matches
the convention already used by `colorizer.py` and respects the gravity
leveling applied by stage 0 (`colored_map.ply` and `trajectory.csv` are
both Z-up; do *not* re-level here).

### Step 2 — Candidate frames per bbox

For each bbox, iterate all camera frames. A frame is a **candidate** iff:

- The bbox center projects in front of the camera (positive Z in camera
  frame).
- The projected AABB of the 8 bbox corners has non-zero overlap with
  the image rectangle `[0,0,1280,720]` after distortion.

Use `cloud_slam.projection.project_lidar_to_camera` (or an equivalent
that takes `T_world_cam` instead of `T_lidar_cam`) — check whether the
existing function can be reused directly; otherwise extract its core
into a small helper that takes an arbitrary camera pose.

### Step 3 — Per-candidate scores

All four scores are normalized to [0, 1]:

- **`area`** = (projected-AABB pixel area) / (W · H), clipped to 1.
- **`centering`** = `1 − min(1, ‖AABB_center − image_center‖ / (0.5·diag))`.
- **`occlusion`** = fraction of the 8 bbox-corner rays that are
  unobstructed. A corner's ray is **obstructed** iff the voxel.ply
  contains any point within 5 cm of the ray *and* at least 10 cm closer
  to the camera than the corner itself.
- **`sharpness`** = Laplacian-variance of the frame (computed once per
  frame, cached), then percentile-ranked across the candidate pool for
  this bbox so the score is 0 for the blurriest and 1 for the sharpest
  in this bbox's candidate set.

### Step 4 — Composite score and winner selection

```
composite = 0.35·area + 0.15·centering + 0.30·occlusion + 0.20·sharpness
```

Weights live in a module-level `dict` (`SCORE_WEIGHTS`) so tuning is a
one-line edit. Argmax over candidates → winning frame.

If the winner's composite score is below `SCORE_FLOOR` (default 0.2),
the bbox is marked `skipped` in the manifest and no crop is written;
the manifest entry still includes the best-attempted `scores` dict so
the reviewer can see why it was rejected. If the bbox has zero
candidates (never visible), it is marked `skipped: never_visible` and
no `scores` field is emitted.

### Step 5 — Crop

Project the 8 corners into the winning frame, take the 2D AABB, expand
by **10% of the AABB's own width/height on each side**, clip to image
bounds. The 10% expansion is symmetric (top/bottom and left/right each
grow by 10% of the AABB height / width respectively). Crop → save as
`best_views/<class>_<NN>.jpg`, where `NN` is the bbox_id from
`layout_merged.txt` (zero-padded to 2 digits).

### Step 6 — Emit manifest

```json
{
  "weights": {"area": 0.35, "centering": 0.15, "occlusion": 0.30, "sharpness": 0.20},
  "score_floor": 0.2,
  "entries": [
    {
      "bbox_id": 0,
      "class": "sofa",
      "bbox_3d": [cx, cy, cz, yaw, sx, sy, sz],
      "frame_timestamp": 1713976423.472,
      "image_path": "best_views/sofa_00.jpg",
      "pixel_aabb": [x0, y0, x1, y1],
      "crop_aabb": [x0, y0, x1, y1],
      "scores": {
        "area": 0.42,
        "centering": 0.81,
        "occlusion": 1.0,
        "sharpness": 0.67,
        "composite": 0.63
      },
      "camera_distance_m": 2.31
    },
    { "bbox_id": 3, "class": "mirror", "skipped": "no_candidate_above_floor" }
  ]
}
```

## Output layout

```
<output>/best_views/
  sofa_00.jpg
  chair_01.jpg
  chair_02.jpg
  ...
  best_views.json
```

## CLI surface

`scripts/rosbag_to_bboxes.py` gains an implicit stage 8 (no new flag —
runs by default). An escape hatch `--skip-best-views` is added for the
rare case of wanting to skip it.

For standalone re-runs during weight tuning:

```bash
python3 -m cloud_slam.spatiallm_pipeline.best_views <output_dir> [mcap_path]
```

If `mcap_path` is omitted, the path recorded in `slam/frames_index.json`
is used. An explicit CLI argument overrides the recorded path (handy
when the rosbag has been moved to a different host).

## Integration points

- `cloud_slam/mcap_reader.py` already exposes timestamped byte access;
  add a helper `read_frame_by_offset(mcap_path, offset)` or pass the
  offsets through an existing API. If the mcap library doesn't expose
  random-access by offset, fall back to timestamp-based seek.
- `cloud_slam/projection.py::project_lidar_to_camera` — confirm it
  accepts an arbitrary `T_world_cam` (vs. the hard-wired lidar→cam
  extrinsic). If not, extract a thin helper alongside it that takes a
  4×4 pose directly.
- `scripts/rosbag_to_bboxes.py` — add the stage 8 call after stage 7's
  embed step and before the final summary print.
- Stage 0 (`cloud_slam/slam_backends/kiss_icp_backend.py` or its post
  processing) — emit `slam/frames_index.json` alongside existing outputs.

## Edge cases and failure modes

- **Bbox entirely outside camera FOV for the whole scan** → skip with
  `"skipped": "never_visible"`.
- **All candidates fail occlusion/sharpness floor** → skip with
  `"skipped": "no_candidate_above_floor"`, manifest still records
  the best *attempted* composite score.
- **Two bboxes of same class** → `<class>_<NN>` uses the bbox_id from
  `layout_merged.txt`, not a re-numbering; collisions impossible.
- **Camera frame timestamp outside trajectory range** → frame dropped
  as a candidate (extrapolation not attempted).
- **Projected AABB fully off-image after distortion** → not a candidate.
- **Crop window clips to zero area after image-bound clamp** → skip with
  `"skipped": "crop_empty"`.
- **Mirror reflections** — expected failure mode. Occlusion and area
  scores will prefer a direct view; if the only view is a mirror, the
  crop will show the mirror. Document as a known limitation; detecting
  mirrors is out of scope.

## Testing

Unit tests in `tests/test_best_views.py`:

1. **Geometry**: synthetic 3D box at known pose + synthetic camera
   pose → projected AABB matches hand-computed values within 1 px.
2. **Occlusion — clear**: box + empty voxel cloud → occlusion = 1.0.
3. **Occlusion — blocked**: box + synthetic occluder cloud on every
   corner ray → occlusion = 0.0.
4. **Crop clipping**: projected AABB partially off-image → crop window
   is clamped to image bounds and non-empty.
5. **Crop padding**: AABB of 100×100 at image center → crop becomes
   120×120 (10% padding each side).
6. **Scoring sanity**: two candidate frames for the same bbox — one
   close+centered, one far+edge — close+centered wins.
7. **Skip path**: bbox never visible → manifest entry has
   `skipped = "never_visible"` and no image file exists.

Integration check (manual, not automated):

Run on `TEST_SCAN/` rosbag and eyeball `best_views/`:

- `sofa_00.jpg` should obviously show the sectional sofa.
- `chair_*.jpg` should show dining chairs (8 of them).
- `table_*.jpg` should show tables (5–6 of them).
- Mirrors and doors may or may not crop well; record observations.

Copy `best_views/` to the Windows-side scan dir per project convention
so the user can browse in File Explorer.

## Open questions (resolved during brainstorming)

- "Clearest" = weighted combination of area + centering + occlusion +
  sharpness (D).
- Output = one JPG per bbox + `best_views.json` manifest (B).
- Integration = stage 8 inside `rosbag_to_bboxes.py` (B).
- Crop = 2D AABB of projected 3D box, 10% padding on each side (user).
- Occlusion = 8-corner-ray sampling against `voxel.ply` (B).

## Out of scope

- Multi-frame fusion / HDR.
- Automatic mirror detection.
- VLM-based re-labeling from the crop.
- Upstream changes to SpatialLM inference (stages 5–6).
