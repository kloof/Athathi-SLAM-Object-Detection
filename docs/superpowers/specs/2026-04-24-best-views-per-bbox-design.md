# Stage 8 — Best-View-Per-Bbox

**Date**: 2026-04-24
**Branch**: `stage-8-best-views`
**Status**: Design approved; revised after codebase audit (see "Audit revisions").

## Audit revisions

A codebase audit (see commit history) found four blockers in the initial
draft that this revision fixes:

1. `trajectory.csv` is in the **raw SLAM frame**, not leveled or
   Manhattan-rotated — but bboxes live in the leveled + Manhattan-rotated
   world. Stage 8 must apply both transforms to the trajectory before
   projecting. We surface these transforms via a new index file.
2. The mcap library (`mcap_ros2`) does **not** support byte-offset random
   access; it only supports time-ranged reads. The frames index stores
   nanosecond timestamps instead of offsets.
3. `colorizer.load_calibration` returns a matrix called `T_lidar_cam`
   that is actually the **camera-from-lidar** transform (used as
   `p_cam = T @ p_lidar`). Composition must invert it: `T_world_cam =
   T_world_lidar · inv(calib['T_lidar_cam'])`, or equivalently, use the
   pose directly as `T_cam_world = calib['T_lidar_cam'] · inv(T_world_lidar)`.
4. `cloud_slam/projection.py::project_lidar_to_camera` hard-reads
   `calib['T_lidar_cam']` and is not usable with an arbitrary camera
   pose. A small sibling helper `project_world_to_image` is added.

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

## Guardrails — **DO NOT BREAK THE WORKING PIPELINE**

The production pipeline (stages 0–7) is empirically tuned and validated
(beam=4 single-shot SpatialLM inference, 3-Llama-pass ≥2-vote fallback,
voxel 2.5 cm, temp 0.3, top_k 3). The user spent months getting this
working. This feature must be **strictly additive**:

- **No changes to existing artifacts** — `trajectory.csv`,
  `colored_map.ply`, `colored_map_cropped.ply`, `colored_map_manhattan.ply`,
  `voxel.ply`, `spatiallm_input.ply`, `layout_qwen.txt`, `layout_llama.txt`,
  `layout_merged.txt`, `scene_with_boxes.ply`, `metrics.json` — byte-for-byte
  identical to today after this change.
- **No changes to SpatialLM inference** (`infer.py`, `merge.py`). Do not
  touch beam size, temperature, rep_penalty, top_k, consensus vote count,
  or any model-loading/weight path.
- **No changes to stage timings** that could affect determinism of
  SpatialLM sampling (e.g. don't reorder the pipeline around stage 5/6).
- **New additions only**: `slam/frames_index.json`, `best_views/` directory,
  `best_views/best_views.json`, a new `project_world_to_image` helper
  alongside `project_lidar_to_camera` (existing function untouched), and a
  new stage-8 function call in `rosbag_to_bboxes.py` placed **after**
  stage 7's `embed` step.
- **Stage 2 emitter addition**: `manhattan.py` must save its yaw angle
  (it currently only logs it). The change is one `json.dump` of a scalar;
  cloud output and cropping behavior stay identical.
- **Stage 0 emitter addition**: `mcap_reader.read_mcap` already iterates
  `/camera/image_raw/compressed` messages; tap the generator (or a sibling
  path that does) to collect `log_time` ns timestamps and dump them
  alongside existing outputs.
- **CLI**: the default `rosbag_to_bboxes.py` invocation gains stage 8 but
  `--skip-best-views` lets the user reproduce the old exact behavior.
- **Verification required**: before merging, run the canonical TEST_SCAN
  both with and without stage 8 and confirm `diff` on the existing
  artifacts is empty (or structurally identical for non-deterministic
  files like metrics.json if any such fields exist — call those out).

## Architecture

New module `cloud_slam/spatiallm_pipeline/best_views.py`, called as
stage 8 from `scripts/rosbag_to_bboxes.py`. Stage 0 gains a tiny emitter
for `slam/frames_index.json`.

### Image threading strategy

**Approach 2: frames-index + on-demand time-ranged decode.**

Stage 0 writes `slam/frames_index.json` with per-frame nanosecond
timestamps, the mcap path, the camera topic, and the world-frame
transforms needed to reconcile `trajectory.csv` with the leveled +
Manhattan-rotated world the bboxes live in:

```json
{
  "mcap_path": "/path/to/scan.mcap",
  "topic": "/camera/image_raw/compressed",
  "frames": [
    {"t_ns": 1713976420001000000},
    ...
  ],
  "level_rotation": [[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]],
  "level_z_shift_m": 0.047,
  "manhattan_yaw_deg": -3.2
}
```

`level_rotation` is the 3×3 matrix applied by
`cloud_slam/level.level_points` during stage 0 post-processing;
`level_z_shift_m` is the scalar floor-Z offset subtracted from the
cloud. `manhattan_yaw_deg` is the yaw angle applied by stage 2
(`manhattan.py`). Stage 2 must emit this value (currently it only logs
it); see Integration Points below.

Stage 8 opens the mcap once, uses `mcap_ros2.reader.read_ros2_messages`
with `start_time=t_ns, end_time=t_ns+1` to fetch each winning frame,
decodes only those. No heavy image buffer is threaded through stages
1–7; RAM and disk cost are both near zero.

Rationale: matches the "each stage emits artifacts" pattern already used
by CLAUDE.md; keeps stage 8 re-runnable standalone, which is critical
while tuning scoring weights.

## Inputs

| Artifact | Source | Purpose |
|---|---|---|
| `layout_merged.txt` | stage 6 | 3D bboxes (world Z-up) |
| `trajectory.csv` | stage 0 | per-scan LiDAR pose in **raw SLAM frame** (not leveled, not Manhattan) |
| `slam/frames_index.json` | **new**, stage 0+2 | camera frame timestamps (ns) + level/Manhattan transforms |
| `calibration/intrinsics.yaml` | vendored | K (3×3), D (plumb_bob 5-vec) |
| `calibration/extrinsics.yaml` | vendored | `T_lidar_cam` (already loaded by `colorizer.py`) |
| `voxel.ply` | stage 3 | occlusion point cloud (2.5 cm) |
| mcap path | `metrics.json` / CLI arg | re-opened to decode winning frames |

## Processing pipeline

### Step 1 — Load poses, transform into leveled+Manhattan world, interpolate

Read `trajectory.csv` (`timestamp, x, y, z, qw, qx, qy, qz`). These poses
are in the **raw SLAM frame** (pre-leveling, pre-Manhattan). Transform
each pose into the bbox world frame by applying, in order:

1. **Level**: `T_level = [[R_level, -R_level·[0,0,z_shift]^T], [0,0,0,1]]`
   where `R_level` is `level_rotation` from `frames_index.json` and
   `z_shift` is `level_z_shift_m`. (The floor-Z shift is applied after
   rotation, matching `post_process.level_points`.)
2. **Manhattan yaw**: `T_manhattan = Rz(manhattan_yaw_deg)` (pure yaw
   around world +Z).

So: `T_world_lidar(t) = T_manhattan · T_level · T_raw_lidar(t)`.

For each camera frame timestamp `t_ns`, compute `T_world_lidar(t)` by
SLERP on rotation and LERP on translation between the two bracketing
(already-transformed) scan poses. Frames outside the trajectory range
are dropped.

Compose camera pose — **note the inversion** required by
`load_calibration`'s convention (returned `T_lidar_cam` is actually
`T_cam←lidar`, used as `p_cam = T @ p_lidar`):

```
T_world_cam(t) = T_world_lidar(t) · inv(calib["T_lidar_cam"])
T_cam_world(t) = inv(T_world_cam(t))
              = calib["T_lidar_cam"] · inv(T_world_lidar(t))
```

The spec uses `T_cam_world` when projecting (it's what `cv2.projectPoints`
effectively wants). The inversion is encapsulated in the helper; module
code should never build the chain by hand.

### Step 2 — Candidate frames per bbox

For each bbox, iterate all camera frames. A frame is a **candidate** iff:

- The bbox center projects in front of the camera (positive Z in camera
  frame).
- The projected AABB of the 8 bbox corners has non-zero overlap with
  the image rectangle `[0,0,1280,720]` after distortion.

Use the new helper `cloud_slam.projection.project_world_to_image(
xyz_world, T_cam_world, K, D)` — leaves the existing
`project_lidar_to_camera` untouched, and accepts any camera pose.
Internally it transforms world points to camera frame and calls
`cv2.projectPoints` with zero rvec/tvec and the plumb_bob distortion
coefficients, matching the convention of the existing function.

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
`best_views/<class>_<NNN>.jpg`, where `NNN` is the bbox_id from
`layout_merged.txt` (zero-padded to 3 digits — `merge.py` writes the
id unpadded; a 3-digit file name accommodates cluttered scenes without
collisions). Slashes or spaces in the class string are replaced with
underscores to keep the filename shell-safe.

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
      "frame_timestamp_ns": 1713976423472000000,
      "image_path": "best_views/sofa_000.jpg",
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
  sofa_000.jpg
  chair_001.jpg
  chair_002.jpg
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

- **`cloud_slam/mcap_reader.py`** — add `read_frames_by_time_ns(mcap_path,
  topic, t_ns_list) -> list[(t_ns, bytes, format)]`. Internally uses
  `mcap_ros2.reader.read_ros2_messages(..., start_time=t, end_time=t+1)`
  per timestamp (no byte-offset API exists). Does **not** modify existing
  `read_mcap` behavior.
- **`cloud_slam/projection.py`** — add `project_world_to_image(xyz_world,
  T_cam_world, K, D)` as a sibling to `project_lidar_to_camera`.
  Existing function untouched.
- **Stage 0 emitter** — in the MCAP-iteration path (either `mcap_reader.py`
  itself by capturing `msg.log_time` per camera frame into an optional
  out-list, or in `post_process.py` / the SLAM runner where images are
  already held) collect camera `log_time` ns values and write them to
  `slam/frames_index.json`. Writer also embeds `level_rotation` and
  `level_z_shift_m` (already in `metrics.json` per the audit; re-surface
  here for self-contained consumption) and the mcap path.
- **`cloud_slam/spatiallm_pipeline/manhattan.py`** — at end of stage 2,
  append `manhattan_yaw_deg` to `slam/frames_index.json` (read existing,
  add field, re-write). Cloud processing and cropping stay identical.
- **`scripts/rosbag_to_bboxes.py`** — add the stage 8 call after stage 7's
  `embed` step and before the final summary print. Add
  `--skip-best-views` flag (default false). Keep everything above stage 8
  untouched, including all SpatialLM inference arguments and timings.
- **`cloud_slam/spatiallm_pipeline/merge.py`** — reuse `parse_layout`
  read-only for loading `layout_merged.txt`. No modification.
- **`cloud_slam/colorizer.py`** — reuse `load_calibration` read-only.
  No modification.

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

- `sofa_000.jpg` (or whichever id) should obviously show the sectional sofa.
- `chair_*.jpg` should show dining chairs (8 of them).
- `table_*.jpg` should show tables (5–6 of them).
- Mirrors and doors may or may not crop well; record observations.

**Non-regression verification (required before merging):**

Run the canonical TEST_SCAN pipeline twice:

1. With `--skip-best-views`: must produce byte-identical artifacts to a
   pre-change baseline run. Capture a reference hash of every file in
   `<output>/` before making changes; after changes, rerun and `diff`.
   Any diff must be explained and accepted (e.g. a deliberate addition
   like `slam/frames_index.json`).
2. Without `--skip-best-views`: produces the same artifacts plus the
   `best_views/` tree. SpatialLM output (`layout_qwen.txt`,
   `layout_llama.txt`, `layout_merged.txt`) must still match the
   skip-best-views run byte-for-byte — stage 8 must not perturb the
   upstream RNG / timing / GPU state used by beam=4 inference.

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
- Upstream changes to SpatialLM inference (stages 5–6) — beam=4 single-shot
  and 3-Llama-pass ≥2-vote fallback are empirically tuned and frozen for
  this work.

## Performance notes

Dominant costs (see discussion): per-bbox cheap-score filter is O(N_frames)
with `cv2.projectPoints`; occlusion + sharpness are O(K) over top-K
candidates per bbox (prefilter). Expected wall-clock for a 1-min scan
with ~20 bboxes: 15–30 s. Optimizations documented in module docstring:

- Prefilter to top-K candidates (default K=30) by cheap scores before
  running occlusion / sharpness.
- Build one KDTree over `voxel.ply` up front; reuse for all rays.
- Decode each unique winning frame at most once (cache by `t_ns`).
- Compute Laplacian variance only on decoded (top-K ∪ winners) frames.
