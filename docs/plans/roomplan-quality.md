# Toward RoomPlan-quality indoor scanning

Sub-2 cm wall lengths + structured doors / windows / passages. Based on a
6-agent deep research pass across 2024-2026 literature, then 5 review
agents sanity-checked the plan against the codebase (fixed the Stage 8
math formulation, corrected the accuracy budget downward by ~25 %,
flagged `scripts/loop_closure.py` as non-adaptable, tightened the M3
opening bounds, etc.).

Branch: `feature/roomplan-quality` (off `feature/vision-wall-typing`,
which carries the Mask2Former wall typing work in `2208447` + `dab3ebc`).

---

## Context

Current state (commits through `dab3ebc`):

- D_refined walls with ~3 cm RANSAC residual, confidence 1.0
- Vision wall typing (Mask2Former ADE20K) + per-wall `features`
- YOLOE-26L for furniture (person + door removed — door is now a vision
  feature on walls, de-duplicated)

User goals (from a brainstorming session):

- **Sub-2 cm tolerance on wall lengths** (currently off by 10 cm+ vs
  tape measure, random direction)
- **RoomPlan-style structured openings** — each door, window, and
  passage as a structured object with position-along-wall + dimensions,
  not just a label on the wall
- Same input pipeline (MCAP batch, Unitree L2 + Logitech Brio)

---

## Honest accuracy expectations (calibrating "sub-2 cm")

From the research sweep:

| Constraint | Physical floor |
| --- | --- |
| Unitree L2 raw precision | ~2 cm (sensor noise) |
| ICP registration per-frame | ~5–10 mm RMSE on walls |
| Cumulative SLAM drift over 100 frames | ~50–150 mm (√N accumulation) |
| Closure error (scan start vs end) | typically 10–50 mm |

Sub-2 cm is right at the lidar noise floor. Achievable on camera-covered
walls with line anchors; not guaranteed on walls occluded by furniture.
Mobile scanners with similar hardware (Polycam, Canvas / Twindo,
Scaniverse) publicly claim 5–15 mm interior range — the target is tight
but not unrealistic.

Step-by-step impact estimate (revised after review — the earlier
table was ~25 % optimistic; these numbers are the expected median,
not the best case):

| After milestone | Typical median | Worst-case (occluded / reflective) |
| --- | --- | --- |
| Current (`dab3ebc`) | 30-100 mm | 100 mm+ |
| + M1a Stage 8 polygon closure | 25-50 mm | 50-80 mm |
| + M1b GTSAM pose-graph refinement (optional) | 20-40 mm | 40-60 mm |
| + M2 DeepLSD line anchors | 15-25 mm on camera-covered walls | unchanged on blind walls |
| + tight lidar-camera calibration | 10-20 mm best case | 30-50 mm |

**Realistic outcome after all milestones:** median 18-25 mm, worst-case
50-70 mm. Sub-2 cm is achievable only for pristine rooms with full
camera sweep + minimal occlusion. Users should expect 2-4 cm as the
typical result on real scans; sub-2 cm is the aspirational ceiling.

---

## Strategy — 3 milestones, dependency order

Walls must be accurate before openings (openings reference walls by
`wall_id` + `along_start/end`). Sequence is strict.

### M1 — Wall accuracy

#### M1a — Stage 8 weighted polygon closure (classical, ~180 LOC)

After Stage 7 produces the refined walls, the polygon loop has a
residual closure vector `Δ = Σ (p2_i − p1_i)` that should be zero.
Redistribute `Δ` across walls using a confidence-weighted variant of
the Bowditch / Compass Rule. Note: the classical Compass Rule
distributes closure error proportional to *wall lengths*; we use
inverse-confidence weighting instead as a principled heuristic (low-
confidence corners absorb more error). This is no longer "the
classical technique" — it's a variant. Document in code.

Key references:
- [Compass Rule — Mahun](https://jerrymahun.com/index.php/home/open-access/17-trav-comps/44-travcomps-chap-e)
- [Adjustment Computations — Bethel/Purdue](https://engineering.purdue.edu/~bethel/adjcmp.pdf)
- [O-Snap — ACM TOG 2013](https://dl.acm.org/doi/10.1145/2421636.2421642)

Algorithm (`_stage8_polygon_closure` in `cloud_slam/floorplan.py`):

```
1. Compute Δ = Σ (p2_i − p1_i)    # closure gap, typically 10-50 mm
2. If |Δ| < 1 cm: return walls unchanged (strict no-op;
                   guarantees zero regression on already-closed polygons)
3. Constraint formulation (DO NOT process corners sequentially —
   sequential greedy adjustment can fail when snapped + free walls
   meet at a shared corner. Instead, build a constrained least-
   squares system):
     Variables: corner positions {c_j}
     Soft cost: Σ w_j · ||c_j − c_j_original||²
                where w_j = 1 / (confidence_left_j * confidence_right_j + ε)
     Hard constraints:
        (a) Polygon must close: c_0_end == c_0_start (i.e. the chain
             of wall vectors must sum to zero).
        (b) For each wall with snapped_to ∈ {dominant, manhattan,
             diagonal}: the wall direction is fixed — corner movement
             is restricted to the 1D subspace along that wall's line.
        (c) For free walls: corners are free 2D variables.
     Solve via scipy.optimize.minimize (SLSQP) or closed-form
     block-Lagrangian.
4. Safeguards post-solve:
     - No wall length < 15 cm → revert that corner to pre-closure
     - No wall direction flip
     - Per-corner |shift| < 50 mm (else reject entire solve, log warning)
5. Log per-corner shift + weight for diagnostics
```

Expected impact: 5-10 cm random error → 2.5-5 cm (revised per Agent
A+D review — the original 2-4 cm estimate was optimistic because
Stage 8 only redistributes closure error, it can't fix drift that
doesn't manifest as polygon closure).

**Invariant:** when `|Δ| < 1 cm`, Stage 8 is a strict no-op by
construction. That's how we get "zero regression" on already-clean
polygons: we don't touch them at all.

#### M1b — OPTIONAL GTSAM pose-graph refinement (~180 LOC, behind a flag)

If M1a isn't enough, correct SLAM drift itself rather than patch wall
geometry. **Important note** (per earlier code-reviewer agent): our
existing `scripts/loop_closure.py` is tightly coupled to FAST-LIO2
(hard-coded `pos_log.txt` 25-column format, per-frame PCD files on
disk, position-only optimization). It is **not adaptable** — only the
helper functions are reusable. M1b is effectively a NEW script.

Reusable helpers from the existing file:
- `make_scan_context`, `sc_distance`, `detect_loops` (lines 47-163)

Write a new `scripts/pose_graph_refine.py`:

1. Build ScanContext descriptors per frame (reuse existing helpers).
2. Detect loops with a tighter threshold (0.12) and looser frame gap
   (min_gap = 10) — single-room scans revisit the start quickly.
3. ICP between loop pairs, stricter settings (300 iters, reject RMSE
   > 5 cm).
4. **GTSAM iSAM2 6DoF pose graph** (new code):
   - Odometry edge: relative pose between consecutive `poses` from
     the in-memory `detect_pipeline.run()` output (no disk round-trip).
   - Loop edge: refined relative pose + info weight from ICP fitness.
   - Prior: gravity-aligned rotation from detected floor plane.
5. Apply corrected poses to the PER-FRAME xyz arrays captured during
   SLAM (requires exporting them from `icp_imu_pipeline.run` — new
   plumbing, ~50 LOC extra).
6. Re-merge point cloud with corrected poses in memory.
7. Re-run Stages 1-8 on the corrected cloud.

Expected impact: 5-15 mm additional reduction on top of M1a (per
Agent D's audit — more conservative than the earlier "1-2 cm" claim).
Not recommended for V1 unless M1a + M2 still fall short of sub-2 cm.

### M2 — DeepLSD line anchors for sub-pixel endpoint precision

RANSAC on sparse lidar can't do better than lidar's native precision
floor. [DeepLSD](https://github.com/cvg/DeepLSD)
([arxiv 2212.07766](https://arxiv.org/abs/2212.07766), CVPR 2023)
detects line segments in RGB images with sub-pixel endpoint accuracy.
Projecting camera-detected ceiling-wall seams to 3D gives tighter wall
endpoints than lidar alone.

Why DeepLSD:

- Hybrid classical + learned → robust to domain shift.
- Sub-pixel endpoint accuracy (beats pure-learned LETR / HAWP on VP
  error 1.63 vs 1.76).
- Indoor-tuned pretrained weights available.
- ~20 M params, ~50 ms / frame on GPU.
- Alternative: [LINEA](https://github.com/SebastianJanampa/LINEA)
  (ICIP 2025, [arxiv 2505.16264](https://arxiv.org/abs/2505.16264))
  if > 100 FPS needed — we don't.

Integration:

New module `cloud_slam/line_detector.py` (lazy-load, graceful disable,
FP16 on CUDA — mirrors `wall_segmenter.py` structure).

Per-frame (inside `detect_pipeline.on_frame`, only when
`--label-walls`):

1. DeepLSD → 2D line segments `{(u1, v1), (u2, v2), score}`.
2. Filter: keep lines whose 3D reconstruction is within 0.5 m of the
   detected ceiling plane — removes furniture clutter.
3. Look up per-pixel lidar depth at line endpoints (bilinear from the
   projected lidar sweep).
4. Back-project to world coords via the existing
   `project_lidar_to_camera` inverse.
5. Append `(xyz_world_start, xyz_world_end, score, frame_id)` to a
   global buffer.

Post-merge (inside `generate_floorplan` BEFORE Stage 8):

6. For each D_refined wall, find 3D lines whose XY direction is
   within 3° of the wall AND perpendicular distance < 10 cm AND
   whose along-wall projection falls within the wall segment + 20 cm
   slack (this third filter prevents corner-clutter contamination —
   a line belonging to the next wall at a 27° corner could match
   direction and proximity but belong to the wrong wall).
7. **Require ≥ 5 matching lines per wall** (per Agent A review —
   with only 2-3 lines the 20/80 percentile is identical to raw
   min/max and loses its robustness). If < 5 matching lines, skip
   DeepLSD anchor for that wall and fall back to Stage 7 endpoints.
8. Project the matching line endpoints onto the wall's line →
   candidate along-wall extents.
9. Take robust percentiles (20th min / 80th max) as the refined
   endpoints.
10. Run Stage 8 closure on the refined endpoints.

Expected impact on walls with camera coverage and ≥ 5 matching lines:
5-10 mm off the residual. Walls without camera coverage or with < 5
lines unchanged.

**DeepLSD install note** (per earlier code-reviewer agent): the
`cvg/DeepLSD` repo requires compiling Ceres + GFlags + GLog for the
full refinement step. Use `quickstart_install.sh` for a Ceres-free
install — the inference path still works (we lose the gradient-descent
refinement but keep the primary line detection, which is the big
win). Pin the repo to a specific commit SHA in `requirements.txt`.

### M3 — Structured openings (doors / windows / passages)

Schema — mirrors Apple's `CapturedOpening`:

```json
"D_refined": {
  "walls": [...],
  "openings": [
    {
      "id": "open_0", "type": "door",
      "wall_id": 11, "along_start": 1.52, "along_end": 2.42,
      "z_bottom": 0.00, "z_top": 2.05,
      "width_m": 0.90, "height_m": 2.05,
      "source": "vision+lidar", "is_open": false,
      "confidence": 0.82
    }
  ]
}
```

Algorithm (classical histogram + vision overlay — the
[Cloud2BIM](https://arxiv.org/abs/2503.11498) pattern with vision
re-weighting; beats end-to-end ML for this task per the research
sweep).

Per wall:

1. Collect wall-band points **directly** (do NOT reuse
   `_collect_wall_band()` — it applies a 0.30 m margin that would
   cut off baseboard and transom regions). Use a custom filter:
   perpendicular distance < 0.20 m AND Z in `[floor_z, ceiling_z]`
   (no margin). Per point record `(t, z, bucket_id)` with `t` =
   along-wall distance from `p1`.
2. Build 2D grid at **0.03 m resolution** (tighter than the earlier
   0.05 m spec — small bathroom windows at 0.25-0.30 m wide would
   be only 5-6 cells with 0.05 m; at 0.03 m they get ~10 cells,
   safer for connectivity). Per cell record:
   - (a) lidar occupancy (any point present),
   - (b) majority vision bucket,
   - (c) point density.
3. **Vision blobs** — connected-component label with **8-connectivity**
   (explicit — better for elongated rectangular door blobs than
   4-connectivity) over cells with bucket ∈ {door, window, glass}.
   Two blob maps: one for door bucket, one for window + glass.
4. **Gap detector** — cells with NO lidar point and 8-neighbors with
   points → candidate passage cells.
5. For each blob / gap region:
   - Fit OBB in (t, z) → extract `along_start/end, z_bottom/top,
     width, height`.
   - Reject by plausible dimension bounds (loosened per Agent C):
     - door: 0.4-2.5 m wide × 1.5-2.8 m tall (covers pocket and
       double doors up to 2.5 m)
     - window: 0.2-3.5 m × 0.3-2.5 m (transoms & small bathroom
       windows go down to 0.2 m)
     - passage: 0.5-4.0 m × 1.8-3.0 m (widened to catch 4 m
       archways; tall-ceiling passages up to 3 m)
   - Reject support < 30 labeled cells (vision) or < 20 empty cells
     (gap).
   - Reject blobs whose (t, z) bbox touches the wall ends within
     0.1 m (corner artifact).
6. **Open/closed classification** for doors:
   - `density_ratio = point density in blob / avg density on wall`
   - `is_open = density_ratio < 0.25`
   - Note: 0.25 is a starting threshold. Validate on real scans and
     tune; 0.25-0.75 range is ambiguous ("ajar"), report as `null`
     instead of forcing binary.
7. **Dedup:** vision blob and passage gap overlapping by **IoU > 0.3**
   (intersection over union, explicitly specified) → keep the vision
   blob (more specific type).
8. **Confidence:**
   - vision: `support / blob_bbox_cells`, capped at 1.0
   - passage: `min(gap_height / wall_height, 1.0)`

Expected recall / precision (Agent 3 synthesis of Cloud2BIM numbers +
vision overlay):

- Closed doors / windows: ~93–95 % F1
- Open passages: ~75–80 % F1 (fundamentally ambiguous from static
  scans)

PNG rendering in `_export_refined_png`:

- `door` → wall line with ~1 m gap + two ⊥ ticks pointing inward
- `window` → wall line with gap + single ⊤ tick
- `passage` → wall line with plain gap (no overlay)
- Secondary legend `Openings` with the three symbol swatches

---

## Files to modify

| Path | Change | LOC |
| --- | --- | --- |
| `cloud_slam/floorplan.py` | `_stage8_polygon_closure()`; `_detect_openings()`; opening rendering in `_export_refined_png`; metadata additions | +400 |
| `cloud_slam/line_detector.py` | NEW — DeepLSD wrapper (lazy-load, graceful disable, FP16 CUDA; mirrors `wall_segmenter.py`) | ~150 |
| `cloud_slam/pipelines/detect_pipeline.py` | Optional `line_detector` kwarg; per-frame line-segment accumulation alongside wall labels; z-buffer check on line endpoints | +60 |
| `scripts/detect_and_slam.py` | Instantiate `LineDetector` when `--label-walls` is set; pass through to pipeline + floorplan; transform line endpoints in lockstep with `merged` at each rotation | +30 |
| `cloud_slam/requirements.txt` | Install instructions for DeepLSD (or drop-in equivalent) | +1 |
| `scripts/loop_closure.py` (M1b only, optional) | Adapt FAST-LIO2 → ICP+IMU output; swap `scipy.least_squares` → GTSAM iSAM2 | +180 |

Total M1a + M2 + M3 (without M1b): **~640 new LOC**. Zero changes to
Stage 1–7 or YOLOE.

---

## Dependencies

| Package | Reason | Size |
| --- | --- | --- |
| `deeplsd` (git install from [cvg/DeepLSD](https://github.com/cvg/DeepLSD)) | M2 line detection | ~100 MB weights |
| `gtsam` (optional, M1b only) | 6DoF pose-graph optimization | `pip install gtsam` |

No other new deps. Keeping `transformers` (Mask2Former, already in).

---

## Verification

### Per-milestone checkpoints

**After M1a:** re-run each scan, compare against its previous run.

- D_refined area stable within ±1 %
- Log per-wall endpoint shift from Stage 8, per-corner weight
- Tape-measure a room: if still > 3 cm off, proceed to M1b or M2

**After M2:** compare camera-covered walls vs camera-blind walls.

- Camera-covered walls should be tighter (target sub-2 cm)
- Camera-blind walls match M1a output (no regression)
- Log per-wall DeepLSD anchor status (anchored / not / conflict)

**After M3:** on the three SLAM_test scans:

- Bedroom (`scan_20260411_121711`) — expect ≥ 1 `door` opening on the
  −1.8 m wall (YOLOE previously detected 3 doors here).
- `scan_20260412_180132` — expect ≥ 1 `door` on wall [1], ≥ 1
  `window` / `glass`-type on wall [3].
- L-shape (`scan_20260412_180513`) — expect ≥ 2 openings total.

Structural checks on every opening:

- `0 ≤ along_start < along_end ≤ wall.length_m`
- `floor_z ≤ z_bottom < z_top ≤ ceiling_z`
- `width_m = along_end − along_start`
- `height_m = z_top − z_bottom`

### Regression (no `--label-walls`)

- No `openings` key in metadata.
- No DeepLSD / Mask2Former loads (no HF downloads, no GPU allocs).
- Stage 8 STILL runs (classical, no model dep) — but it is a **strict
  no-op** when `|Δ| < 1 cm` (already-closed polygon). So wall
  positions are byte-identical to `dab3ebc` on the regression path,
  not "< 1 cm drift" as originally claimed.

Regression criterion: D_refined area within ±1 % of `dab3ebc`, wall
count identical on the three test scans. On scans where the closure
gap is already < 1 cm (every D_refined output to date), Stage 8 makes
no changes at all — regression is mathematically zero.

### Ground-truth benchmark (new, added from Agent C review)

Before claiming sub-2 cm is met, establish physical ground truth:

1. Pick **2-3 rooms** with distinct geometry:
   - A simple 4-wall rectangular room (baseline easy case)
   - An L-shape room with 6+ walls (harder; tests closure)
   - A room with visible features (door, window, glass) for M3
2. Before scanning, tape-measure each wall to the nearest mm. Record
   in `docs/plans/ground-truth/<room>.yaml`:
   ```yaml
   room_id: simple_bedroom
   date: 2026-04-20
   walls:
     - id: N, length_m: 3.625
     - id: E, length_m: 4.810
     - id: S, length_m: 3.628
     - id: W, length_m: 4.815
   corners:
     - id: NE, xy: [3.625, 0.000]  # local room frame
     ...
   openings:
     - {type: door, wall_id: W, along_start: 0.80, width_m: 0.90}
   ```
3. Scan each room and run with `--label-walls`.
4. Score each output:
   - `max_wall_error_mm` = max over walls of |measured - scanned|
   - `median_wall_error_mm` = median same
   - `opening_match_count` = #ground-truth openings within 10 cm
5. Sub-2 cm goal met when `max_wall_error_mm < 20` on ≥ 2 rooms.

Commit the ground-truth YAMLs to the repo so future runs can
regression-test the accuracy claim, not just the schema.

---

## Risks and honest caveats

1. **Sub-2 cm is aspirational, not guaranteed.** Walls fully occluded
   by furniture can't be recovered. Expect bimodal accuracy:
   camera+lidar-visible walls sub-2 cm, occluded walls 3-5 cm.
2. **ScanNet++-level accuracy is impossible without Leica-class
   hardware.** Mobile lidar + camera + algorithms max out at 5-15 mm
   per the 2024-2026 literature. Unitree L2 at ~2 cm native precision
   is slightly worse than iPhone / Matterport.
3. **Random 10 cm+ errors suggest loop closure IS the real fix.**
   Our current SLAM is open-loop ICP+IMU; single-room scans DO
   revisit the start; M1b is the correct long-term answer. M1a is a
   band-aid that works surprisingly well but doesn't fix the root
   cause.
4. **DeepLSD depends on camera visibility.** Walls the camera never
   sees fall back to Stage 7/8 values. Check per-scan whether the
   camera swept around.
5. **Open passages are hard.** RoomPlan sidesteps this with real-time
   re-scan prompts; our static-scan detector hits ~75 % recall.
   Expect some false negatives on open doorways.
6. **We can't measure wall thickness from a one-sided scan.**
   RoomPlan gets it from multi-room connectivity; we don't. Wall
   thickness stays implicitly 0 (1D line walls). Not fixable in
   M1-M3.
7. **GPU OOM on mid-range cards (8 GB)** — Mask2Former + DeepLSD +
   YOLOE concurrent loading can exceed 8 GB VRAM. Mitigations:
   (a) keep FP16 for all three, (b) add a `--cpu-seg` fallback that
   puts Mask2Former on CPU if CUDA OOMs, (c) document minimum GPU
   requirement in README.
8. **Double-buffer sync is a maintenance hazard.** We already apply
   post-SLAM rotations to `merged` and `wall_labels['xyz']` in
   lockstep inside `scripts/detect_and_slam.py`. M2 adds a third
   buffer (line-endpoint pairs) that must follow the same
   transformations. Refactor: extract a `_apply_transform_to_buffers()`
   helper so all three buffers move together by construction. Failing
   to do this = silent bug where line anchors drift relative to
   `merged` by frame N.
9. **Model weight versioning** — DeepLSD weights and Mask2Former
   ADE20K weights aren't version-pinned today. HuggingFace / cvg can
   re-train or remove models. Pin specific commit SHAs in
   `requirements.txt` and cache weights under source control or an
   internal mirror.
10. **No tape-measure ground truth** — the plan's verification
    criteria are relative (area delta ±1 %). Before claiming sub-2 cm
    is met, we need actual tape-measure benchmarks on ≥ 2 rooms. See
    the new "Ground-truth benchmark" section below.
11. **Branch divergence cost grows over time.** `feature/roomplan-quality`
    branches off `feature/vision-wall-typing` which branches off
    `master`. Merge `feature/vision-wall-typing` to `master` before
    starting M1a implementation, so the new work stacks directly on
    `master`.

---

## Deliberately NOT in scope

Tempting-but-rejected items surfaced by the 6-agent survey:

- **MASt3R-SLAM** (CVPR 2025, [arxiv 2412.12392](https://arxiv.org/abs/2412.12392))
  — could replace / augment lidar SLAM with dense feed-forward stereo.
  Genuinely transformative but ~1000+ LOC, changes the input model
  fundamentally. V2 consideration.
- **Depth Pro** (Apple, Oct 2024, [arxiv 2410.02073](https://arxiv.org/abs/2410.02073))
  / **Depth Anything v2** — monocular metric depth for glass/mirror
  validation. Useful for M4 (sensor fusion on reflective surfaces),
  not needed for M1–M3.
- **Florence-2** — possibly better than Mask2Former on open-vocab
  wall types but not obviously better on our 4-class task.
- **3D Gaussian Splatting** (GaussianRoom, PGSR) — great for
  rendering, not for parametric output.
- **Multi-room / whole-house** — scans are single-room.
- **Wall thickness** — impossible from single-sided scanning.
- **USDZ / IFC / GLTF export** — can be added later; metadata
  already carries everything a conversion script needs.
- **Live / real-time capture** — we're offline batch.
- **Fine-tuning on ScanNet++** — no ground truth for our specific
  scans; would require collecting our own dataset.

---

## References

Stage 8 math:

- [Compass Rule — Mahun](https://jerrymahun.com/index.php/home/open-access/17-trav-comps/44-travcomps-chap-e)
- [Adjustment Computations — Bethel/Purdue](https://engineering.purdue.edu/~bethel/adjcmp.pdf)
- [O-Snap — ACM TOG 2013](https://dl.acm.org/doi/10.1145/2421636.2421642)

Line detection:

- [DeepLSD — arxiv 2212.07766](https://arxiv.org/abs/2212.07766)
- [LINEA — arxiv 2505.16264](https://arxiv.org/abs/2505.16264)
- [HAWP — arxiv 2003.01663](https://arxiv.org/abs/2003.01663)

Openings:

- [Cloud2BIM — arxiv 2503.11498](https://arxiv.org/abs/2503.11498)
- [DoorDet — arxiv 2508.07714](https://arxiv.org/abs/2508.07714)
- [PolyRoom — arxiv 2407.10439](https://arxiv.org/abs/2407.10439)

SLAM drift:

- [KISS-ICP — arxiv 2209.15397](https://arxiv.org/abs/2209.15397)
- [FAST-LIO2 — arxiv 2302.04031](https://arxiv.org/abs/2302.04031)

Datasets / bleeding-edge:

- [ScanNet++ — arxiv 2308.11417](https://arxiv.org/abs/2308.11417)
- MASt3R-SLAM ([arxiv 2412.12392](https://arxiv.org/abs/2412.12392))
- Depth Pro ([arxiv 2410.02073](https://arxiv.org/abs/2410.02073))
- [GRASS — glass reflection suppression](https://www.mdpi.com/2072-4192/18/2/332)
