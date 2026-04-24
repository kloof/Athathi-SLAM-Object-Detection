# Cloud SLAM ICP — Modal API (v1) Design

Branch: `modal-api`
Date: 2026-04-24

## Purpose

Deploy the production `rosbag_to_bboxes.py` pipeline as a Modal service
that accepts a rosbag upload (raw or `zstd -1`) and returns a structured
JSON with the detected floorplan, furniture bounding boxes, and
best-view images. Async execution with HTTP polling.

## Decisions locked in

| Area | Decision |
|---|---|
| Upload | Direct `POST /jobs` (binary body, up to 4 GiB). No S3, no presigned URLs. Accepts `.mcap`, `.mcap.zst`, `.tar`, or `.tar.zst` (the tar forms carry a ROS2 `rosbag2` directory; server extracts and picks the first `*.mcap` inside). |
| Execution | Async — submit spawns a background Modal Function; client polls. Recommended polling interval 2–5 s; endpoints emit `Retry-After: 3` while status ≠ `done`/`failed`. |
| Compute | `gpu="H100"`, `cpu=8`, `memory=32768` MiB (SLAM is CPU-bound; SpatialLM is GPU-bound; one container runs both). `max_containers=4` caps concurrent GPU spend. H100 picked over L4/A10G because the local dev machine is a 4070 Ti (504 GB/s bw, ~160 TFLOPS FP16) — L4 is actually *slower* than local, A10G is a wash, and SpatialLM-Llama at batch=1 is memory-bandwidth-bound. H100 at 3350 GB/s gives ~5× faster LLM decode; Modal auto-upgrades to H200 when available at no cost. |
| Volume consistency | Runner calls `volume.commit()` after every `status.json`/`result.json` write; endpoints call `volume.reload()` at the top of every `GET /jobs/{id}` to see cross-container writes. This is Modal's canonical footgun — without it the polling UX silently serves stale state. |
| Orchestration | Monolith — one Modal function wraps `scripts/rosbag_to_bboxes.py` unchanged. |
| Storage | Modal Volume `slam-jobs` mounted at `/jobs` in the container. |
| Retention | 7 days; a daily scheduled function deletes `job_id/` dirs older than 7 days. |
| Auth | Single static API key in header `X-API-Key`, stored in a Modal Secret. `job_id` whitelist regex on every path param: `^j_\d{4}-\d{2}-\d{2}_[0-9a-f]{8}$`. |
| Output | Full JSON inline in the status response once `status="done"`. Also persisted as `/jobs/<id>/result.json` on the volume so it can be re-fetched any time within retention. |
| Images / PLYs | Served via `GET /jobs/<id>/image/<idx>` and `GET /jobs/<id>/artifact/<name>`. |
| Compressed input | `.mcap.zst` is auto-detected by filename suffix and stream-decompressed server-side via `zstandard.ZstdDecompressor.copy_stream`. |
| Auto mode gate | Input validated up front: must be MCAP (magic bytes `0x89 MCAP0`) after optional zstd decode; reject otherwise with 400. |

## Non-goals (v1)

- Live / streaming upload while scanning.
- Live SLAM (KISS-ICP running incrementally as frames arrive).
- Partial-result SSE / webhook callbacks.
- Multi-tenant users or quotas beyond one global API key.
- Resumable uploads.

Each is a reasonable v2+ extension; none block v1.

## Architecture

```
 ┌──────────────────────┐   POST /jobs   ┌─────────────────────────────┐
 │ Client (curl / SDK)  │───bag bytes───▶│ submit endpoint (FastAPI)   │
 └──────────────────────┘                │  - validate auth+MIME        │
          ▲                              │  - write /jobs/<id>/input   │
          │ poll GET /jobs/<id>          │  - run_pipeline.spawn(id)    │
          │                              │  - return {job_id}          │
          │                              └────────────┬────────────────┘
          │                                           │ .spawn()
          │    ┌──────────────────────────────────────▼───────────────┐
          │    │ run_pipeline (background Modal Function)             │
          │    │  cpu=8, memory=32G, gpu="H100", timeout=3600         │
          │    │  max_containers=4                                    │
          │    │  1. decode .zst / untar if needed                    │
          │    │  2. subprocess: python rosbag_to_bboxes.py \         │
          │    │       <input.mcap> <artifacts_dir>                   │
          │    │     (calibration is the vendored default, no flag)   │
          │    │  3. parse layout_merged.txt + best_views.json        │
          │    │  4. write /jobs/<id>/result.json  +  volume.commit() │
          │    │  5. update /jobs/<id>/status.json + volume.commit()  │
          │    └──────────────────────────────────────────────────────┘
          │                                           │
          │                                           │ writes
          │    ┌──────────────────────────────────────▼───────────────┐
          └────│ status / image / artifact endpoints (FastAPI)        │
               │  all read-only views onto Modal Volume /jobs         │
               └──────────────────────────────────────────────────────┘
```

## Job lifecycle states

`status.json` on the volume holds a single field `status` progressing:

1. `queued` — bag received, pipeline spawn scheduled
2. `decoding` — zstd decompression if applicable
3. `stage_0_slam` through `stage_7_embed` — per-stage progress (set before running each stage)
4. `stage_8_best_views`
5. `done` — `result.json` available
6. `failed` — `error.json` written with traceback

Status is written via atomic rename (`tmp` → `status.json`) to avoid torn reads from the status endpoint.

## Endpoint contract

### `POST /jobs`

Request:
- Header `X-API-Key: <key>`
- Header `Content-Type: application/octet-stream` or `application/zstd`
- Query `?filename=scan.mcap.zst` — used only to pick the decode path
- Body: raw bytes (up to 4 GiB)

Response `200`:
```json
{
  "job_id": "j_2026-04-24_ab12cd34",
  "status_url": "https://.../jobs/j_2026-04-24_ab12cd34",
  "submitted_at": "2026-04-24T14:03:22Z"
}
```

Errors: `401` (bad key), `400` (not MCAP / not decompressable), `413` (over 4 GiB — Modal itself enforces this).

### `GET /jobs/{id}`

Response while running:
```json
{
  "job_id": "...",
  "status": "stage_5_infer",
  "submitted_at": "...",
  "started_at": "...",
  "stages_completed": ["stage_0_slam","stage_1_crop",…],
  "progress_hint": 0.62
}
```

Response when done (status: `done`): the same envelope PLUS a full `result` object (schema below).

Response on failure (status: `failed`): envelope PLUS `error: {type, message, stage}`.

### `GET /jobs/{id}/image/{idx}`

Streams the i-th best-view JPEG. `idx` is the index into `result.best_images[]`. 404 if out of range.

### `GET /jobs/{id}/artifact/{name}`

Streams one of the named artifact files. `name` ∈ {`colored_map.ply`, `scene_with_boxes.ply`, `layout_merged.txt`, `result.json`}. 404 for anything else (whitelist).

### `DELETE /jobs/{id}` (optional v1)

Immediately removes the job dir. Useful for debugging; not required.

## Result JSON schema

```json
{
  "job_id": "j_2026-04-24_ab12cd34",
  "status": "done",
  "submitted_at": "2026-04-24T14:03:22Z",
  "finished_at": "2026-04-24T14:14:09Z",
  "metrics": {
    "slam": { "duration_s": 42.3, "frame_count": 312, "point_count": 1420000 },
    "spatiallm": { "duration_s": 184.7, "qwen_count": 14, "llama_count": 23 },
    "total_duration_s": 651
  },
  "floorplan": {
    "walls":   [ { "id": "wall_0",   "start": [x,y,z], "end": [x,y,z], "thickness": 0.1, "height": 2.7 } ],
    "doors":   [ { "id": "door_0",   "wall": "wall_0", "center": [x,y,z], "width": 0.8, "height": 2.1 } ],
    "windows": [ { "id": "window_0", "wall": "wall_2", "center": [x,y,z], "width": 1.2, "height": 1.5 } ]
  },
  "furniture": [
    {
      "id": "bbox_0",
      "class": "sofa",
      "center": [1.2, 2.3, 0.4],
      "size":   [2.5, 1.0, 0.8],
      "yaw":    1.57
    }
  ],
  "best_images": [
    {
      "bbox_id": "bbox_0",
      "class": "sofa",
      "url": "https://.../jobs/j_…/image/0",
      "frame_timestamp_ns": 1712345678000,
      "camera_distance_m": 2.14,
      "pixel_aabb": [x0, y0, x1, y1]
    }
  ],
  "artifacts": {
    "colored_map_ply":    "https://.../jobs/j_…/artifact/colored_map.ply",
    "scene_with_boxes_ply":"https://.../jobs/j_…/artifact/scene_with_boxes.ply",
    "layout_merged_txt":  "https://.../jobs/j_…/artifact/layout_merged.txt",
    "result_json":        "https://.../jobs/j_…/artifact/result.json"
  }
}
```

`result.json` on the volume is exactly this document — the status endpoint re-renders URLs with the current host, so the volume copy stores relative paths and the endpoint fills in the base URL when serving.

## Files to add / modify

### New
- `modal_app/app.py` — Modal `App` definition, image spec, Volume, Secret, Function, endpoints.
- `modal_app/pipeline_runner.py` — background function body: decode → subprocess → parse → write `result.json`.
- `cloud_slam/api/parse_outputs.py` — pure-Python parsers:
  - `parse_layout_merged(path) -> {walls, doors, windows, furniture}`
  - `load_best_views_manifest(path) -> best_images[]`
  - `load_slam_metrics(path) -> metrics.slam`
- `modal_app/schemas.py` — Pydantic models for request / response envelopes.
- `modal_app/README.md` — short deploy + `curl` usage.

### Modified
- `cloud_slam/spatiallm_pipeline/infer.py` — replace hardcoded `/home/klof/...` paths (lines 20–21) with env-var driven paths:
  ```python
  REPO = Path(__file__).resolve().parents[2]
  SPATIALLM_DIR = Path(os.getenv("SPATIALLM_DIR", REPO / "third_party" / "SpatialLM"))
  SPATIALLM_PY  = Path(os.getenv("SPATIALLM_PY",  Path.home() / "spatiallm_env" / "bin" / "python"))
  ```
  Same env-var pattern for `CODE_TEMPLATE`. Behaviour is unchanged for the current local developer workflow (defaults resolve to today's paths); Modal sets the env vars to the in-container locations.
- `cloud_slam/requirements.txt` — add `modal`, `zstandard`, `python-multipart`, `fastapi[standard]`. Keep existing pins.

## Modal image build

Base: `modal.Image.debian_slim(python_version="3.10")` (matches current dev env).

Steps:
1. `apt_install` build-essential, cmake, git, libeigen3-dev (for KISS-ICP native bits), libgl1 (OpenCV).
2. `pip_install` from `cloud_slam/requirements.txt`.
3. `COPY` the repo (`cloud_slam/`, `scripts/`, `calibration/`) into `/root/cloud_slam_icp`.
4. `COPY third_party/SpatialLM` into `/opt/spatiallm` and `run_commands(...)` the setup_spatiallm.sh pattern:
   - create venv at `/opt/spatiallm_env`
   - `pip install -r requirements.txt` from SpatialLM's pyproject equivalent
   - build flash-attn, spconv-cu120, torch-scatter against the image's torch 2.4.1+cu124
5. `run_commands` that pre-download the two HuggingFace weights into an `HF_HOME` baked into the image:
   ```
   huggingface-cli download manycore-research/SpatialLM1.1-Qwen-0.5B
   huggingface-cli download manycore-research/SpatialLM1.1-Llama-1B
   ```
6. `env({"SPATIALLM_DIR": "/opt/spatiallm", "SPATIALLM_PY": "/opt/spatiallm_env/bin/python", "HF_HOME": "/root/.cache/hf"})`

The image build is expensive (~20 min first run, especially flash-attn). Subsequent cold starts reuse the image layer cache.

Note: `cloud_slam/spatiallm_pipeline/infer.py:79,140` runs the SpatialLM
inference in a subprocess **without** an explicit `env=` kwarg — so the
child inherits the Modal function's process env. The image-level `env(...)`
above therefore propagates through both `infer.py` and its SpatialLM child
Python without further glue. If a future refactor switches to `env=...`
in those `subprocess.run` calls, it must explicitly forward `HF_HOME`,
`SPATIALLM_DIR`, `SPATIALLM_PY`, `CODE_TEMPLATE`.

## Volume layout

```
/jobs/
  j_2026-04-24_ab12cd34/
    input.mcap                    # raw or decompressed
    status.json                   # atomically-updated state
    error.json                    # only on failure
    result.json                   # final JSON, written on success
    artifacts/
      slam/{colored_map.ply,trajectory.csv,metrics.json,frames_index.json}
      colored_map_cropped.ply
      colored_map_manhattan.ply
      voxel.ply
      spatiallm_input.ply
      layout_qwen.txt
      layout_llama.txt
      layout_merged.txt
      scene_with_boxes.ply
      best_views/
        best_views.json
        *.jpg
```

A sweeper Modal function (`@app.function(schedule=modal.Period(days=1))`) deletes any `j_*/` dir older than 7 days based on `status.json` mtime.

## Error handling

- **Zstd decode fail** → status=`failed`, error.type=`decode_error`.
- **MCAP magic-byte validation fail** → status=`failed` before pipeline spawn, so the GPU is never acquired.
- **`rosbag_to_bboxes.py` non-zero exit** → capture stderr tail (last 4 KB), status=`failed`, error.type=`pipeline_error`, error.stage=last known stage from `status.json`.
- **Stage 8 failure or no camera frames in bag** → non-fatal; result.json sets `best_images: []` and adds `warnings: ["best_views_failed: <msg>"]`. All other fields populated from stages 0–7.
- **Modal function timeout (>1 h)** → status=`failed`, error.type=`timeout`.
- **Upload >4 GiB** → Modal's ingress may close the connection rather than returning a clean 413. The submit endpoint catches `ClientDisconnect`/`LimitOverrunError` and maps to a JSON 413 where it can, but clients should also respect the 4 GiB ceiling themselves.

## Cost safety (don't burn credits)

These are hard rails, not aspirations — every one has a specific
enforcement point in code.

| # | Risk | Safeguard | Enforcement |
|---|---|---|---|
| 1 | Pipeline hangs → 1 h of H100 | outer function `timeout=3600`, inner `subprocess.run(..., timeout=3300)` (55 min) — inner fires first so we always get a clean `TimeoutExpired` and log it | `modal_app/pipeline_runner.py` |
| 2 | Modal auto-retries → 2× billing | `@app.function(retries=0)` on the runner and the submit function | `modal_app/app.py` |
| 3 | On-demand image build (flash-attn ~20 min) during first request | deploy builds the image during `modal deploy`; requests never trigger layer builds. Runner asserts `SPATIALLM_PY.is_file()` at start and fails fast if env is broken. | `modal_app/app.py` + M3 acceptance |
| 4 | Bad input reaches GPU | validation runs **in the submit endpoint** (cheap ASGI container, no GPU): check extension whitelist, decompress-probe the first 64 KB, check MCAP magic. Only after all checks pass does submit call `runner.spawn(job_id)`. | `modal_app/app.py::submit` |
| 5 | Runaway concurrency spawning many H100s | runner declared with `max_containers=2`; request rate limit on submit at 1 req/sec/key | `modal_app/app.py` |
| 6 | Client retries POST on transient 500 → we do the work twice | `X-Idempotency-Key` header (client-generated UUID). If an existing job with that key is found, return the existing `job_id` instead of starting a new run. Key→job_id map stored at `/jobs/_idempotency/<hash>` on the volume, 24 h TTL. | `modal_app/app.py::submit` |
| 7 | Failed job gives no useful info → user re-runs paying again | `error.json` captures: last 4 KB of subprocess stderr, Python traceback, last known `status` value, container hostname, wall-time elapsed. One paid run is enough to debug. | `modal_app/pipeline_runner.py` |
| 8 | Zstd decompression bomb | decompress into a size-capped writer (10 GiB ceiling); abort and delete on overflow | decode helper in runner |
| 9 | DELETE /jobs/{id} doesn't actually cancel the function call | submit stores the `FunctionCall` object ID on the volume; DELETE loads it and invokes `FunctionCall.from_id(...).cancel()` before removing the dir | `modal_app/app.py::cancel` |
| 10 | Multiple image versions / orphaned deploys | single-named app (`cloud-slam-icp`). Re-deploys replace the previous version atomically. | `modal_app/app.py` |
| 11 | Deploy-time model-weight download baked into image (2 GB) fails halfway and bloats image | pre-download step uses `huggingface-cli download --local-dir` with hash verification; build fails loud if hash mismatches | image build stage |

Operational check before any request is accepted:

```
$ modal deploy modal_app/app.py        # builds + deploys; 15-25 min first time
$ modal app list | grep cloud-slam-icp # confirms deploy succeeded
```

If those two commands haven't run successfully, there's nothing to POST to.

## Security

- API key comes from Modal Secret `slam-api-key`. Never committed to git.
- `job_id` is `j_<date>_<random-8-hex>` — unguessable enough for an internal MVP; not a substitute for auth.
- The token used to auth Modal itself (`modal token set`) stays in `~/.modal.toml` on the developer machine; not in the repo.

## Testing strategy

1. **Unit** — `tests/test_parse_outputs.py`: feed a captured `layout_merged.txt` and `best_views.json` from a prior TEST_SCAN run; assert the JSON schema matches.
2. **Local Modal dev** — `modal serve modal_app/app.py` against a stripped-down toy rosbag (<100 MB). Confirms image builds, pipeline runs, polling returns `done`.
3. **Full smoke** — `modal deploy modal_app/app.py` + curl POST with TEST_SCAN rosbag. Compare `result.json` against the known-good TEST_SCAN expectations (1 sectional sofa, 5–6 tables, 8 dining chairs).

## Cost model

- Image build: one-time ~20 min of CI-class compute (~$0.30).
- Per pipeline run: ~6 min on H100 (vs ~10 min on 4070 Ti local) × (H100 ~$3.95/hr + 8 vCPU + 32 GB) ≈ **$0.50–0.75 per rosbag**. A10G at ~$0.22/run is cheaper but no speed win over local.
- Volume storage: $0.20/GB-month; a typical job dir is ~3 GB → ~$0.02 for a 7-day retention window.
- No keep-warm cost; cold start (~30 s) is negligible on a 10-min job.

## Open questions resolved during brainstorm

- **Upload size cap** — Modal web endpoints allow up to **4 GiB** request bodies (confirmed from current Modal docs). Covers our compressed bags.
- **CPU cores** — KISS-ICP is CPU-only and multi-threaded. `cpu=8` gives real headroom; prior memory on "multi-core scales well" applies.
- **GPU** — SpatialLM Qwen-0.5B + Llama-1B fit in ~6 GB VRAM, so all tiers are size-sufficient. Choice is driven by **memory bandwidth** (the actual bottleneck for batch=1 LLM decode), not VRAM. H100 @ 3350 GB/s wins ~5× over the local 4070 Ti; L4 is slower than local so rejected; A10G is roughly equivalent so only worthwhile if H100 quota is unavailable.
- **JSON delivery** — inline in the polling response once `done`, AND persisted as `result.json` on the volume so it survives and can be re-fetched later during retention.
