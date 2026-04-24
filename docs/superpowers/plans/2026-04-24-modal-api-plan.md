# Modal API — Implementation Plan

Companion to `docs/superpowers/specs/2026-04-24-modal-api-design.md`.
Branch: `modal-api`. Each milestone ends with a commit and (M3+) a
`code-reviewer` subagent pass before moving on. Per project memory:
*"Review agent after every milestone — fold findings back before
proceeding"*.

---

## M1 — De-hardcode `infer.py` paths

**Files:**
- `cloud_slam/spatiallm_pipeline/infer.py` (edit only lines 15–25)

**Change:**
```python
import os
REPO = Path(__file__).resolve().parents[2]
SPATIALLM_DIR = Path(os.getenv("SPATIALLM_DIR", REPO / "third_party" / "SpatialLM"))
SPATIALLM_PY  = Path(os.getenv("SPATIALLM_PY",  Path.home() / "spatiallm_env" / "bin" / "python"))
CODE_TEMPLATE = Path(os.getenv("SPATIALLM_CODE_TEMPLATE", SPATIALLM_DIR / "code_template.txt"))
```

**Acceptance:**
- `python -c "from cloud_slam.spatiallm_pipeline.infer import SPATIALLM_PY; print(SPATIALLM_PY)"` returns the same path locally as today.
- Setting `SPATIALLM_PY=/opt/spatiallm_env/bin/python` in env overrides it.
- All existing local runs still work (no behaviour change for dev workflow).

**Commit message:** `refactor(infer): env-driven SpatialLM paths for Modal portability`

---

## M2 — Output parser module

**Files:**
- `cloud_slam/api/__init__.py` (new, empty)
- `cloud_slam/api/parse_outputs.py` (new)
- `tests/test_parse_outputs.py` (new)
- `tests/fixtures/sample_layout_merged.txt` (new, copied from a prior TEST_SCAN run)
- `tests/fixtures/sample_best_views.json` (new, ditto)
- `tests/fixtures/empty_best_views.json` (new, `{"entries": []}` — covers bag-with-no-camera-frames and Stage 8 failure fallback)

**Functions in `parse_outputs.py`:**
- `parse_layout_merged(path: Path) -> dict` — regexes already exist in
  `cloud_slam/spatiallm_pipeline/merge.py:_WALL_RX,_DOOR_RX,_WIN_RX,_BBOX_RX`;
  reuse them. Return shape matches `floorplan` + `furniture` sub-trees of the
  result JSON.
- `load_best_views_manifest(path: Path) -> list[dict]` — re-shape entries to
  the API's `best_images` schema (drop internal fields like `scores`, `crop_aabb`).
- `load_slam_metrics(slam_dir: Path) -> dict` — read `metrics.json`.
- `build_result_json(output_dir: Path, job_id: str, submitted_at, finished_at) -> dict`
  — orchestrator that stitches all three and returns the envelope *with
  relative artifact paths*. The endpoint layer substitutes the base URL at
  serve time.

**Acceptance:**
- `pytest tests/test_parse_outputs.py -q` passes (6 tests: walls, doors,
  windows, bboxes, full envelope roundtrip, empty-manifest fallback).
- Fixture files committed; no network or GPU required to run the tests.
- `best_images[]` schema matches spec (no `visible_fraction`; uses
  `pixel_aabb` + `camera_distance_m` which *are* in the real manifest
  at `best_views.py:698-708`).

**Commit message:** `feat(api): parse layout_merged + best_views into result JSON`

**Review checkpoint:** dispatch `code-reviewer` subagent on the new module.

---

## M3 — Modal app skeleton (image + Volume + Secret wiring)

**Files:**
- `modal_app/__init__.py`
- `modal_app/app.py` — `modal.App("cloud-slam-icp")`, Image builder, Volume, Secret.
- `modal_app/README.md` — curl usage.

**Image build steps:** exactly as listed in spec §"Modal image build",
INCLUDING the offline-load verification probe at the end of the same
`run_commands(...)` block that did the `huggingface-cli download`. Without
the probe, a silent partial download will produce a green image that fails
on first real run.

**Verification functions added in this milestone:**
- `@app.function(...) def verify_image()` — manual preflight smoke; loads
  both models in offline mode + runs a trivial inference. Returns dict.
- FastAPI `GET /health` (added in M5, but shape decided here): returns
  `{status, image_built_at, weights: {qwen, llama}, uptime_s}`.

**Acceptance:**
- `modal build modal_app/app.py` succeeds. If the weight-cache probe
  fails, the build fails loudly and we fix the download step before M4.
- `modal run modal_app/app.py::verify_image` returns `{"qwen": "ok", "llama": "ok"}`.
- No heavy Function logic yet — just the image, `verify_image`, App object.

**Commit message:** `feat(modal): app skeleton with prebuilt SpatialLM image`

**Review checkpoint:** dispatch `code-reviewer` on the image spec — particularly
the HuggingFace pre-cache step (must verify model paths are current) and
the flash-attn build step (must pin CUDA version).

---

## M4 — Pipeline runner function

**Files:**
- `modal_app/pipeline_runner.py` — the background `@app.function(...)`.
- Touches Volume at `/jobs/<id>/`.

**Logic:**
0. **`verify_cache_or_die()`** — offline `from_pretrained(..., local_files_only=True)` for both models; if either is missing or corrupt, write `error.json` with `error.type="image_cache_broken"` and abort. This guards against the case where a deploy succeeded but the HF cache was partially populated.
1. Read `input.mcap[.zst|.tar|.tar.zst]` from `/jobs/<id>/`.
2. Decode path by suffix: stream-unzstd if `.zst` (with 10 GiB decompressed cap); extract first `*.mcap` if tar.
3. Validate MCAP magic bytes on the resolved file; raise with error JSON on fail.
4. Write `status.json` atomically (tmp → rename) for each stage transition
   using an in-Python context manager `stage(name)`. **Immediately call
   `volume.commit()` after every rename** so the status endpoint in another
   container can see the update.
5. `subprocess.check_call([sys.executable, "/root/cloud_slam_icp/scripts/rosbag_to_bboxes.py", input_mcap, artifacts_dir])`.
   Calibration is the vendored default baked into the image — no CLI flag needed.
6. On success: `build_result_json(...)` → write `result.json`, `status="done"`,
   `volume.commit()`.
7. On `CalledProcessError`: capture last 4 KB of stderr, write `error.json`,
   `status="failed"`, `volume.commit()`.

**Acceptance (local):**
- `modal run modal_app/app.py::pipeline_runner --job-id test001` against
  a pre-staged small bag on the volume produces `result.json` + final
  artifacts.

**Commit message:** `feat(modal): pipeline runner function with staged status updates`

**Review checkpoint:** subagent specifically checks atomic-write correctness,
subprocess env handling (PYTHONPATH), and timeout behaviour.

---

## M5 — FastAPI endpoints

**Files:** `modal_app/app.py` (extend with `@modal.asgi_app()`).

**Endpoints:**
- `POST /jobs` — header auth, stream body to `/jobs/<id>/input.(mcap|mcap.zst|tar|tar.zst)`, call `pipeline_runner.spawn(job_id)`.
- `GET /jobs/{id}` — **call `volume.reload()` first**; read status.json; if
  `done`, merge with result.json and rewrite artifact URLs to absolute URLs
  using `request.base_url`. If status ≠ `done`/`failed`, include header
  `Retry-After: 3` to suggest a polling cadence.
- `GET /jobs/{id}/image/{idx}` — `volume.reload()`, read best_views.json, stream JPEG.
- `GET /jobs/{id}/artifact/{name}` — `volume.reload()`, whitelisted file streamer.
- `DELETE /jobs/{id}` (optional).

**Path-param validation:** every endpoint that takes `{id}` runs the regex
`^j_\d{4}-\d{2}-\d{2}_[0-9a-f]{8}$` via FastAPI `Path(..., pattern=...)` to
block path traversal.

**Function decorator:**
```python
@app.function(image=..., volumes={"/jobs": volume}, secrets=[api_key_secret],
              max_containers=4)
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def web(): ...
```
`max_containers=4` caps simultaneous warm ASGI containers; the compute-heavy
`pipeline_runner` has its own (lower) cap.

**Acceptance:**
- `curl -X POST -H "X-API-Key: …" --data-binary @small.mcap.zst https://…/jobs` returns `{job_id}`.
- `curl https://…/jobs/<id>` goes through `queued → stage_* → done`.
- `curl -o image.jpg https://…/jobs/<id>/image/0` retrieves a valid JPEG.

**Commit message:** `feat(modal): FastAPI submit / status / artifact endpoints`

**Review checkpoint:** subagent audits auth handling, path traversal
(job_id regex whitelist), streaming response backpressure.

---

## M6 — Cleanup sweeper + API key Secret

**Files:** `modal_app/app.py` (add scheduled function).

**Logic:**
- `@app.function(schedule=modal.Period(days=1), volumes={"/jobs": volume})`
- Iterate `/jobs/*/status.json`, check mtime; if > 7 days, rm dir.

**Secret provisioning (manual):**
- `modal secret create slam-api-key API_KEY=$(openssl rand -hex 32)`
- Record the value to a local gitignored `.env.modal` for dev; user can rotate later.

**Acceptance:**
- Deploy and confirm `modal app list` shows the scheduled function.
- Manual aging test: touch a fake `/jobs/j_old/status.json -d '8 days ago'`,
  trigger sweeper, confirm removal.

**Commit message:** `feat(modal): daily retention sweeper + api key secret`

---

## M7 — Full smoke on TEST_SCAN

**No code changes.** Upload the canonical TEST_SCAN bag. Note: TEST_SCAN's
rosbag is a `rosbag2` directory (`metadata.yaml` + `*.mcap`), so the client
must tar it first:
```
tar -C /mnt/c/.../TEST_SCAN -cf - rosbag | zstd -1 > scan.tar.zst
curl -X POST -H "X-API-Key: $KEY" --data-binary @scan.tar.zst \
     "https://…/jobs?filename=scan.tar.zst"
```
Compare `result.json` against expected counts:
- 1 sectional sofa
- 5–6 tables
- ≥6 dining chairs (out of 8 ground truth — SpatialLM typically merges a couple)
- Floorplan with 4+ walls and ≥1 door

If counts deviate materially: investigate whether it's the pipeline
(compare to local `rosbag_to_bboxes.py` output on same bag) or the Modal
wrapper (path/env drift).

**Acceptance:** numbers within expected range for TEST_SCAN.

**Commit message:** `docs(modal): TEST_SCAN smoke results`

---

## Summary of all files touched

### New (11)
- `modal_app/__init__.py`
- `modal_app/app.py`
- `modal_app/pipeline_runner.py`
- `modal_app/README.md`
- `cloud_slam/api/__init__.py`
- `cloud_slam/api/parse_outputs.py`
- `tests/test_parse_outputs.py`
- `tests/fixtures/sample_layout_merged.txt`
- `tests/fixtures/sample_best_views.json`
- `tests/fixtures/empty_best_views.json`
- `docs/superpowers/{specs,plans}/2026-04-24-modal-api-*.md` (this file + spec)

### Modified (2)
- `cloud_slam/spatiallm_pipeline/infer.py` — env-driven paths.
- `cloud_slam/requirements.txt` — add `modal`, `zstandard`, `python-multipart`, `fastapi[standard]`.

## Rollback

All work lives on `modal-api` branch. Merging to master is gated on
passing M7. Until then, `master` continues to build and run the local
pipeline exactly as today (M1's env-var defaults preserve behaviour).
