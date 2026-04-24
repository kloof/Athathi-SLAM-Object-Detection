# modal_app

Modal deploy target for the cloud_slam_icp pipeline.

End-to-end: POST a rosbag (`.mcap`, `.mcap.zst`, `.tar`, or `.tar.zst`),
poll until done, fetch a JSON result with the detected floorplan,
furniture bounding boxes, and best-view image URLs.

## Prerequisites

- `modal` installed locally; token set (`modal token new` or
  `modal token set --token-id … --token-secret …`).
- API-key Secret created (once per workspace):

```bash
API_KEY=$(openssl rand -hex 32)
modal secret create slam-api-key API_KEY=$API_KEY
echo $API_KEY > ~/.cloud_slam_icp_api_key   # save it — the secret is not re-readable
```

The Volume `slam-jobs` is created on first deploy (`create_if_missing=True`).

## Deploy

From the repo root (so `add_local_dir("third_party/SpatialLM", ...)` and
friends resolve):

```bash
modal deploy modal_app/app.py
```

First deploy takes ~20 min — flash-attn compiles against torch 2.4.1+cu124
and the two SpatialLM checkpoints (~2 GB) are pre-downloaded into the
image's HF cache. Subsequent deploys reuse the flash-attn / weights
layers unless you edit `third_party/SpatialLM`.

The image build aborts loudly if either checkpoint's tokenizer or config
is not loadable offline after download — this is the build-time probe,
layer 1 of three (see `docs/superpowers/specs/2026-04-24-modal-api-design.md`
§Model-cache verification).

## Smoke tests

### Layer 3 — `verify_image` on H100 (~$0.01)

```bash
modal run modal_app/app.py::verify_image
```

Expected output:

```python
{
    'qwen': 'ok',
    'llama': 'ok',
    'cuda': True,
    'device': 'NVIDIA H100 80GB HBM3',  # or H200 when Modal auto-upgrades
}
```

Anything other than `'ok'` for qwen/llama means the baked HF cache is
corrupt — rebuild the image.

### `/health` (free, no GPU)

```bash
curl https://<workspace>--cloud-slam-icp-web.modal.run/health
```

Returns `{status: "ok", uptime_s, image_env: {...}}`. Use this to
confirm the ASGI container is alive before posting a bag.

## Submitting a rosbag

### Single MCAP file (`.mcap` or `.mcap.zst`)

```bash
# Compress your bag with zstd -1 for faster uploads (~2-3x smaller):
zstd -1 scan.mcap   # -> scan.mcap.zst

curl -X POST \
     -H "X-API-Key: $(cat ~/.cloud_slam_icp_api_key)" \
     --data-binary @scan.mcap.zst \
     "https://<workspace>--cloud-slam-icp-web.modal.run/jobs?filename=scan.mcap.zst"
```

### ROS2 rosbag2 directory

If your recorder emits a `metadata.yaml + *.mcap` directory (e.g.
`ros2 bag record -s mcap`), tar it first — the server extracts the
first `*.mcap` inside:

```bash
tar -C /path/to/parent -cf - rosbag | zstd -1 > scan.tar.zst
curl -X POST \
     -H "X-API-Key: $(cat ~/.cloud_slam_icp_api_key)" \
     --data-binary @scan.tar.zst \
     "https://<workspace>--cloud-slam-icp-web.modal.run/jobs?filename=scan.tar.zst"
```

Response:

```json
{ "job_id": "j_2026-04-24_ab12cd34", "status_url": "...", "submitted_at": "..." }
```

### Polling

```bash
JOB=j_2026-04-24_ab12cd34
API=https://<workspace>--cloud-slam-icp-web.modal.run
KEY=$(cat ~/.cloud_slam_icp_api_key)

while true; do
    curl -sH "X-API-Key: $KEY" "$API/jobs/$JOB" | jq '.status'
    STATUS=$(curl -sH "X-API-Key: $KEY" "$API/jobs/$JOB" | jq -r '.status')
    [[ "$STATUS" == "done" || "$STATUS" == "failed" ]] && break
    sleep 3     # server suggests Retry-After: 3
done

curl -sH "X-API-Key: $KEY" "$API/jobs/$JOB" > result.json
```

### Idempotent submission (safe client retry)

```bash
curl -X POST \
     -H "X-API-Key: $KEY" \
     -H "X-Idempotency-Key: $(uuidgen)" \
     --data-binary @scan.mcap.zst \
     "$API/jobs?filename=scan.mcap.zst"
```

If the same key is reused within 24 h, the server returns the original
`job_id` with `reused: true` instead of starting a new run.

## Artifact downloads

```bash
# Best-view JPEGs (index into result.best_images[])
curl -o img0.jpg -H "X-API-Key: $KEY" "$API/jobs/$JOB/image/0"

# Whitelisted artifact files
curl -o scene_with_boxes.ply -H "X-API-Key: $KEY" "$API/jobs/$JOB/artifact/scene_with_boxes.ply"
curl -o layout_merged.txt    -H "X-API-Key: $KEY" "$API/jobs/$JOB/artifact/layout_merged.txt"
```

## Cancel + cleanup

```bash
curl -X DELETE -H "X-API-Key: $KEY" "$API/jobs/$JOB"
```

Cancels the in-flight Modal Function and removes the job dir from the
volume. Without explicit DELETE, jobs auto-purge after 7 days via the
daily `retention_sweeper` scheduled function.

## Cost per rosbag

Roughly $0.50–0.75 per run at TEST_SCAN scale:

- H100 (auto-upgraded to H200 when available) for 5–10 min pipeline time.
- 8 vCPU + 32 GB RAM attached while SLAM runs (CPU-bound).
- Volume storage ~3 GB × 7 days ≈ $0.02.

Runner is hard-capped at `max_containers=2` and `retries=0` —
concurrent spend cannot exceed 2 H100s, and a failed job fails once
(no auto-retry double billing).

## Troubleshooting

- **Image build fails at flash-attn step.** Usually `MAX_JOBS=2` isn't
  enough RAM headroom. Drop to `MAX_JOBS=1` in `app.py` and redeploy.
- **`modal run ::verify_image` returns `'failed: ...'`.** HF cache is
  corrupt. Force a fresh build: `modal deploy --force modal_app/app.py`.
- **POST returns 401 unexpectedly.** Check the secret was actually
  populated: `modal secret list`. An unpopulated secret makes every
  non-empty key succeed (first-deploy convenience), so the opposite
  surprise is more common.
- **`GET /jobs/{id}` reads stale status.** `volume.reload()` is called
  at the top of every GET, so staleness >1 s usually means the runner
  missed a `volume.commit()` — check `modal app logs`.
