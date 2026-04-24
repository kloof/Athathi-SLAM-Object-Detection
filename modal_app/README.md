# modal_app

Modal deploy target for the cloud_slam_icp pipeline.

M3 state: image + Volume + Secret stub + `verify_image` smoke test only.
Pipeline runner (M4) and FastAPI endpoints (M5) are not yet wired.

## Prerequisites

- `modal` installed locally and a token set (`modal token new`).
- API-key Secret created (once per workspace):

```bash
modal secret create slam-api-key API_KEY=$(openssl rand -hex 32)
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

## Smoke test (layer 3)

After every fresh deploy, run the manual verify function before trusting
the image:

```bash
modal run modal_app/app.py::verify_image
```

Expected output (printed by the Modal CLI as the function's return value):

```python
{
    'qwen': 'ok',
    'llama': 'ok',
    'cuda': True,
    'device': 'NVIDIA H100 80GB HBM3',  # or H200 when Modal auto-upgrades
}
```

Cost is ~$0.01 per invocation (one H100 container, cold start ~30 s, load
probe runs in a few seconds). Any value other than `'ok'` on `qwen` or
`llama` means the baked HF cache is corrupt and the deploy must be
rebuilt.

## What's NOT here yet

- `POST /jobs`, `GET /jobs/{id}`, `GET /jobs/{id}/image/{idx}`,
  `GET /jobs/{id}/artifact/{name}` — M5.
- `pipeline_runner` background function that wraps `rosbag_to_bboxes.py` — M4.
- Daily retention sweeper — M6.

See `docs/superpowers/plans/2026-04-24-modal-api-plan.md` for the full
milestone breakdown.
