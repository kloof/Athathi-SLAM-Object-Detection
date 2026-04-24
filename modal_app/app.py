"""Modal app skeleton for cloud_slam_icp (M3 — image + verify_image only).

This module defines the Modal App, container Image, Volume, and API-key
Secret stub, plus a single `verify_image` function that performs an
offline load probe of the two SpatialLM checkpoints on an H100 and
returns a dict.

Pipeline runner and FastAPI endpoints are added in M4 / M5.

Reference:
- docs/superpowers/specs/2026-04-24-modal-api-design.md
- docs/superpowers/plans/2026-04-24-modal-api-plan.md §M3
- third_party/setup_spatiallm.sh (authoritative local install recipe)
"""
from __future__ import annotations

import modal


# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------
#
# Base: CUDA 12.4 devel Ubuntu 22.04 — flash-attn compiles against nvcc from
# the -devel tag; the -runtime tag does not carry nvcc and would break the
# Sonata encoder deps step. `add_python="3.10"` keeps us in lockstep with
# SpatialLM's pyproject pin (python = ">=3.10,<=3.12") and the local
# setup_spatiallm.sh venv.
#
# Step ordering is deliberate: low-churn steps (base image, apt, virtualenv
# bootstrap) come first; SpatialLM install comes before the main repo copy
# so an edit under cloud_slam/ does not invalidate the flash-attn layer
# (~15 min rebuild).

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10"
    )
    .apt_install(
        "build-essential",
        "cmake",
        "git",
        "libeigen3-dev",
        "libgl1",
        "libglib2.0-0",
        "curl",
    )
    # Make nvcc reachable by non-interactive shells (flash-attn's setup.py
    # calls `nvcc --version` to choose arch flags; the -devel image ships
    # nvcc at /usr/local/cuda/bin but may not add it to PATH in scripted
    # `run_commands` contexts).
    .env({
        "PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "CUDA_HOME": "/usr/local/cuda",
    })
    # NOTE: we deliberately do NOT create a venv.
    #
    # The local setup_spatiallm.sh uses a venv because the dev machine has
    # other Python environments to isolate from. A Modal container is
    # already isolated — a venv only adds a second interpreter path that
    # Modal's function body doesn't use, causing `import transformers` to
    # ImportError when the function runs under Modal's `add_python` Python
    # while the deps were installed into the venv's python.
    #
    # Install everything into the single Python that Modal's runtime uses.
    # `pip`, `python`, `python3` all resolve to Modal's add_python=3.10.
    #
    # We do NOT use `poetry install` on SpatialLM's pyproject.toml. First
    # attempt tried to — poetry 1.x completed the actual installs
    # successfully (torch 2.4.1+cu124, transformers, etc. all landed) but
    # then crashed in its post-install reporter (poetry-core API skew:
    # `'ProjectPackage' object has no attribute 'readme_content'` →
    # `ModuleNotFoundError: poetry.mixology.solutions`). The crash returned
    # non-zero and aborted the image build.
    #
    # Cleaner: pip-install SpatialLM's runtime deps directly, mirroring
    # third_party/SpatialLM/pyproject.toml `[tool.poetry.dependencies]`
    # with two deliberate omissions — poethepoet (dev/test tool that pulls
    # poetry back as a transitive dep, recreating the bug) and rerun-sdk
    # (visualization-only; unused by the pipeline). SpatialLM is used
    # as a source package via PYTHONPATH, not `pip install .`, so nothing
    # imports it as an installed distribution.
    .pip_install("huggingface_hub[cli]")
    .add_local_dir(
        "third_party/SpatialLM",
        remote_path="/opt/spatiallm",
        copy=True,
    )
    .run_commands(
        # SpatialLM runtime deps. Torch wheels pulled from pytorch's
        # cu124 index via --extra-index-url (equivalent to poetry's
        # supplemental-source setup in the skipped poetry install).
        "pip install --extra-index-url https://download.pytorch.org/whl/cu124 "
        "'torch==2.4.1+cu124' 'torchvision==0.19.1+cu124' 'torchaudio==2.4.1+cu124'",
        "pip install "
        "'transformers>=4.41.2,<=4.46.1' 'safetensors>=0.4.5,<0.5' "
        "'pandas>=2.2.3,<3' 'einops>=0.8.1,<0.9' 'numpy>=1.26,<2' "
        "'scipy>=1.15.2,<2' 'scikit-learn>=1.6.1,<2' 'toml>=0.10.2,<0.11' "
        "'tokenizers>=0.19.0,<0.20.4' 'huggingface_hub>=0.25.0' "
        "'shapely>=2.0.7,<3' 'bbox>=0.9.4,<1' 'terminaltables>=3.1.10,<4' "
        "'open3d>=0.18.0,<0.19' 'addict>=2.4.0,<3' "
        "'nvidia-cudnn-cu12' 'nvidia-nccl-cu12'",
        # Sonata encoder deps — mirrors third_party/setup_spatiallm.sh.
        "pip install ninja psutil timm",
        # flash-attn is the slow step (~15 min compile against torch 2.4.1+cu124).
        # MAX_JOBS=2 caps compile parallelism; flash-attn's per-job RSS can peak
        # ~10-14 GB and Modal builders are not guaranteed to have headroom for
        # the default (MAX_JOBS=nproc). OOM here would fail the image build.
        "MAX_JOBS=2 pip install flash-attn --no-build-isolation",
        "pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu124.html",
        # spconv-cu120 is binary-compatible with cu124 at runtime — this is
        # what the local working setup uses (see setup_spatiallm.sh).
        "pip install spconv-cu120",
    )
    # Bring the main repo in AFTER the heavy SpatialLM layer so edits to
    # cloud_slam/ do not bust flash-attn's cache.
    .add_local_dir(
        "cloud_slam",
        remote_path="/root/cloud_slam_icp/cloud_slam",
        copy=True,
    )
    .add_local_dir(
        "scripts",
        remote_path="/root/cloud_slam_icp/scripts",
        copy=True,
    )
    .add_local_dir(
        "calibration",
        remote_path="/root/cloud_slam_icp/calibration",
        copy=True,
    )
    .add_local_file(
        "cloud_slam/requirements.txt",
        remote_path="/root/cloud_slam_icp/requirements.txt",
        copy=True,
    )
    # Main-repo deps into the same Python. Any overlap with SpatialLM's
    # poetry install will be idempotent or upgrade-in-place.
    .run_commands(
        "pip install -r /root/cloud_slam_icp/requirements.txt",
    )
    # Pre-download HuggingFace weights into the image's baked-in cache.
    # HF_HOME must be set BEFORE the download so the files land where
    # runtime will look for them. Modal's .env() applies to subsequent
    # build steps in the same chain.
    .env({"HF_HOME": "/root/.cache/hf"})
    .run_commands(
        "mkdir -p /root/.cache/hf",
        "huggingface-cli download manycore-research/SpatialLM1.1-Qwen-0.5B",
        "huggingface-cli download manycore-research/SpatialLM1.1-Llama-1B",
        # Build-time verification probe — layer 1 of the three-layer
        # model-cache verification (see spec §Model-cache verification).
        # Runs offline (HF_HUB_OFFLINE=1, local_files_only=True); if either
        # checkpoint's tokenizer/config is not loadable from the on-disk
        # cache, the shell exits non-zero and Modal aborts the image build.
        # Written to a file first so we don't have to battle shell-quote
        # rules for a multi-line python -c "..." arg.
        (
            "cat > /tmp/verify_cache.py <<'PY'\n"
            "from transformers import AutoTokenizer, AutoConfig\n"
            "for m in ['manycore-research/SpatialLM1.1-Qwen-0.5B','manycore-research/SpatialLM1.1-Llama-1B']:\n"
            "    AutoTokenizer.from_pretrained(m, local_files_only=True)\n"
            "    AutoConfig.from_pretrained(m, local_files_only=True)\n"
            "print('cache verified')\n"
            "PY"
        ),
        "HF_HUB_OFFLINE=1 python /tmp/verify_cache.py",
    )
    # Capture the resolved python path at a stable absolute location so
    # cloud_slam/spatiallm_pipeline/infer.py can launch SpatialLM via
    # subprocess without depending on PATH inheritance. `ln -sf $(which
    # python3) ...` resolves the Modal-provided python at build time.
    .run_commands(
        "ln -sf $(which python3) /usr/local/bin/cloud_slam_python",
    )
    .env(
        {
            "SPATIALLM_DIR": "/opt/spatiallm",
            "SPATIALLM_PY": "/usr/local/bin/cloud_slam_python",
            "SPATIALLM_CODE_TEMPLATE": "/opt/spatiallm/code_template.txt",
            # /opt/spatiallm is included so `import spatiallm` works when
            # SpatialLM's inference.py is subprocess-launched from cwd=/opt/spatiallm
            # (we skipped the poetry install that would have otherwise
            # registered it as a site-packages distribution).
            "PYTHONPATH": "/root/cloud_slam_icp:/opt/spatiallm",
            "HF_HUB_OFFLINE": "1",
        }
    )
    .workdir("/root/cloud_slam_icp")
)


# ---------------------------------------------------------------------------
# Volume & Secret
# ---------------------------------------------------------------------------
#
# Volume holds per-job state at /jobs/<job_id>/ — see spec §"Volume layout".
# create_if_missing lets the first deploy provision it without a manual
# `modal volume create` step.
volume = modal.Volume.from_name("slam-jobs", create_if_missing=True)

# API-key Secret stub. Populated via:
#     modal secret create slam-api-key API_KEY=$(openssl rand -hex 32)
# Not attached to verify_image (no user input in that path), but declared
# here so M4/M5 can reference it without circular-import gymnastics.
api_key_secret = modal.Secret.from_name("slam-api-key")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = modal.App("cloud-slam-icp")


# ---------------------------------------------------------------------------
# verify_image — manual preflight smoke (layer 3 of model-cache verification)
# ---------------------------------------------------------------------------
#
# Run with:
#     modal run modal_app/app.py::verify_image
#
# Spins one H100 container, does an offline load probe of both models,
# reports CUDA + device, returns a dict. ~$0.01 per run; recommended
# after every fresh `modal deploy`.
#
# retries=0 is a hard cost-safety rail — a flaky model load should fail
# loudly once, not burn two H100-minutes silently (spec §"Cost safety" #2).
# Volume and Secret are intentionally NOT attached: this probe only reads
# the baked-in image cache and has no user-supplied input.
@app.function(image=image, gpu="H100", retries=0, timeout=600)
def verify_image() -> dict:
    """Offline load probe for both SpatialLM checkpoints.

    Returns a dict of the form::

        {
            "qwen":   "ok" | "failed: <ExceptionType>: <msg>",
            "llama":  "ok" | "failed: <ExceptionType>: <msg>",
            "cuda":   bool,
            "device": str | None,
        }

    Any entry other than ``"ok"`` means the image's HF cache is broken for
    that checkpoint and the deploy should be re-built before accepting
    real work.
    """
    import os

    # Belt-and-braces — the image already sets HF_HUB_OFFLINE=1, but if a
    # future change moves that env var, we still want this probe to prove
    # offline loadability.
    os.environ["HF_HUB_OFFLINE"] = "1"

    from transformers import AutoConfig, AutoTokenizer

    out: dict[str, object] = {}
    for name, repo in [
        ("qwen", "manycore-research/SpatialLM1.1-Qwen-0.5B"),
        ("llama", "manycore-research/SpatialLM1.1-Llama-1B"),
    ]:
        try:
            AutoTokenizer.from_pretrained(repo, local_files_only=True)
            AutoConfig.from_pretrained(repo, local_files_only=True)
            out[name] = "ok"
        except Exception as e:  # pragma: no cover — runs in Modal container
            out[name] = f"failed: {type(e).__name__}: {e}"

    import torch

    out["cuda"] = torch.cuda.is_available()
    out["device"] = torch.cuda.get_device_name(0) if out["cuda"] else None
    return out


# ---------------------------------------------------------------------------
# pipeline_runner — background Modal Function (M4)
# ---------------------------------------------------------------------------
#
# The submit endpoint (M5) calls `pipeline_runner.spawn(job_id)`. All real
# logic lives in `modal_app/pipeline_runner.py::run` to keep this file
# scanning-friendly and to keep the heavy cloud_slam import graph off the
# top-level module (it would otherwise load on every ASGI cold-start).
#
# Knobs:
# - `gpu="H100"` — SpatialLM-Llama decode is memory-bandwidth-bound at
#   batch=1; H100 at 3350 GB/s wins ~5× over the local 4070 Ti. See spec
#   §"Decisions locked in" for the L4/A10G rejection rationale.
# - `cpu=8.0, memory=32768` — KISS-ICP is CPU-bound, SLAM stage needs
#   headroom for Open3D point-cloud ops.
# - `retries=0` — hard cost-safety rail (spec §"Cost safety" row 2). A
#   flaky run fails once; Modal must not silently double-bill us.
# - `timeout=3600` — outer cap. Inner subprocess timeout is 3300 s so we
#   always raise `TimeoutExpired` first and log a clean pipeline_timeout.
# - `max_containers=2` — caps simultaneous H100 spend (spec §"Cost safety"
#   row 5; spec section §"Decisions locked in" mentions a 4-container
#   plan — 2 is the deliberately tighter M4 default, easy to raise later).
@app.function(
    image=image,
    gpu="H100",
    cpu=8.0,
    memory=32768,
    volumes={"/jobs": volume},
    retries=0,
    timeout=3600,
    max_containers=2,
)
def pipeline_runner(job_id: str) -> None:
    """Background runner invoked by the submit endpoint (M5) via `.spawn`.

    Lazy-imports `modal_app.pipeline_runner.run` so the heavy cloud_slam
    + SpatialLM module graph stays off the top level of `app.py`.
    """
    from modal_app.pipeline_runner import run

    run(volume=volume, job_id=job_id)


# ---------------------------------------------------------------------------
# web — FastAPI ASGI endpoints (M5)
# ---------------------------------------------------------------------------
#
# Endpoints: POST /jobs, GET /jobs/{id}, GET /jobs/{id}/image/{idx},
# GET /jobs/{id}/artifact/{name}, DELETE /jobs/{id}, GET /health.
# All routing + logic lives in modal_app.web.build_web_app so it's
# testable without a Modal container (see tests/test_web.py).
#
# Knobs:
# - `timeout=900` — 15-min wall cap per request. Uploads stream; the
#   cap is mostly for runaway handlers, not body I/O.
# - `max_containers=4` — cap warm ASGI containers independently of the
#   heavier pipeline_runner's `max_containers=2`.
# - `@modal.concurrent(max_inputs=50)` — lets one ASGI worker handle
#   many in-flight polling GETs concurrently; POST /jobs is the only
#   write-heavy path and it's still serialized per-container by the
#   streaming body read.
# - `api_key_secret_value` reads from env because the Secret
#   `slam-api-key` stores the key under `API_KEY` by Modal convention.
#   If the secret is not yet populated, `os.environ.get("API_KEY")`
#   returns None and the endpoint code treats every request as "no
#   auth configured yet" — accepts any non-empty X-API-Key header.
#   See modal_app/web.py module docstring for the full rule.
# NOTE: Modal rejects `retries=0` on web endpoints ("Web endpoints do not
# support retries"). Retry behaviour for an ASGI app is the client's job.
@app.function(
    image=image,
    volumes={"/jobs": volume},
    secrets=[api_key_secret],
    timeout=900,
    max_containers=4,
)
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def web():
    import os

    from modal_app.web import build_web_app

    return build_web_app(
        volume=volume,
        runner_fn=pipeline_runner,
        api_key_secret_value=os.environ.get("API_KEY"),
    )


# ---------------------------------------------------------------------------
# retention_sweeper — daily cleanup of aged job dirs (M6)
# ---------------------------------------------------------------------------
#
# Deletes any /jobs/<job_id>/ directory whose status.json mtime is older
# than 7 days. Idempotency records under /jobs/_idempotency/ are swept by
# the same rule (their mtime reflects last write, not last read — a client
# retrying under an old idempotency key is the only way to keep a record
# alive, which is exactly the correct TTL semantics).
#
# Runs on CPU only (no GPU attached), so the cost is negligible —
# order of $0.001 per daily run. retries=0 because a one-day miss is
# harmless; the next run will pick up what this one missed.
@app.function(
    volumes={"/jobs": volume},
    schedule=modal.Period(days=1),
    retries=0,
    timeout=600,
)
def retention_sweeper() -> dict:
    """Remove /jobs/<id>/ dirs older than RETENTION_DAYS.

    Returns a summary dict for observability via `modal app logs`.
    """
    import shutil
    import time
    from pathlib import Path

    RETENTION_DAYS = 7
    cutoff = time.time() - RETENTION_DAYS * 86400

    jobs_root = Path("/jobs")
    volume.reload()

    removed: list[str] = []
    kept: list[str] = []
    errors: list[dict] = []

    for entry in sorted(jobs_root.iterdir()):
        if not entry.is_dir():
            continue
        # Consider both real job dirs (j_YYYY-MM-DD_xxxxxxxx) and the
        # idempotency bookkeeping dir. We key on status.json if present,
        # otherwise on the dir's own mtime.
        status_file = entry / "status.json"
        try:
            mtime = (
                status_file.stat().st_mtime
                if status_file.is_file()
                else entry.stat().st_mtime
            )
        except OSError as e:
            errors.append({"path": str(entry), "error": str(e)})
            continue

        if mtime >= cutoff:
            kept.append(entry.name)
            continue

        try:
            shutil.rmtree(entry)
            removed.append(entry.name)
        except OSError as e:
            errors.append({"path": str(entry), "error": str(e)})

    if removed or errors:
        volume.commit()

    summary = {
        "retention_days": RETENTION_DAYS,
        "removed_count": len(removed),
        "kept_count": len(kept),
        "errors_count": len(errors),
        "removed": removed[:100],     # cap log size
        "errors": errors[:20],
    }
    print(f"[retention_sweeper] {summary}")
    return summary
