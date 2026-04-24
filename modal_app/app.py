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
    # Bootstrap: virtualenv only. huggingface-cli must be installed INTO
    # the SpatialLM venv later so /opt/spatiallm_env/bin/huggingface-cli
    # actually exists (the HF download + build-time probe call it by path).
    .pip_install("virtualenv")
    .run_commands("virtualenv /opt/spatiallm_env")
    # Copy SpatialLM source into the image, then run the poetry-driven
    # install (torch 2.4.1+cu124, transformers, etc.) inside the venv.
    .add_local_dir(
        "third_party/SpatialLM",
        remote_path="/opt/spatiallm",
        copy=True,
    )
    .run_commands(
        # Pin poetry <2 — poetry 2.x changed supplemental-source resolution
        # and may skip the pytorch cu124 index during lock-less install,
        # causing "could not find torch 2.4.1+cu124". 1.x is what the local
        # setup_spatiallm.sh was validated on.
        "/opt/spatiallm_env/bin/pip install 'poetry<2.0'",
        "/opt/spatiallm_env/bin/pip install 'huggingface_hub[cli]'",
        # poetry reads /opt/spatiallm/pyproject.toml; `virtualenvs.create
        # false` forces it to install into the active venv
        # (/opt/spatiallm_env) instead of a nested one.
        "cd /opt/spatiallm && /opt/spatiallm_env/bin/python -m poetry config virtualenvs.create false --local && /opt/spatiallm_env/bin/python -m poetry install --no-interaction",
        # Sonata encoder deps — mirrors third_party/setup_spatiallm.sh.
        "/opt/spatiallm_env/bin/pip install ninja psutil timm",
        # flash-attn is the slow step (~15 min compile against torch 2.4.1+cu124).
        # MAX_JOBS=2 caps compile parallelism; flash-attn's per-job RSS can peak
        # ~10-14 GB and Modal builders are not guaranteed to have headroom for
        # the default (MAX_JOBS=nproc). OOM here would fail the image build.
        "MAX_JOBS=2 /opt/spatiallm_env/bin/pip install flash-attn --no-build-isolation",
        "/opt/spatiallm_env/bin/pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu124.html",
        # spconv-cu120 is binary-compatible with cu124 at runtime — this is
        # what the local working setup uses (see setup_spatiallm.sh).
        "/opt/spatiallm_env/bin/pip install spconv-cu120",
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
    # Install main-repo deps INTO the SpatialLM venv. One venv for
    # everything — when infer.py spawns SpatialLM via subprocess with
    # SPATIALLM_PY pointed at this interpreter, both sides use the same
    # transformers / torch build.
    .run_commands(
        "/opt/spatiallm_env/bin/pip install -r /root/cloud_slam_icp/requirements.txt",
    )
    # Pre-download HuggingFace weights into the image's baked-in cache.
    # HF_HOME must be set BEFORE the download so the files land where
    # runtime will look for them. Modal's .env() applies to subsequent
    # build steps in the same chain.
    .env({"HF_HOME": "/root/.cache/hf"})
    .run_commands(
        "mkdir -p /root/.cache/hf",
        "/opt/spatiallm_env/bin/huggingface-cli download manycore-research/SpatialLM1.1-Qwen-0.5B",
        "/opt/spatiallm_env/bin/huggingface-cli download manycore-research/SpatialLM1.1-Llama-1B",
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
        "HF_HUB_OFFLINE=1 /opt/spatiallm_env/bin/python /tmp/verify_cache.py",
    )
    .env(
        {
            "SPATIALLM_DIR": "/opt/spatiallm",
            "SPATIALLM_PY": "/opt/spatiallm_env/bin/python",
            "SPATIALLM_CODE_TEMPLATE": "/opt/spatiallm/code_template.txt",
            "PYTHONPATH": "/root/cloud_slam_icp",
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
