"""Background pipeline runner for the Modal API (M4).

Entry point: :func:`run`. Called by ``modal_app.app.pipeline_runner``
(which does nothing but ``volume`` wiring + a thin ``.spawn`` target) so
that the heavy imports (cloud_slam, SpatialLM venv glue) stay lazy.

Contract — executed in order:

1. ``verify_cache_or_die()``  runtime startup probe (layer 2 of the
   three-layer model-cache verification, spec §"Model-cache verification").
2. Resolve ``job_dir = /jobs/<job_id>``, read ``submitted_at`` from any
   pre-existing ``status.json`` written by the submit endpoint.
3. Decode input via :func:`modal_app.decode.resolve_mcap`
   (handles ``.mcap`` / ``.mcap.zst`` / ``.tar`` / ``.tar.zst``).
4. Validate MCAP magic bytes.
5. Transition status to ``stage_0_slam`` (with ``progress_hint=0.05``)
   and shell out to ``scripts/rosbag_to_bboxes.py``. That script runs
   stages 0-7 internally; we cannot observe per-stage progress from
   here without intrusive changes to the script. Flagged in the code.
6. On ``subprocess.TimeoutExpired``  ``error.type="pipeline_timeout"``.
7. On non-zero exit  ``error.type="pipeline_error"`` with last 4 KiB
   of stderr+stdout captured.
8. On success  :func:`cloud_slam.api.parse_outputs.build_result_json`,
   write ``result.json``, flip status to ``done``.

Every persistent write is followed by ``volume.commit()`` via
:class:`StatusWriter` / :func:`write_error`  the polling endpoint lives
in another container and sees nothing until we commit.
"""
from __future__ import annotations

import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from modal_app.decode import (
    DecodeError,
    InvalidMcapError,
    resolve_mcap,
    validate_mcap_magic,
)
from modal_app.status import StatusWriter, write_error


# Inner subprocess timeout. Must be less than the outer Modal function
# timeout (3600 s) so we always hit TimeoutExpired first and log a clean
# pipeline_timeout rather than Modal's generic container-killed signal.
# Spec §"Cost safety" row 1.
SUBPROCESS_TIMEOUT_S = 3300

# Tail size for stderr/stdout captured on pipeline error. Spec §"Error
# handling" says "last 4 KB" but asks for last 4096 chars of stderr+stdout
# combined in M4 deliverable body  go with 4096 chars each to keep
# debuggability high.
STDERR_TAIL_CHARS = 4096


def run(*, volume: Any, job_id: str) -> None:
    """Execute the full pipeline for ``job_id`` on the attached volume.

    ``volume`` is a ``modal.Volume`` handle (or any duck-typed object with
    a ``.commit()`` method for tests). All state lives under
    ``/jobs/<job_id>/``.
    """
    job_dir = Path("/jobs") / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = job_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    writer = StatusWriter(volume, job_dir)

    # Preserve submitted_at from the submit endpoint's initial status.json.
    # If missing (e.g. running the function directly for a dev smoke),
    # fall back to now().
    submitted_at = _read_submitted_at(job_dir) or datetime.utcnow()

    # -------------------------------------------------------------------
    # Step 1  startup cache probe (BEFORE any GPU work)
    # -------------------------------------------------------------------
    try:
        verify_cache_or_die()
    except Exception as exc:
        write_error(
            volume,
            job_dir,
            error_type="image_cache_broken",
            message=f"SpatialLM weights not loadable offline: {exc}",
            stage="verify_cache",
            traceback=traceback.format_exc(),
        )
        return

    # -------------------------------------------------------------------
    # Step 2/3  decode input
    # -------------------------------------------------------------------
    writer.write(status="decoding", progress_hint=0.01)

    try:
        input_mcap = resolve_mcap(job_dir)
    except DecodeError as exc:
        write_error(
            volume,
            job_dir,
            error_type="decode_error",
            message=str(exc),
            stage="decoding",
            traceback=traceback.format_exc(),
        )
        return
    except Exception as exc:  # unexpected decode crash
        write_error(
            volume,
            job_dir,
            error_type="decode_error",
            message=f"unexpected error during decode: {exc}",
            stage="decoding",
            traceback=traceback.format_exc(),
        )
        return

    # -------------------------------------------------------------------
    # Step 4  MCAP magic-byte validation
    # -------------------------------------------------------------------
    try:
        validate_mcap_magic(input_mcap)
    except InvalidMcapError as exc:
        write_error(
            volume,
            job_dir,
            error_type="invalid_mcap",
            message=str(exc),
            stage="validate_mcap",
            traceback=traceback.format_exc(),
        )
        return

    # -------------------------------------------------------------------
    # Step 5  run the pipeline
    # -------------------------------------------------------------------
    # NOTE: scripts/rosbag_to_bboxes.py runs ALL 8 stages (0-7) internally
    # in a single process. We can't observe per-stage transitions from
    # here without intrusive changes to that script (out of scope for M4).
    # So we set a single "stage_0_slam" marker with a low progress_hint
    # and leave it until the subprocess returns. Per-stage live updates
    # require hooking the script's stage-completion callbacks in a later
    # milestone.
    writer.write(
        status="stage_0_slam",
        progress_hint=0.05,
        started_at=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    env = {**os.environ, "PYTHONPATH": "/root/cloud_slam_icp"}
    cmd = [
        sys.executable,
        "/root/cloud_slam_icp/scripts/rosbag_to_bboxes.py",
        str(input_mcap),
        str(artifacts_dir),
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_S,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        # exc.stdout/stderr are bytes-or-None; coerce to safe tail strings.
        write_error(
            volume,
            job_dir,
            error_type="pipeline_timeout",
            message=f"subprocess did not return within {SUBPROCESS_TIMEOUT_S}s",
            stage="stage_0_slam",  # last known stage marker we wrote
            stderr_tail=_tail(_to_text(exc.stderr)),
            traceback=traceback.format_exc(),
            extra={"stdout_tail": _tail(_to_text(exc.stdout))},
        )
        return
    except Exception as exc:
        # subprocess.run itself raised (OSError, etc.) before the child ran.
        write_error(
            volume,
            job_dir,
            error_type="pipeline_error",
            message=f"failed to launch pipeline subprocess: {exc}",
            stage="stage_0_slam",
            traceback=traceback.format_exc(),
        )
        return

    if proc.returncode != 0:
        write_error(
            volume,
            job_dir,
            error_type="pipeline_error",
            message=f"rosbag_to_bboxes.py exited with returncode {proc.returncode}",
            stage=_last_known_stage(job_dir),
            returncode=proc.returncode,
            stderr_tail=_tail(proc.stderr or ""),
            extra={"stdout_tail": _tail(proc.stdout or "")},
        )
        return

    # -------------------------------------------------------------------
    # Step 8  success path
    # -------------------------------------------------------------------
    try:
        # Lazy import: parse_outputs pulls in cloud_slam.spatiallm_pipeline.merge,
        # which is heavy and only available in the Modal image / main venv.
        from cloud_slam.api.parse_outputs import build_result_json

        finished_at = datetime.utcnow()
        envelope = build_result_json(
            output_dir=artifacts_dir,
            job_id=job_id,
            submitted_at=submitted_at,
            finished_at=finished_at,
        )
    except Exception as exc:
        write_error(
            volume,
            job_dir,
            error_type="pipeline_error",
            message=f"result-envelope build failed: {exc}",
            stage="build_result_json",
            traceback=traceback.format_exc(),
        )
        return

    _write_result_json(job_dir, envelope)
    writer.write(
        status="done",
        progress_hint=1.0,
        finished_at=finished_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


# ---------------------------------------------------------------------------
# Runtime cache probe (layer 2)
# ---------------------------------------------------------------------------

def verify_cache_or_die() -> None:
    """Offline-load both SpatialLM checkpoints; raise on any failure.

    Guards against the case where the image build's cache verification
    step passed but the mounted cache dir in this container is different
    or incomplete. A failure here writes ``error.type=image_cache_broken``
    in the caller and aborts before any GPU work starts.
    """
    # Belt-and-braces  the image already sets HF_HUB_OFFLINE=1.
    os.environ["HF_HUB_OFFLINE"] = "1"

    # transformers is a heavy import (~1 s cold); keep it inside this
    # function so unit tests that mock this function never pay for it.
    from transformers import AutoConfig, AutoTokenizer

    for repo in (
        "manycore-research/SpatialLM1.1-Qwen-0.5B",
        "manycore-research/SpatialLM1.1-Llama-1B",
    ):
        AutoTokenizer.from_pretrained(repo, local_files_only=True)
        AutoConfig.from_pretrained(repo, local_files_only=True)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _read_submitted_at(job_dir: Path) -> datetime | None:
    """Return the ISO-Z ``submitted_at`` from ``status.json`` as a datetime."""
    import json as _json

    status_path = job_dir / "status.json"
    if not status_path.is_file():
        return None
    try:
        raw = _json.loads(status_path.read_text())
    except (_json.JSONDecodeError, OSError):
        return None
    ts = raw.get("submitted_at")
    if not isinstance(ts, str):
        return None
    # Accept both "...Z" and an RFC3339 form with +00:00.
    ts_clean = ts[:-1] if ts.endswith("Z") else ts
    try:
        return datetime.fromisoformat(ts_clean)
    except ValueError:
        return None


def _last_known_stage(job_dir: Path) -> str | None:
    """Peek ``status.json`` for its current ``status`` field."""
    import json as _json

    status_path = job_dir / "status.json"
    if not status_path.is_file():
        return None
    try:
        raw = _json.loads(status_path.read_text())
    except (_json.JSONDecodeError, OSError):
        return None
    val = raw.get("status")
    return val if isinstance(val, str) else None


def _tail(s: str, n: int = STDERR_TAIL_CHARS) -> str:
    """Return the last ``n`` chars of ``s`` (empty string passes through)."""
    if not s:
        return ""
    return s[-n:]


def _to_text(maybe_bytes: Any) -> str:
    """Coerce ``subprocess.TimeoutExpired.stderr/stdout`` (bytes|str|None) to str."""
    if maybe_bytes is None:
        return ""
    if isinstance(maybe_bytes, bytes):
        return maybe_bytes.decode("utf-8", errors="replace")
    return str(maybe_bytes)


def _write_result_json(job_dir: Path, envelope: dict) -> None:
    """Persist the API result envelope to ``<job_dir>/result.json``."""
    import json as _json

    out = job_dir / "result.json"
    tmp = out.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        _json.dump(envelope, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out)
