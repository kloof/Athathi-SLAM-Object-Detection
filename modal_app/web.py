"""FastAPI endpoints for the Modal API (M5).

This module is deliberately decoupled from the ``modal`` SDK so it can
be unit-tested without a Modal container. All Modal-specific imports
(``modal.FunctionCall``) are deferred into the function bodies that
need them; a ``try``/``except ImportError`` in the DELETE path lets
tests run without ``modal`` installed.

Entry point: :func:`build_web_app`. The Modal-side glue lives in
``modal_app/app.py`` and simply wires a real ``modal.Volume``,
:func:`pipeline_runner`, and the secret-sourced API key into this
factory.

``api_key_secret_value`` contract
---------------------------------

The Modal Secret ``slam-api-key`` stores the API key under the env var
``API_KEY``. The factory reads ``os.environ.get("API_KEY")`` at import
time in ``app.py`` and passes it here.

If the secret is not yet populated (first deploy, before the operator
has run ``modal secret create slam-api-key ...``), ``API_KEY`` is
``None``. To unblock first-deploy smoke tests without introducing a
security regression when the secret IS set, we adopt the following
rule:

- If ``api_key_secret_value`` is ``None``: accept ANY non-empty
  ``X-API-Key`` header (including garbage) — the deployment is
  unauthenticated but "show me something" still works.
- If ``api_key_secret_value`` is set: constant-time compare against
  the client header; mismatch → 401.
- In both modes, a MISSING or EMPTY ``X-API-Key`` header → 401.
  ("Accept anything" is not "accept nothing" — that would let an
  anonymous drive-by start a GPU run.)

This behaviour is loud by design; once the secret is set, regressing
to the accept-any-key mode requires actively unsetting it.
"""
from __future__ import annotations

import hmac
import json
import os
import re
import secrets
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator

from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Path as FPath,
    Query,
    Request,
    Response,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from modal_app.idempotency import check_or_record


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Spec §"Auth": job_id regex whitelist on every path param. Prevents path
# traversal (any `..` / `/` in the param would fail the match).
JOB_ID_PATTERN = r"^j_\d{4}-\d{2}-\d{2}_[0-9a-f]{8}$"
_JOB_ID_RX = re.compile(JOB_ID_PATTERN)

# Spec §"Upload": accept raw bytes up to 4 GiB. Enforced by a running
# byte counter on the streaming write so we don't load the whole body
# into memory before rejecting it (FastAPI's ``await request.body()``
# would buffer everything — we intentionally use ``stream()``).
MAX_UPLOAD_BYTES = 4 * 1024 * 1024 * 1024  # 4 GiB

# The four suffixes the runner knows how to decode.
_ALLOWED_SUFFIXES = (".mcap.zst", ".tar.zst", ".mcap", ".tar")

# Whitelist for GET /jobs/{id}/artifact/{name} — anything not listed
# here returns 404 (not 403; we don't want to leak file existence).
_ARTIFACT_WHITELIST: dict[str, str] = {
    "colored_map.ply": "artifacts/slam/colored_map.ply",
    "scene_with_boxes.ply": "artifacts/scene_with_boxes.ply",
    "layout_merged.txt": "artifacts/layout_merged.txt",
    "result.json": "result.json",
}

# Chunk size for StreamingResponse body reads + upload writes. 1 MiB is
# the same size the decode helper uses; big enough to amortize syscall
# overhead, small enough to avoid GPU-container memory pressure.
_CHUNK = 1 << 20  # 1 MiB

# ``Retry-After`` seconds while the job is still running. Matches spec
# §"Endpoint contract" ("Retry-After: 3 while status != done/failed").
RETRY_AFTER_S = "3"

# Jobs root mount point on the Modal Volume. Override via the
# ``CLOUD_SLAM_JOBS_ROOT`` env var for tests (TestClient points it at
# a ``tempfile.TemporaryDirectory``).
def _jobs_root() -> Path:
    return Path(os.environ.get("CLOUD_SLAM_JOBS_ROOT", "/jobs"))


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _utcnow_iso() -> str:
    """UTC ``YYYY-MM-DDTHH:MM:SSZ`` (spec-shape)."""
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _generate_job_id() -> str:
    """``j_YYYY-MM-DD_<hex8>`` — matches spec regex."""
    today = datetime.utcnow().strftime("%Y-%m-%d")
    return f"j_{today}_{secrets.token_hex(4)}"


def _pick_suffix(filename: str) -> str | None:
    """Return the longest allowed suffix matching ``filename`` or None."""
    fn = filename.lower()
    for suffix in _ALLOWED_SUFFIXES:
        if fn.endswith(suffix):
            return suffix
    return None


async def _safe_reload(volume: Any) -> None:
    """Best-effort ``volume.reload()`` — tests pass duck-typed doubles.

    Prefers the async ``.reload.aio(...)`` variant when running on real
    Modal (Modal warns when a blocking reload() is called from an async
    FastAPI handler). Tests pass a simple object with a sync ``reload``
    method; the fallback handles that case.

    Modal raises ``RuntimeError: there are open files preventing the
    operation`` when any container on the same Volume is holding a file
    open (e.g. a runner streaming ``input.mcap.zst``). That makes the
    reload itself failable through no fault of the caller — and we'd
    rather serve slightly stale status than 500 every status/DELETE
    while a job is running. Swallow it; the next reload will succeed
    once the runner closes the file.
    """
    reload = getattr(volume, "reload", None)
    if reload is None:
        return
    aio = getattr(reload, "aio", None)
    try:
        if callable(aio):
            await aio()
        elif callable(reload):
            reload()
    except RuntimeError as e:
        print(f"[_safe_reload] swallowed: {e}")


async def _safe_commit(volume: Any) -> None:
    """Best-effort ``volume.commit()`` — same sync/async dispatch as reload."""
    commit = getattr(volume, "commit", None)
    if commit is None:
        return
    aio = getattr(commit, "aio", None)
    try:
        if callable(aio):
            await aio()
        elif callable(commit):
            commit()
    except RuntimeError as e:
        print(f"[_safe_commit] swallowed: {e}")


def _read_json(path: Path) -> dict:
    """Read JSON from ``path`` or return ``{}`` on any failure."""
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _file_iter(path: Path, chunk: int = _CHUNK) -> Iterator[bytes]:
    """Yield ``chunk``-sized byte reads from ``path`` for StreamingResponse."""
    with path.open("rb") as f:
        while True:
            data = f.read(chunk)
            if not data:
                return
            yield data


def _image_media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    # best_views.py only emits jpg today, but stay forgiving.
    return "application/octet-stream"


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

def _auth_dep_factory(api_key_secret_value: str | None) -> Callable[[str | None], None]:
    """Return a FastAPI dependency that enforces X-API-Key per the rules
    in the module docstring.

    Constant-time compare via :func:`hmac.compare_digest` when the
    secret IS set. An absent/empty header is always 401.
    """

    def _check(x_api_key: str | None = Header(default=None)) -> None:
        if not x_api_key:
            # Missing or empty header. Even in "no secret configured yet"
            # mode we reject — otherwise an anonymous GET could spawn a
            # $0.75 GPU run.
            raise HTTPException(status_code=401, detail="missing X-API-Key")
        if api_key_secret_value is None:
            # Accept any non-empty key. Loudly documented in the module
            # docstring and app.py.
            return
        if not hmac.compare_digest(x_api_key, api_key_secret_value):
            raise HTTPException(status_code=401, detail="invalid X-API-Key")

    return _check


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_web_app(
    *,
    volume: Any,
    runner_fn: Any,
    api_key_secret_value: str | None = None,
) -> FastAPI:
    """Build and return the FastAPI application.

    Parameters
    ----------
    volume
        Duck-typed object with ``.commit()`` / ``.reload()`` methods
        (a real ``modal.Volume`` in production, a ``SimpleNamespace``
        in tests).
    runner_fn
        The ``pipeline_runner`` Modal Function. We call
        ``runner_fn.spawn(job_id)`` to kick off the background job;
        tests mock this with a ``MagicMock``.
    api_key_secret_value
        The API key value, or None if the Secret is not yet populated.
        See module docstring for the resolution rules.
    """
    started_at = time.monotonic()
    auth_dep = _auth_dep_factory(api_key_secret_value)

    app = FastAPI(
        title="Cloud SLAM ICP — Modal API",
        version="1.0.0",
        # Disable the default docs on the public URL (this is an
        # internal API; docs would just make path-scanning easier).
        docs_url=None,
        redoc_url=None,
    )

    # Spec §"Endpoint contract": path-param regex mismatch must return
    # 404, NOT FastAPI's default 422 (422 would leak the fact that the
    # route matched the shape). Rewrite path-param validation failures
    # on job_id to a 404.
    @app.exception_handler(RequestValidationError)
    async def _validation_handler(request: Request, exc: RequestValidationError):
        for err in exc.errors():
            loc = err.get("loc") or ()
            if len(loc) >= 2 and loc[0] == "path" and loc[1] == "job_id":
                return JSONResponse(
                    status_code=404,
                    content={"detail": "job not found"},
                )
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    # ------------------------------------------------------------------
    # POST /jobs
    # ------------------------------------------------------------------
    @app.post("/jobs")
    async def submit_job(
        request: Request,
        filename: str = Query(default="input.mcap"),
        x_idempotency_key: str | None = Header(default=None),
        _auth: None = Depends(auth_dep),
    ) -> JSONResponse:
        """Accept a rosbag upload, spawn the pipeline runner."""
        suffix = _pick_suffix(filename)
        if suffix is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "filename must end with .mcap, .mcap.zst, .tar, or "
                    ".tar.zst"
                ),
            )

        # Idempotency short-circuit — do this BEFORE writing any bytes
        # so retried POSTs don't even consume bandwidth we'll throw away.
        if x_idempotency_key:
            await _safe_reload(volume)
            # Use a provisional job_id for the record; it's only
            # written if no prior record exists.
            provisional = _generate_job_id()
            resolved_id, was_new = check_or_record(
                volume,
                x_idempotency_key,
                provisional,
                jobs_root=_jobs_root(),
            )
            if not was_new:
                return JSONResponse(
                    status_code=200,
                    content={
                        "job_id": resolved_id,
                        "reused": True,
                        "status_url": _status_url(request, resolved_id),
                        "submitted_at": _utcnow_iso(),
                    },
                )
            job_id = resolved_id
        else:
            job_id = _generate_job_id()

        submitted_at = _utcnow_iso()

        # Stream body to disk. FastAPI's request.stream() yields chunks
        # without buffering the full body in memory — critical for the
        # 4 GiB upper bound.
        job_dir = _jobs_root() / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        input_path = job_dir / f"input{suffix}"

        total = 0
        try:
            with input_path.open("wb") as f:
                async for chunk in request.stream():
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > MAX_UPLOAD_BYTES:
                        # Truncate and clean up before raising. Modal's
                        # ingress layer also enforces 4 GiB, but we
                        # defend in depth here for the local TestClient
                        # case.
                        try:
                            f.truncate(0)
                        except OSError:
                            pass
                        raise HTTPException(
                            status_code=413,
                            detail=f"upload exceeds {MAX_UPLOAD_BYTES} bytes",
                        )
                    f.write(chunk)
        except HTTPException:
            # Cleanup partial upload before re-raising.
            _cleanup_partial(job_dir)
            raise
        except Exception as exc:  # pragma: no cover — unexpected IO
            _cleanup_partial(job_dir)
            raise HTTPException(
                status_code=500,
                detail=f"failed to store upload: {exc}",
            ) from exc

        # Initial status.json — pipeline_runner will merge on top.
        status_payload = {
            "status": "queued",
            "submitted_at": submitted_at,
            "updated_at": submitted_at,
            "job_id": job_id,
            "filename": filename,
        }
        _write_json_atomic(job_dir / "status.json", status_payload)
        await _safe_commit(volume)

        # Spawn the background runner. ``call.object_id`` is Modal's
        # FunctionCall handle; we persist it so DELETE can cancel.
        # Tests pass a MagicMock where ``.object_id`` may not exist —
        # swallow AttributeError in that case.
        try:
            call = runner_fn.spawn(job_id)
        except Exception as exc:  # pragma: no cover — real Modal path
            # If spawn itself fails, mark the job failed. Still return
            # 200 with the job_id so the client can poll and see the
            # failure reason, rather than silently losing the upload.
            from modal_app.status import write_error
            write_error(
                volume,
                job_dir,
                error_type="spawn_failed",
                message=f"pipeline_runner.spawn failed: {exc}",
                stage="queued",
            )
        else:
            try:
                object_id = getattr(call, "object_id", None)
                if object_id:
                    (job_dir / "function_call_id.txt").write_text(
                        str(object_id), encoding="utf-8"
                    )
                    await _safe_commit(volume)
            except AttributeError:
                # Tests mock .spawn() without an object_id — fine.
                pass

        return JSONResponse(
            status_code=200,
            content={
                "job_id": job_id,
                "status_url": _status_url(request, job_id),
                "submitted_at": submitted_at,
            },
        )

    # ------------------------------------------------------------------
    # GET /jobs/{job_id}
    # ------------------------------------------------------------------
    @app.get("/jobs/{job_id}")
    async def get_job(
        request: Request,
        job_id: str = FPath(..., pattern=JOB_ID_PATTERN),
    ) -> Response:
        await _safe_reload(volume)
        job_dir = _jobs_root() / job_id
        if not job_dir.is_dir():
            raise HTTPException(status_code=404, detail="job not found")

        status_payload = _read_json(job_dir / "status.json")
        if not status_payload:
            raise HTTPException(status_code=404, detail="status.json missing")

        status = status_payload.get("status")

        if status == "done":
            result = _read_json(job_dir / "result.json")
            # Merge result into status payload. status_payload already
            # has status=done, job_id, submitted_at, etc.; result brings
            # metrics, floorplan, furniture, best_images, artifacts.
            payload = dict(result)
            for k, v in status_payload.items():
                payload.setdefault(k, v)
            _absolutize_urls(payload, request, job_id)
            return JSONResponse(status_code=200, content=payload)

        if status == "failed":
            # error may already be inlined in status.json by
            # modal_app.status.write_error; fall back to error.json.
            if "error" not in status_payload:
                err = _read_json(job_dir / "error.json")
                if err:
                    status_payload["error"] = err
            return JSONResponse(status_code=200, content=status_payload)

        # Still running — include Retry-After hint.
        return JSONResponse(
            status_code=200,
            content=status_payload,
            headers={"Retry-After": RETRY_AFTER_S},
        )

    # ------------------------------------------------------------------
    # GET /jobs/{job_id}/image/{idx}
    # ------------------------------------------------------------------
    @app.get("/jobs/{job_id}/image/{idx}")
    async def get_image(
        job_id: str = FPath(..., pattern=JOB_ID_PATTERN),
        idx: int = FPath(...),
    ) -> Response:
        # No volume.reload() here: images are written once by the runner
        # before `status=done` is committed. If the polling endpoint has
        # already seen the `done` state, the other container has already
        # synced; reloading again before every large-file GET adds minutes
        # of latency for zero correctness benefit.
        job_dir = _jobs_root() / job_id
        if not job_dir.is_dir():
            raise HTTPException(status_code=404, detail="job not found")

        manifest_path = job_dir / "artifacts" / "best_views" / "best_views.json"
        if not manifest_path.is_file():
            raise HTTPException(status_code=404, detail="best_views manifest missing")

        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            raise HTTPException(status_code=404, detail="best_views manifest unreadable")

        entries = raw.get("entries") or []
        # Apply the same filter as load_best_views_manifest: skip
        # entries without an image_path. This keeps the idx here in
        # lockstep with result.best_images[] on the client side.
        kept = [e for e in entries if e.get("image_path")]
        if idx < 0 or idx >= len(kept):
            raise HTTPException(status_code=404, detail="image index out of range")

        rel = kept[idx].get("image_path")
        if not rel:
            raise HTTPException(status_code=404, detail="image path missing from manifest")

        image_path = (job_dir / "artifacts" / rel).resolve()
        # Defensive path-traversal check — make sure the resolved path
        # is still inside the job's artifacts dir.
        artifacts_root = (job_dir / "artifacts").resolve()
        try:
            image_path.relative_to(artifacts_root)
        except ValueError:
            raise HTTPException(status_code=404, detail="image path escapes artifacts dir")

        if not image_path.is_file():
            raise HTTPException(status_code=404, detail="image file missing")

        return StreamingResponse(
            _file_iter(image_path),
            media_type=_image_media_type(image_path),
        )

    # ------------------------------------------------------------------
    # GET /jobs/{job_id}/artifact/{name}
    # ------------------------------------------------------------------
    @app.get("/jobs/{job_id}/artifact/{name}")
    async def get_artifact(
        job_id: str = FPath(..., pattern=JOB_ID_PATTERN),
        name: str = FPath(...),
    ) -> Response:
        # No volume.reload() — static artifacts don't change after the
        # runner writes them. Reload on every GET made 100 MB PLY
        # downloads take 10+ minutes.
        rel = _ARTIFACT_WHITELIST.get(name)
        if rel is None:
            # Do NOT return 403 — we must not leak existence info.
            raise HTTPException(status_code=404, detail="artifact not found")

        job_dir = _jobs_root() / job_id
        if not job_dir.is_dir():
            raise HTTPException(status_code=404, detail="job not found")

        path = job_dir / rel
        if not path.is_file():
            raise HTTPException(status_code=404, detail="artifact not found")

        media_type = _artifact_media_type(name)
        return StreamingResponse(_file_iter(path), media_type=media_type)

    # ------------------------------------------------------------------
    # DELETE /jobs/{job_id}
    # ------------------------------------------------------------------
    @app.delete("/jobs/{job_id}")
    async def delete_job(
        job_id: str = FPath(..., pattern=JOB_ID_PATTERN),
        _auth: None = Depends(auth_dep),
    ) -> JSONResponse:
        await _safe_reload(volume)
        job_dir = _jobs_root() / job_id
        if not job_dir.is_dir():
            raise HTTPException(status_code=404, detail="job not found")

        # Attempt to cancel the running function call first (spec
        # §"Cost safety" row 9). Any failure is logged and swallowed so
        # the directory cleanup still happens.
        fc_path = job_dir / "function_call_id.txt"
        if fc_path.is_file():
            object_id = fc_path.read_text(encoding="utf-8").strip()
            if object_id:
                try:
                    import modal  # deferred: tests run without modal

                    modal.FunctionCall.from_id(object_id).cancel()
                except Exception:  # pragma: no cover — best-effort cancel
                    pass

        shutil.rmtree(job_dir, ignore_errors=True)
        await _safe_commit(volume)
        return JSONResponse(status_code=200, content={"status": "deleted"})

    # ------------------------------------------------------------------
    # GET /health
    # ------------------------------------------------------------------
    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "uptime_s": int(time.monotonic() - started_at),
            "image_env": {
                "SPATIALLM_PY": os.environ.get("SPATIALLM_PY"),
                "HF_HOME": os.environ.get("HF_HOME"),
            },
        }

    return app


# ---------------------------------------------------------------------------
# Private helpers used by endpoint bodies
# ---------------------------------------------------------------------------

def _write_json_atomic(path: Path, payload: dict) -> None:
    """Atomic tmp-rename writer for status.json during submit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _cleanup_partial(job_dir: Path) -> None:
    """Remove a half-created job_dir on upload failure."""
    try:
        shutil.rmtree(job_dir, ignore_errors=True)
    except OSError:
        pass


def _status_url(request: Request, job_id: str) -> str:
    """Build the absolute ``status_url`` returned by POST /jobs."""
    base = str(request.base_url).rstrip("/")
    return f"{base}/jobs/{job_id}"


def _absolutize_urls(payload: dict, request: Request, job_id: str) -> None:
    """Rewrite relative artifact + image paths in ``payload`` to absolute
    URLs, in place.

    Only operates on the keys the envelope guarantees (spec schema):

    - ``best_images[i]``: sets ``url`` to
      ``<base>/jobs/<job_id>/image/<i>`` and drops the internal
      ``relative_image_path`` (clients should never see filename).
    - ``artifacts.colored_map_ply`` etc.: rewritten to the public
      ``/artifact/<whitelist-name>`` URLs.
    """
    base = str(request.base_url).rstrip("/")

    best_images = payload.get("best_images")
    if isinstance(best_images, list):
        for i, entry in enumerate(best_images):
            if not isinstance(entry, dict):
                continue
            # Index-based URL — matches the public contract. We
            # intentionally do NOT leak the on-disk filename.
            entry["url"] = f"{base}/jobs/{job_id}/image/{i}"
            entry.pop("relative_image_path", None)

    # Artifacts: the envelope stores relative on-disk paths; the public
    # URL uses the whitelist keys, not the disk paths. Map by key.
    _ARTIFACT_URL_KEYS = {
        "colored_map_ply": "colored_map.ply",
        "scene_with_boxes_ply": "scene_with_boxes.ply",
        "layout_merged_txt": "layout_merged.txt",
        "result_json": "result.json",
    }
    artifacts = payload.get("artifacts")
    if isinstance(artifacts, dict):
        for key, whitelist_name in _ARTIFACT_URL_KEYS.items():
            if key in artifacts:
                artifacts[key] = (
                    f"{base}/jobs/{job_id}/artifact/{whitelist_name}"
                )


def _artifact_media_type(name: str) -> str:
    """Best-effort Content-Type for whitelisted artifacts."""
    if name.endswith(".json"):
        return "application/json"
    if name.endswith(".txt"):
        return "text/plain; charset=utf-8"
    if name.endswith(".ply"):
        # PLY has no registered MIME; application/octet-stream keeps
        # CloudCompare + browsers happy (forces download).
        return "application/octet-stream"
    return "application/octet-stream"
