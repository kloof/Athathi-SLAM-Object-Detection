"""Atomic ``status.json`` writer for the Modal pipeline runner (M4).

Modal Volumes buffer writes inside each container until ``volume.commit()``
is called. The polling endpoints run in a *separate* ASGI container, so
without a commit after every status update the client sees stale state.
:class:`StatusWriter` bundles three things:

1. **Atomic rename** — write to a tmp sibling then ``os.replace`` into
   ``status.json`` so the endpoint never reads a half-written file.
2. **Merge with existing fields** — each ``.write(**fields)`` merges on
   top of whatever is already on disk, so transient keys like
   ``submitted_at`` (set by the submit endpoint) survive the runner's
   later status transitions.
3. **``volume.commit()`` after every rename** — makes the update visible
   to the polling container. This is the single footgun the spec calls
   out (see §"Architecture": "Modal's canonical footgun — without it the
   polling UX silently serves stale state").

``write_error`` is a companion helper that writes both ``error.json``
(the failure detail) and a ``status.json`` with ``status="failed"`` in
one shot, and commits once.
"""
from __future__ import annotations

import json
import os
import traceback as tb_mod
from datetime import datetime
from pathlib import Path
from typing import Any


def _utcnow_iso() -> str:
    """UTC ``YYYY-MM-DDTHH:MM:SSZ`` timestamp (spec-shape)."""
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write ``payload`` to ``path`` via tmp-rename (no torn reads)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    # os.replace is the atomic-rename primitive that works on both POSIX
    # and Windows. On POSIX it's a single rename(2) syscall.
    os.replace(tmp, path)


def _read_json_or_empty(path: Path) -> dict:
    """Return the JSON object at ``path`` or ``{}`` on any read/parse error."""
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


class StatusWriter:
    """Atomically update ``<job_dir>/status.json`` and commit the volume.

    Each :meth:`write` merges the given fields onto whatever's currently
    on disk, sets ``updated_at`` to *now*, writes via tmp-rename, and
    calls ``volume.commit()``.

    The ``volume`` argument is duck-typed — any object with a
    ``.commit()`` method works, which keeps the class trivially testable
    without spinning up Modal.
    """

    def __init__(self, volume: Any, job_dir: Path) -> None:
        self.volume = volume
        self.job_dir = Path(job_dir)
        self.path = self.job_dir / "status.json"

    def write(self, **fields: Any) -> dict:
        """Merge ``fields`` into ``status.json`` and commit.

        Returns the full merged payload for caller introspection (handy
        for logging and tests).
        """
        current = _read_json_or_empty(self.path)
        current.update(fields)
        current["updated_at"] = _utcnow_iso()
        _atomic_write_json(self.path, current)
        _safe_commit(self.volume)
        return current


def write_error(
    volume: Any,
    job_dir: Path,
    *,
    error_type: str,
    message: str,
    stage: str | None = None,
    stderr_tail: str | None = None,
    returncode: int | None = None,
    traceback: str | None = None,
    extra: dict | None = None,
) -> None:
    """Write ``error.json`` + a ``status.json`` with ``status="failed"``.

    Both files land in ``job_dir``; the volume is committed exactly once
    after both renames succeed. ``traceback`` defaults to
    :func:`traceback.format_exc` when an exception is currently being
    handled (check ``sys.exc_info``); pass an explicit string to override.

    ``extra`` merges into the error payload for ad-hoc fields
    (``container_hostname``, ``wall_time_s``, etc. — spec §"Cost safety"
    row 7).
    """
    job_dir = Path(job_dir)

    error_payload: dict[str, Any] = {
        "type": error_type,
        "message": message,
    }
    if stage is not None:
        error_payload["stage"] = stage
    if stderr_tail is not None:
        error_payload["stderr_tail"] = stderr_tail
    if returncode is not None:
        error_payload["returncode"] = returncode
    if traceback is None:
        # Only capture if we're inside an except block — otherwise
        # format_exc() returns the useless string "NoneType: None\n".
        fmt = tb_mod.format_exc()
        if fmt and not fmt.startswith("NoneType: None"):
            traceback = fmt
    if traceback is not None:
        error_payload["traceback"] = traceback
    if extra:
        error_payload.update(extra)

    _atomic_write_json(job_dir / "error.json", error_payload)

    # Merge-into-existing so submit-endpoint fields (submitted_at) survive.
    status_path = job_dir / "status.json"
    current = _read_json_or_empty(status_path)
    current.update({
        "status": "failed",
        "error": error_payload,
        "updated_at": _utcnow_iso(),
    })
    _atomic_write_json(status_path, current)
    _safe_commit(volume)


def _safe_commit(volume: Any) -> None:
    """Call ``volume.commit()`` if the duck-typed handle exposes one.

    Tests pass plain objects without a ``commit`` method, and that's
    fine — silently skipping the call keeps the writer unit-testable.
    """
    commit = getattr(volume, "commit", None)
    if callable(commit):
        commit()
