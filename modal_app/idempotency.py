"""Idempotency-key → job_id mapping, persisted on the Modal Volume.

When the client sends an ``X-Idempotency-Key`` header on a retried
POST /jobs, we want to return the FIRST job's id rather than spinning
up a duplicate H100 run (spec §"Cost safety" row 6).

Storage shape::

    /jobs/_idempotency/<sha256(key)>.txt   # file contents = job_id

Using ``sha256(key)`` avoids path-traversal / weird-filename concerns
while keeping lookups O(1) with no extra index file. The file's
``mtime`` is the TTL anchor — entries older than 24 h are treated as
if they did not exist (and overwritten on the next POST).

The function is pure-file-IO so it's trivially unit-testable against a
``tempfile.TemporaryDirectory`` mounted at ``/jobs`` via monkeypatch;
the ``volume.commit()`` hook is duck-typed (any object with a
``.commit()`` method works, objects without it silently skip).
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any


# TTL for idempotency records, in seconds (24 h per spec §"Cost safety" row 6).
IDEMPOTENCY_TTL_S = 24 * 60 * 60

# Directory on the volume that holds the key→job_id mapping files.
# Leading underscore keeps it distinct from real ``j_*`` job dirs.
_IDEMPOTENCY_DIR_NAME = "_idempotency"


def check_or_record(
    volume: Any,
    key: str,
    job_id: str,
    *,
    jobs_root: Path | str = "/jobs",
) -> tuple[str, bool]:
    """Look up ``key`` or record ``(key → job_id)`` atomically.

    Returns ``(existing_or_new_job_id, was_new)``:

    - If a non-expired record exists for ``key``: returns
      ``(existing_job_id, False)`` — caller should return the existing
      job in the response instead of spawning a new runner.
    - Otherwise: records ``(key → job_id)`` and returns
      ``(job_id, True)`` — caller proceeds with the new job.

    Expired entries (mtime older than :data:`IDEMPOTENCY_TTL_S`) are
    treated as absent and overwritten.

    Directory layout on the volume::

        <jobs_root>/_idempotency/<sha256(key)>.txt   # contents = job_id

    ``volume.commit()`` is called after any write so cross-container
    reads see the record on their next ``.reload()``. The ``volume``
    argument is duck-typed; tests may pass a simple ``types.SimpleNamespace``
    that no-ops.
    """
    root = Path(jobs_root) / _IDEMPOTENCY_DIR_NAME
    root.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    record = root / f"{digest}.txt"

    if record.is_file():
        # Respect TTL: old records are treated as not present and get
        # overwritten below. Using mtime (not ctime) so an unrelated
        # chmod wouldn't reset the clock.
        age_s = time.time() - record.stat().st_mtime
        if age_s <= IDEMPOTENCY_TTL_S:
            existing = record.read_text(encoding="utf-8").strip()
            if existing:
                return existing, False
            # Empty file means a prior write was interrupted — treat as
            # absent and fall through to overwrite.

    # Record not present or expired — write the new mapping.
    record.write_text(job_id, encoding="utf-8")
    _safe_commit(volume)
    return job_id, True


def _safe_commit(volume: Any) -> None:
    """Best-effort ``volume.commit()`` — no-ops for duck-typed test doubles."""
    commit = getattr(volume, "commit", None)
    if callable(commit):
        commit()
