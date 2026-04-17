"""Main-env client for the SpatialLM worker subprocess.

The worker lives in ``~/spatiallm_env`` (torch 2.4); the main pipeline lives in
the system env (torch 2.11). This module is the bridge: spawn the worker,
drop PLY jobs into its inbox, poll the outbox for results. Stdlib + open3d
only — never imports torch, transformers, or spatiallm.
"""

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SpatialLMClient:
    """Manages the SpatialLM worker subprocess and provides an async job API."""

    def __init__(
        self,
        worker_env_python: str = "~/spatiallm_env/bin/python",
        worker_script: str = "cloud_slam/detectors/spatiallm_worker.py",
        ipc_dir: Optional[str] = None,
        model_path: str = "manycore-research/SpatialLM1.1-Qwen-0.5B",
        categories: Optional[List[str]] = None,
        detect_type: str = "all",
        max_points: int = 200000,
        inference_dtype: str = "bfloat16",
        ready_timeout_s: float = 120.0,
    ):
        self.worker_env_python = os.path.expanduser(worker_env_python)
        if not os.path.isabs(worker_script):
            worker_script = os.path.abspath(worker_script)
        self.worker_script = worker_script
        self.model_path = model_path
        self.categories = list(categories) if categories else []
        self.detect_type = detect_type
        self.max_points = int(max_points)
        self.inference_dtype = inference_dtype
        self.ready_timeout_s = float(ready_timeout_s)

        self._owns_ipc_dir = ipc_dir is None
        if ipc_dir is None:
            ipc_dir = tempfile.mkdtemp(prefix="spatiallm_ipc_")
        self.ipc_dir = Path(ipc_dir).resolve()
        self.inbox = self.ipc_dir / "inbox"
        self.outbox = self.ipc_dir / "outbox"
        self.inbox.mkdir(parents=True, exist_ok=True)
        self.outbox.mkdir(parents=True, exist_ok=True)

        self._proc: Optional[subprocess.Popen] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._stderr_tail: List[str] = []
        self._stderr_lock = threading.Lock()
        self._ready = threading.Event()
        self._submitted: set = set()
        self._completed: Dict[str, Dict[str, Any]] = {}

    # ---- lifecycle ----

    def start(self) -> None:
        """Spawn the worker and block until it emits ``WORKER_READY``."""
        if self._proc is not None:
            raise RuntimeError("client already started")
        if not os.path.exists(self.worker_env_python):
            raise FileNotFoundError(
                f"worker python interpreter not found: {self.worker_env_python}")
        if not os.path.exists(self.worker_script):
            raise FileNotFoundError(
                f"worker script not found: {self.worker_script}")

        cmd = [
            self.worker_env_python, self.worker_script,
            "--inbox", str(self.inbox),
            "--outbox", str(self.outbox),
            "--model_path", self.model_path,
            "--detect_type", self.detect_type,
            "--max_points", str(self.max_points),
            "--inference_dtype", self.inference_dtype,
        ]
        if self.categories:
            cmd.extend(["--category", *self.categories])

        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        logger.info("starting SpatialLM worker: %s", " ".join(cmd))
        self._proc = subprocess.Popen(
            cmd,
            env=env,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

        if not self._ready.wait(self.ready_timeout_s):
            tail = "\n".join(self._stderr_tail[-40:])
            self.stop(timeout=2.0)
            raise TimeoutError(
                f"worker did not emit WORKER_READY within "
                f"{self.ready_timeout_s:.0f}s. Last stderr:\n{tail}")
        logger.info("SpatialLM worker ready")

    def _drain_stderr(self) -> None:
        if self._proc is None or self._proc.stderr is None:
            return
        for line in self._proc.stderr:
            line = line.rstrip("\n")
            with self._stderr_lock:
                self._stderr_tail.append(line)
                if len(self._stderr_tail) > 400:
                    self._stderr_tail = self._stderr_tail[-200:]
            if "WORKER_READY" in line:
                self._ready.set()
            else:
                logger.debug("[worker] %s", line)

    def stop(self, timeout: float = 10.0) -> None:
        """SIGTERM the worker, wait, kill on timeout, remove ipc dir."""
        if self._proc is not None and self._proc.poll() is None:
            logger.info("stopping SpatialLM worker")
            try:
                self._proc.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self._proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                logger.warning("worker did not exit; killing")
                self._proc.kill()
                try:
                    self._proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    pass
        self._proc = None

        if self._owns_ipc_dir and self.ipc_dir.exists():
            shutil.rmtree(self.ipc_dir, ignore_errors=True)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()

    # ---- submit / poll ----

    def submit(self, pcd_open3d, job_id: Optional[str] = None) -> str:
        """Atomically write a job into the inbox. Returns the job id."""
        if self._proc is None:
            raise RuntimeError("client not started")
        if job_id is None:
            job_id = uuid.uuid4().hex
        if job_id in self._submitted:
            raise ValueError(f"job_id already used: {job_id}")

        import open3d as o3d
        ply_path = self.inbox / f"{job_id}.ply"
        meta_path = self.inbox / f"{job_id}.meta.json"

        # Open3D rejects non-.ply extensions, so we stage under a hidden-prefix
        # name and rename to the visible name. Worker's filter on ``.ply`` +
        # ``not endswith('.ply.tmp')`` skips both the dot-prefix and any in-flight
        # renames.
        staging_ply = self.inbox / f".staging_{job_id}.ply"
        staging_meta = self.inbox / f".staging_{job_id}.meta.json"
        o3d.io.write_point_cloud(str(staging_ply), pcd_open3d, write_ascii=False)
        staging_meta.write_text(json.dumps({
            "job_id": job_id, "categories": self.categories}))
        # Rename meta first so the worker sees it by the time it notices .ply.
        staging_meta.rename(meta_path)
        staging_ply.rename(ply_path)

        self._submitted.add(job_id)
        logger.debug("submitted job %s (%d points)", job_id, len(pcd_open3d.points))
        return job_id

    def poll(self, job_id: Optional[str] = None,
             timeout: float = 0.0) -> Optional[Dict[str, Any]]:
        """Non-blocking (timeout=0) check for a completed job.

        If ``job_id`` is None, returns the first completed job found.
        When a job is done, its outbox files are read, deleted, and
        the result dict is returned. Returns None if not yet ready.
        """
        deadline = time.monotonic() + timeout if timeout > 0 else 0.0
        while True:
            if job_id is not None:
                if job_id in self._completed:
                    return self._completed.pop(job_id)
                result = self._try_read(job_id)
                if result is not None:
                    return result
            else:
                if self._completed:
                    first = next(iter(self._completed))
                    return self._completed.pop(first)
                found = self._scan_outbox()
                if found:
                    return self._completed.pop(found)

            # Check worker liveness — a crashed worker will never answer.
            if self._proc is not None and self._proc.poll() is not None:
                tail = "\n".join(self._stderr_tail[-20:])
                raise RuntimeError(
                    f"worker exited with code {self._proc.returncode}. "
                    f"Last stderr:\n{tail}")

            if timeout <= 0 or time.monotonic() >= deadline:
                return None
            time.sleep(0.05)

    def _scan_outbox(self) -> Optional[str]:
        """Look for any completed (non-.tmp) .txt result. Returns a job id."""
        try:
            for p in sorted(self.outbox.iterdir()):
                if p.suffix != ".txt" or p.name.endswith(".txt.tmp"):
                    continue
                job_id = p.stem
                result = self._try_read(job_id)
                if result is not None:
                    return job_id
        except FileNotFoundError:
            pass
        return None

    def _try_read(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Read + cache a result if both .txt and .meta.json are present."""
        txt_path = self.outbox / f"{job_id}.txt"
        meta_path = self.outbox / f"{job_id}.meta.json"
        if not (txt_path.exists() and meta_path.exists()):
            return None
        # Defensive: if the worker is mid-rename, skip this round.
        if txt_path.with_suffix(".txt.tmp").exists():
            return None
        if meta_path.with_suffix(".meta.json.tmp").exists():
            return None
        try:
            language_string = txt_path.read_text()
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("poll read race for %s: %s", job_id, exc)
            return None

        try:
            txt_path.unlink()
        except FileNotFoundError:
            pass
        try:
            meta_path.unlink()
        except FileNotFoundError:
            pass

        self._submitted.discard(job_id)
        result = {
            "job_id": job_id,
            "status": meta.get("status", "ok"),
            "language_string": language_string,
            "meta": meta,
        }
        self._completed[job_id] = result
        return result

    def wait_for(self, job_id: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Block until ``job_id`` completes or ``timeout`` elapses.

        Raises :class:`TimeoutError` on timeout.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            result = self.poll(job_id=job_id, timeout=0.0)
            if result is not None:
                return result
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"job {job_id} did not complete within "
                                   f"{timeout:.1f}s")
            time.sleep(0.1)

    def drain_pending(self, timeout: float = 0.0) -> List[Dict[str, Any]]:
        """Return every completed job currently available.

        Non-blocking when ``timeout=0``. When ``timeout>0``, waits up to
        ``timeout`` seconds for at least one result.
        """
        results: List[Dict[str, Any]] = []
        deadline = time.monotonic() + timeout if timeout > 0 else None
        while True:
            # Drain anything cached.
            while self._completed:
                first = next(iter(self._completed))
                results.append(self._completed.pop(first))
            # Scan outbox for new completions.
            found_any = False
            try:
                files = sorted(self.outbox.iterdir())
            except FileNotFoundError:
                files = []
            for p in files:
                if p.suffix != ".txt" or p.name.endswith(".txt.tmp"):
                    continue
                result = self._try_read(p.stem)
                if result is not None:
                    results.append(self._completed.pop(p.stem))
                    found_any = True
            if results:
                return results
            if deadline is None or time.monotonic() >= deadline:
                return results
            if found_any:
                continue
            time.sleep(0.05)
