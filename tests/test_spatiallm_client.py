"""Unit tests for SpatialLMClient using a fake-worker subprocess.

These tests never invoke the real SpatialLM model — they spawn a tiny
Python subprocess that implements the same inbox/outbox protocol with a
pre-computed language string, so the client + IPC plumbing can be
exercised in any env.

Integration tests that require the real ``~/spatiallm_env`` are gated
behind a ``skipif`` and live in ``tests/test_spatiallm_pipeline_smoke.py``.
"""

import json
import os
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest

from cloud_slam.detectors.spatiallm_client import SpatialLMClient


# --------------------------------------------------------------------------- #
# Fake worker script — writes a canned reply for every .ply it sees.
# --------------------------------------------------------------------------- #

FAKE_WORKER = textwrap.dedent(r"""
    import argparse
    import json
    import os
    import signal
    import sys
    import time
    from pathlib import Path

    REPLY = "wall_0=Wall(0.0,0.0,0.0,4.0,0.0,0.0,2.8,0.0)\nbbox_0=Bbox(sofa,2.0,2.0,0.4,0.0,2.0,0.85,0.9)"

    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--inbox", required=True)
        parser.add_argument("--outbox", required=True)
        parser.add_argument("--model_path", default="")
        parser.add_argument("--detect_type", default="all")
        parser.add_argument("--category", nargs="*", default=[])
        parser.add_argument("--max_points", type=int, default=200000)
        parser.add_argument("--inference_dtype", default="bfloat16")
        parser.add_argument("--mode", default="reply",
                            choices=["reply", "silent", "empty", "error"])
        args, _ = parser.parse_known_args()

        inbox = Path(args.inbox)
        outbox = Path(args.outbox)
        inbox.mkdir(parents=True, exist_ok=True)
        outbox.mkdir(parents=True, exist_ok=True)

        stop = {"flag": False}
        def _handle(sig, frame):
            stop["flag"] = True
        signal.signal(signal.SIGTERM, _handle)
        signal.signal(signal.SIGINT, _handle)

        print("WORKER_READY", file=sys.stderr, flush=True)

        while not stop["flag"]:
            entries = sorted(p for p in inbox.iterdir()
                             if p.suffix == ".ply"
                             and not p.name.endswith(".ply.tmp")
                             and not p.name.startswith("."))
            if not entries:
                time.sleep(0.05)
                continue
            for ply in entries:
                if stop["flag"]:
                    break
                job_id = ply.stem
                meta_in_path = ply.with_suffix(".meta.json")
                # Read input meta (throw-away; proves worker CAN read it)
                if meta_in_path.exists():
                    try:
                        json.loads(meta_in_path.read_text())
                    except Exception:
                        pass

                if args.mode == "silent":
                    # Never reply.
                    try:
                        ply.unlink()
                    except FileNotFoundError:
                        pass
                    if meta_in_path.exists():
                        meta_in_path.unlink()
                    continue

                if args.mode == "empty":
                    reply = ""
                    status = "empty"
                elif args.mode == "error":
                    reply = ""
                    status = "error"
                else:
                    reply = REPLY
                    status = "ok"

                out_txt = outbox / f"{job_id}.txt"
                out_meta = outbox / f"{job_id}.meta.json"
                tmp_txt = out_txt.with_suffix(".txt.tmp")
                tmp_meta = out_meta.with_suffix(".meta.json.tmp")
                tmp_txt.write_text(reply)
                tmp_meta.write_text(json.dumps({
                    "status": status,
                    "inference_seconds": 0.01,
                    "point_count": 100,
                    "model_path": args.model_path,
                    "error": "" if status != "error" else "fake error",
                }))
                tmp_meta.rename(out_meta)
                tmp_txt.rename(out_txt)

                try:
                    ply.unlink()
                except FileNotFoundError:
                    pass
                if meta_in_path.exists():
                    meta_in_path.unlink()

        print("worker exiting", file=sys.stderr, flush=True)

    if __name__ == "__main__":
        main()
""").strip()


@pytest.fixture
def fake_worker_script(tmp_path):
    path = tmp_path / "fake_worker.py"
    path.write_text(FAKE_WORKER)
    return path


def _make_pcd(n: int = 500) -> o3d.geometry.PointCloud:
    rng = np.random.default_rng(0)
    pts = rng.uniform(-1.0, 1.0, size=(n, 3))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_roundtrip_single_job(fake_worker_script, tmp_path):
    """submit -> wait_for returns a parsable language string."""
    ipc = tmp_path / "ipc"
    client = SpatialLMClient(
        worker_env_python=sys.executable,
        worker_script=str(fake_worker_script),
        ipc_dir=str(ipc),
        model_path="fake",
        ready_timeout_s=30.0,
    )
    client.start()
    try:
        job_id = client.submit(_make_pcd(200))
        result = client.wait_for(job_id, timeout=15.0)
        assert result["job_id"] == job_id
        assert result["status"] == "ok"
        assert "wall_0=Wall" in result["language_string"]
        assert "bbox_0=Bbox" in result["language_string"]
        from cloud_slam.detectors.scene import Scene
        scene = Scene.from_language_string(result["language_string"])
        assert len(scene.walls) == 1
        assert len(scene.bboxes) == 1
        assert scene.bboxes[0].class_name == "sofa"
    finally:
        client.stop()


def test_wait_for_timeout(fake_worker_script, tmp_path, monkeypatch):
    """wait_for raises TimeoutError if the worker never answers."""
    # Start fake worker in 'silent' mode (it deletes inputs but never writes output).
    silent_script = tmp_path / "silent_worker.py"
    silent_script.write_text(FAKE_WORKER)
    ipc = tmp_path / "ipc"

    # Monkey-patch the Popen args to inject --mode silent.
    import subprocess
    orig_popen = subprocess.Popen

    def patched_popen(cmd, *a, **kw):
        return orig_popen(cmd + ["--mode", "silent"], *a, **kw)

    monkeypatch.setattr(subprocess, "Popen", patched_popen)

    client = SpatialLMClient(
        worker_env_python=sys.executable,
        worker_script=str(silent_script),
        ipc_dir=str(ipc),
        model_path="fake",
        ready_timeout_s=30.0,
    )
    client.start()
    try:
        job_id = client.submit(_make_pcd(100))
        with pytest.raises(TimeoutError):
            client.wait_for(job_id, timeout=0.5)
    finally:
        client.stop()


def test_multiple_jobs_drain(fake_worker_script, tmp_path):
    """submit 3 jobs, drain_pending returns all 3."""
    ipc = tmp_path / "ipc"
    client = SpatialLMClient(
        worker_env_python=sys.executable,
        worker_script=str(fake_worker_script),
        ipc_dir=str(ipc),
        model_path="fake",
        ready_timeout_s=30.0,
    )
    client.start()
    try:
        ids = [client.submit(_make_pcd(100), job_id=f"job{i}") for i in range(3)]
        # Wait until all 3 appear.
        deadline = time.monotonic() + 15.0
        collected = []
        while time.monotonic() < deadline and len(collected) < 3:
            collected.extend(client.drain_pending(timeout=0.2))
        assert len(collected) == 3
        assert {r["job_id"] for r in collected} == set(ids)
    finally:
        client.stop()


def test_atomic_writes_ignored(fake_worker_script, tmp_path):
    """poll() ignores .txt.tmp files, only picks up .txt after rename."""
    ipc = tmp_path / "ipc"
    client = SpatialLMClient(
        worker_env_python=sys.executable,
        worker_script=str(fake_worker_script),
        ipc_dir=str(ipc),
        model_path="fake",
        ready_timeout_s=30.0,
    )
    client.start()
    try:
        # Stick a bogus .tmp file into outbox — poll must ignore it.
        bogus = client.outbox / "ghost.txt.tmp"
        bogus.write_text("should_be_ignored")
        bogus_meta = client.outbox / "ghost.meta.json.tmp"
        bogus_meta.write_text("{}")

        # Submit a real job — poll should return ONLY the real result,
        # never "ghost".
        job_id = client.submit(_make_pcd(100))
        result = client.wait_for(job_id, timeout=15.0)
        assert result["job_id"] == job_id
        assert "ghost" not in result["job_id"]

        # And ghost files must still be sitting on disk.
        assert bogus.exists()
    finally:
        client.stop()


def test_cleanup_on_stop(fake_worker_script, tmp_path):
    """stop() removes the auto-created ipc_dir."""
    # Let the client auto-create its ipc_dir (the cleanup path we want to test).
    client = SpatialLMClient(
        worker_env_python=sys.executable,
        worker_script=str(fake_worker_script),
        model_path="fake",
        ready_timeout_s=30.0,
    )
    client.start()
    ipc_dir = client.ipc_dir
    assert ipc_dir.exists()
    client.stop()
    assert not ipc_dir.exists()


def test_cleanup_preserves_user_ipc_dir(fake_worker_script, tmp_path):
    """If ipc_dir was user-supplied, stop() must NOT delete it."""
    ipc = tmp_path / "user_ipc"
    ipc.mkdir()
    client = SpatialLMClient(
        worker_env_python=sys.executable,
        worker_script=str(fake_worker_script),
        ipc_dir=str(ipc),
        model_path="fake",
        ready_timeout_s=30.0,
    )
    client.start()
    client.stop()
    assert ipc.exists(), "user-supplied ipc_dir should survive stop()"


@pytest.mark.skipif(
    not Path("~/spatiallm_env/bin/python").expanduser().exists()
    or os.environ.get("SPATIALLM_INTEGRATION_TESTS", "0") != "1",
    reason="spatiallm_env not installed or SPATIALLM_INTEGRATION_TESTS!=1",
)
def test_real_worker_smoke(tmp_path):
    """Real-env smoke test — gated behind SPATIALLM_INTEGRATION_TESTS=1.

    Smoke-starts the real worker script to confirm it can at least boot
    (model load failure will propagate as a RuntimeError via _stderr check).
    """
    worker_script = Path(__file__).resolve().parent.parent / (
        "cloud_slam/detectors/spatiallm_worker.py")
    client = SpatialLMClient(
        worker_env_python="~/spatiallm_env/bin/python",
        worker_script=str(worker_script),
        ipc_dir=str(tmp_path / "real_ipc"),
        ready_timeout_s=180.0,
    )
    try:
        client.start()
    finally:
        client.stop()
