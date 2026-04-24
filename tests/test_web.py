"""Unit tests for modal_app.web (the FastAPI endpoints).

No Modal, no GPU, no network. The strategy is:

- ``volume`` is a ``types.SimpleNamespace`` with no-op ``.commit()``
  and ``.reload()`` methods (duck-typed).
- ``/jobs`` root is redirected to a ``tempfile.TemporaryDirectory``
  via the ``CLOUD_SLAM_JOBS_ROOT`` env var (monkeypatched per-test).
- ``runner_fn`` is a ``MagicMock`` whose ``.spawn()`` returns another
  ``MagicMock`` — no actual background function runs.

Run:

    pytest tests/test_web.py -q
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from modal_app.web import JOB_ID_PATTERN, build_web_app


# The fixture-generated MCAP payload must at least have the right
# first-8-bytes magic; decode validation isn't called at submit time
# but keeps us honest if a later refactor enables it.
MCAP_MAGIC = b"\x89MCAP0\r\n"
SAMPLE_MCAP = MCAP_MAGIC + b"\x00" * 256

API_KEY = "test-api-key-abcdef"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def jobs_root(tmp_path, monkeypatch):
    """Redirect the web module's /jobs root to a per-test tempdir."""
    monkeypatch.setenv("CLOUD_SLAM_JOBS_ROOT", str(tmp_path))
    return tmp_path


@pytest.fixture
def fake_volume():
    """Duck-typed Modal Volume stub with no-op commit/reload."""
    return SimpleNamespace(commit=lambda: None, reload=lambda: None)


@pytest.fixture
def runner_fn():
    """Mock pipeline_runner — .spawn returns a Mock with .object_id."""
    fn = MagicMock()
    call = MagicMock()
    call.object_id = "fc-test-1234"
    fn.spawn.return_value = call
    return fn


@pytest.fixture
def client(jobs_root, fake_volume, runner_fn):
    """FastAPI TestClient with a fully wired-up app."""
    app = build_web_app(
        volume=fake_volume,
        runner_fn=runner_fn,
        api_key_secret_value=API_KEY,
    )
    return TestClient(app)


# ---------------------------------------------------------------------------
# POST /jobs
# ---------------------------------------------------------------------------


def test_post_jobs_happy_path_returns_job_id(client, runner_fn, jobs_root):
    """Valid upload + valid key returns 200 with a job_id matching regex."""
    r = client.post(
        "/jobs",
        params={"filename": "scan.mcap"},
        headers={"X-API-Key": API_KEY},
        content=SAMPLE_MCAP,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert re.match(JOB_ID_PATTERN, body["job_id"]), body
    assert body["status_url"].endswith(f"/jobs/{body['job_id']}")
    assert body["submitted_at"].endswith("Z")

    # Body was persisted.
    job_dir = jobs_root / body["job_id"]
    assert (job_dir / "input.mcap").read_bytes() == SAMPLE_MCAP

    # Initial status.json is "queued".
    status = json.loads((job_dir / "status.json").read_text())
    assert status["status"] == "queued"
    assert status["job_id"] == body["job_id"]

    # spawn was invoked with the job_id.
    runner_fn.spawn.assert_called_once_with(body["job_id"])

    # function_call_id.txt persisted so DELETE can cancel.
    assert (job_dir / "function_call_id.txt").read_text() == "fc-test-1234"


def test_post_jobs_without_api_key_is_401(client):
    r = client.post(
        "/jobs",
        params={"filename": "scan.mcap"},
        content=SAMPLE_MCAP,
    )
    assert r.status_code == 401
    assert "X-API-Key" in r.json()["detail"]


def test_post_jobs_with_wrong_api_key_is_401(client):
    r = client.post(
        "/jobs",
        params={"filename": "scan.mcap"},
        headers={"X-API-Key": "nope-this-is-not-the-key"},
        content=SAMPLE_MCAP,
    )
    assert r.status_code == 401


def test_post_jobs_bad_filename_suffix_is_400(client):
    r = client.post(
        "/jobs",
        params={"filename": "scan.bag"},
        headers={"X-API-Key": API_KEY},
        content=SAMPLE_MCAP,
    )
    assert r.status_code == 400


def test_post_jobs_idempotency_returns_first_job_id(client, runner_fn):
    """Two POSTs with the same X-Idempotency-Key → second returns first job_id."""
    idem = "client-uuid-abcdef-123456"
    first = client.post(
        "/jobs",
        params={"filename": "scan.mcap"},
        headers={"X-API-Key": API_KEY, "X-Idempotency-Key": idem},
        content=SAMPLE_MCAP,
    )
    assert first.status_code == 200
    first_id = first.json()["job_id"]

    second = client.post(
        "/jobs",
        params={"filename": "scan.mcap"},
        headers={"X-API-Key": API_KEY, "X-Idempotency-Key": idem},
        content=SAMPLE_MCAP,
    )
    assert second.status_code == 200
    body = second.json()
    assert body["job_id"] == first_id
    assert body.get("reused") is True

    # runner_fn.spawn fired only ONCE (for the first job); second call
    # short-circuits before spawn.
    assert runner_fn.spawn.call_count == 1


# ---------------------------------------------------------------------------
# GET /jobs/{id}
# ---------------------------------------------------------------------------


def test_get_job_with_bad_regex_is_404(client):
    """Non-matching job_id → FastAPI 404 (route match fails)."""
    r = client.get("/jobs/bad_id")
    # Path regex mismatch → 404 (route not matched).
    assert r.status_code == 404


def test_get_job_queued_returns_retry_after(client, jobs_root):
    """A job with status=queued returns 200 + Retry-After: 3."""
    # Seed a valid job dir directly (bypass POST to keep the test
    # focused on the GET path).
    job_id = "j_2026-04-24_abcdef01"
    job_dir = jobs_root / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "status.json").write_text(
        json.dumps({
            "status": "queued",
            "job_id": job_id,
            "submitted_at": "2026-04-24T12:00:00Z",
            "updated_at": "2026-04-24T12:00:00Z",
        })
    )

    r = client.get(f"/jobs/{job_id}")
    assert r.status_code == 200
    assert r.headers.get("Retry-After") == "3"
    assert r.json()["status"] == "queued"


def test_get_job_done_merges_result_and_rewrites_urls(client, jobs_root):
    """A done job returns the merged result envelope with absolute URLs."""
    job_id = "j_2026-04-24_abcdef02"
    job_dir = jobs_root / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "status.json").write_text(
        json.dumps({
            "status": "done",
            "job_id": job_id,
            "submitted_at": "2026-04-24T12:00:00Z",
            "updated_at": "2026-04-24T12:05:00Z",
        })
    )
    (job_dir / "result.json").write_text(
        json.dumps({
            "job_id": job_id,
            "status": "done",
            "best_images": [
                {
                    "bbox_id": "bbox_0",
                    "class": "sofa",
                    "relative_image_path": "best_views/sofa_000.jpg",
                    "pixel_aabb": [0, 0, 1, 1],
                }
            ],
            "artifacts": {
                "colored_map_ply": "artifacts/slam/colored_map.ply",
                "scene_with_boxes_ply": "artifacts/scene_with_boxes.ply",
                "layout_merged_txt": "artifacts/layout_merged.txt",
                "result_json": "result.json",
            },
        })
    )

    r = client.get(f"/jobs/{job_id}")
    assert r.status_code == 200, r.text
    # Done jobs do NOT get Retry-After.
    assert "retry-after" not in {k.lower() for k in r.headers.keys()}
    body = r.json()

    # Best-image URL is index-based and does NOT leak the filename.
    assert body["best_images"][0]["url"].endswith(f"/jobs/{job_id}/image/0")
    assert "relative_image_path" not in body["best_images"][0]
    assert "sofa_000.jpg" not in body["best_images"][0]["url"]

    # Artifact URLs use the whitelist names.
    assert body["artifacts"]["colored_map_ply"].endswith(
        f"/jobs/{job_id}/artifact/colored_map.ply"
    )
    assert body["artifacts"]["result_json"].endswith(
        f"/jobs/{job_id}/artifact/result.json"
    )


def test_get_job_missing_returns_404(client):
    """Valid-regex job_id that doesn't exist on the volume → 404."""
    r = client.get("/jobs/j_2026-04-24_deadbeef")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# GET /jobs/{id}/image/{idx}
# ---------------------------------------------------------------------------


def test_get_image_returns_jpeg(client, jobs_root):
    """Valid idx returns the JPEG bytes with image/jpeg content-type."""
    job_id = "j_2026-04-24_abcdef03"
    job_dir = jobs_root / job_id
    best_views = job_dir / "artifacts" / "best_views"
    best_views.mkdir(parents=True)

    jpeg_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 50  # JFIF-ish placeholder
    (best_views / "sofa_000.jpg").write_bytes(jpeg_bytes)
    (best_views / "best_views.json").write_text(
        json.dumps({
            "entries": [
                # First entry is score-floor skipped (no image_path) —
                # it must NOT count in the public index.
                {"bbox_id": 0, "class": "skipped"},
                {
                    "bbox_id": 1,
                    "class": "sofa",
                    "image_path": "best_views/sofa_000.jpg",
                },
            ]
        })
    )

    r = client.get(f"/jobs/{job_id}/image/0")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/jpeg")
    assert r.content == jpeg_bytes


def test_get_image_out_of_range_is_404(client, jobs_root):
    job_id = "j_2026-04-24_abcdef04"
    job_dir = jobs_root / job_id
    best_views = job_dir / "artifacts" / "best_views"
    best_views.mkdir(parents=True)
    (best_views / "best_views.json").write_text(json.dumps({"entries": []}))

    r = client.get(f"/jobs/{job_id}/image/0")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# GET /jobs/{id}/artifact/{name}
# ---------------------------------------------------------------------------


def test_get_artifact_not_whitelisted_is_404(client, jobs_root):
    """Anything outside the whitelist returns 404 (not 403)."""
    job_id = "j_2026-04-24_abcdef05"
    (jobs_root / job_id).mkdir(parents=True)

    r = client.get(f"/jobs/{job_id}/artifact/notawhitelist")
    assert r.status_code == 404
    # Specifically: NOT 403 (don't leak existence).
    assert r.status_code != 403


def test_get_artifact_whitelisted_returns_bytes(client, jobs_root):
    """A whitelisted artifact on disk streams back its bytes."""
    job_id = "j_2026-04-24_abcdef06"
    job_dir = jobs_root / job_id
    (job_dir / "artifacts").mkdir(parents=True)
    layout_bytes = b"wall_0=Wall(0,0,0,1,0,0,2.5,0.1)\n"
    (job_dir / "artifacts" / "layout_merged.txt").write_bytes(layout_bytes)

    r = client.get(f"/jobs/{job_id}/artifact/layout_merged.txt")
    assert r.status_code == 200
    assert r.content == layout_bytes
    assert r.headers["content-type"].startswith("text/plain")


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


def test_health_no_auth_required(client):
    """/health returns 200 without X-API-Key and exposes env vars."""
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert isinstance(body["uptime_s"], int)
    assert "image_env" in body
    assert set(body["image_env"].keys()) == {"SPATIALLM_PY", "HF_HOME"}
