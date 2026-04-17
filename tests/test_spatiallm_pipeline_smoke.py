"""Integration smoke test for the SpatialLM pipeline.

Auto-skipped unless the real ``~/spatiallm_env`` is installed *and* a
known test PLY is available. When enabled it spawns the real worker,
submits one offline cloud, and asserts the result contains geometry.
"""

import os
import sys
from pathlib import Path

import pytest


pytestmark = [
    pytest.mark.skipif(
        not Path("~/spatiallm_env/bin/python").expanduser().exists()
        or os.environ.get("SPATIALLM_INTEGRATION_TESTS", "0") != "1",
        reason="spatiallm_env not installed or SPATIALLM_INTEGRATION_TESTS!=1",
    ),
]


TEST_PLY_CANDIDATES = [
    Path("/mnt/c/Users/klof/Desktop/SLAM_test/Scans/"
         "scan_20260414_143317/colored_map_leveled.ply"),
]


def _first_existing_ply():
    for p in TEST_PLY_CANDIDATES:
        if p.exists():
            return p
    return None


@pytest.mark.skipif(_first_existing_ply() is None,
                    reason="reference test PLY not available")
def test_real_inference_produces_scene(tmp_path):
    import open3d as o3d

    from cloud_slam.detectors.scene import Scene
    from cloud_slam.detectors.spatiallm_client import SpatialLMClient

    ply = _first_existing_ply()
    pcd = o3d.io.read_point_cloud(str(ply))
    assert len(pcd.points) > 0, f"{ply} empty"

    worker_script = Path(__file__).resolve().parent.parent / (
        "cloud_slam/detectors/spatiallm_worker.py")

    client = SpatialLMClient(
        worker_env_python="~/spatiallm_env/bin/python",
        worker_script=str(worker_script),
        ipc_dir=str(tmp_path / "ipc"),
        model_path="manycore-research/SpatialLM1.1-Qwen-0.5B",
        ready_timeout_s=240.0,
    )
    client.start()
    try:
        job_id = client.submit(pcd)
        result = client.wait_for(job_id, timeout=300.0)
        assert result["status"] in ("ok", "empty"), (
            f"unexpected status: {result['status']} ({result['meta']})")
        scene = Scene.from_language_string(result["language_string"])
        # Defends against silent empty (SpatialLM issue #81).
        assert len(scene.walls) >= 1, (
            f"no walls in output — possible #81 empty bug. "
            f"full meta: {result['meta']}")
        assert len(scene.bboxes) >= 1, (
            f"no bboxes in output — unexpected for this scene. "
            f"full meta: {result['meta']}")
    finally:
        client.stop()
