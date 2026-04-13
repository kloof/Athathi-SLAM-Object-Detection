"""pytest configuration for cloud_slam tests.

Ensures deterministic numpy/open3d behavior, provides skipif markers for
heavy optional deps (DeepLSD, GTSAM, Mask2Former), and exposes common
fixtures.
"""
from __future__ import annotations

import importlib.util
import os
import random
import sys

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _deterministic_seeds():
    """Reseed numpy + random before every test.

    Open3D's segment_plane() is the actual non-determinism culprit (see
    M0b) but this fixture at least pins numpy and python random so
    unit tests are reproducible.
    """
    np.random.seed(0)
    random.seed(0)
    os.environ.setdefault("PYTHONHASHSEED", "0")


def _have(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


# --- skipif markers for optional heavy deps -------------------------------

requires_deeplsd = pytest.mark.skipif(
    not _have("deeplsd"),
    reason="DeepLSD not installed — install via the pinned SHA in requirements.txt",
)
requires_gtsam = pytest.mark.skipif(
    not _have("gtsam"),
    reason="GTSAM not installed — install gtsam==4.2.0 on Py 3.10/3.11",
)
requires_mask2former = pytest.mark.skipif(
    not _have("transformers"),
    reason="transformers (Mask2Former) not installed",
)
