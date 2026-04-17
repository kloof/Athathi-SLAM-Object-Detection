"""SLAM backend registry for comparing registration algorithms.

Each backend implements a single contract: take the raw MCAP data (clouds +
IMU samples), return per-frame world-from-scan poses. Everything downstream
— deskew, per-point colorization, color-aware voxel reduction, floor
leveling, PLY export, metrics — is shared post-processing so backends are
strictly apples-to-apples.
"""

from cloud_slam.slam_backends.base import Backend, BackendResult

BACKENDS: dict[str, type[Backend]] = {}


def register(name: str):
    """Decorator for registering a backend class under a CLI name."""
    def deco(cls):
        BACKENDS[name] = cls
        return cls
    return deco


def get_backend(name: str) -> Backend:
    if name not in BACKENDS:
        raise KeyError(
            f"unknown backend {name!r}; available: {sorted(BACKENDS)}")
    return BACKENDS[name]()


# Import modules for their @register side-effects.
from cloud_slam.slam_backends import baseline  # noqa: E402, F401
