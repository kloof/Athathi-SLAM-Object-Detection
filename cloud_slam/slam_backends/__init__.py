"""SLAM backend registry.

Two backends ship here:

- ``kiss_icp``: primary. Modern self-adaptive voxel scan-to-map ICP.
  Empirically ties with the baseline on wall/floor RMSE while running
  ~4.5x faster.
- ``baseline``: fallback. Open3D point-to-plane ICP with IMU-gyro
  initial guess. Used automatically when KISS-ICP can't be imported or
  when its poses diverge. Preserved as the known-good path.

The experimental alternatives (FAST-LIO2, Point-LIO, DLIO,
open3d_multiway, small_gicp) that were benchmarked earlier lived in this
directory but have been removed — they either matched or underperformed
KISS-ICP on this sensor/scene. Git history has the full comparison
harness if you want to re-run it; look for the commit that deletes
``ros2_runner.py``.
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
from cloud_slam.slam_backends import kiss_icp_backend  # noqa: E402, F401
