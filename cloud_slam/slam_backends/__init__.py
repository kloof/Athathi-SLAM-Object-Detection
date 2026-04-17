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
from cloud_slam.slam_backends import kiss_icp_backend  # noqa: E402, F401
from cloud_slam.slam_backends import open3d_multiway  # noqa: E402, F401
from cloud_slam.slam_backends import small_gicp_backend  # noqa: E402, F401

# ROS2-based backends — only register if rclpy imports (so the harness
# still works in a plain Python env without a ROS2 install sourced).
try:
    import rclpy  # noqa: F401
    _HAS_RCLPY = True
except ImportError:
    _HAS_RCLPY = False
if _HAS_RCLPY:
    from cloud_slam.slam_backends import fast_lio2  # noqa: E402, F401
    from cloud_slam.slam_backends import point_lio  # noqa: E402, F401
    from cloud_slam.slam_backends import dlio  # noqa: E402, F401
