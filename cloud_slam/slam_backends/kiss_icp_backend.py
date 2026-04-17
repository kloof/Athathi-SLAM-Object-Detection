"""KISS-ICP backend.

KISS-ICP is a modern, simple, high-performing LiDAR-odometry registration
pipeline — scan-to-map ICP with a self-adaptive motion-model initial
guess, voxel-hash local map, and built-in robust deskew via point
timestamps. Does not fuse IMU.

Registration is run on points that we pre-deskew with the shared
deskew_scan function, with KISS-ICP's own internal deskew disabled. This
keeps the deskew identical across all backends so per-frame poses are
comparable without mixing different deskew conventions.
"""

import time

import numpy as np

from cloud_slam.deskew import deskew_scan
from cloud_slam.slam_backends import register
from cloud_slam.slam_backends.base import BackendResult


@register("kiss_icp")
class KissICPBackend:
    name = "kiss_icp"
    description = "KISS-ICP self-adaptive voxel ICP, no IMU"

    def __init__(self,
                 max_range: float = 20.0,
                 min_range: float = 0.3,
                 voxel_size: float = 0.1,
                 initial_threshold: float = 2.0):
        self.max_range = max_range
        self.min_range = min_range
        self.voxel_size = voxel_size
        self.initial_threshold = initial_threshold

    def _make_icp(self):
        from kiss_icp.kiss_icp import KissICP
        from kiss_icp.config import KISSConfig
        cfg = KISSConfig()
        cfg.data.max_range = self.max_range
        cfg.data.min_range = self.min_range
        cfg.data.deskew = False
        cfg.mapping.voxel_size = self.voxel_size
        cfg.adaptive_threshold.initial_threshold = self.initial_threshold
        return KissICP(config=cfg)

    def run(self,
            clouds: list[tuple[float, np.ndarray, np.ndarray]],
            imus: list[tuple[float, np.ndarray, np.ndarray]]
            ) -> BackendResult:
        t0 = time.time()

        if imus:
            imu_times = np.array([t for t, _, _ in imus], dtype=np.float64)
            imu_gyros = np.array([g for _, g, _ in imus], dtype=np.float64)
        else:
            imu_times = np.array([], dtype=np.float64)
            imu_gyros = np.empty((0, 3), dtype=np.float64)

        icp = self._make_icp()
        poses: list[np.ndarray] = []

        for stamp, xyz, time_offsets in clouds:
            xyz = np.asarray(xyz, dtype=np.float64)

            if len(imu_times) > 0 and len(time_offsets) > 0 and len(xyz) > 0:
                xyz_desk = deskew_scan(xyz, time_offsets, stamp,
                                       imu_times, imu_gyros)
            else:
                xyz_desk = xyz

            if len(xyz_desk) < 10:
                poses.append(
                    icp.last_pose.copy() if len(poses) else np.eye(4))
                continue

            # KISS-ICP needs per-point timestamps, but we've already
            # deskewed. Pass zeros so its internal deskew (disabled
            # above) is fully a no-op regardless.
            dummy_times = np.zeros(len(xyz_desk), dtype=np.float64)
            icp.register_frame(xyz_desk, dummy_times)
            poses.append(icp.last_pose.copy())

        return BackendResult(
            poses=poses,
            runtime_s=time.time() - t0,
            backend_name=self.name,
            extra={
                "kiss_icp_voxel_size": self.voxel_size,
                "kiss_icp_max_range": self.max_range,
            },
        )
