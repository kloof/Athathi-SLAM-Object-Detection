"""Baseline SLAM backend: wraps the existing ICP + IMU-gyro pipeline.

The pipeline's merging / colorization branches are disabled here (images
and calib are not forwarded) so the shared post-processor can do all
colorization and reduction uniformly across every backend being compared.
"""

import time

import numpy as np

from cloud_slam.pipelines.icp_imu_pipeline import run as icp_imu_run
from cloud_slam.slam_backends import register
from cloud_slam.slam_backends.base import BackendResult


@register("baseline")
class BaselineBackend:
    """Open3D point-to-plane ICP with IMU-gyro initial guess (current default)."""

    name = "baseline"
    description = "Open3D point-to-plane ICP + IMU-gyro init (current default)"

    def run(self,
            clouds: list[tuple[float, np.ndarray, np.ndarray]],
            imus: list[tuple[float, np.ndarray, np.ndarray]]
            ) -> BackendResult:
        t0 = time.time()
        _, poses_raw, _ = icp_imu_run(
            clouds, imus,
            voxel_size=0.005,
            images=None,
            calib=None,
        )
        # The baseline pipeline appends the initial identity as poses[0] and
        # one pose per frame thereafter; strip the prefix to align 1:1 with
        # the input clouds.
        poses = poses_raw[1:]
        if len(poses) != len(clouds):
            raise RuntimeError(
                f"baseline: pose count {len(poses)} != cloud count {len(clouds)}")
        return BackendResult(
            poses=poses,
            runtime_s=time.time() - t0,
            backend_name=self.name,
            extra={},
        )
