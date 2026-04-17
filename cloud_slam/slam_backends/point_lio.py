"""Point-LIO backend (ROS2 offline).

Point-LIO (HKU-MARS) is a continuous-time tightly-coupled LIO with
rate-invariant registration — designed to handle aggressive motion
where FAST-LIO2's per-frame IMU undistortion struggles. Uses a
different state representation (explicit velocity in IMU frame) and
can process each point at its own timestamp.

Wraps the LycanW/Point-LiO-ROS2-Unilidar port (Unitree-L2 specific)
built locally in ~/slam_ws.
"""

import os
from pathlib import Path

import numpy as np

from cloud_slam.slam_backends import register
from cloud_slam.slam_backends.base import BackendResult
from cloud_slam.slam_backends.ros2_runner import (
    match_poses_to_clouds, run_ros_slam)


@register("point_lio")
class PointLIOBackend:
    name = "point_lio"
    description = ("Point-LIO continuous-time tight-coupled LiDAR-IMU "
                   "odometry (HKU-MARS), rate-invariant — ROS2 offline")

    slam_ws = Path.home() / "slam_ws"
    setup_sh = slam_ws / "install/setup.bash"
    config_rel = "src/Point_LIO/src/point_lio/config/unilidar_l2.yaml"

    def run(self,
            clouds,
            imus,
            rosbag_path: str | None = None,
            ) -> BackendResult:
        import subprocess
        import time
        t0 = time.time()

        if not rosbag_path:
            raise RuntimeError(
                "point_lio backend requires rosbag_path (ROS2-driven)")

        config_path = self.slam_ws / self.config_rel
        if not config_path.exists():
            raise FileNotFoundError(f"point_lio config missing: {config_path}")

        env = os.environ.copy()
        source_cmd = (
            "source /opt/ros/humble/setup.bash && "
            f"source {self.setup_sh} && env")
        sourced = subprocess.check_output(
            ["bash", "-lc", source_cmd]).decode("utf-8")
        for line in sourced.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                env[k] = v

        exe = (self.slam_ws / "install/point_lio/lib/point_lio"
               / "pointlio_mapping")
        if not exe.exists():
            raise FileNotFoundError(
                f"pointlio_mapping binary missing at {exe}")

        launch_argv = [
            str(exe),
            "--ros-args",
            "--params-file", str(config_path),
        ]

        pose_samples = run_ros_slam(
            launch_argv=launch_argv,
            rosbag_path=rosbag_path,
            pose_topic="/Odometry",
            warmup_seconds=4.0,
            post_bag_drain_seconds=5.0,
            play_rate=1.0,
            env=env,
            verbose=False,
        )

        poses = match_poses_to_clouds(pose_samples, clouds)
        matched = sum(1 for T in poses if not np.allclose(T, np.eye(4)))

        return BackendResult(
            poses=poses,
            runtime_s=time.time() - t0,
            backend_name=self.name,
            extra={
                "point_lio_pose_samples": len(pose_samples),
                "point_lio_clouds_matched": matched,
                "point_lio_config": str(config_path),
            },
        )
