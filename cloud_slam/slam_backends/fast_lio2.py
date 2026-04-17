"""FAST-LIO2 backend (ROS2 offline).

FAST-LIO2 (HKU-MARS) is the canonical tightly-coupled IEKF LiDAR-IMU
odometry for small LiDARs — considered the accuracy gold standard for
this class of sensor. C++/ROS2, no Python bindings, so we run the node
as a subprocess and feed it the user's MCAP via `ros2 bag play`.

This backend wraps the Ericsii/FAST_LIO_ROS2 port (built locally in
~/slam_ws) and a custom config tuned for the Unitree L2.
"""

import os
from pathlib import Path

import numpy as np

from cloud_slam.slam_backends import register
from cloud_slam.slam_backends.base import BackendResult
from cloud_slam.slam_backends.ros2_runner import (
    match_poses_to_clouds, run_ros_slam)


@register("fast_lio2")
class FastLIO2Backend:
    name = "fast_lio2"
    description = ("FAST-LIO2 tight IEKF LiDAR-IMU odometry (HKU-MARS), "
                   "ROS2 offline via rosbag playback")

    slam_ws = Path.home() / "slam_ws"
    setup_sh = slam_ws / "install/setup.bash"
    config_rel = "src/FAST_LIO/config/unilidar_l2.yaml"

    def _require_rosbag(self, rosbag_path: str | None) -> str:
        if not rosbag_path:
            raise RuntimeError(
                "fast_lio2 backend requires rosbag_path (ROS2-driven); "
                "pass --rosbag on compare_slam.py")
        if not Path(rosbag_path).exists():
            raise FileNotFoundError(rosbag_path)
        return str(rosbag_path)

    def run(self,
            clouds,
            imus,
            rosbag_path: str | None = None,
            ) -> BackendResult:
        import time
        t0 = time.time()

        rosbag_path = self._require_rosbag(rosbag_path)

        config_path = self.slam_ws / self.config_rel
        if not config_path.exists():
            raise FileNotFoundError(f"fast_lio2 config not found: {config_path}")

        env = os.environ.copy()
        source_cmd = (
            "source /opt/ros/humble/setup.bash && "
            f"source {self.setup_sh} && env")
        # Capture env after sourcing — simpler than trying to pass through
        # a shell. We need ROS2 setup vars (AMENT_PREFIX_PATH, etc.).
        import subprocess
        sourced = subprocess.check_output(
            ["bash", "-lc", source_cmd]).decode("utf-8")
        for line in sourced.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                env[k] = v

        # Launch via the underlying executable. `ros2 run` would work
        # too but adds a bash indirection.
        fastlio_exe = (self.slam_ws / "install/fast_lio/lib/fast_lio"
                       / "fastlio_mapping")
        if not fastlio_exe.exists():
            raise FileNotFoundError(
                f"fastlio_mapping binary missing at {fastlio_exe} "
                f"— run `colcon build --packages-select fast_lio` first")

        launch_argv = [
            str(fastlio_exe),
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
                "fast_lio2_pose_samples": len(pose_samples),
                "fast_lio2_clouds_matched": matched,
                "fast_lio2_config": str(config_path),
            },
        )
