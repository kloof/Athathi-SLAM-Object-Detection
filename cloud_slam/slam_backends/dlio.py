"""DLIO backend (Direct LiDAR-Inertial Odometry, UCLA VECTR Lab, ROS2).

DLIO is a lightweight direct LIO with continuous-time motion correction
built from a coarse-to-fine approach. Runs both an `odom` node (pose
estimation) and a `map` node (keyframe map). We subscribe to the
`dlio/odom_node/odom` topic.

Wraps the vectr-ucla feature/ros2 branch built locally in ~/slam_ws.
"""

import os
from pathlib import Path

import numpy as np

from cloud_slam.slam_backends import register
from cloud_slam.slam_backends.base import BackendResult
from cloud_slam.slam_backends.ros2_runner import (
    match_poses_to_clouds, run_ros_slam)


@register("dlio")
class DLIOBackend:
    name = "dlio"
    description = ("DLIO Direct LiDAR-Inertial Odometry (UCLA VECTR), "
                   "continuous-time motion correction — ROS2 offline")

    slam_ws = Path.home() / "slam_ws"
    setup_sh = slam_ws / "install/setup.bash"

    def run(self,
            clouds,
            imus,
            rosbag_path: str | None = None,
            ) -> BackendResult:
        import subprocess
        import time
        t0 = time.time()

        if not rosbag_path:
            raise RuntimeError("dlio backend requires rosbag_path")

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

        # Use ros2 launch with topic remappings for our Unitree L2 data.
        launch_argv = [
            "ros2", "launch", "direct_lidar_inertial_odometry",
            "dlio.launch.py",
            "pointcloud_topic:=/unilidar/cloud",
            "imu_topic:=/unilidar/imu",
            "rviz:=false",
        ]

        pose_samples = run_ros_slam(
            launch_argv=launch_argv,
            rosbag_path=rosbag_path,
            pose_topic="/dlio/odom_node/odom",
            warmup_seconds=5.0,
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
                "dlio_pose_samples": len(pose_samples),
                "dlio_clouds_matched": matched,
            },
        )
