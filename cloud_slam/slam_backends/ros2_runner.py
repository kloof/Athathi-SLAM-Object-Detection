"""Offline runner for ROS2-based SLAM nodes.

Orchestration:
  1. Start a ROS2 SLAM node as a subprocess with the given launch command.
  2. Wait briefly for the subscriber to come up.
  3. Subscribe (from this process) to the node's /Odometry output.
  4. Play the user's MCAP rosbag via `ros2 bag play`.
  5. Collect (timestamp, 4x4 pose) tuples until bag play exits.
  6. Shut the node down cleanly.
  7. Return the collected poses.

This is the bridge that lets FAST-LIO2, Point-LIO, DLIO, and GLIM (all
native ROS2 nodes) slot into the same Backend protocol as the Python-
native backends.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class PoseSample:
    stamp: float
    T: np.ndarray  # 4x4


def _pose_from_odom(msg) -> np.ndarray:
    """Nav2 Odometry message → 4x4 world-from-body."""
    from scipy.spatial.transform import Rotation
    T = np.eye(4)
    T[0, 3] = msg.pose.pose.position.x
    T[1, 3] = msg.pose.pose.position.y
    T[2, 3] = msg.pose.pose.position.z
    q = msg.pose.pose.orientation
    T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    return T


def run_ros_slam(launch_argv: list[str],
                 rosbag_path: str,
                 pose_topic: str = "/Odometry",
                 warmup_seconds: float = 3.0,
                 post_bag_drain_seconds: float = 3.0,
                 play_rate: float = 1.0,
                 env: dict[str, str] | None = None,
                 verbose: bool = False,
                 ) -> list[PoseSample]:
    """Run a ROS2 SLAM node offline on a rosbag; collect pose samples.

    Parameters
    ----------
    launch_argv
        Full argv for the SLAM node (already has --ros-args with params).
    rosbag_path
        Path to the user's MCAP rosbag (file or directory).
    pose_topic
        Topic to subscribe for nav_msgs/Odometry output.
    warmup_seconds
        Pause between starting the SLAM node and starting bag play.
    post_bag_drain_seconds
        How long to keep subscribing after bag play exits (to catch
        the final odometry emissions queued up in the node).
    play_rate
        `ros2 bag play --rate` value. 1.0 = real-time; lower = gives
        slower nodes more time per frame.

    Returns
    -------
    list[PoseSample]
        All (timestamp, 4x4) samples observed on pose_topic. Empty list
        if the node never published.
    """
    import rclpy
    from rclpy.node import Node
    from nav_msgs.msg import Odometry

    effective_env = os.environ.copy()
    if env:
        effective_env.update(env)

    print(f"[ros2_runner] launching SLAM node: {' '.join(launch_argv)}")
    slam_proc = subprocess.Popen(
        launch_argv,
        stdout=(None if verbose else subprocess.DEVNULL),
        stderr=(None if verbose else subprocess.DEVNULL),
        env=effective_env,
    )

    if not rclpy.ok():
        rclpy.init()
    collector = _PoseCollector(pose_topic)
    t_warmup_end = time.time() + warmup_seconds
    while time.time() < t_warmup_end:
        rclpy.spin_once(collector, timeout_sec=0.1)

    print(f"[ros2_runner] starting bag playback: {rosbag_path}")
    play_argv = [
        "ros2", "bag", "play", rosbag_path,
        "--rate", str(play_rate),
        "--clock",
    ]
    play_proc = subprocess.Popen(
        play_argv,
        stdout=(None if verbose else subprocess.DEVNULL),
        stderr=(None if verbose else subprocess.DEVNULL),
        env=effective_env,
    )

    while play_proc.poll() is None:
        rclpy.spin_once(collector, timeout_sec=0.1)
        if slam_proc.poll() is not None:
            print(f"[ros2_runner] SLAM node exited unexpectedly "
                  f"(code {slam_proc.returncode})")
            break

    drain_end = time.time() + post_bag_drain_seconds
    while time.time() < drain_end:
        rclpy.spin_once(collector, timeout_sec=0.1)

    print(f"[ros2_runner] collected {len(collector.samples)} pose samples")

    if slam_proc.poll() is None:
        slam_proc.send_signal(signal.SIGINT)
        try:
            slam_proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            slam_proc.kill()
            slam_proc.wait()

    collector.destroy_node()
    # Caller may run more nodes — leave rclpy up.

    return collector.samples


class _PoseCollector:
    """Internal rclpy node that accumulates Odometry samples."""
    def __init__(self, topic: str):
        from rclpy.node import Node
        from nav_msgs.msg import Odometry
        self.samples: list[PoseSample] = []
        self._node = Node("slam_pose_collector")
        self._sub = self._node.create_subscription(
            Odometry, topic, self._cb, 100)

    def destroy_node(self) -> None:
        self._node.destroy_node()

    def _cb(self, msg) -> None:
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        T = _pose_from_odom(msg)
        self.samples.append(PoseSample(stamp=stamp, T=T))

    # Forward rclpy's spin_once target lookup
    def __getattr__(self, name):
        return getattr(self._node, name)


def match_poses_to_clouds(pose_samples: list[PoseSample],
                          clouds: list[tuple[float, np.ndarray, np.ndarray]],
                          ) -> list[np.ndarray]:
    """For each input cloud, return the pose sample nearest in time.

    Falls back to identity for clouds that have no close pose — can
    happen for frames before the node fully warms up.
    """
    if not pose_samples:
        return [np.eye(4) for _ in clouds]

    stamps = np.array([p.stamp for p in pose_samples])
    Ts = np.stack([p.T for p in pose_samples])

    out: list[np.ndarray] = []
    for stamp, _, _ in clouds:
        idx = int(np.argmin(np.abs(stamps - stamp)))
        if abs(stamps[idx] - stamp) > 0.5:
            out.append(np.eye(4))
        else:
            out.append(Ts[idx].copy())
    return out
