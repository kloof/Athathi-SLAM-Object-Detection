"""
Shared MCAP reader — extracts point clouds and IMU data as numpy arrays.
Uses structured numpy parsing for speed (~1.4s for 534 frames).
"""

import numpy as np
from pathlib import Path

# Unitree L2 PointCloud2 layout (32 bytes/point)
# x(0) y(4) z(8) [gap12] intensity(16) ring(20) [gap22] time(24) [gap28]
PC_DTYPE = np.dtype({
    'names': ['x', 'y', 'z', 'intensity', 'ring', 'time'],
    'formats': ['<f4', '<f4', '<f4', '<f4', '<u2', '<f4'],
    'offsets': [0, 4, 8, 16, 20, 24],
    'itemsize': 32
})


def read_mcap(mcap_path):
    """
    Read an MCAP file and return cloud frames + IMU data.

    Args:
        mcap_path: Path to .mcap file or directory containing .mcap files

    Returns:
        clouds: list of (timestamp, xyz_Nx3_float64, time_offsets_N_float64)
        imus: list of (timestamp, gyro_xyz_3, acc_xyz_3)
    """
    from mcap_ros2.reader import read_ros2_messages

    mcap_path = Path(mcap_path)
    if mcap_path.is_dir():
        mcap_files = sorted(mcap_path.glob("*.mcap"))
    else:
        mcap_files = [mcap_path]

    if not mcap_files:
        raise FileNotFoundError(f"No .mcap files found at {mcap_path}")

    clouds = []
    imus = []

    for mcap_file in mcap_files:
        for msg in read_ros2_messages(str(mcap_file)):
            if msg.channel.topic == "/unilidar/cloud":
                ros_msg = msg.ros_msg
                stamp = ros_msg.header.stamp.sec + ros_msg.header.stamp.nanosec * 1e-9
                data = bytes(ros_msg.data)
                pts = np.frombuffer(data, dtype=PC_DTYPE)
                xyz = np.column_stack([pts['x'], pts['y'], pts['z']]).astype(np.float64)
                timestamps = pts['time'].astype(np.float64)
                valid = np.linalg.norm(xyz, axis=1) > 0.05
                clouds.append((stamp, xyz[valid], timestamps[valid]))

            elif msg.channel.topic == "/unilidar/imu":
                ros_msg = msg.ros_msg
                stamp = ros_msg.header.stamp.sec + ros_msg.header.stamp.nanosec * 1e-9
                gyro = np.array([
                    ros_msg.angular_velocity.x,
                    ros_msg.angular_velocity.y,
                    ros_msg.angular_velocity.z
                ])
                acc = np.array([
                    ros_msg.linear_acceleration.x,
                    ros_msg.linear_acceleration.y,
                    ros_msg.linear_acceleration.z
                ])
                imus.append((stamp, gyro, acc))

    clouds.sort(key=lambda x: x[0])
    imus.sort(key=lambda x: x[0])
    return clouds, imus
