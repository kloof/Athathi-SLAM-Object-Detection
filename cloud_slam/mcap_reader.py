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
    Read an MCAP file and return cloud frames, IMU data, and camera images.

    Args:
        mcap_path: Path to .mcap file or directory containing .mcap files

    Returns:
        clouds: list of (timestamp, xyz_Nx3_float64, time_offsets_N_float64)
        imus: list of (timestamp, gyro_xyz_3, acc_xyz_3)
        images: list of (timestamp, compressed_bytes, format_str)
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
    images = []

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

            elif msg.channel.topic == "/camera/image_raw/compressed":
                ros_msg = msg.ros_msg
                stamp = ros_msg.header.stamp.sec + ros_msg.header.stamp.nanosec * 1e-9
                images.append((stamp, bytes(ros_msg.data), ros_msg.format))

    clouds.sort(key=lambda x: x[0])
    imus.sort(key=lambda x: x[0])
    images.sort(key=lambda x: x[0])
    return clouds, imus, images


def _mcap_files(mcap_path):
    mcap_path = Path(mcap_path)
    if mcap_path.is_dir():
        files = sorted(mcap_path.glob("*.mcap"))
    else:
        files = [mcap_path]
    if not files:
        raise FileNotFoundError(f"No .mcap files found at {mcap_path}")
    return files


def list_camera_frame_times_ns(mcap_path, topic="/camera/image_raw/compressed"):
    """Iterate the mcap(s) and return per-camera-message ``log_time`` in ns.

    Does not decode image data — only scans message metadata. Used by
    stage 0 to emit ``slam/frames_index.json`` so stage 8 can reopen the
    rosbag and fetch winning frames by timestamp.

    Returns:
        list[int]: sorted ascending log-time ns values; duplicates preserved.
    """
    from mcap_ros2.reader import read_ros2_messages

    times: list[int] = []
    for mcap_file in _mcap_files(mcap_path):
        for msg in read_ros2_messages(str(mcap_file), topics=[topic]):
            if msg.channel.topic == topic:
                times.append(int(msg.log_time))
    times.sort()
    return times


def read_frames_by_time_ns(mcap_path,
                           topic,
                           t_ns_list,
                           *,
                           window_ns=1):
    """Fetch compressed-image messages by ``log_time`` nanoseconds.

    Decodes no image data — just returns the compressed bytes + format
    string per matched message so callers can cv2.imdecode only what they
    need. Requests are de-duplicated internally; missing frames are
    skipped with a warning.

    Args:
        mcap_path:   Path to .mcap file or directory.
        topic:       Camera topic, e.g. ``/camera/image_raw/compressed``.
        t_ns_list:   Iterable of log-time ns values to fetch.
        window_ns:   Widening half-window (ns) around each target time.
                     Default 1 (the ns is the exact log_time emitted by
                     ``list_camera_frame_times_ns`` so a window of 1 is
                     enough; supports slop if a caller rounds).

    Returns:
        list[tuple[int, bytes, str]]: ``(t_ns, compressed_bytes, format)``
        per successfully matched request, in input order.
    """
    from mcap_ros2.reader import read_ros2_messages

    requests = sorted({int(t) for t in t_ns_list})
    if not requests:
        return []

    files = _mcap_files(mcap_path)
    out_by_t: dict[int, tuple[int, bytes, str]] = {}

    for t_ns in requests:
        start = max(0, t_ns - window_ns)
        end = t_ns + window_ns + 1
        best = None
        best_dt = None
        for mcap_file in files:
            for msg in read_ros2_messages(
                str(mcap_file),
                topics=[topic],
                start_time=start,
                end_time=end,
            ):
                if msg.channel.topic != topic:
                    continue
                dt = abs(int(msg.log_time) - t_ns)
                if best is None or dt < best_dt:
                    ros_msg = msg.ros_msg
                    best = (int(msg.log_time), bytes(ros_msg.data), ros_msg.format)
                    best_dt = dt
        if best is None:
            print(f"[read_frames_by_time_ns] warning: no message near "
                  f"t_ns={t_ns} on topic {topic}")
            continue
        out_by_t[t_ns] = best

    # return in input order (after de-dup)
    return [out_by_t[t] for t in requests if t in out_by_t]
