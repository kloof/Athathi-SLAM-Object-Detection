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


def _header_stamp_ns(ros_msg) -> int:
    """Extract header.stamp as nanoseconds since epoch.

    Matches the timestamp convention used by ``read_mcap`` (which stores
    ``stamp = sec + nanosec * 1e-9`` for clouds and cameras) and by
    ``trajectory.csv`` (written from cloud header stamps). This is the
    authoritative capture-time clock; ``msg.log_time`` (when the message
    was written to the rosbag) is a different, later clock we do not use.
    """
    s = ros_msg.header.stamp
    return int(s.sec) * 1_000_000_000 + int(s.nanosec)


def list_camera_frame_times_ns(mcap_path, topic="/camera/image_raw/compressed"):
    """Iterate the mcap(s) and return per-camera-message header-stamp ns.

    Used by stage 0 to emit ``slam/frames_index.json`` so stage 8 can
    reopen the rosbag and fetch winning frames by timestamp. Uses the
    ROS header stamp (capture time), matching the convention in
    ``read_mcap`` and ``trajectory.csv`` so the emitted timestamps
    interpolate correctly against SLAM poses.

    Returns:
        list[int]: sorted ascending header-stamp ns values; duplicates
        preserved.
    """
    from mcap_ros2.reader import read_ros2_messages

    times: list[int] = []
    for mcap_file in _mcap_files(mcap_path):
        for msg in read_ros2_messages(str(mcap_file), topics=[topic]):
            if msg.channel.topic == topic:
                times.append(_header_stamp_ns(msg.ros_msg))
    times.sort()
    return times


def read_frames_by_time_ns(mcap_path,
                           topic,
                           t_ns_list,
                           *,
                           window_ns=250_000_000):
    """Fetch compressed-image messages by header-stamp nanoseconds.

    Returns compressed bytes + format string per matched message so
    callers can ``cv2.imdecode`` only what they need. Requests are
    de-duplicated internally; missing frames are skipped with a warning.

    The request `t_ns` values are ROS header-stamp nanoseconds (matching
    ``list_camera_frame_times_ns``). The mcap library seeks by log_time
    internally, which is ≥ header_stamp by the rosbag recording latency.
    We use a wide time window around each request, then pick the message
    whose *header stamp* is nearest to the target. The default 250 ms
    window comfortably covers typical rosbag latency.

    Args:
        mcap_path:   Path to .mcap file or directory.
        topic:       Camera topic, e.g. ``/camera/image_raw/compressed``.
        t_ns_list:   Iterable of header-stamp ns values to fetch.
        window_ns:   Half-window (ns) to scan around each target.

    Returns:
        list[tuple[int, bytes, str]]: ``(header_ns, compressed_bytes,
        format)`` per successfully matched request, in input order.
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
                ros_msg = msg.ros_msg
                msg_t_ns = _header_stamp_ns(ros_msg)
                dt = abs(msg_t_ns - t_ns)
                if best is None or dt < best_dt:
                    best = (msg_t_ns, bytes(ros_msg.data), ros_msg.format)
                    best_dt = dt
        if best is None:
            print(f"[read_frames_by_time_ns] warning: no message near "
                  f"t_ns={t_ns} on topic {topic}")
            continue
        out_by_t[t_ns] = best

    # return in input order (after de-dup)
    return [out_by_t[t] for t in requests if t in out_by_t]
