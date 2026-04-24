"""Tests for the mcap_reader helpers added for stage 8.

Uses monkeypatch to fake ``mcap_ros2.reader.read_ros2_messages`` so we
can exercise the seeking / dedup / missing-frame handling paths without
a real rosbag on disk. Timestamps throughout are ROS header stamps
(capture time), matching what the emitter and consumer actually use.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _make_msg(topic: str, header_ns: int, data: bytes = b"x",
              fmt: str = "jpeg", log_ns: int | None = None):
    """Minimal stand-in for an mcap_ros2 message object.

    header_ns is what our helpers extract. log_ns simulates the rosbag's
    write time (always >= header_ns in real data); defaults to
    header_ns + 50 ms to match realistic recording latency.
    """
    if log_ns is None:
        log_ns = header_ns + 50_000_000
    channel = SimpleNamespace(topic=topic)
    stamp = SimpleNamespace(sec=header_ns // 1_000_000_000,
                            nanosec=header_ns % 1_000_000_000)
    header = SimpleNamespace(stamp=stamp)
    ros_msg = SimpleNamespace(data=data, format=fmt, header=header)
    return SimpleNamespace(
        channel=channel,
        ros_msg=ros_msg,
        log_time=log_ns,
    )


def _fake_reader_factory(messages):
    """Build a fake ``read_ros2_messages`` that filters by topic + log_time.

    Mimics mcap's real behavior: start_time/end_time filter by the
    message's log_time (write time to the rosbag), not header stamp.
    """

    def fake_reader(path, topics=None, start_time=None, end_time=None):
        for msg in messages:
            if topics is not None and msg.channel.topic not in topics:
                continue
            if start_time is not None and msg.log_time < start_time:
                continue
            if end_time is not None and msg.log_time >= end_time:
                continue
            yield msg

    return fake_reader


def test_read_frames_by_time_ns_returns_matching_frames(monkeypatch, tmp_path):
    """Three header-stamp requests, all present -> returns three frames."""
    mcap = tmp_path / "fake.mcap"
    mcap.write_bytes(b"")

    topic = "/camera/image_raw/compressed"
    # Header stamps 1_000_000_000, 2_000_000_000, 3_000_000_000 (1s, 2s, 3s)
    base = 1_000_000_000
    messages = [
        _make_msg(topic, base + 0 * base, b"a"),
        _make_msg(topic, base + 1 * base, b"b"),
        _make_msg(topic, base + 2 * base, b"c"),
        _make_msg("/other/topic", base + 1_500_000_000, b"z"),
    ]

    import mcap_ros2.reader
    monkeypatch.setattr(mcap_ros2.reader, "read_ros2_messages",
                        _fake_reader_factory(messages))

    import cloud_slam.mcap_reader as m
    out = m.read_frames_by_time_ns(
        mcap, topic,
        [base, 2 * base, 3 * base]
    )
    assert len(out) == 3
    assert [t for t, _, _ in out] == [base, 2 * base, 3 * base]
    assert [data for _, data, _ in out] == [b"a", b"b", b"c"]
    assert all(fmt == "jpeg" for _, _, fmt in out)


def test_read_frames_by_time_ns_dedupes_requests(monkeypatch, tmp_path):
    """Duplicate timestamps -> one result each, sorted asc."""
    mcap = tmp_path / "fake.mcap"
    mcap.write_bytes(b"")

    topic = "/camera/image_raw/compressed"
    messages = [
        _make_msg(topic, 2_000_000_000, b"b"),
        _make_msg(topic, 1_000_000_000, b"a"),
    ]

    import mcap_ros2.reader
    monkeypatch.setattr(mcap_ros2.reader, "read_ros2_messages",
                        _fake_reader_factory(messages))

    import cloud_slam.mcap_reader as m
    out = m.read_frames_by_time_ns(
        mcap, topic,
        [2_000_000_000, 1_000_000_000, 2_000_000_000, 1_000_000_000]
    )
    assert [t for t, _, _ in out] == [1_000_000_000, 2_000_000_000]


def test_read_frames_by_time_ns_tolerates_missing(monkeypatch, tmp_path,
                                                    capsys):
    """Missing frame -> skipped with warning, others still returned."""
    mcap = tmp_path / "fake.mcap"
    mcap.write_bytes(b"")

    topic = "/camera/image_raw/compressed"
    # Request 2_000_000_000 but no message exists near there
    messages = [
        _make_msg(topic, 1_000_000_000, b"a"),
        _make_msg(topic, 5_000_000_000, b"c"),  # far from 2s request
    ]

    import mcap_ros2.reader
    monkeypatch.setattr(mcap_ros2.reader, "read_ros2_messages",
                        _fake_reader_factory(messages))

    import cloud_slam.mcap_reader as m
    out = m.read_frames_by_time_ns(
        mcap, topic,
        [1_000_000_000, 2_000_000_000, 5_000_000_000]
    )
    assert [t for t, _, _ in out] == [1_000_000_000, 5_000_000_000]
    captured = capsys.readouterr().out
    assert "2000000000" in captured.replace("_", "")
    assert "warning" in captured.lower()


def test_read_frames_by_time_ns_empty_request(monkeypatch, tmp_path):
    """Empty request list -> empty result, no reader calls."""
    mcap = tmp_path / "fake.mcap"
    mcap.write_bytes(b"")

    calls = []

    def boom(*args, **kwargs):
        calls.append(1)
        return iter([])

    import mcap_ros2.reader
    monkeypatch.setattr(mcap_ros2.reader, "read_ros2_messages", boom)

    import cloud_slam.mcap_reader as m
    out = m.read_frames_by_time_ns(mcap, "/topic", [])
    assert out == []
    assert calls == []


def test_read_frames_by_time_ns_log_time_offset(monkeypatch, tmp_path):
    """Message log_time is header_ns + recording latency; seek by wide
    window and match by header stamp still finds the right frame."""
    mcap = tmp_path / "fake.mcap"
    mcap.write_bytes(b"")

    topic = "/camera/image_raw/compressed"
    # Header stamp 2s, log_time 2s + 100ms latency — the helper must
    # seek over a wide enough window to find this.
    messages = [
        _make_msg(topic, 2_000_000_000, b"b",
                  log_ns=2_100_000_000),
    ]

    import mcap_ros2.reader
    monkeypatch.setattr(mcap_ros2.reader, "read_ros2_messages",
                        _fake_reader_factory(messages))

    import cloud_slam.mcap_reader as m
    out = m.read_frames_by_time_ns(mcap, topic, [2_000_000_000])
    assert len(out) == 1
    assert out[0][0] == 2_000_000_000  # returns header stamp, not log_time
    assert out[0][1] == b"b"


def test_read_frames_by_time_ns_picks_nearest_in_window(monkeypatch,
                                                         tmp_path):
    """Multiple candidates within window -> the one with nearest
    header stamp to the request wins."""
    mcap = tmp_path / "fake.mcap"
    mcap.write_bytes(b"")

    topic = "/camera/image_raw/compressed"
    messages = [
        _make_msg(topic, 1_900_000_000, b"a"),  # 100 ms before request
        _make_msg(topic, 2_050_000_000, b"b"),  # 50 ms after request
        _make_msg(topic, 2_150_000_000, b"c"),  # 150 ms after request
    ]

    import mcap_ros2.reader
    monkeypatch.setattr(mcap_ros2.reader, "read_ros2_messages",
                        _fake_reader_factory(messages))

    import cloud_slam.mcap_reader as m
    out = m.read_frames_by_time_ns(mcap, topic, [2_000_000_000])
    assert len(out) == 1
    # Nearest by header stamp is b (50 ms delta vs 100 and 150)
    assert out[0][1] == b"b"


def test_list_camera_frame_times_ns_returns_header_stamps_sorted(
        monkeypatch, tmp_path):
    """list_camera_frame_times_ns returns header-stamp nanoseconds (NOT
    log_time) from camera topic only, sorted ascending."""
    mcap = tmp_path / "fake.mcap"
    mcap.write_bytes(b"")

    topic = "/camera/image_raw/compressed"
    messages = [
        _make_msg(topic, 3_000_000_000),
        _make_msg(topic, 1_000_000_000),
        _make_msg("/other", 2_500_000_000),  # ignored
        _make_msg(topic, 2_000_000_000),
    ]

    import mcap_ros2.reader
    monkeypatch.setattr(mcap_ros2.reader, "read_ros2_messages",
                        _fake_reader_factory(messages))

    import cloud_slam.mcap_reader as m
    times = m.list_camera_frame_times_ns(mcap, topic)
    assert times == [1_000_000_000, 2_000_000_000, 3_000_000_000]


def test_list_camera_frame_times_ns_rejects_real_datetime_log_time(
        monkeypatch, tmp_path):
    """Regression: earlier version tried `int(msg.log_time)` which fails
    on the real datetime object returned by mcap_ros2. Ensure we never
    touch msg.log_time in this helper."""
    mcap = tmp_path / "fake.mcap"
    mcap.write_bytes(b"")

    topic = "/camera/image_raw/compressed"
    from datetime import datetime, timezone
    dt = datetime(2026, 4, 24, 12, 0, 0, tzinfo=timezone.utc)
    # Craft a message whose log_time is a datetime — if the helper
    # called int() on it, this would raise TypeError. Header stamp is
    # deliberately NOT aligned with the datetime to prove we ignore
    # log_time entirely.
    msg = _make_msg(topic, 42_000_000_000, log_ns=dt)

    import mcap_ros2.reader
    monkeypatch.setattr(mcap_ros2.reader, "read_ros2_messages",
                        _fake_reader_factory([msg]))

    import cloud_slam.mcap_reader as m
    times = m.list_camera_frame_times_ns(mcap, topic)
    assert times == [42_000_000_000]
