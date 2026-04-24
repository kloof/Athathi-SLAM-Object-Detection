"""Tests for the mcap_reader helpers added for stage 8.

Uses monkeypatch to fake ``mcap_ros2.reader.read_ros2_messages`` so we
can exercise the seeking / dedup / missing-frame handling paths without
a real rosbag on disk.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


def _make_msg(topic: str, log_time_ns: int, data: bytes = b"x",
              fmt: str = "jpeg"):
    """Minimal stand-in for an mcap_ros2 message object."""
    channel = SimpleNamespace(topic=topic)
    ros_msg = SimpleNamespace(data=data, format=fmt)
    return SimpleNamespace(
        channel=channel,
        ros_msg=ros_msg,
        log_time=log_time_ns,
    )


def _fake_reader_factory(messages):
    """Build a fake ``read_ros2_messages`` that filters by time + topic."""

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
    """Requests three timestamps, all present -> returns three frames."""
    mcap = tmp_path / "fake.mcap"
    mcap.write_bytes(b"")

    topic = "/camera/image_raw/compressed"
    messages = [
        _make_msg(topic, 1_000, b"a"),
        _make_msg(topic, 2_000, b"b"),
        _make_msg(topic, 3_000, b"c"),
        _make_msg("/other/topic", 2_500, b"z"),  # wrong topic, must be ignored
    ]

    import cloud_slam.mcap_reader as m
    monkeypatch.setattr(m, "__name__", m.__name__)  # noop; hold ref to module

    import mcap_ros2.reader
    monkeypatch.setattr(mcap_ros2.reader, "read_ros2_messages",
                        _fake_reader_factory(messages))

    out = m.read_frames_by_time_ns(mcap, topic, [1_000, 2_000, 3_000])
    assert len(out) == 3
    assert [t for t, _, _ in out] == [1_000, 2_000, 3_000]
    assert [data for _, data, _ in out] == [b"a", b"b", b"c"]
    assert all(fmt == "jpeg" for _, _, fmt in out)


def test_read_frames_by_time_ns_dedupes_requests(monkeypatch, tmp_path):
    """Duplicate timestamps -> one result each, sorted asc."""
    mcap = tmp_path / "fake.mcap"
    mcap.write_bytes(b"")

    topic = "/camera/image_raw/compressed"
    messages = [
        _make_msg(topic, 2_000, b"b"),
        _make_msg(topic, 1_000, b"a"),
    ]

    import cloud_slam.mcap_reader as m
    import mcap_ros2.reader
    monkeypatch.setattr(mcap_ros2.reader, "read_ros2_messages",
                        _fake_reader_factory(messages))

    out = m.read_frames_by_time_ns(
        mcap, topic, [2_000, 1_000, 2_000, 1_000]
    )
    assert [t for t, _, _ in out] == [1_000, 2_000]


def test_read_frames_by_time_ns_tolerates_missing(monkeypatch, tmp_path,
                                                    capsys):
    """Missing frame -> skipped with warning, others still returned."""
    mcap = tmp_path / "fake.mcap"
    mcap.write_bytes(b"")

    topic = "/camera/image_raw/compressed"
    messages = [
        _make_msg(topic, 1_000, b"a"),
        # no message at 2_000
        _make_msg(topic, 3_000, b"c"),
    ]

    import cloud_slam.mcap_reader as m
    import mcap_ros2.reader
    monkeypatch.setattr(mcap_ros2.reader, "read_ros2_messages",
                        _fake_reader_factory(messages))

    out = m.read_frames_by_time_ns(mcap, topic, [1_000, 2_000, 3_000])
    assert [t for t, _, _ in out] == [1_000, 3_000]
    captured = capsys.readouterr().out
    assert "2000" in captured.replace("_", "")
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


def test_read_frames_by_time_ns_window_absorbs_rounding(monkeypatch, tmp_path):
    """Message sits 1 ns off the request -> still matched via window."""
    mcap = tmp_path / "fake.mcap"
    mcap.write_bytes(b"")

    topic = "/camera/image_raw/compressed"
    messages = [_make_msg(topic, 2_001, b"b")]  # 1 ns later than requested

    import mcap_ros2.reader
    monkeypatch.setattr(mcap_ros2.reader, "read_ros2_messages",
                        _fake_reader_factory(messages))

    import cloud_slam.mcap_reader as m
    out = m.read_frames_by_time_ns(mcap, topic, [2_000])
    assert len(out) == 1
    assert out[0][0] == 2_001


def test_list_camera_frame_times_ns_returns_sorted(monkeypatch, tmp_path):
    """list_camera_frame_times_ns returns all camera-topic log_times sorted."""
    mcap = tmp_path / "fake.mcap"
    mcap.write_bytes(b"")

    topic = "/camera/image_raw/compressed"
    messages = [
        _make_msg(topic, 3_000),
        _make_msg(topic, 1_000),
        _make_msg("/other", 2_500),
        _make_msg(topic, 2_000),
    ]

    import mcap_ros2.reader
    monkeypatch.setattr(mcap_ros2.reader, "read_ros2_messages",
                        _fake_reader_factory(messages))

    import cloud_slam.mcap_reader as m
    times = m.list_camera_frame_times_ns(mcap, topic)
    assert times == [1_000, 2_000, 3_000]
