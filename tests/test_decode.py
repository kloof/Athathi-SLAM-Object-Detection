"""Unit tests for modal_app.decode.

No network, no Modal, no GPU. All fixtures are generated in-process from
the MCAP magic-byte header so the tests don't depend on any binary
fixture files on disk.

Run:

    pytest tests/test_decode.py -q
"""
from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest
import zstandard

from modal_app.decode import (
    MAX_DECOMPRESSED_BYTES,
    MCAP_MAGIC,
    DecodeError,
    InvalidMcapError,
    resolve_mcap,
    validate_mcap_magic,
)


# A "just enough to be valid-looking" MCAP: 8 magic bytes + a 128-byte
# payload. The decode path doesn't inspect the body beyond the magic, so
# this is exactly what we need to exercise the file-shape logic.
def _fake_mcap_bytes(body_size: int = 128) -> bytes:
    return MCAP_MAGIC + (b"\x00" * body_size)


def test_zstd_mcap_roundtrip(tmp_path: Path) -> None:
    """A small input.mcap.zst decompresses and keeps valid MCAP magic."""
    payload = _fake_mcap_bytes(1024)
    job_dir = tmp_path / "j_2026-04-24_deadbeef"
    job_dir.mkdir()

    cctx = zstandard.ZstdCompressor()
    (job_dir / "input.mcap.zst").write_bytes(cctx.compress(payload))

    out = resolve_mcap(job_dir)

    assert out == job_dir / "work" / "input.mcap"
    assert out.is_file()
    # Byte-identical roundtrip.
    assert out.read_bytes() == payload
    # Sanity: magic-byte validator is happy.
    validate_mcap_magic(out)


def test_tar_with_nested_mcap_roundtrip(tmp_path: Path) -> None:
    """A rosbag2-style input.tar extracts the first *.mcap member."""
    payload = _fake_mcap_bytes(512)
    job_dir = tmp_path / "j_2026-04-24_cafebabe"
    job_dir.mkdir()

    # Build an in-memory tar with a rosbag2-style directory + mcap inside.
    # This mirrors the TEST_SCAN `tar -C … -cf - rosbag` layout.
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        # A plain directory entry (tarfile extracts it as a member but
        # isfile() returns False → skipped by our first-match loop).
        dir_info = tarfile.TarInfo(name="rosbag")
        dir_info.type = tarfile.DIRTYPE
        dir_info.mode = 0o755
        tf.addfile(dir_info)

        # Second: a non-mcap sibling the decoder should skip.
        meta = b"version: 4\n"
        meta_info = tarfile.TarInfo(name="rosbag/metadata.yaml")
        meta_info.size = len(meta)
        tf.addfile(meta_info, io.BytesIO(meta))

        # Third: the real mcap that must win.
        mcap_info = tarfile.TarInfo(name="rosbag/rosbag_0.mcap")
        mcap_info.size = len(payload)
        tf.addfile(mcap_info, io.BytesIO(payload))

    (job_dir / "input.tar").write_bytes(buf.getvalue())

    out = resolve_mcap(job_dir)

    assert out == job_dir / "work" / "input.mcap"
    assert out.read_bytes() == payload
    validate_mcap_magic(out)


def test_zstd_oversize_aborts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A zstd stream that decompresses past the size cap raises DecodeError.

    Uses monkeypatch to drop MAX_DECOMPRESSED_BYTES to a small value so we
    don't need to actually produce 10 GiB of data in a unit test.
    """
    # 4 KiB cap for this test; real code uses 10 GiB.
    monkeypatch.setattr(
        "modal_app.decode.MAX_DECOMPRESSED_BYTES", 4 * 1024
    )

    # 64 KiB of zero bytes — zstd compresses this to <100 B, so the input
    # file is tiny but the decompressed stream will blow past 4 KiB well
    # before it finishes.
    payload = b"\x00" * (64 * 1024)
    compressed = zstandard.ZstdCompressor().compress(payload)

    job_dir = tmp_path / "j_2026-04-24_0badf00d"
    job_dir.mkdir()
    (job_dir / "input.mcap.zst").write_bytes(compressed)

    with pytest.raises(DecodeError, match="decompression bomb"):
        resolve_mcap(job_dir)


def test_tar_without_mcap_rejected(tmp_path: Path) -> None:
    """A tar that contains no *.mcap file raises DecodeError."""
    job_dir = tmp_path / "j_2026-04-24_abad1dea"
    job_dir.mkdir()

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        # Everything is .yaml / .txt; no mcap in sight.
        for name, data in [
            ("rosbag/metadata.yaml", b"version: 4\n"),
            ("rosbag/notes.txt", b"hello world\n"),
        ]:
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))

    (job_dir / "input.tar").write_bytes(buf.getvalue())

    with pytest.raises(DecodeError, match="no \\*\\.mcap"):
        resolve_mcap(job_dir)


def test_mcap_magic_validator_on_bad_bytes(tmp_path: Path) -> None:
    """Sanity: validator rejects a file that lacks the MCAP magic prefix.

    Not one of the four deliverables, but cheap and catches regressions
    in the magic constant.
    """
    bad = tmp_path / "not.mcap"
    bad.write_bytes(b"NOTMCAP!" + b"\x00" * 100)
    with pytest.raises(InvalidMcapError):
        validate_mcap_magic(bad)

    # While we're here: confirm the module-level constant didn't drift.
    assert MCAP_MAGIC == b"\x89MCAP0\r\n"
    assert MAX_DECOMPRESSED_BYTES == 10 * 1024 * 1024 * 1024
