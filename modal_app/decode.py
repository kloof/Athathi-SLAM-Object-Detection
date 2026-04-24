"""Input-decode helpers for the Modal pipeline runner (M4).

The submit endpoint writes the upload to the volume at one of:

- ``/jobs/<id>/input.mcap``        — already a raw MCAP file
- ``/jobs/<id>/input.mcap.zst``    — zstd-compressed MCAP
- ``/jobs/<id>/input.tar``         — tarball that contains a ROS2 ``rosbag2``
  directory with one or more ``*.mcap`` files inside
- ``/jobs/<id>/input.tar.zst``     — tar + zstd combo (what the spec's
  M7 smoke procedure produces)

:func:`resolve_mcap` picks whichever one exists, decompresses/extracts as
needed, and returns a ``Path`` to a usable ``.mcap`` file under
``<job_dir>/work/``.

Cost-safety rails (spec §"Cost safety" row 8):

- :data:`MAX_DECOMPRESSED_BYTES` = 10 GiB hard cap. Both zstd streams and
  tar-member streams honour this — a decompression bomb is aborted and the
  partial file on disk is removed before :class:`DecodeError` is raised.

No network, no Modal, no GPU. Importable and unit-testable standalone.
"""
from __future__ import annotations

import shutil
import tarfile
from pathlib import Path
from typing import BinaryIO

import zstandard


# 10 GiB decompressed-size ceiling. Anything larger is rejected as a
# potential zip-bomb (row 8 of spec §"Cost safety"). 10 GiB is well above
# the largest TEST_SCAN bag (~3 GB decompressed) but far below runaway.
MAX_DECOMPRESSED_BYTES = 10 * 1024 * 1024 * 1024

# Bytes-per-read chunk for streaming copies. 1 MiB keeps the inner loop
# hot-cache friendly while still checking the size cap frequently enough
# to bail out quickly on a bomb.
_CHUNK = 1 << 20  # 1 MiB


class DecodeError(Exception):
    """Raised for any decode-path failure (missing input, oversize, bad tar)."""


class InvalidMcapError(Exception):
    """Raised when a resolved file is not an MCAP (magic-byte check failed).

    Kept separate from :class:`DecodeError` so the runner can map it to
    ``error.type="invalid_mcap"`` instead of ``"decode_error"``.
    """


def resolve_mcap(job_dir: Path) -> Path:
    """Locate and decode the job's input into ``job_dir/work/input.mcap``.

    Resolution order (first-match wins):

    1. ``input.mcap``          -> copy to ``work/input.mcap``
    2. ``input.mcap.zst``      -> stream-decompress zstd to ``work/input.mcap``
    3. ``input.tar``           -> extract first ``*.mcap`` member
    4. ``input.tar.zst``       -> stream zstd into a tar reader, extract first ``*.mcap``

    Raises :class:`DecodeError` if no input file is found, if
    decompression produces more than :data:`MAX_DECOMPRESSED_BYTES`, or
    if a tar has no ``*.mcap`` member.
    """
    job_dir = Path(job_dir)
    work = job_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    out = work / "input.mcap"

    raw = job_dir / "input.mcap"
    zst = job_dir / "input.mcap.zst"
    tar = job_dir / "input.tar"
    tar_zst = job_dir / "input.tar.zst"

    if raw.is_file():
        # Straight copy — shutil.copyfile is faster than read/write loops
        # and does not need the size cap (the upload was already bounded
        # by Modal's 4 GiB request-body ceiling).
        shutil.copyfile(raw, out)
        return out

    if zst.is_file():
        with zst.open("rb") as src, out.open("wb") as dst:
            _stream_zstd_to_file(src, dst)
        return out

    if tar.is_file():
        with tar.open("rb") as src:
            _extract_first_mcap_from_tar_stream(src, out)
        return out

    if tar_zst.is_file():
        # zstandard's stream_reader gives us a file-like object; tarfile
        # opens it in streaming mode (mode="r|") which only does forward
        # reads — that matches stream_reader's non-seekable contract.
        with tar_zst.open("rb") as src_z:
            dctx = zstandard.ZstdDecompressor()
            with dctx.stream_reader(src_z) as zstream:
                _extract_first_mcap_from_tar_stream(zstream, out, streaming=True)
        return out

    raise DecodeError(
        f"no input.{{mcap,mcap.zst,tar,tar.zst}} under {job_dir}"
    )


# ---------------------------------------------------------------------------
# zstd streaming with size cap
# ---------------------------------------------------------------------------

def _stream_zstd_to_file(src: BinaryIO, dst: BinaryIO) -> None:
    """Stream-decompress ``src`` into ``dst``, aborting over the size cap.

    On overflow the (still-open) destination file is truncated so the
    caller can release it cleanly; the on-disk path is left in place and
    the caller is expected to delete it on the ``DecodeError`` path.
    """
    dctx = zstandard.ZstdDecompressor()
    total = 0
    with dctx.stream_reader(src) as reader:
        while True:
            chunk = reader.read(_CHUNK)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_DECOMPRESSED_BYTES:
                # Best-effort cleanup: truncate the output, then let the
                # caller (typically resolve_mcap) unlink the file.
                try:
                    dst.truncate(0)
                except OSError:
                    pass
                raise DecodeError(
                    f"zstd stream exceeded {MAX_DECOMPRESSED_BYTES} bytes "
                    f"(decompression bomb guard)"
                )
            dst.write(chunk)


# ---------------------------------------------------------------------------
# tar extraction (handles both seekable and streaming sources)
# ---------------------------------------------------------------------------

def _extract_first_mcap_from_tar_stream(
    fileobj: BinaryIO,
    out_path: Path,
    *,
    streaming: bool = False,
) -> None:
    """Find the first ``*.mcap`` in ``fileobj`` and write it to ``out_path``.

    ``streaming=True`` switches tarfile to ``"r|"`` (forward-only) mode,
    which is required when ``fileobj`` is a zstd stream_reader (not
    seekable). ``streaming=False`` uses ``"r:"`` (seekable) which is
    faster for on-disk tar files.

    Raises :class:`DecodeError` if no ``*.mcap`` member is found or if
    decompression during member extraction exceeds
    :data:`MAX_DECOMPRESSED_BYTES`.
    """
    mode = "r|" if streaming else "r:"
    with tarfile.open(fileobj=fileobj, mode=mode) as tf:
        for member in tf:
            # Only regular files named "<anything>.mcap" qualify. ROS2
            # rosbag2 dirs look like "rosbag/rosbag_0.mcap" — the path
            # has leading segments we can safely ignore.
            if not member.isfile():
                continue
            if not member.name.lower().endswith(".mcap"):
                continue

            extracted = tf.extractfile(member)
            if extracted is None:
                # tarfile returns None for special members that slipped
                # past isfile() (rare, but guard anyway).
                continue

            total = 0
            with out_path.open("wb") as dst:
                while True:
                    chunk = extracted.read(_CHUNK)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_DECOMPRESSED_BYTES:
                        try:
                            dst.truncate(0)
                        except OSError:
                            pass
                        raise DecodeError(
                            f"tar member '{member.name}' exceeded "
                            f"{MAX_DECOMPRESSED_BYTES} bytes during extraction"
                        )
                    dst.write(chunk)
            return  # first-match-wins

    raise DecodeError("tar archive contains no *.mcap member")


# ---------------------------------------------------------------------------
# MCAP magic-byte validation
# ---------------------------------------------------------------------------

# First 8 bytes of any valid MCAP file (per the MCAP specification).
MCAP_MAGIC = b"\x89MCAP0\r\n"


def validate_mcap_magic(path: Path) -> None:
    """Check the MCAP magic bytes on ``path``.

    Raises :class:`InvalidMcapError` if the first 8 bytes don't match
    :data:`MCAP_MAGIC`. Files shorter than 8 bytes are also rejected.
    """
    path = Path(path)
    try:
        with path.open("rb") as f:
            head = f.read(len(MCAP_MAGIC))
    except OSError as exc:
        raise InvalidMcapError(f"could not read {path}: {exc}") from exc

    if head != MCAP_MAGIC:
        raise InvalidMcapError(
            f"{path} is not an MCAP file "
            f"(expected magic {MCAP_MAGIC!r}, got {head!r})"
        )
