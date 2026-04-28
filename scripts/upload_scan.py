#!/usr/bin/env python3
"""Upload a rosbag to the cloud_slam_icp Modal API.

Handles Modal's "delayed response" pattern. When a slow upload exceeds
Modal's synchronous-response timeout, the edge returns 303/307 with
``Location:`` pointing to a Modal-internal result URL that GET-blocks
until the upstream handler finishes. ``curl -L`` would re-POST the
body to that URL and get rejected with ``modal-http: bad redirect
method``; this script GETs it instead.

Stdlib only (http.client), so the Pi doesn't need ``requests``. Streams
the request body in 1 MiB chunks rather than loading the whole file
into RAM.

Usage:
    upload_scan.py <file> [--filename SCAN.mcap.zst]
                          [--api https://...modal.run]
                          [--key-file ~/.cloud_slam_icp_api_key]
                          [--key 64HEX]
                          [--idempotency-key UUID]
                          [--poll]
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlencode, urlparse


CHUNK = 1 << 20  # 1 MiB

DEFAULT_API = os.environ.get(
    "CLOUD_SLAM_API",
    "https://tiktokredditkw--cloud-slam-icp-web.modal.run",
)
DEFAULT_KEY_FILE = str(Path.home() / ".cloud_slam_icp_api_key")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Upload a rosbag to cloud_slam_icp Modal API",
    )
    p.add_argument("file", help="Path to .mcap, .mcap.zst, .tar, or .tar.zst")
    p.add_argument(
        "--filename",
        help="Server-side filename (default: basename of <file>)",
    )
    p.add_argument("--api", default=DEFAULT_API, help=f"Base URL (default: {DEFAULT_API})")
    p.add_argument(
        "--key",
        default=os.environ.get("CLOUD_SLAM_API_KEY"),
        help="API key (or use --key-file / CLOUD_SLAM_API_KEY env)",
    )
    p.add_argument("--key-file", default=DEFAULT_KEY_FILE)
    p.add_argument("--idempotency-key", help="Optional X-Idempotency-Key value")
    p.add_argument("--poll", action="store_true", help="Poll until status==done/failed")
    p.add_argument("--poll-interval", type=float, default=3.0)
    p.add_argument("--timeout", type=float, default=1800.0, help="Per-request socket timeout (s)")
    return p.parse_args()


def _read_key(key_arg: str | None, key_file: str) -> str:
    if key_arg:
        return key_arg.strip()
    p = Path(key_file).expanduser()
    if p.is_file():
        v = p.read_text().strip()
        if v:
            return v
    sys.exit(
        f"ERROR: no API key. Pass --key, set $CLOUD_SLAM_API_KEY, or "
        f"populate {key_file} (mode 600)."
    )


def _connect(parsed, timeout: float) -> http.client.HTTPConnection:
    if parsed.scheme == "https":
        return http.client.HTTPSConnection(parsed.netloc, timeout=timeout)
    if parsed.scheme == "http":
        return http.client.HTTPConnection(parsed.netloc, timeout=timeout)
    raise ValueError(f"unsupported scheme: {parsed.scheme}")


def _path_qs(parsed) -> str:
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query
    return path


def post_streamed(
    url: str,
    headers: dict[str, str],
    file_path: Path,
    timeout: float,
) -> tuple[int, dict[str, str], bytes]:
    """POST a file body in chunks. Returns (status, headers, body)."""
    parsed = urlparse(url)
    conn = _connect(parsed, timeout)
    file_size = file_path.stat().st_size
    conn.putrequest("POST", _path_qs(parsed))
    for k, v in headers.items():
        conn.putheader(k, v)
    conn.putheader("Content-Length", str(file_size))
    conn.endheaders()
    with file_path.open("rb") as f:
        while True:
            chunk = f.read(CHUNK)
            if not chunk:
                break
            conn.send(chunk)
    resp = conn.getresponse()
    body = resp.read()
    h = {k.lower(): v for k, v in resp.getheaders()}
    return resp.status, h, body


def get_url(
    url: str,
    headers: dict[str, str],
    timeout: float,
) -> tuple[int, dict[str, str], bytes]:
    parsed = urlparse(url)
    conn = _connect(parsed, timeout)
    conn.request("GET", _path_qs(parsed), headers=headers)
    resp = conn.getresponse()
    body = resp.read()
    h = {k.lower(): v for k, v in resp.getheaders()}
    return resp.status, h, body


def main() -> int:
    args = parse_args()
    key = _read_key(args.key, args.key_file)
    file_path = Path(args.file).expanduser().resolve()
    if not file_path.is_file():
        sys.exit(f"ERROR: not a file: {file_path}")
    filename = args.filename or file_path.name
    api = args.api.rstrip("/")
    url = f"{api}/jobs?{urlencode({'filename': filename})}"
    headers = {
        "X-API-Key": key,
        "Content-Type": "application/octet-stream",
    }
    if args.idempotency_key:
        headers["X-Idempotency-Key"] = args.idempotency_key

    size = file_path.stat().st_size
    print(f"POST {url}")
    print(f"  body: {file_path} ({size:,} bytes)")

    t0 = time.time()
    status, h, body = post_streamed(url, headers, file_path, args.timeout)
    print(f"  -> {status} in {time.time() - t0:.1f}s")

    # Modal "delayed response" loop: 303/307 -> GET Location until 2xx.
    hops = 0
    while status in (303, 307) and "location" in h:
        hops += 1
        if hops > 5:
            print(f"ERROR: redirect loop (>5 hops). last Location: {h.get('location')}",
                  file=sys.stderr)
            return 1
        loc = h["location"]
        if loc.startswith("/"):
            loc = f"{api}{loc}"
        print(f"  redirect ({status}) -> GET {loc}")
        status, h, body = get_url(loc, {"X-API-Key": key}, args.timeout)
        print(f"  -> {status}")

    if status != 200:
        print(f"ERROR: status={status}", file=sys.stderr)
        print(f"  content-type: {h.get('content-type', '')}", file=sys.stderr)
        print(f"  body: {body[:500].decode('utf-8', 'replace')}", file=sys.stderr)
        return 1

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        print("ERROR: 200 with non-JSON body", file=sys.stderr)
        print(body[:500].decode("utf-8", "replace"), file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=2))

    job_id = payload.get("job_id")
    if not args.poll or not job_id:
        return 0

    print(f"\nPolling {api}/jobs/{job_id}")
    while True:
        s, _, sb = get_url(f"{api}/jobs/{job_id}", {"X-API-Key": key}, args.timeout)
        if s != 200:
            print(f"  {s} {sb[:200].decode('utf-8', 'replace')}")
            time.sleep(args.poll_interval)
            continue
        try:
            sp = json.loads(sb)
        except json.JSONDecodeError:
            print(f"  bad JSON: {sb[:200].decode('utf-8', 'replace')}")
            time.sleep(args.poll_interval)
            continue
        st = sp.get("status", "?")
        stage = sp.get("stage", "")
        print(f"  status={st} stage={stage}")
        if st in ("done", "failed"):
            print(json.dumps(sp, indent=2))
            return 0 if st == "done" else 2
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    sys.exit(main())
