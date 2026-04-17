"""Standalone SpatialLM inference worker.

This script runs inside ``~/spatiallm_env`` (Python 3.10 + torch 2.4.1 +
flash-attn + spconv + spatiallm). It is NOT imported by the main pipeline —
the main env (torch 2.11) spawns it as a subprocess and communicates via
file-based IPC (drop .ply into inbox, pick up .txt from outbox).

Protocol
--------
  inbox/<job>.ply            (point cloud; colors optional)
  inbox/<job>.meta.json      ({"job_id": str, "categories": [str, ...]})
  outbox/<job>.txt           (SpatialLM structured-language output)
  outbox/<job>.meta.json     ({"status": "ok"|"empty"|"error", ...})

All writes are atomic (write .tmp then rename). The worker prints
``WORKER_READY`` on stderr once the model is loaded so the client can
unblock. On SIGTERM/SIGINT it logs ``worker exiting`` and returns cleanly.
"""

import argparse
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path


def _log(msg: str) -> None:
    """Timestamped stderr logger (flushed eagerly since client tails it)."""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] spatiallm_worker: {msg}", file=sys.stderr, flush=True)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.rename(path)


def _atomic_write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data))
    tmp.rename(path)


def _count_entities(language_string: str) -> int:
    """Count wall/door/window/bbox entity lines in a SpatialLM output."""
    if not language_string:
        return 0
    n = 0
    for line in language_string.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        label = line.split("=", 1)[0]
        head = label.split("_", 1)[0]
        if head in ("wall", "door", "window", "bbox"):
            n += 1
    return n


def _load_model(model_path: str, inference_dtype: str):
    """Load tokenizer + model once; called exactly at startup."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    # Import spatiallm to register its custom `spatiallm_qwen` /
    # `spatiallm_llama` model types with transformers' AutoConfig (side-effect
    # of spatiallm/__init__.py → spatiallm/model/*). Without this, AutoModel
    # raises KeyError: 'spatiallm_qwen' (upstream issue #107).
    import spatiallm  # noqa: F401

    _log(f"loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    _log(f"loading model from {model_path} (dtype={inference_dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=getattr(torch, inference_dtype)
    )
    model.to("cuda")
    model.set_point_backbone_dtype(torch.float32)
    model.eval()
    _log(f"model loaded; num_bins={model.config.point_config['num_bins']}")
    return tokenizer, model


def _downsample_to_cap(pcd, max_points: int, voxel_start: float):
    """Progressively voxel-downsample until point count is at or below cap."""
    import numpy as np

    voxel = voxel_start
    current = pcd
    attempts = 0
    while len(current.points) > max_points and attempts < 12:
        voxel *= 1.3
        current = pcd.voxel_down_sample(voxel)
        attempts += 1
    if attempts > 0:
        _log(f"downsampled to {len(current.points)} pts (voxel={voxel:.4f}m, attempts={attempts})")
    return current, voxel


def _run_inference(tokenizer, model, pcd, code_template_file: str,
                   categories: list, detect_type: str,
                   max_points: int) -> tuple:
    """One inference pass. Returns (language_string, point_count, retry_voxel)."""
    import numpy as np
    import torch
    from spatiallm import Layout
    from spatiallm.pcd import get_points_and_colors, cleanup_pcd

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent
                           / "third_party" / "SpatialLM"))
    from inference import preprocess_point_cloud, generate_layout

    num_bins = model.config.point_config["num_bins"]
    grid_size = Layout.get_grid_size(num_bins)

    pcd = cleanup_pcd(pcd, voxel_size=grid_size)
    pcd, final_voxel = _downsample_to_cap(pcd, max_points, grid_size)

    points, colors = get_points_and_colors(pcd)
    if len(points) == 0:
        return "", 0, final_voxel

    min_extent = np.min(points, axis=0)
    input_pcd = preprocess_point_cloud(points, colors, grid_size, num_bins)

    layout = generate_layout(
        model,
        input_pcd,
        tokenizer,
        code_template_file,
        temperature=0.6,
        seed=-1,
        detect_type=detect_type,
        categories=categories,
    )
    layout.translate(min_extent)
    return layout.to_language_string(), len(points), final_voxel


def _process_job(ply_path: Path, meta_path: Path, outbox: Path,
                 tokenizer, model, code_template_file: str,
                 default_categories: list, detect_type: str,
                 max_points: int) -> None:
    """Run a single inference job end-to-end; always writes a meta.json."""
    import open3d as o3d

    t0 = time.time()
    job_id = ply_path.stem
    categories = default_categories
    if meta_path.exists():
        try:
            meta_in = json.loads(meta_path.read_text())
            if "categories" in meta_in and meta_in["categories"]:
                categories = list(meta_in["categories"])
            if "job_id" in meta_in:
                job_id = meta_in["job_id"]
        except Exception as exc:
            _log(f"meta parse failed for {meta_path.name}: {exc}")

    status = "ok"
    error = ""
    language_string = ""
    point_count = 0
    try:
        _log(f"job {job_id}: loading {ply_path.name}")
        pcd = o3d.io.read_point_cloud(str(ply_path))
        if len(pcd.points) == 0:
            raise RuntimeError("input cloud has zero points")

        language_string, point_count, voxel = _run_inference(
            tokenizer, model, pcd, code_template_file,
            categories, detect_type, max_points)

        if _count_entities(language_string) == 0:
            _log(f"job {job_id}: empty first pass; retrying with 2x voxel")
            import copy
            pcd_retry = copy.deepcopy(pcd).voxel_down_sample(voxel * 2)
            if len(pcd_retry.points) > 0:
                language_string, point_count, _ = _run_inference(
                    tokenizer, model, pcd_retry, code_template_file,
                    categories, detect_type, max_points)
            if _count_entities(language_string) == 0:
                status = "empty"
                _log(f"job {job_id}: still empty after retry — SpatialLM issue #81")
    except Exception as exc:
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
        _log(f"job {job_id}: ERROR {error}")
        _log(traceback.format_exc())

    elapsed = time.time() - t0
    out_txt = outbox / f"{job_id}.txt"
    out_meta = outbox / f"{job_id}.meta.json"
    _atomic_write_text(out_txt, language_string)
    _atomic_write_json(out_meta, {
        "status": status,
        "inference_seconds": round(elapsed, 3),
        "point_count": int(point_count),
        "model_path": str(args.model_path) if 'args' in globals() else "",
        "error": error,
    })
    _log(f"job {job_id}: {status} in {elapsed:.2f}s ({point_count} pts, "
         f"{_count_entities(language_string)} entities)")

    try:
        ply_path.unlink()
    except FileNotFoundError:
        pass
    try:
        if meta_path.exists():
            meta_path.unlink()
    except FileNotFoundError:
        pass


_SHUTDOWN = False


def _handle_signal(signum, frame):
    global _SHUTDOWN
    _SHUTDOWN = True
    _log(f"received signal {signum}; will exit after current job")


def main() -> int:
    global args
    parser = argparse.ArgumentParser("SpatialLM inference worker")
    parser.add_argument("--inbox", type=str, required=True,
                        help="Directory to poll for .ply job files")
    parser.add_argument("--outbox", type=str, required=True,
                        help="Directory to write .txt results")
    parser.add_argument("--model_path", type=str,
                        default="manycore-research/SpatialLM1.1-Qwen-0.5B")
    parser.add_argument("--detect_type", type=str, default="all",
                        choices=["all", "arch", "object"])
    parser.add_argument("--category", nargs="*", default=[])
    parser.add_argument("--max_points", type=int, default=200000)
    parser.add_argument("--inference_dtype", type=str, default="bfloat16")
    parser.add_argument("--code_template_file", type=str, default="")
    parser.add_argument("--poll_interval", type=float, default=0.5)
    args = parser.parse_args()

    inbox = Path(args.inbox).resolve()
    outbox = Path(args.outbox).resolve()
    inbox.mkdir(parents=True, exist_ok=True)
    outbox.mkdir(parents=True, exist_ok=True)

    code_template_file = args.code_template_file or str(
        Path(__file__).resolve().parent.parent.parent
        / "third_party" / "SpatialLM" / "code_template.txt")
    if not Path(code_template_file).exists():
        _log(f"FATAL: code_template_file not found at {code_template_file}")
        return 2

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent
                               / "third_party" / "SpatialLM"))
        tokenizer, model = _load_model(args.model_path, args.inference_dtype)
    except Exception as exc:
        _log(f"FATAL: model load failed: {exc}")
        _log(traceback.format_exc())
        return 2

    print("WORKER_READY", file=sys.stderr, flush=True)
    _log(f"polling {inbox} for jobs")

    while not _SHUTDOWN:
        try:
            entries = sorted(p for p in inbox.iterdir()
                             if p.suffix == ".ply"
                             and not p.name.endswith(".ply.tmp")
                             and not p.name.startswith("."))
        except FileNotFoundError:
            time.sleep(args.poll_interval)
            continue

        if not entries:
            time.sleep(args.poll_interval)
            continue

        for ply_path in entries:
            if _SHUTDOWN:
                break
            meta_path = ply_path.with_suffix(".meta.json")
            try:
                _process_job(ply_path, meta_path, outbox, tokenizer, model,
                             code_template_file, args.category,
                             args.detect_type, args.max_points)
            except Exception as exc:
                _log(f"unhandled job error: {exc}")
                _log(traceback.format_exc())

    _log("worker exiting")
    return 0


if __name__ == "__main__":
    sys.exit(main())
