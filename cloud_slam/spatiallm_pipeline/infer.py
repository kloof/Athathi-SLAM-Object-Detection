"""Subprocess wrapper around third_party/SpatialLM/inference.py.

SpatialLM requires torch/transformers + flash-attn + spconv pinned
differently from the main rgbd venv; it lives in ~/spatiallm_env. We
shell out to its python and parse the resulting layout.txt.

Tuned defaults:
  temperature=0.3, top_k=3 — empirically avoids the hallucination-loop
  failure mode while keeping class diversity.
  repetition_penalty=1.15 — suppresses Llama's window-repetition loop
  (200+ duplicate window_N=Window(...) lines).
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path


_REPO = Path(__file__).resolve().parents[2]
SPATIALLM_DIR = Path(os.getenv("SPATIALLM_DIR", _REPO / "third_party" / "SpatialLM"))
SPATIALLM_PY = Path(os.getenv("SPATIALLM_PY", Path.home() / "spatiallm_env" / "bin" / "python"))
CODE_TEMPLATE = Path(os.getenv("SPATIALLM_CODE_TEMPLATE", SPATIALLM_DIR / "code_template.txt"))

MODEL_QWEN = "manycore-research/SpatialLM1.1-Qwen-0.5B"
MODEL_LLAMA = "manycore-research/SpatialLM1.1-Llama-1B"


def run_spatiallm(
    input_ply: Path | str,
    output_txt: Path | str,
    *,
    model: str = MODEL_QWEN,
    detect_type: str = "all",
    temperature: float = 0.3,
    top_k: int = 3,
    repetition_penalty: float = 1.15,
    seed: int = -1,
    num_beams: int = 1,
    timeout_s: int = 900,
    verbose: bool = True,
) -> Path:
    """Run a single SpatialLM inference pass on the given PLY.

    When num_beams>1 the worker uses deterministic beam search
    (do_sample=False) instead of sampling — trades multi-pass consensus
    for a single deterministic pass that explores num_beams hypotheses.
    """
    input_ply = Path(input_ply).resolve()
    output_txt = Path(output_txt).resolve()
    if not input_ply.is_file():
        raise FileNotFoundError(input_ply)
    if not SPATIALLM_PY.is_file():
        raise FileNotFoundError(f"SpatialLM venv python missing at {SPATIALLM_PY}")
    if not CODE_TEMPLATE.is_file():
        raise FileNotFoundError(f"code_template.txt missing at {CODE_TEMPLATE}")

    cmd = [
        str(SPATIALLM_PY),
        "inference.py",
        "-p", str(input_ply),
        "-o", str(output_txt),
        "-m", model,
        "-d", detect_type,
        "-t", str(CODE_TEMPLATE),
        "--temperature", str(temperature),
        "--top_k", str(top_k),
        "--repetition_penalty", str(repetition_penalty),
        "--seed", str(seed),
        "--num_beams", str(num_beams),
    ]

    if verbose:
        mode = f"beam(n={num_beams})" if num_beams > 1 else f"sample(t={temperature} k={top_k})"
        print(f"[infer] model={model}  det={detect_type}  mode={mode}  "
              f"rep_pen={repetition_penalty}  seed={seed}")
        print(f"        in:  {input_ply}")
        print(f"        out: {output_txt}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(SPATIALLM_DIR),
                          capture_output=True, text=True, timeout=timeout_s)
    if verbose:
        print(f"[infer] finished exit={proc.returncode} in {time.time()-t0:.1f}s")
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-20:])
        raise RuntimeError(f"SpatialLM failed (exit {proc.returncode}):\n{tail}")
    if not output_txt.is_file():
        raise RuntimeError(f"SpatialLM did not write {output_txt}")
    return output_txt


def run_spatiallm_multi_seed(
    input_ply: Path | str,
    output_dir: Path | str,
    seeds: list[int],
    *,
    output_stem: str = "layout",
    model: str = MODEL_LLAMA,
    detect_type: str = "all",
    temperature: float = 0.3,
    top_k: int = 3,
    repetition_penalty: float = 1.15,
    timeout_s: int = 1800,
    verbose: bool = True,
) -> list[Path]:
    """Run N SpatialLM passes sharing a single model load.

    Returns paths to the N generated layout files, one per seed, named
    ``<output_stem>_seed<N>.txt`` inside ``output_dir``.
    """
    input_ply = Path(input_ply).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not seeds:
        raise ValueError("seeds must be non-empty")

    # Use a temp directory-style "-o" so inference.py's multi-seed branch
    # writes <ply-stem>_seed<N>.txt; rename to <output_stem>_seed<N>.txt.
    seeds_str = ",".join(str(s) for s in seeds)
    cmd = [
        str(SPATIALLM_PY),
        "inference.py",
        "-p", str(input_ply),
        "-o", str(output_dir),
        "-m", model,
        "-d", detect_type,
        "-t", str(CODE_TEMPLATE),
        "--temperature", str(temperature),
        "--top_k", str(top_k),
        "--repetition_penalty", str(repetition_penalty),
        "--seeds", seeds_str,
    ]

    if verbose:
        print(f"[infer] multi-seed model={model} seeds={seeds} "
              f"t={temperature} top_k={top_k} rep_pen={repetition_penalty}")
        print(f"        in:  {input_ply}")
        print(f"        out: {output_dir}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(SPATIALLM_DIR),
                          capture_output=True, text=True, timeout=timeout_s)
    if verbose:
        print(f"[infer] finished exit={proc.returncode} "
              f"in {time.time()-t0:.1f}s ({len(seeds)} passes)")
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-20:])
        raise RuntimeError(f"SpatialLM failed (exit {proc.returncode}):\n{tail}")

    ply_stem = input_ply.stem
    written = []
    for seed in seeds:
        src = output_dir / f"{ply_stem}_seed{seed}.txt"
        dst = output_dir / f"{output_stem}_seed{seed}.txt"
        if not src.is_file():
            raise RuntimeError(f"SpatialLM did not write {src}")
        if src != dst:
            src.replace(dst)
        written.append(dst)
    return written
