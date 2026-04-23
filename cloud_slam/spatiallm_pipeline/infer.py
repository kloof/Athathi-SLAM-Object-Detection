"""Subprocess wrapper around third_party/SpatialLM/inference.py.

SpatialLM requires torch/transformers + flash-attn + spconv pinned
differently from the main rgbd venv; it lives in ~/spatiallm_env. We
shell out to its python and parse the resulting layout.txt.

Tuned defaults: temperature=0.3, top_k=3 (empirically the setting that
avoids the hallucination-loop failure mode while keeping class diversity).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path


SPATIALLM_DIR = Path("/home/klof/cloud_slam_icp/third_party/SpatialLM")
SPATIALLM_PY = Path("/home/klof/spatiallm_env/bin/python")
CODE_TEMPLATE = SPATIALLM_DIR / "code_template.txt"

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
    timeout_s: int = 900,
    verbose: bool = True,
) -> Path:
    """Run SpatialLM inference on the given PLY and dump layout to TXT."""
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
    ]

    if verbose:
        print(f"[infer] model={model}  det={detect_type}  t={temperature} "
              f"top_k={top_k}")
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
