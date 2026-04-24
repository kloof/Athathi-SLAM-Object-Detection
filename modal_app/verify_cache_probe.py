"""Build-time HF cache verification probe.

Runs inside the Modal image build with HF_HUB_OFFLINE=1 — if the
local cache cannot satisfy both SpatialLM checkpoints' tokenizers via
``local_files_only=True``, the script exits non-zero and Modal aborts
the image build. Layer 1 of the three-layer model-cache verification
(see docs/superpowers/specs/2026-04-24-modal-api-design.md).

Intentionally loads only ``AutoTokenizer``, not ``AutoConfig``. The
SpatialLM checkpoints declare a custom ``model_type`` (e.g.
``spatiallm_qwen``) that transformers does not recognize until
``import spatiallm`` registers it via its entrypoints — which is a
heavyweight import (torch/spconv/flash-attn) unnecessary here. The
tokenizer alone is enough to prove the cache files are present and
offline-loadable; the actual inference path (SpatialLM's own
``inference.py``) handles custom-arch registration on its own.
"""
from transformers import AutoTokenizer

for m in (
    "manycore-research/SpatialLM1.1-Qwen-0.5B",
    "manycore-research/SpatialLM1.1-Llama-1B",
):
    AutoTokenizer.from_pretrained(m, local_files_only=True)

print("cache verified")
