"""Load HF Transformers models for baseline evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation_lib.config import MODEL_DISPLAY_NAMES, MODEL_PATHS

DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def load_model_and_tokenizer(
    model_key: str,
    device: str,
    dtype_name: str = "float16",
    model_path: Path | None = None,
) -> tuple[Any, Any]:
    """Load the tokenizer and model for *model_key*, placing the model on *device*.

    Parameters
    ----------
    dtype_name:
        One of ``"float32"``, ``"bfloat16"``, ``"float16"``. Note that
        float16 matmuls on CPU fall back to slow/unoptimized kernels in
        PyTorch (no oneDNN/MKL-DNN fast path), so float32 or bfloat16 is
        strongly recommended for CPU benchmarking.

    Notes
    -----
    The model is first loaded to CPU and then moved to *device*.  Loading
    directly with ``device_map="mps"`` causes a segfault on some transformers
    builds; the two-step approach is safe on all backends.
    """
    model_path = model_path or MODEL_PATHS[model_key]
    print(f"[model] Loading tokenizer from {model_path}")
    tokenizer: Any = AutoTokenizer.from_pretrained(str(model_path))
    template_path = model_path / "chat_template.jinja"
    if not getattr(tokenizer, "chat_template", None) and template_path.is_file():
        tokenizer.chat_template = template_path.read_text(encoding="utf-8")
        print(f"[model] Loaded chat template from {template_path.name}")

    dtype = DTYPE_MAP[dtype_name]
    print(
        f"[model] Loading {MODEL_DISPLAY_NAMES.get(model_key, model_key)}"
        f" → device={device}, dtype={dtype}"
    )

    model: Any = AutoModelForCausalLM.from_pretrained(str(model_path), torch_dtype=dtype)
    if device != "cpu":
        model = model.to(device)  # type: ignore[arg-type]
    model.eval()
    print("[model] Ready.\n")
    return model, tokenizer
