"""Load quantized GGUF models via llama-cpp-python for evaluation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from llama_cpp import Llama

from evaluation_lib.config import (
    MODEL_DISPLAY_NAMES,
    MODEL_GGUF_DIRS,
    MODEL_GGUF_STEMS,
    N_CTX_DEFAULT,
)


def gguf_model_path(model_key: str, quant: str) -> Path:
    """Return the expected GGUF file path for *model_key* at *quant* level."""
    stem = MODEL_GGUF_STEMS[model_key]
    return MODEL_GGUF_DIRS[model_key] / f"{stem}-{quant}.gguf"


def load_model(
    model_key: str,
    quant: str,
    device: str,
    n_ctx: int = N_CTX_DEFAULT,
    model_path: Path | None = None,
) -> Llama:
    """Load the quantized GGUF model for *model_key* at *quant* level.

    Parameters
    ----------
    device:
        ``"mps"`` offloads all layers to the Metal GPU (``n_gpu_layers=-1``).
        ``"cpu"`` forces CPU-only inference (``n_gpu_layers=0``).
    n_ctx:
        Context window size (must be >= the longest prompt in the dataset).
    """
    model_path = model_path or gguf_model_path(model_key, quant)
    if not model_path.exists():
        raise FileNotFoundError(
            f"GGUF model not found: {model_path}\n"
            "Convert + quantize the HF checkpoint first -- see "
            "evaluation_llama_cpp/readme.md for the conversion steps."
        )

    n_gpu_layers = -1 if device == "mps" else 0
    print(
        f"[model] Loading {MODEL_DISPLAY_NAMES.get(model_key, model_key)} ({quant})"
        f" from {model_path.name} → device={device} (n_gpu_layers={n_gpu_layers})"
    )
    llm = Llama(
        model_path=str(model_path),
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        n_threads=os.cpu_count(),
        verbose=False,
    )
    print("[model] Ready.\n")
    return llm


def gguf_model_size_mb(model_key: str, quant: str, model_path: Path | None = None) -> float:
    """Return the on-disk GGUF file size in MB (analogous to model_weights_mb)."""
    model_path = model_path or gguf_model_path(model_key, quant)
    return round(model_path.stat().st_size / (1024 * 1024), 2)


def load_text_tokenizer(model_path: Any) -> Any:
    """Load the original HF tokenizer, used ONLY for chat-template text rendering.

    ``evaluation_lib.prompt`` builds prompt text via
    ``tokenizer.apply_chat_template()`` — that logic is reused unmodified so
    both benchmarks (HF and llama.cpp) construct byte-identical prompt text.
    The actual token IDs fed to the GGUF model come from ``Llama.tokenize()``
    instead (see evaluation_llama_cpp/inference.py), so no model weights are
    loaded here — this is metadata/text-only.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(model_path))
