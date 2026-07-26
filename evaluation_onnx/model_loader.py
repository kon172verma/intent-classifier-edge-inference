"""Load ONNX Runtime sessions for the evaluation_onnx benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import onnxruntime as ort

from evaluation_lib.config import MODEL_DISPLAY_NAMES, MODEL_ONNX_STEMS, ONNX_DIR

# Execution providers to try, in priority order, per --device value. ORT
# falls back to the next provider in the list for any op it can't run on
# the preceding one (CoreML support for LLM decoder graphs is partial), so
# CPUExecutionProvider is always included as a safety net.
_PROVIDERS_BY_DEVICE: dict[str, list[str]] = {
    "cpu": ["CPUExecutionProvider"],
    "coreml": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
}


def onnx_model_path(model_key: str, precision: str) -> Path:
    """Return the ``model.onnx`` path for *model_key* at *precision*."""
    stem = MODEL_ONNX_STEMS[model_key]
    return ONNX_DIR / f"{stem}-{precision}" / "model.onnx"


def onnx_model_size_mb(model_key: str, precision: str) -> float:
    """Return the on-disk size (MB) of the model, including any external-data file.

    Mirrors ``model_weights_mb``/``gguf_model_size_mb`` in the other
    evaluation packages -- a static, precision-dependent weights size to
    report alongside runtime metrics.
    """
    model_path = onnx_model_path(model_key, precision)
    total_bytes = model_path.stat().st_size
    external_data = model_path.with_name(model_path.name + "_data")
    if external_data.exists():
        total_bytes += external_data.stat().st_size
    else:
        # onnxruntime.quantization writes the sibling file as
        # "<name>.data" instead of "<name>_data" -- check both spellings.
        external_data = model_path.with_suffix(model_path.suffix + ".data")
        if external_data.exists():
            total_bytes += external_data.stat().st_size
    return round(total_bytes / (1024 * 1024), 2)


def load_session(model_key: str, precision: str, device: str) -> ort.InferenceSession:
    """Load an ONNX Runtime inference session for *model_key* at *precision*.

    Parameters
    ----------
    device:
        ``"cpu"`` or ``"coreml"``. CoreML routes ops through Apple's ANE/GPU
        via the CoreML framework; unsupported ops silently fall back to CPU
        within the same session (ORT's provider-fallback mechanism).
    """
    model_path = onnx_model_path(model_key, precision)
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found -- export/quantize this model first "
            f"(see evaluation_onnx/readme.md)."
        )

    providers = _PROVIDERS_BY_DEVICE[device]
    print(
        f"[model] Loading {MODEL_DISPLAY_NAMES[model_key]} ({precision})"
        f" from {model_path.parent.name} → providers={providers}"
    )
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(model_path), sess_options=sess_options, providers=providers
    )
    print(f"[model] Ready. (active providers: {session.get_providers()})\n")
    return session


def load_text_tokenizer(model_path: Any) -> Any:
    """Load the original HF tokenizer, used for chat-template rendering AND
    actual tokenization (unlike evaluation_llama_cpp, the ONNX graph has no
    embedded tokenizer of its own -- the HF tokenizer's token ids are fed
    directly to the ONNX session).
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(model_path), clean_up_tokenization_spaces=False
    )
