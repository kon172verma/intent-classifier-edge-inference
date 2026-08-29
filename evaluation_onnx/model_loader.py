"""Load ONNX Runtime sessions for the evaluation_onnx benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import onnxruntime as ort

from evaluation_lib.config import MODEL_DISPLAY_NAMES, MODEL_ONNX_DIRS

# Execution providers to try, in priority order, per --device value. ORT
# falls back to the next provider in the list for any op it can't run on
# the preceding one (CoreML support for LLM decoder graphs is partial), so
# CPUExecutionProvider is always included as a safety net.
#
# QNN is handled separately (see _qnn_providers()) because it requires
# per-call provider options (backend_path, etc.).
_PROVIDERS_BY_DEVICE: dict[str, list[str]] = {
    "cpu": ["CPUExecutionProvider"],
    "coreml": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
}

# QNN backend names -> shared-library stem (prefix/suffix added at runtime).
_QNN_BACKEND_LIB: dict[str, str] = {
    "htp": "QnnHtp",  # Hexagon DSP -- primary target for LLM inference
    "gpu": "QnnGpu",  # Adreno GPU  -- good for fp32/fp16
    "cpu": "QnnCpu",  # CPU reference backend (slow; debug use only)
}


def _qnn_lib_name(backend: str) -> str:
    """Return the OS-appropriate QNN shared-library filename for *backend*.

    On Linux/Android the convention is ``lib<Name>.so``; on Windows it is
    ``<Name>.dll``.  The caller can override the full path via
    ``load_session(..., qnn_lib_path=...)``.
    """
    import platform as _platform

    stem = _QNN_BACKEND_LIB[backend]
    if _platform.system() == "Windows":
        return f"{stem}.dll"
    return f"lib{stem}.so"


def _qnn_providers(backend: str, precision: str, lib_path: str | None) -> list:
    """Build the QNN provider tuple consumed by ``ort.InferenceSession``.

    Returns a list whose first element is a ``(provider_name, options_dict)``
    tuple (the ORT API accepts either bare strings or 2-tuples for providers
    that need options), followed by ``CPUExecutionProvider`` as fallback.

    Parameters
    ----------
    backend:
        One of ``"htp"``, ``"gpu"``, or ``"cpu"``.
    precision:
        The ONNX model precision string (``"fp32"``, ``"fp16"``, etc.).
        Used to set ``enable_htp_fp16_precision`` automatically.
    lib_path:
        Full path to the QNN backend library, or ``None`` to use the
        OS-default name (resolved by ``_qnn_lib_name``).
    """
    resolved_lib = lib_path if lib_path else _qnn_lib_name(backend)
    options: dict[str, str] = {
        "backend_path": resolved_lib,
        # Request FP16 arithmetic on HTP when the model is FP16; ignored by
        # other backends.  String values are required by the ORT C++ API.
        "enable_htp_fp16_precision": "1" if precision == "fp16" else "0",
        # "burst" keeps the HTP clocks high during inference for lowest
        # latency; use "balanced" or "low_power" for battery-sensitive runs.
        "htp_performance_mode": "burst",
    }
    return [("QnnExecutionProvider", options), "CPUExecutionProvider"]


def onnx_model_path(model_key: str, precision: str) -> Path:
    """Return the ``model.onnx`` path for *model_key* at *precision*."""
    return MODEL_ONNX_DIRS[model_key] / precision / "model.onnx"


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


def load_session(
    model_key: str,
    precision: str,
    device: str,
    qnn_backend: str = "htp",
    qnn_lib_path: str | None = None,
) -> ort.InferenceSession:
    """Load an ONNX Runtime inference session for *model_key* at *precision*.

    Parameters
    ----------
    device:
        ``"cpu"``, ``"coreml"``, or ``"qnn"``.
        CoreML routes ops through Apple's ANE/GPU via the CoreML framework;
        unsupported ops fall back to CPU within the same session.
        QNN routes ops through the Qualcomm AI Engine (HTP/Adreno/CPU
        backend) via the QNN SDK; unsupported ops also fall back to CPU.
    qnn_backend:
        QNN backend to use when *device* is ``"qnn"``.
        ``"htp"`` (Hexagon DSP, default) gives the best throughput for
        LLM inference on Snapdragon SoCs; ``"gpu"`` targets the Adreno GPU;
        ``"cpu"`` is a reference backend for debugging.
    qnn_lib_path:
        Optional explicit path to the QNN backend shared library
        (e.g. ``"/opt/qcom/qnn/lib/aarch64-android/libQnnHtp.so"``).
        When ``None`` the OS-default library name is used.
    """
    model_path = onnx_model_path(model_key, precision)
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found -- export/quantize this model first "
            f"(see evaluation_onnx/readme.md)."
        )

    if device == "qnn":
        providers = _qnn_providers(qnn_backend, precision, qnn_lib_path)
    else:
        providers = _PROVIDERS_BY_DEVICE[device]
    print(
        f"[model] Loading {MODEL_DISPLAY_NAMES[model_key]} ({precision})"
        f" from {model_path.parent.name} → providers={providers}"
    )
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(model_path), sess_options=sess_options, providers=providers)
    print(f"[model] Ready. (active providers: {session.get_providers()})\n")
    return session


def load_text_tokenizer(model_path: Any) -> Any:
    """Load the original HF tokenizer, used for chat-template rendering AND
    actual tokenization (unlike evaluation_llama_cpp, the ONNX graph has no
    embedded tokenizer of its own -- the HF tokenizer's token ids are fed
    directly to the ONNX session).
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(model_path), clean_up_tokenization_spaces=False)
