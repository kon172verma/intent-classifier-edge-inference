"""Load a compiled TensorRT-LLM engine and the matching HF tokenizer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evaluation_lib.config import (
    MODEL_DISPLAY_NAMES,
    MODEL_PATHS,
    TENSORRT_DIR,
    MODEL_TENSORRT_STEMS,
    TENSORRT_DTYPES,
)


def engine_dir(model_key: str, dtype: str) -> Path:
    """Return the compiled-engine directory path for *model_key* at *dtype*."""
    stem = MODEL_TENSORRT_STEMS[model_key]
    return TENSORRT_DIR / f"{stem}-{dtype}"


def load_session(model_key: str, dtype: str) -> tuple[Any, Path]:
    """Load the TensorRT-LLM ``ModelRunner`` for *model_key* at *dtype*.

    The ModelRunner wraps the compiled ``.engine`` file and exposes a
    ``generate()`` method that handles KV-cache management internally.

    Returns
    -------
    runner : tensorrt_llm.runtime.ModelRunner
        Ready-to-use runner instance.
    engine_dir_path : Path
        Path to the directory that holds the engine + config.

    Raises
    ------
    FileNotFoundError
        If the compiled engine directory is missing (build it first -- see
        evaluation_tensorrt/readme.md for the convert + build steps).
    ImportError
        If ``tensorrt_llm`` is not installed in the current environment.
    """
    try:
        from tensorrt_llm.runtime import ModelRunner  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "tensorrt_llm is not installed. "
            "Install it on a system with a compatible NVIDIA GPU following "
            "https://nvidia.github.io/TensorRT-LLM/installation/linux.html"
        ) from exc

    dir_path = engine_dir(model_key, dtype)
    if not dir_path.exists():
        raise FileNotFoundError(
            f"{dir_path} not found -- convert the HF checkpoint and build "
            "the TensorRT-LLM engine first (see evaluation_tensorrt/readme.md)."
        )

    print(
        f"[model] Loading {MODEL_DISPLAY_NAMES[model_key]} ({dtype})"
        f" from {dir_path.name}"
    )
    runner = ModelRunner.from_dir(engine_dir=str(dir_path), rank=0)
    print("[model] Ready.\n")
    return runner, dir_path


def load_tokenizer(model_key: str) -> Any:
    """Load the HF tokenizer for *model_key*.

    TensorRT-LLM does not bundle a tokenizer inside the compiled engine, so
    we load the original HF tokenizer directly from the HF checkpoint
    directory (same source the ONNX and baseline packages use).
    """
    from transformers import AutoTokenizer  # type: ignore[import]

    model_path = MODEL_PATHS[model_key]
    return AutoTokenizer.from_pretrained(
        str(model_path), clean_up_tokenization_spaces=False
    )


def engine_size_mb(dir_path: Path) -> float:
    """Return the total size in MB of all ``.engine`` files in *dir_path*."""
    total = sum(p.stat().st_size for p in dir_path.iterdir() if p.suffix == ".engine")
    return round(total / (1024**2), 2)
