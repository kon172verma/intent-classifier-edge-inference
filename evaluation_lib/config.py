"""Project-wide constants: model registry, dataset paths, generation settings."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT: Path = Path(__file__).parent.parent

MODEL_RUNS: dict[str, str] = {
    "qwen3": "qwen3-0.6b_LoRA_C_1k",
    "llama3": "llama3.2-1b_LoRA_C_1k",
}

MODEL_PATHS: dict[str, Path] = {
    key: REPO_ROOT / "models" / f"{run}_merged" / "safetensors" for key, run in MODEL_RUNS.items()
}

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "qwen3": "Qwen3-0.6B",
    "llama3": "Llama-3.2-1B",
}

DATASET_DEFAULT: Path = REPO_ROOT / "dataset_full" / "sample_0001.json"

MAX_NEW_TOKENS: int = 32
WARMUP_EXAMPLES: int = 2

# GGUF model files for the llama.cpp evaluation (see evaluation_llama_cpp/).
# Filenames follow <stem>-<QUANT>.gguf inside each model's gguf/ subfolder.
MODEL_GGUF_DIRS: dict[str, Path] = {
    key: REPO_ROOT / "models" / f"{run}_merged" / "gguf" for key, run in MODEL_RUNS.items()
}

MODEL_GGUF_STEMS: dict[str, str] = {
    "qwen3": "qwen3-0.6b",
    "llama3": "llama3.2-1b",
}

# Quantization levels benchmarked for evaluation_llama_cpp.
QUANT_LEVELS: list[str] = ["Q8_0", "Q6_K", "Q4_K_M"]

N_CTX_DEFAULT: int = 2048

# ONNX model directories for the evaluation_onnx benchmark. Precision
# directories live under each model's onnx/ subfolder.
MODEL_ONNX_DIRS: dict[str, Path] = {
    key: REPO_ROOT / "models" / f"{run}_merged" / "onnx" for key, run in MODEL_RUNS.items()
}

MODEL_ONNX_STEMS: dict[str, str] = {
    "qwen3": "qwen3-0.6b",
    "llama3": "llama3.2-1b",
}

# Precision variants benchmarked for evaluation_onnx. Directory names follow
# <precision>/model.onnx under MODEL_ONNX_DIRS[model_key].
ONNX_PRECISIONS: list[str] = ["fp32", "fp16", "dynamic-int8", "static-int8"]

# TensorRT-LLM engine directory for the evaluation_tensorrt evaluation (see
# evaluation_tensorrt/readme.md for the convert + build steps).
TENSORRT_DIR: Path = REPO_ROOT / "models" / "tensorrt"

MODEL_TENSORRT_STEMS: dict[str, str] = {
    "qwen3": "qwen3-0.6b",
    "llama3": "llama3.2-1b",
}

# Dtype variants benchmarked for evaluation_tensorrt. Directory names follow
# <stem>-<DTYPE>/, produced by scripts/build_trt_engine.py.
# fp16 is the primary Jetson Orin target (SM87, no BF16 hardware support).
# bf16 is available on Ampere/Ada/Hopper data-centre GPUs.
TENSORRT_DTYPES: list[str] = ["fp16", "bf16", "int8", "int4"]

# Static system prompt used for all tool-routing evaluations.
# This is also the cacheable prefix in prefix_cache mode.
SYSTEM_PROMPT: str = (
    "You are a tool router.\n\n"
    "Rules:\n"
    "- Return only the tool name.\n"
    '- Return "none" if no tool matches.\n'
    "- Do not explain."
)
