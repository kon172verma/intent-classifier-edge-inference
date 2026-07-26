"""Project-wide constants: model registry, dataset paths, generation settings."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT: Path = Path(__file__).parent.parent

MODEL_PATHS: dict[str, Path] = {
    "qwen3": REPO_ROOT / "models" / "intent-classifier-qwen3-0.6b_C_1k_merged",
    "llama3": REPO_ROOT / "models" / "intent-classifier-llama3.2-1b_C_1k_merged",
}

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "qwen3": "Qwen3-0.6B",
    "llama3": "Llama-3.2-1B",
}

DATASET_DEFAULT: Path = REPO_ROOT / "dataset_full" / "sample_0001.json"

MAX_NEW_TOKENS: int = 32
WARMUP_EXAMPLES: int = 2

# GGUF model files for the llama.cpp evaluation (see evaluation_llama_cpp/).
# Filenames follow <stem>-<QUANT>.gguf, produced by convert_hf_to_gguf.py +
# llama-quantize (see evaluation_llama_cpp/readme.md for the conversion steps).
GGUF_DIR: Path = REPO_ROOT / "models" / "gguf"

MODEL_GGUF_STEMS: dict[str, str] = {
    "qwen3": "qwen3-0.6b",
    "llama3": "llama3.2-1b",
}

# Quantization levels benchmarked for evaluation_llama_cpp.
QUANT_LEVELS: list[str] = ["Q8_0", "Q6_K", "Q4_K_M"]

N_CTX_DEFAULT: int = 2048

# ONNX model directory for the evaluation_onnx evaluation (see
# evaluation_onnx/readme.md for the export + quantization steps).
ONNX_DIR: Path = REPO_ROOT / "models" / "onnx"

MODEL_ONNX_STEMS: dict[str, str] = {
    "qwen3": "qwen3-0.6b",
    "llama3": "llama3.2-1b",
}

# Precision variants benchmarked for evaluation_onnx. Directory names follow
# <stem>-<PRECISION>/model.onnx, produced by scripts/quantize_onnx.py (and
# optimum-cli export onnx for the fp32/fp16 base exports).
ONNX_PRECISIONS: list[str] = ["fp32", "fp16", "dynamic-int8", "static-int8"]

# Static system prompt used for all tool-routing evaluations.
# This is also the cacheable prefix in prefix_cache mode.
SYSTEM_PROMPT: str = (
    "You are a tool router.\n\n"
    "Rules:\n"
    "- Return only the tool name.\n"
    '- Return "none" if no tool matches.\n'
    "- Do not explain."
)
