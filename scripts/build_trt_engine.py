"""Build TensorRT-LLM engine files for the intent-classifier models.

This script automates the two-step TensorRT-LLM engine build pipeline:

    Step 1: ``convert_checkpoint.py``
        Converts the HF SafeTensors checkpoint into TRT-LLM weight shards
        + a ``config.json`` that ``trtllm-build`` can consume.

    Step 2: ``trtllm-build``
        Compiles the weight shards into an optimised ``.engine`` file for a
        specific GPU architecture.

Both steps are invoked as subprocesses so that this script can be run from
the project's standard Python environment -- the heavy TRT-LLM packages only
need to be on ``PATH``/``PYTHONPATH``.

Supported dtypes
-----------------
fp16        FP16 weights + FP16 activations  (default; Jetson Orin primary)
bf16        BF16 weights + BF16 activations  (Ampere/Ada/Hopper only)
int8        INT8 weights + INT8 activations  (SmoothQuant W8A8)
int4        INT4 weight-only quantisation    (AWQ W4A16)

Usage
------
    # Activate the TRT-LLM Python environment first, then:
    python scripts/build_trt_engine.py --model qwen3 --dtype fp16
    python scripts/build_trt_engine.py --model llama3 --dtype int8 --tp 1

Output
------
    models/tensorrt/<stem>-<dtype>/
        rank0.engine          compiled engine (rename from config.json)
        config.json           build metadata
        (+ rank1.engine etc. for TP > 1)

Prerequisites
--------------
- NVIDIA GPU with the target SM architecture.
  - SM72: Jetson Xavier AGX / NX (JetPack 5, experimental -- TRT-LLM does not
    officially support SM72; prefer ONNX Runtime or bare TensorRT on Xavier).
  - SM87: Jetson Orin AGX / NX / Nano (JetPack 6, fully supported).
  - SM80+: A100 / H100 / L40S (data-centre).
- TensorRT-LLM installed; ``trtllm-build`` on PATH
- TRT-LLM convert_checkpoint.py scripts bundled with trtllm install (found
  under <trtllm_pkg>/examples/<arch>/)
- ``pip install tensorrt_llm`` in the active Python env

For Jetson Orin JetPack 6 installation, follow:
    https://nvidia.github.io/TensorRT-LLM/installation/jetson.html
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation_lib.config import (
    MODEL_DISPLAY_NAMES,
    MODEL_PATHS,
    MODEL_TENSORRT_STEMS,
    TENSORRT_DIR,
    TENSORRT_DTYPES,
)

# Map model_key -> architecture directory name inside the trtllm examples tree.
# Qwen3 uses TRT-LLM's dedicated qwen converter; Llama-3.2 uses the llama one.
_TRTLLM_ARCH_DIR: dict[str, str] = {
    "qwen3": "qwen",
    "llama3": "llama",
}


# SM architectures that do NOT support BF16 (Volta and earlier).
_NO_BF16_SM: frozenset[str] = frozenset({"sm70", "sm72"})


def _require(cmd: str) -> str:
    """Return the resolved path of *cmd* or raise if it is not on PATH."""
    path = shutil.which(cmd)
    if path is None:
        raise RuntimeError(
            f"'{cmd}' not found on PATH. "
            "Install TensorRT-LLM and ensure its binaries are on PATH."
        )
    return path


def _find_convert_script(arch: str) -> Path:
    """Locate the ``convert_checkpoint.py`` script for *arch* from the trtllm package."""
    try:
        import tensorrt_llm  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "tensorrt_llm is not installed. "
            "See https://nvidia.github.io/TensorRT-LLM/installation/linux.html"
        ) from exc

    pkg_root = Path(tensorrt_llm.__file__).parent
    candidates = [
        pkg_root / "examples" / arch / "convert_checkpoint.py",
        # Some trtllm distributions install examples one level up
        pkg_root.parent / "examples" / arch / "convert_checkpoint.py",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"convert_checkpoint.py not found for architecture '{arch}' in the "
        f"tensorrt_llm package tree. "
        f"Searched: {[str(c) for c in candidates]}"
    )


def _dtype_to_convert_args(dtype: str) -> list[str]:
    """Return extra CLI flags for convert_checkpoint.py for *dtype*."""
    if dtype == "fp16":
        return ["--dtype", "float16"]
    if dtype == "bf16":
        return ["--dtype", "bfloat16"]
    if dtype == "int8":
        # SmoothQuant W8A8
        return [
            "--dtype",
            "float16",
            "--smoothquant",
            "0.5",
            "--per-token",
            "--per-channel",
        ]
    if dtype == "int4":
        # AWQ INT4 weight-only
        return [
            "--dtype",
            "float16",
            "--use-weight-only",
            "--weight-only-precision",
            "int4_awq",
        ]
    raise ValueError(f"Unsupported dtype: {dtype}")


def _dtype_to_build_args(dtype: str) -> list[str]:
    """Return extra ``trtllm-build`` flags for *dtype*."""
    if dtype in ("fp16", "bf16"):
        return []
    if dtype == "int8":
        return ["--strongly-typed"]
    if dtype == "int4":
        return ["--use-fused-mlp"]
    return []


def build(
    model_key: str, dtype: str, tp: int, max_seq_len: int, device_sm: str | None
) -> None:
    """Run the full convert → build pipeline for *model_key* at *dtype*."""
    if device_sm is not None and device_sm in _NO_BF16_SM and dtype == "bf16":
        raise ValueError(
            f"BF16 is not supported on {device_sm} (Volta architecture). "
            "Use fp16 or int8 instead."
        )
    model_path = MODEL_PATHS[model_key]
    stem = MODEL_TENSORRT_STEMS[model_key]
    arch = _TRTLLM_ARCH_DIR[model_key]
    out_dir = TENSORRT_DIR / f"{stem}-{dtype}"

    print(f"\n{'=' * 60}")
    print(f" Building {MODEL_DISPLAY_NAMES[model_key]} ({dtype})")
    print(f"   source    : {model_path}")
    print(f"   arch      : {arch}")
    print(f"   device SM : {device_sm or 'auto-detected'}")
    print(f"   output    : {out_dir}")
    print(f"   TP        : {tp}")
    print(f"{'=' * 60}\n")

    trtllm_build = _require("trtllm-build")
    convert_script = _find_convert_script(arch)

    with tempfile.TemporaryDirectory(prefix="trt_ckpt_") as ckpt_dir:
        # ── Step 1: convert HF checkpoint to TRT-LLM checkpoint ──────────
        convert_cmd = [
            sys.executable,
            str(convert_script),
            "--model_dir",
            str(model_path),
            "--output_dir",
            ckpt_dir,
            "--tp_size",
            str(tp),
            *_dtype_to_convert_args(dtype),
        ]
        print("[step 1/2] Converting HF checkpoint to TRT-LLM checkpoint ...")
        print("  cmd:", " ".join(convert_cmd))
        subprocess.run(convert_cmd, check=True)

        # ── Step 2: compile engine ────────────────────────────────────────
        out_dir.mkdir(parents=True, exist_ok=True)
        build_cmd = [
            trtllm_build,
            "--checkpoint_dir",
            ckpt_dir,
            "--output_dir",
            str(out_dir),
            "--max_seq_len",
            str(max_seq_len),
            "--max_batch_size",
            "1",
            "--gpt_attention_plugin",
            "float16" if dtype != "bf16" else "bfloat16",
            "--gemm_plugin",
            "float16" if dtype != "bf16" else "bfloat16",
            "--tp_size",
            str(tp),
            *_dtype_to_build_args(dtype),
        ]
        print("\n[step 2/2] Building TensorRT engine ...")
        print("  cmd:", " ".join(build_cmd))
        subprocess.run(build_cmd, check=True)

    print(f"\n[done] Engine written to: {out_dir}")
    engine_files = list(out_dir.glob("*.engine"))
    total_mb = sum(p.stat().st_size for p in engine_files) / (1024**2)
    print(f"  {len(engine_files)} engine file(s), {total_mb:.1f} MB total")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build TensorRT-LLM engine files for intent-classifier models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        choices=list(MODEL_PATHS),
        required=True,
        help="Model to build the engine for",
    )
    p.add_argument(
        "--dtype",
        choices=TENSORRT_DTYPES,
        default="fp16",
        help="Engine dtype / quantisation",
    )
    p.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor-parallelism degree (1 = single GPU, default for Jetson)",
    )
    p.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Maximum context length compiled into the engine",
    )
    p.add_argument(
        "--device-sm",
        type=str,
        default=None,
        metavar="SM",
        help=(
            "Target GPU SM architecture (e.g. sm72 for Jetson Xavier, "
            "sm87 for Jetson Orin). Used for validation only -- "
            "trtllm-build auto-detects the arch from the CUDA device present. "
            "Set explicitly to catch incompatible dtype+arch combinations "
            "before starting the build (e.g. bf16 is rejected on sm72)."
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(args.model, args.dtype, args.tp, args.max_seq_len, args.device_sm)
