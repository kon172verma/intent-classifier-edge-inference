#!/usr/bin/env python3
"""
ONNX Runtime baseline benchmark: mirrors evaluation_baseline/ and
evaluation_llama_cpp/, but runs the model via ONNX Runtime instead of HF
Transformers/PyTorch or llama.cpp.

Precisions
-----------
fp32          — Unquantized FP32 ONNX export.
fp16          — Unquantized FP16 ONNX export (CoreML/ANE-friendly; not
                recommended on plain CPU -- no fast FP16 SIMD path on
                ARM/x86, see evaluation_onnx/readme.md).
dynamic-int8  — Post-training dynamic quantization (weights INT8,
                activations quantized on-the-fly at runtime; no calibration
                needed).
static-int8   — Post-training static quantization (weights AND activations
                INT8, calibrated offline on real dataset prompts; full
                speedup, needs the calibration step in
                scripts/quantize_onnx.py).

Devices (Execution Providers)
------------------------------
cpu     — CPUExecutionProvider (all platforms).
coreml  — CoreMLExecutionProvider (Apple Silicon only; routes ops through
          the ANE/GPU via CoreML, falls back to CPU per-op for anything
          unsupported).
qnn     — QnnExecutionProvider (Qualcomm Snapdragon devices; Windows on
          Snapdragon, Linux ARM64 with QNN SDK, Android).  Backend options:
          htp  → Hexagon DSP / HTP (default; best throughput for LLM inference)
          gpu  → Adreno GPU (fp32/fp16)
          cpu  → QNN CPU reference (debug only)

Mode
-----
Only prefix_cache-style caching is implemented (system prompt ingested once,
Reused via a cloned KV-cache dict for every example) since ONNX Runtime's
decoder-with-past graphs are inherently a from-scratch-only ("no_cache"
would just mean an empty starting cache each phase) or KV-cache design --
see evaluation_onnx/cache.py.

Usage
------
    # Activate the project venv first: source .venv/bin/activate
    python evaluation_onnx/run.py --model qwen3 --precision static-int8 --device cpu
    python evaluation_onnx/run.py --model llama3 --precision fp16 --device coreml

Output
------
    evaluation_onnx/results/<model>_<machine>_<device>_<mode>_<precision>_<timestamp>.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import transformers

_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation_lib.boundary import find_tools_query_boundary
from evaluation_lib.config import (
    DATASET_DEFAULT,
    MODEL_DISPLAY_NAMES,
    MODEL_PATHS,
    ONNX_PRECISIONS,
    WARMUP_EXAMPLES,
)
from evaluation_lib.metrics import aggregate_metrics, compute_quality
from evaluation_lib.output_parser import extract_predicted_tool
from evaluation_lib.prompt import (
    build_full_prompt,
    build_system_prefix_text,
    build_tools_only_prompt,
)
from evaluation_lib.reporting import build_prefill_split_info, print_run_summary
from evaluation_onnx.cache import (
    Cache,
    clone_cache,
    compute_prefix_cache,
    kv_cache_bytes,
    run_segment,
)
from evaluation_onnx.inference import run_inference
from evaluation_onnx.model_loader import (
    load_cpu_bootstrap_session,
    load_session,
    load_text_tokenizer,
    onnx_model_size_mb,
)

_RESULTS_DIR = _REPO_ROOT / "evaluation_onnx" / "results"


def resolve_device(device_arg: str) -> str:
    """Return the concrete device string for a given ``--device`` argument."""
    if device_arg != "auto":
        return device_arg
    available = ort.get_available_providers()
    if platform.system() == "Darwin" and "CoreMLExecutionProvider" in available:
        return "coreml"
    if "QnnExecutionProvider" in available:
        return "qnn"
    return "cpu"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ONNX Runtime baseline benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        choices=list(MODEL_PATHS),
        required=True,
        help="Model to benchmark",
    )
    p.add_argument(
        "--precision",
        choices=ONNX_PRECISIONS,
        required=True,
        help=(
            "ONNX model precision/quantization. Recommended matrix: "
            "cpu -> {fp32, dynamic-int8, static-int8}; "
            "coreml -> {fp16}; "
            "qnn -> {fp32, fp16} (INT8 requires QDQ-quantized models; "
            "our static/dynamic-int8 use ORT QOperator format which QNN EP "
            "does not support -- use fp32 or fp16 on QNN)."
        ),
    )
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "coreml", "qnn"],
        default="auto",
        help=(
            "Execution provider. "
            "coreml: Apple ANE/GPU (macOS only). "
            "qnn: Qualcomm AI Engine via QNN SDK (Snapdragon devices). "
            "auto: coreml on Apple Silicon, qnn if QnnExecutionProvider is "
            "available, otherwise cpu."
        ),
    )
    p.add_argument(
        "--qnn-backend",
        choices=["htp", "gpu", "cpu"],
        default="htp",
        help=(
            "QNN backend to use when --device=qnn. "
            "htp: Hexagon DSP (best latency for LLM inference on Snapdragon). "
            "gpu: Adreno GPU (fp32/fp16). "
            "cpu: QNN CPU reference backend (debug only)."
        ),
    )
    p.add_argument(
        "--qnn-lib-path",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Explicit path to the QNN backend shared library "
            "(e.g. /opt/qcom/qnn/lib/aarch64-android/libQnnHtp.so). "
            "Defaults to the OS-standard library name resolved from PATH."
        ),
    )
    p.add_argument(
        "--machine",
        type=str,
        default=platform.node() or "unknown",
        help="Label identifying the physical machine this run was executed on",
    )
    p.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_DEFAULT,
        help="Path to dataset JSON file",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_RESULTS_DIR,
        help="Directory to write JSON results",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=WARMUP_EXAMPLES,
        help="Number of warmup examples excluded from measurements",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    print("=== ONNX Runtime Benchmark ===")
    print(f"  model     : {args.model} ({MODEL_DISPLAY_NAMES[args.model]})")
    print(f"  machine   : {args.machine}")
    print(f"  precision : {args.precision}")
    print(f"  device    : {device}")
    print(f"  dataset   : {args.dataset}")
    print(f"  warmup    : {args.warmup} examples\n")

    with open(args.dataset) as f:
        dataset: list[dict] = json.load(f)
    print(f"[data] Loaded {len(dataset)} examples from {args.dataset.name}\n")

    session = load_session(
        args.model,
        args.precision,
        device,
        qnn_backend=args.qnn_backend,
        qnn_lib_path=args.qnn_lib_path,
    )
    bootstrap_session = load_cpu_bootstrap_session(args.model, args.precision, device)
    weights_mb = onnx_model_size_mb(args.model, args.precision)
    print(f"[model] ONNX file size: {weights_mb:.1f} MB ({args.precision} on {device})\n")

    text_tokenizer = load_text_tokenizer(MODEL_PATHS[args.model])
    eos_token_ids: set[int] = set(
        transformers.GenerationConfig.from_pretrained(str(MODEL_PATHS[args.model])).eos_token_id
        or []
    )
    if text_tokenizer.eos_token_id is not None:
        eos_token_ids.add(text_tokenizer.eos_token_id)

    def tok(text: str) -> np.ndarray:
        return text_tokenizer(text, return_tensors="np").input_ids.astype(np.int64)

    system_prefix_text = build_system_prefix_text(text_tokenizer)
    system_tokens_template = (
        tok(system_prefix_text) if system_prefix_text else (np.empty((1, 0), dtype=np.int64))
    )
    system_len = system_tokens_template.shape[1]

    # ------------------------------------------------------------------
    # Prefix-cache setup (computed once; cloned per example -- see module
    # docstring on why ONNX Runtime only supports this caching style).
    # ------------------------------------------------------------------
    prefix_cache: Cache | None
    prefix_cache, prefix_len, prefix_creation_ms = compute_prefix_cache(
        session, system_tokens_template, bootstrap_session=bootstrap_session
    )
    # The CPU session is only needed for CoreML's first non-empty KV cache.
    # Keep benchmark memory and all request-time execution on the main session.
    del bootstrap_session
    prefix_cache_size_bytes = kv_cache_bytes(prefix_cache)

    _sample_prompt = build_full_prompt(
        text_tokenizer, dataset[0]["user_request"], dataset[0]["available_tools"]
    )
    _full_ids = tok(_sample_prompt)
    if not np.array_equal(_full_ids[:, :system_len], system_tokens_template):
        print(
            "[prefix_cache] WARNING: prefix tokens do not align with full "
            "prompt tokens. Falling back to full prompt (no prefix savings)."
        )
        prefix_cache = None
        prefix_len = 0
    else:
        print(f"[prefix_cache] Prefix alignment verified. prefix_len={prefix_len} tokens.\n")

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------
    per_example: list[dict] = []

    for idx, example in enumerate(dataset):
        user_request = example["user_request"]
        available_tools = example["available_tools"]
        expected = example["answer"]
        tool_names = {t["name"] for t in available_tools}

        full_prompt = build_full_prompt(text_tokenizer, user_request, available_tools)
        full_ids = tok(full_prompt)

        tools_only_prompt = build_tools_only_prompt(text_tokenizer, available_tools)
        tools_only_ids = tok(tools_only_prompt)
        boundary = find_tools_query_boundary(full_ids[0].tolist(), tools_only_ids[0].tolist())

        is_warmup = idx < args.warmup
        tag = (
            "[warmup]"
            if is_warmup
            else f"[{idx - args.warmup + 1:3d}/{len(dataset) - args.warmup}]"
        )

        # System prompt was already ingested once outside the loop
        # (compute_prefix_cache): zero incremental cost per example.
        cache_after_system = clone_cache(prefix_cache) if prefix_cache else {}
        system_prefill_ms = 0.0
        system_prefill_tokens = prefix_len

        tools_ids = full_ids[:, system_len:boundary]
        query_ids = full_ids[:, boundary:]

        cache_after_tools, _tools_logits, tools_prefill_ms = run_segment(
            session, tools_ids, cache_after_system
        )

        timing = run_inference(
            session,
            text_tokenizer,
            query_ids,
            cache_after_tools,
            eos_token_ids,
            system_prefill_ms=system_prefill_ms,
            system_prefill_tokens=system_prefill_tokens,
            tools_prefill_ms=tools_prefill_ms,
            tools_prefill_tokens=tools_ids.shape[1],
        )

        predicted = extract_predicted_tool(timing["generated_text"], tool_names)
        correct = predicted == expected

        if not is_warmup:
            print(
                f"{tag} e2e={timing['e2e_latency_ms']:.0f}ms"
                f"  ttft={timing['ttft_ms']:.0f}ms"
                f"  expected={expected!r}  predicted={predicted!r}"
                f"  {'OK' if correct else 'WRONG'}"
            )
            per_example.append(
                {
                    "id": idx - args.warmup,
                    "user_request": user_request,
                    "expected": expected,
                    "predicted": predicted,
                    "correct": correct,
                    **timing,
                }
            )
        else:
            print(f"{tag} e2e={timing['e2e_latency_ms']:.0f}ms  (warmup, not recorded)")

    print(f"\n[done] Measured {len(per_example)} examples.")

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------
    aggregate = aggregate_metrics(per_example)
    quality = compute_quality(per_example, dataset, args.warmup)

    print_run_summary(aggregate, quality, weights_mb, args.precision)

    # ------------------------------------------------------------------
    # Write JSON output
    # ------------------------------------------------------------------
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_config: dict[str, Any] = {
        "model_key": args.model,
        "model_name": MODEL_DISPLAY_NAMES[args.model],
        "model_path": str(MODEL_PATHS[args.model]),
        "machine": args.machine,
        "mode": "prefix_cache",
        "device": device,
        "precision": args.precision,
        "dataset": str(args.dataset),
        "n_dataset_examples": len(dataset),
        "n_measured_examples": len(per_example),
        "warmup_examples": args.warmup,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "os": platform.system(),
        "python_version": sys.version,
        "onnxruntime_version": ort.__version__,
        "onnxruntime_providers": session.get_providers(),
        "qnn_backend": args.qnn_backend if device == "qnn" else None,
        "qnn_lib_path": args.qnn_lib_path if device == "qnn" else None,
        "transformers_version": transformers.__version__,
        "model_weights_mb": weights_mb,
        "prefill_split_info": build_prefill_split_info(),
        "prefix_cache_info": {
            "prefix_tokens": prefix_len,
            "creation_time_ms": round(prefix_creation_ms, 3),
            "cache_size_kb": round(prefix_cache_size_bytes / 1024, 2),
            "note": (
                "One-time cost to compute the system-prompt KV cache, "
                "excluded from per-example measurements above."
            ),
            "cpu_bootstrap_tokens": 1 if device == "coreml" and prefix_len else 0,
        },
    }

    output = {
        "run_config": run_config,
        "aggregate": aggregate,
        "quality": quality,
        "per_example": per_example,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        args.output_dir
        / f"{args.model}_{args.machine}_{device}_prefix_cache_{args.precision}_{ts}.json"
    )
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[output] Results written to {out_path}")


if __name__ == "__main__":
    main()
