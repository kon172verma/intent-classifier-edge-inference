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
Only prefix-cache-style caching is implemented: the system prompt is ingested
once and its KV-cache dict is cloned for every example. See
``evaluation_onnx/cache.py``.

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
from evaluation_lib.compatibility import canonical_expected, legacy_prompt_spec, parse_prediction
from evaluation_lib.config import (
    DATASET_DEFAULT,
    MODEL_DISPLAY_NAMES,
    MODEL_PATHS,
    ONNX_PRECISIONS,
    SYSTEM_PROMPT,
    WARMUP_EXAMPLES,
)
from evaluation_lib.generation import normalize_token_ids
from evaluation_lib.metrics import aggregate_metrics, compute_quality
from evaluation_lib.prompt import (
    build_full_prompt,
    build_system_prefix_text,
    build_tools_only_prompt,
)
from evaluation_lib.report_paths import resolve_report_path
from evaluation_lib.reporting import build_prefill_split_info, print_run_summary
from evaluation_lib.run_context import load_prompt_spec
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
    onnx_model_path,
    onnx_model_size_mb,
)

_REPORTS_DIR = _REPO_ROOT / "evaluation_onnx" / "reports"


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
        required=True,
        help="Model label (legacy key, or exact manifest model name with explicit paths).",
    )
    p.add_argument("--onnx-path", type=Path, default=None, help="Explicit ONNX model.onnx path.")
    p.add_argument(
        "--tokenizer-path",
        type=Path,
        default=None,
        help="Explicit merged Transformers checkpoint used for tokenization.",
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
        "--manifest", type=Path, default=None, help="Version manifest supplying prompt rules."
    )
    p.add_argument(
        "--run-id", default=None, help="Optional pipeline run identifier recorded in output."
    )
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "coreml", "cuda", "qnn"],
        default="auto",
        help=(
            "Execution provider. "
            "coreml: Apple ANE/GPU (macOS only). "
            "cuda: CUDAExecutionProvider with CPU fallback. "
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
        default=_REPORTS_DIR,
        help="Directory to write JSON reports",
    )
    p.add_argument(
        "--output-file", type=Path, default=None, help="Exact report path (pipeline use)."
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=WARMUP_EXAMPLES,
        help="Number of warmup examples excluded from measurements",
    )
    p.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional dataset prefix limit for a smoke run; never use for candidate selection.",
    )
    p.add_argument(
        "--benchmark-scope",
        choices=["standard", "smoke"],
        default="standard",
        help="Labels whether this is a full anchor benchmark or a short smoke check.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    prompt_spec, manifest_provenance = load_prompt_spec(args.manifest)
    prompt_spec = prompt_spec or legacy_prompt_spec(SYSTEM_PROMPT)
    if (args.onnx_path is None or args.tokenizer_path is None) and args.model not in MODEL_PATHS:
        raise ValueError("--onnx-path and --tokenizer-path are required for a non-legacy --model")
    onnx_path = args.onnx_path
    resolved_onnx_path = onnx_path or onnx_model_path(args.model, args.precision)
    tokenizer_path = args.tokenizer_path or MODEL_PATHS[args.model]
    model_name = MODEL_DISPLAY_NAMES.get(args.model, args.model)

    print("=== ONNX Runtime Benchmark ===")
    print(f"  model     : {args.model} ({model_name})")
    print(f"  machine   : {args.machine}")
    print(f"  precision : {args.precision}")
    print(f"  device    : {device}")
    print(f"  dataset   : {args.dataset}")
    print(f"  warmup    : {args.warmup} examples\n")

    with open(args.dataset) as f:
        dataset: list[dict] = json.load(f)
    if args.max_examples is not None:
        if args.max_examples <= args.warmup:
            raise ValueError("--max-examples must be greater than --warmup")
        dataset = dataset[: args.max_examples]
    if not dataset:
        raise ValueError("Dataset contains no examples after applying --max-examples")
    print(f"[data] Loaded {len(dataset)} examples from {args.dataset.name}\n")

    session = load_session(
        args.model,
        args.precision,
        device,
        qnn_backend=args.qnn_backend,
        qnn_lib_path=args.qnn_lib_path,
        model_path=onnx_path,
    )
    bootstrap_session = load_cpu_bootstrap_session(
        args.model, args.precision, device, model_path=onnx_path
    )
    weights_mb = onnx_model_size_mb(args.model, args.precision, model_path=onnx_path)
    print(f"[model] ONNX file size: {weights_mb:.1f} MB ({args.precision} on {device})\n")

    text_tokenizer = load_text_tokenizer(tokenizer_path)
    eos_token_ids = normalize_token_ids(
        transformers.GenerationConfig.from_pretrained(str(tokenizer_path)).eos_token_id
    )
    if text_tokenizer.eos_token_id is not None:
        eos_token_ids.add(text_tokenizer.eos_token_id)

    def tok(text: str) -> np.ndarray:
        return text_tokenizer(text, return_tensors="np").input_ids.astype(np.int64)

    system_prefix_text = build_system_prefix_text(text_tokenizer, prompt_spec)
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
        text_tokenizer, dataset[0]["user_request"], dataset[0]["available_tools"], prompt_spec
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
        expected = canonical_expected(example["answer"], prompt_spec)

        full_prompt = build_full_prompt(text_tokenizer, user_request, available_tools, prompt_spec)
        full_ids = tok(full_prompt)

        tools_only_prompt = build_tools_only_prompt(text_tokenizer, available_tools, prompt_spec)
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

        parsed = parse_prediction(timing["generated_text"], available_tools, prompt_spec)
        predicted = parsed.canonical_tool_name
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
                    "raw_model_output": parsed.raw_output,
                    "parsed_model_output": parsed.parsed_output,
                    "invalid_output_reason": parsed.invalid_reason,
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
        "model_name": model_name,
        "model_path": str(resolved_onnx_path),
        "tokenizer_path": str(tokenizer_path),
        "machine": args.machine,
        "mode": "prefix_cache",
        "device": device,
        "precision": args.precision,
        "dataset": str(args.dataset),
        "n_dataset_examples": len(dataset),
        "n_measured_examples": len(per_example),
        "warmup_examples": args.warmup,
        "benchmark_scope": args.benchmark_scope,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "os": platform.system(),
        "python_version": sys.version,
        "onnxruntime_version": ort.__version__,
        "onnxruntime_providers": session.get_providers(),
        "qnn_backend": args.qnn_backend if device == "qnn" else None,
        "qnn_lib_path": args.qnn_lib_path if device == "qnn" else None,
        "transformers_version": transformers.__version__,
        "model_weights_mb": weights_mb,
        "run_id": args.run_id,
        "prompt": {
            "template_id": prompt_spec.template_id,
            "output_format": prompt_spec.output_format,
            **manifest_provenance,
        },
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

    out_path = resolve_report_path(
        args.output_dir,
        args.output_file,
        f"{args.model}_{args.machine}_{device}_prefix_cache_{args.precision}_{ts}.json",
    )
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[output] Results written to {out_path}")


if __name__ == "__main__":
    main()
