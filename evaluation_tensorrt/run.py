"""TensorRT-LLM benchmark: CLI entry point.

Mirrors evaluation_onnx/run.py and evaluation_llama_cpp/run.py -- same
CLI flags, same 3-phase prefill measurement, same JSON output schema.

Mode
-----
Only ``prefix_cache`` is implemented (the system-prompt KV state is prefilled
once before the example loop, matching how production deployments amortise the
static system-prompt cost).

Usage
------
    # On a Jetson Orin (JetPack 6), Jetson Xavier (JetPack 5, experimental),
    # or any Linux machine with a compatible NVIDIA GPU:
    #   1. Build the TensorRT engine first (see evaluation_tensorrt/readme.md).
    #   2. Activate the TRT-LLM Python environment.
    python evaluation_tensorrt/run.py \\
        --model qwen3 --dtype fp16 --machine jetson

Output
------
    evaluation_tensorrt/results/<model>_<machine>_cuda_prefix_cache_<dtype>_<ts>.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tracemalloc
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

# Ensure the repo root is importable regardless of invocation directory.
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation_lib.config import (
    DATASET_DEFAULT,
    MODEL_DISPLAY_NAMES,
    MODEL_PATHS,
    TENSORRT_DTYPES,
    WARMUP_EXAMPLES,
)
from evaluation_lib.metrics import aggregate_metrics, compute_quality
from evaluation_lib.output_parser import extract_predicted_tool
from evaluation_lib.prompt import (
    build_full_prompt,
    build_system_prefix_text,
    build_tools_only_prompt,
)
from evaluation_tensorrt.inference import (
    _run_prefill_segment,
    run_inference,
)
from evaluation_lib.boundary import find_tools_query_boundary
from evaluation_lib.reporting import build_prefill_split_info, print_run_summary
from evaluation_tensorrt.model_loader import (
    engine_size_mb,
    load_session,
    load_tokenizer,
)

_RESULTS_DIR = _REPO_ROOT / "evaluation_tensorrt" / "results"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TensorRT-LLM benchmark (NVIDIA GPU / Jetson Orin / Jetson Xavier)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        choices=list(MODEL_PATHS),
        required=True,
        help="Model to benchmark",
    )
    p.add_argument(
        "--dtype",
        choices=TENSORRT_DTYPES,
        required=True,
        help=(
            "Engine precision / quantization. "
            "fp16: recommended for all Jetson devices. "
            "bf16: Jetson Orin (SM87, Ampere) only -- NOT available on Xavier (SM72, Volta). "
            "int8: SmoothQuant W8A8. "
            "int4: AWQ weight-only W4A16."
        ),
    )
    p.add_argument(
        "--machine",
        type=str,
        default=platform.node() or "unknown",
        help="Label for the physical device (e.g. 'jetson', 'a100').",
    )
    p.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_DEFAULT,
        help="Path to the dataset JSON file.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_RESULTS_DIR,
        help="Directory to write JSON result files.",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=WARMUP_EXAMPLES,
        help="Number of warmup examples excluded from measurements.",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Maximum tokens to generate per example.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Dataset ────────────────────────────────────────────────────────
    with open(args.dataset) as f:
        dataset: list[dict[str, Any]] = json.load(f)
    print(f"[data] Loaded {len(dataset)} examples from {args.dataset}")

    # ── Model + tokenizer ──────────────────────────────────────────────
    runner, dir_path = load_session(args.model, args.dtype)
    tokenizer = load_tokenizer(args.model)
    weights_mb = engine_size_mb(dir_path)

    tok = tokenizer.encode  # callable: str -> list[int]
    eos_token_ids: set[int] = {int(i) for i in (tokenizer.eos_token_id,) if i is not None}
    if hasattr(tokenizer, "additional_special_tokens_ids"):
        eos_token_ids.update(
            int(i) for i in tokenizer.additional_special_tokens_ids if i is not None
        )

    # ── System-prompt prefix cache (computed once) ──────────────────────
    system_text = build_system_prefix_text(tokenizer)
    system_token_ids = tok(system_text)
    system_len = len(system_token_ids)

    print(f"[cache] Pre-filling system prompt ({system_len} tokens) into KV cache ...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    prefix_creation_ms, _ = _run_prefill_segment(runner, system_token_ids)
    print(f"[cache] System-prompt KV cache ready ({prefix_creation_ms:.1f} ms).\n")

    # ── Benchmark loop ─────────────────────────────────────────────────
    per_example: list[dict[str, Any]] = []
    tracemalloc.start()

    for idx, example in enumerate(dataset):
        user_request: str = example["user_request"]
        expected: str = example["expected_tool"]
        available_tools: list[dict] = example["available_tools"]
        tool_names: set[str] = {t["name"] for t in available_tools}

        full_prompt = build_full_prompt(tokenizer, user_request, available_tools)
        full_ids = tok(full_prompt)

        tools_only_prompt = build_tools_only_prompt(tokenizer, available_tools)
        tools_only_ids = tok(tools_only_prompt)
        boundary = find_tools_query_boundary(full_ids, tools_only_ids)

        is_warmup = idx < args.warmup
        tag = (
            "[warmup]"
            if is_warmup
            else f"[{idx - args.warmup + 1:3d}/{len(dataset) - args.warmup}]"
        )

        # System prompt is already in the KV cache; zero incremental cost.
        system_prefill_ms = 0.0
        system_prefill_tokens = system_len

        tools_ids = full_ids[system_len:boundary]
        query_ids = full_ids[boundary:]

        # Ingest tools list into KV cache (phase 2 of 3-phase prefill)
        tools_prefill_ms, _ = _run_prefill_segment(runner, tools_ids)

        timing = run_inference(
            runner,
            tokenizer,
            query_ids,
            eos_token_ids,
            system_prefill_ms=system_prefill_ms,
            system_prefill_tokens=system_prefill_tokens,
            tools_prefill_ms=tools_prefill_ms,
            tools_prefill_tokens=len(tools_ids),
            max_new_tokens=args.max_new_tokens,
        )

        predicted = extract_predicted_tool(timing["generated_text"], tool_names)
        correct = predicted == expected

        # Peak CPU RAM (tracemalloc)
        _, peak_ram_bytes = tracemalloc.get_traced_memory()
        timing["peak_ram_mb"] = round(peak_ram_bytes / (1024**2), 2)

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

    tracemalloc.stop()
    print(f"\n[done] Measured {len(per_example)} examples.")

    # ── Summaries ──────────────────────────────────────────────────────
    aggregate = aggregate_metrics(per_example)
    quality = compute_quality(per_example, dataset, args.warmup)

    print_run_summary(aggregate, quality, weights_mb, args.dtype)

    # ── JSON output ────────────────────────────────────────────────────
    try:
        import tensorrt_llm  # type: ignore[import]

        trt_llm_version: str | None = tensorrt_llm.__version__
    except ImportError:
        trt_llm_version = None

    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_config: dict[str, Any] = {
        "model_key": args.model,
        "model_name": MODEL_DISPLAY_NAMES[args.model],
        "model_path": str(MODEL_PATHS[args.model]),
        "machine": args.machine,
        "mode": "prefix_cache",
        "device": "cuda",
        "dtype": args.dtype,
        "dataset": str(args.dataset),
        "n_dataset_examples": len(dataset),
        "n_measured_examples": len(per_example),
        "warmup_examples": args.warmup,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "os": platform.system(),
        "python_version": sys.version,
        "tensorrt_llm_version": trt_llm_version,
        "model_weights_mb": weights_mb,
        "prefill_split_info": build_prefill_split_info(),
        "prefix_cache_info": {
            "prefix_tokens": system_len,
            "creation_time_ms": round(prefix_creation_ms, 3),
            "note": (
                "One-time cost to pre-fill the system-prompt KV cache, "
                "excluded from per-example measurements above."
            ),
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
        args.output_dir / f"{args.model}_{args.machine}_cuda_prefix_cache_{args.dtype}_{ts}.json"
    )
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[output] Results written to {out_path}")


if __name__ == "__main__":
    main()
