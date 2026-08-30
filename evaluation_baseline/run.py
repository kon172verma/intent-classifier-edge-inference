#!/usr/bin/env python3
"""
Baseline inference benchmark: HF Transformers + PyTorch FP16.

All direct and pipeline runs use prefix_cache. A static system prompt is
pre-computed once and cloned per example.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import transformers

# Ensure the repo root is on sys.path so that evaluation_lib and
# evaluation_baseline are importable regardless of invocation directory.
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation_baseline.cache import (
    clone_cache,
    compute_prefix_cache,
    ingest_prefix_segment,
    kv_cache_bytes,
)
from evaluation_baseline.inference import (
    TTFTCapture,
    run_inference,
)
from evaluation_baseline.model_loader import load_model_and_tokenizer
from evaluation_lib.boundary import find_tools_query_boundary
from evaluation_lib.compatibility import canonical_expected, legacy_prompt_spec, parse_prediction
from evaluation_lib.config import (
    DATASET_DEFAULT,
    MODEL_DISPLAY_NAMES,
    MODEL_PATHS,
    SYSTEM_PROMPT,
    WARMUP_EXAMPLES,
)
from evaluation_lib.device import resolve_device
from evaluation_lib.metrics import aggregate_metrics, compute_quality
from evaluation_lib.prompt import (
    build_full_prompt,
    build_system_prefix_text,
    build_tools_only_prompt,
)
from evaluation_lib.reporting import build_prefill_split_info, print_run_summary
from evaluation_lib.run_context import load_prompt_spec
from evaluation_lib.system_info import model_weights_mb

_RESULTS_DIR = _REPO_ROOT / "evaluation_baseline" / "results"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Baseline inference benchmark (HF Transformers, FP16)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        required=True,
        help="Model label (legacy key, or the exact manifest model name with --model-path).",
    )
    p.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Explicit merged Transformers checkpoint path (for pipeline execution).",
    )
    p.add_argument(
        "--device",
        choices=["auto", "mps", "cpu", "cuda"],
        default="auto",
        help="Compute device",
    )
    p.add_argument(
        "--machine",
        type=str,
        default=platform.node() or "unknown",
        help=(
            "Label identifying the physical machine this run was executed "
            "on (e.g. 'rpi', 'mac'). Used to group results for charting "
            "since the same --device (e.g. cpu) can run on different "
            "hardware. Defaults to the machine's hostname."
        ),
    )
    p.add_argument(
        "--dtype",
        choices=["float32", "bfloat16", "float16"],
        default="float16",
        help=(
            "Model weight/compute dtype. float16 on CPU falls back to slow "
            "unoptimized PyTorch kernels (no oneDNN fast path) -- use "
            "float32 or bfloat16 for CPU benchmarking."
        ),
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
    p.add_argument(
        "--manifest", type=Path, default=None, help="Version manifest supplying prompt rules."
    )
    p.add_argument(
        "--run-id", default=None, help="Optional pipeline run identifier recorded in output."
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    mode = "prefix_cache"
    prompt_spec, manifest_provenance = load_prompt_spec(args.manifest)
    prompt_spec = prompt_spec or legacy_prompt_spec(SYSTEM_PROMPT)
    if args.model_path is None and args.model not in MODEL_PATHS:
        raise ValueError("--model-path is required when --model is not a legacy model key")
    model_path = args.model_path or MODEL_PATHS[args.model]
    model_name = MODEL_DISPLAY_NAMES.get(args.model, args.model)

    print("=== Baseline Benchmark ===")
    print(f"  model   : {args.model} ({model_name})")
    print(f"  machine : {args.machine}")
    print(f"  mode    : {mode}")
    print(f"  device  : {device}")
    print(f"  dtype   : {args.dtype}")
    print(f"  dataset : {args.dataset}")
    print(f"  warmup  : {args.warmup} examples\n")

    with open(args.dataset) as f:
        dataset: list[dict] = json.load(f)
    print(f"[data] Loaded {len(dataset)} examples from {args.dataset.name}\n")

    model, tokenizer = load_model_and_tokenizer(args.model, device, args.dtype, model_path)
    weights_mb = model_weights_mb(model)
    print(f"[model] Parameter + buffer size: {weights_mb:.1f} MB ({args.dtype} on {device})\n")

    # ------------------------------------------------------------------
    # Prefix-cache setup (prefix_cache mode only)
    # ------------------------------------------------------------------
    # The system-prompt token length is needed in ALL cached modes (not just
    # prefix_cache) to split prefill into 3 phases: system prompt -> tools
    # list -> user query. system_len is constant across examples since the
    # system prompt text never changes.
    _system_prefix_text = build_system_prefix_text(tokenizer, prompt_spec)
    system_len = (
        tokenizer(_system_prefix_text, return_tensors="pt").input_ids.shape[1]
        if _system_prefix_text
        else 0
    )

    prefix_past_kv = None
    prefix_len = 0
    prefix_creation_ms = 0.0
    prefix_cache_size_bytes = 0

    prefix_past_kv, prefix_len, prefix_creation_ms = compute_prefix_cache(
        model, tokenizer, device, prompt_spec
    )
    prefix_cache_size_bytes = kv_cache_bytes(prefix_past_kv)

    # Verify prefix token alignment against the first dataset example.
    _sample_prompt = build_full_prompt(
        tokenizer,
        dataset[0]["user_request"],
        dataset[0]["available_tools"],
        prompt_spec,
    )
    _full_ids = tokenizer(_sample_prompt, return_tensors="pt").input_ids
    _prefix_ids = tokenizer(_system_prefix_text, return_tensors="pt").input_ids
    computed_prefix_len = _prefix_ids.shape[1]

    if not torch.equal(_full_ids[:, :computed_prefix_len], _prefix_ids):
        print(
            "[prefix_cache] WARNING: prefix tokens do not align with full "
            "prompt tokens. Falling back to full prompt (no prefix savings)."
        )
        prefix_past_kv = None
        prefix_len = 0
    else:
        prefix_len = computed_prefix_len
        print(f"[prefix_cache] Prefix alignment verified. prefix_len={prefix_len} tokens.\n")

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------
    ttft_capture = TTFTCapture(device)
    per_example: list[dict] = []

    for idx, example in enumerate(dataset):
        user_request = example["user_request"]
        available_tools = example["available_tools"]
        expected = canonical_expected(example["answer"], prompt_spec)

        full_prompt = build_full_prompt(tokenizer, user_request, available_tools, prompt_spec)
        full_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to(device)

        is_warmup = idx < args.warmup
        tag = (
            "[warmup]"
            if is_warmup
            else f"[{idx - args.warmup + 1:3d}/{len(dataset) - args.warmup}]"
        )

        tools_only_prompt = build_tools_only_prompt(tokenizer, available_tools, prompt_spec)
        tools_only_ids = tokenizer(tools_only_prompt, return_tensors="pt").input_ids.to(device)
        boundary = find_tools_query_boundary(full_ids[0].tolist(), tools_only_ids[0].tolist())
        if prefix_past_kv is not None:
            cache_after_system = clone_cache(prefix_past_kv)
            system_prefill_ms = 0.0
            system_prefill_tokens = prefix_len
        else:
            system_ids = full_ids[:, :system_len]
            cache_after_system, system_prefill_ms = ingest_prefix_segment(
                model, system_ids, device, past_key_values=None
            )
            system_prefill_tokens = system_len
        tools_ids = full_ids[:, system_len:boundary]
        query_ids = full_ids[:, boundary:]
        total_len = full_ids.shape[1]
        attn_mask = torch.ones(1, total_len, dtype=torch.long, device=device)
        cache_after_tools, tools_prefill_ms = ingest_prefix_segment(
            model, tools_ids, device, past_key_values=cache_after_system
        )
        timing = run_inference(
            model,
            tokenizer,
            query_ids,
            device,
            ttft_capture,
            past_key_values=cache_after_tools,
            attention_mask=attn_mask,
            system_prefill_ms=system_prefill_ms,
            system_prefill_tokens=system_prefill_tokens,
            tools_prefill_ms=tools_prefill_ms,
            tools_prefill_tokens=tools_ids.shape[1],
            report_prefill_split=True,
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

    print_run_summary(aggregate, quality, weights_mb, args.dtype)

    # ------------------------------------------------------------------
    # Write JSON output
    # ------------------------------------------------------------------
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_config: dict[str, Any] = {
        "model_key": args.model,
        "model_name": model_name,
        "model_path": str(model_path),
        "machine": args.machine,
        "mode": mode,
        "device": device,
        "dtype": args.dtype,
        "dataset": str(args.dataset),
        "n_dataset_examples": len(dataset),
        "n_measured_examples": len(per_example),
        "warmup_examples": args.warmup,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "os": platform.system(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "model_weights_mb": weights_mb,
        "run_id": args.run_id,
        "prompt": {
            "template_id": prompt_spec.template_id,
            "output_format": prompt_spec.output_format,
            **manifest_provenance,
        },
    }

    run_config["prefill_split_info"] = build_prefill_split_info()
    run_config["prefix_cache_info"] = {
        "prefix_tokens": prefix_len,
        "cache_creation_ms": round(prefix_creation_ms, 3),
        "cache_size_bytes": prefix_cache_size_bytes,
        "cache_size_kb": round(prefix_cache_size_bytes / 1024, 2),
        "note": (
            "cache_creation_ms is a one-time cost amortised over all examples. "
            "prefill_latency_ms in per_example reflects only dynamic suffix tokens."
        ),
    }

    output_doc = {
        "run_config": run_config,
        "aggregate": aggregate,
        "quality": quality,
        "per_example": per_example,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        args.output_dir / f"{args.model}_{args.machine}_{device}_{mode}_{args.dtype}_{ts}.json"
    )

    with open(out_path, "w") as f:
        json.dump(output_doc, f, indent=2)

    print(f"\n[output] Results written to {out_path}")


if __name__ == "__main__":
    main()
