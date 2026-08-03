#!/usr/bin/env python3
"""
llama.cpp (GGUF, quantized) baseline benchmark: mirrors evaluation_baseline/
but runs the model via llama-cpp-python instead of HF Transformers/PyTorch.

Modes
------
kv_cache     — Standard llama.cpp KV cache; system prompt + tools list are
               re-ingested (and re-timed) fresh for every example.
prefix_cache — Static system-prompt prefix pre-computed once via
               ``save_state()``; each example restores that snapshot via
               ``load_state()`` before ingesting its own tools list.

Note: llama.cpp has no equivalent of HF's ``use_cache=False`` "no_cache"
mode -- ggml's causal attention always maintains a KV cache internally, so
that mode is not offered here.

Usage
------
    # Activate the project venv first: source .venv/bin/activate
    python evaluation_llama_cpp/run.py --model qwen3 --quant Q4_K_M --mode kv_cache
    python evaluation_llama_cpp/run.py --model llama3 --quant Q8_0 --mode prefix_cache --device mps

Output
------
    evaluation_llama_cpp/results/<model>_<machine>_<device>_<mode>_<quant>_<timestamp>.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import llama_cpp

# Ensure the repo root is on sys.path so that evaluation_lib and
# evaluation_llama_cpp are importable regardless of invocation directory.
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation_lib.config import (
    DATASET_DEFAULT,
    MODEL_DISPLAY_NAMES,
    MODEL_PATHS,
    N_CTX_DEFAULT,
    QUANT_LEVELS,
    WARMUP_EXAMPLES,
)
from evaluation_lib.device import resolve_device
from evaluation_lib.metrics import aggregate_metrics, compute_quality
from evaluation_lib.output_parser import extract_predicted_tool
from evaluation_lib.prompt import (
    build_full_prompt,
    build_system_prefix_text,
    build_tools_only_prompt,
)
from evaluation_llama_cpp.cache import (
    clone_prefix_cache,
    compute_prefix_cache,
    ingest_prefix_segment,
    kv_cache_bytes,
)
from evaluation_llama_cpp.inference import run_inference
from evaluation_lib.boundary import find_tools_query_boundary
from evaluation_lib.reporting import build_prefill_split_info, print_run_summary
from evaluation_llama_cpp.model_loader import (
    gguf_model_size_mb,
    load_model,
    load_text_tokenizer,
)

_RESULTS_DIR = _REPO_ROOT / "evaluation_llama_cpp" / "results"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="llama.cpp (GGUF, quantized) baseline benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        choices=list(MODEL_PATHS),
        required=True,
        help="Model to benchmark",
    )
    p.add_argument(
        "--mode",
        choices=["kv_cache", "prefix_cache"],
        default="prefix_cache",
        help="Caching mode",
    )
    p.add_argument(
        "--quant",
        choices=QUANT_LEVELS,
        required=True,
        help="GGUF quantization level",
    )
    p.add_argument(
        "--device",
        choices=["auto", "mps", "cpu"],
        default="auto",
        help="Compute device (mps offloads all layers to Metal GPU)",
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
    p.add_argument(
        "--n-ctx",
        type=int,
        default=N_CTX_DEFAULT,
        help="Context window size (must exceed the longest prompt in the dataset)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if device == "cuda":
        device = "cpu"  # this project only targets mps/cpu for llama.cpp
    mode = args.mode

    print("=== llama.cpp Benchmark ===")
    print(f"  model   : {args.model} ({MODEL_DISPLAY_NAMES[args.model]})")
    print(f"  machine : {args.machine}")
    print(f"  mode    : {mode}")
    print(f"  quant   : {args.quant}")
    print(f"  device  : {device}")
    print(f"  dataset : {args.dataset}")
    print(f"  warmup  : {args.warmup} examples\n")

    with open(args.dataset) as f:
        dataset: list[dict] = json.load(f)
    print(f"[data] Loaded {len(dataset)} examples from {args.dataset.name}\n")

    llm = load_model(args.model, args.quant, device, n_ctx=args.n_ctx)
    weights_mb = gguf_model_size_mb(args.model, args.quant)
    print(f"[model] GGUF file size: {weights_mb:.1f} MB ({args.quant} on {device})\n")

    # Tokenizer used ONLY for chat-template text rendering (see
    # evaluation_llama_cpp/model_loader.py); actual token ids come from
    # llm.tokenize() so they match the GGUF model's own vocab exactly.
    text_tokenizer = load_text_tokenizer(MODEL_PATHS[args.model])

    def tok(text: str) -> list[int]:
        # add_bos=False: the chat template already embeds the literal BOS
        # token text (e.g. "<|begin_of_text|>" for Llama 3.2) when the
        # tokenizer's own add_bos_token is False, exactly mirroring how
        # evaluation_baseline's `tokenizer(text)` call never re-adds a BOS
        # for this model either. Passing add_bos=True here would prepend a
        # *second* BOS on top of that literal one (harmless for Qwen3, which
        # has no BOS token at all, but wrong for Llama 3.2).
        return llm.tokenize(text.encode("utf-8"), add_bos=False, special=True)

    system_prefix_text = build_system_prefix_text(text_tokenizer)
    system_tokens_template = tok(system_prefix_text) if system_prefix_text else []
    system_len = len(system_tokens_template)

    # ------------------------------------------------------------------
    # Prefix-cache setup (prefix_cache mode only)
    # ------------------------------------------------------------------
    prefix_state = None
    prefix_len = 0
    prefix_creation_ms = 0.0
    prefix_cache_size_bytes = 0

    if mode == "prefix_cache":
        prefix_state, prefix_len, prefix_creation_ms = compute_prefix_cache(
            llm, system_tokens_template
        )
        prefix_cache_size_bytes = kv_cache_bytes(llm)

        # Verify prefix token alignment against the first dataset example.
        sample_prompt = build_full_prompt(
            text_tokenizer,
            dataset[0]["user_request"],
            dataset[0]["available_tools"],
        )
        sample_tokens = tok(sample_prompt)
        if sample_tokens[:system_len] != system_tokens_template:
            print(
                "[prefix_cache] WARNING: prefix tokens do not align with full "
                "prompt tokens. Falling back to fresh system-prefix ingestion "
                "per example (no prefix savings)."
            )
            prefix_state = None
            prefix_len = 0
        else:
            prefix_len = system_len
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
        full_tokens = tok(full_prompt)

        tools_only_prompt = build_tools_only_prompt(text_tokenizer, available_tools)
        tools_only_tokens = tok(tools_only_prompt)
        boundary = find_tools_query_boundary(full_tokens, tools_only_tokens)

        is_warmup = idx < args.warmup
        tag = (
            "[warmup]"
            if is_warmup
            else f"[{idx - args.warmup + 1:3d}/{len(dataset) - args.warmup}]"
        )

        if mode == "prefix_cache" and prefix_state is not None:
            # Restore the model to the saved system-prefix state. Cost is
            # deliberately NOT timed (mirrors clone_cache() in
            # evaluation_baseline, also not timed).
            clone_prefix_cache(llm, prefix_state)
            system_prefill_ms = 0.0
            system_prefill_tokens = prefix_len
        else:
            # kv_cache mode: no persistent cache across examples, so the
            # system prompt is re-ingested (and re-timed) every call.
            llm.reset()
            system_tokens = full_tokens[:system_len]
            system_prefill_ms = ingest_prefix_segment(llm, system_tokens)
            system_prefill_tokens = system_len

        tools_tokens = full_tokens[system_len:boundary]
        query_tokens = full_tokens[boundary:]

        tools_prefill_ms = ingest_prefix_segment(llm, tools_tokens)
        timing = run_inference(
            llm,
            query_tokens,
            system_prefill_ms=system_prefill_ms,
            system_prefill_tokens=system_prefill_tokens,
            tools_prefill_ms=tools_prefill_ms,
            tools_prefill_tokens=len(tools_tokens),
            report_prefill_split=True,
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

    print_run_summary(aggregate, quality, weights_mb, args.quant)

    # ------------------------------------------------------------------
    # Write JSON output
    # ------------------------------------------------------------------
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_config: dict[str, Any] = {
        "model_key": args.model,
        "model_name": MODEL_DISPLAY_NAMES[args.model],
        "machine": args.machine,
        "mode": mode,
        "device": device,
        "quant": args.quant,
        "n_ctx": args.n_ctx,
        "dataset": str(args.dataset),
        "n_dataset_examples": len(dataset),
        "n_measured_examples": len(per_example),
        "warmup_examples": args.warmup,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "os": platform.system(),
        "python_version": sys.version,
        "llama_cpp_python_version": llama_cpp.__version__,
        "model_weights_mb": weights_mb,
        "prefill_split_info": build_prefill_split_info(),
    }

    if mode == "prefix_cache":
        run_config["prefix_cache_info"] = {
            "prefix_tokens": prefix_len,
            "cache_creation_ms": round(prefix_creation_ms, 3),
            "cache_size_bytes": prefix_cache_size_bytes,
            "cache_size_kb": round(prefix_cache_size_bytes / 1024, 2),
            "note": ("cache_creation_ms is a one-time cost amortised over all examples."),
        }

    output_doc = {
        "run_config": run_config,
        "aggregate": aggregate,
        "quality": quality,
        "per_example": per_example,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        args.output_dir / f"{args.model}_{args.machine}_{device}_{mode}_{args.quant}_{ts}.json"
    )

    with open(out_path, "w") as f:
        json.dump(output_doc, f, indent=2)

    print(f"\n[output] Results written to {out_path}")


if __name__ == "__main__":
    main()
