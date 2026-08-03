"""Shared reporting helpers used by all evaluation backend run.py files."""

from __future__ import annotations

PREFILL_SPLIT_NOTE: str = (
    "prefill is measured in 3 phases: system_prefill_* covers "
    "ingesting the static system prompt, tools_prefill_* covers "
    "ingesting the available-tools list, query_prefill_* covers "
    "ingesting the dynamic user query. Both system prompt and "
    "tools list are treated as pre-processing that happens ahead "
    "of the live request in production, so ttft_ms/"
    "prefill_latency_ms/e2e_latency_ms cover ONLY the user-query "
    "phase (+ decode for e2e); preprocessing_latency_ms is the "
    "sum of system_prefill_latency_ms + tools_prefill_latency_ms, "
    "reported separately per example. In prefix_cache mode, "
    "system_prefill_latency_ms is 0 per example because the "
    "system prompt is cached once (see prefix_cache_info) rather "
    "than re-ingested every call."
)


def build_prefill_split_info() -> dict:
    """Return the standard prefill_split_info run_config sub-dict."""
    return {"enabled": True, "note": PREFILL_SPLIT_NOTE}


def print_run_summary(
    aggregate: dict,
    quality: dict,
    weights_mb: float,
    variant_label: str,
) -> None:
    """Print Quality / Latency / Throughput / Memory summary to stdout."""
    split_active = aggregate.get("mean_preprocessing_latency_ms") is not None

    print("\n--- Quality ---")
    print(f"  accuracy       : {quality.get('tool_accuracy', 0):.2%}")
    print(f"  invalid rate   : {quality.get('invalid_tool_rate', 0):.2%}")
    print("\n--- Latency (mean) ---")
    if split_active:
        print(
            f"  preprocessing  : {aggregate.get('mean_preprocessing_latency_ms')} ms"
            " (system prompt + tools list; excluded from TTFT/E2E below)"
        )
        print(
            f"    system prompt: {aggregate.get('mean_system_prefill_latency_ms')} ms"
            f" ({aggregate.get('mean_system_prefill_tokens')} tok)"
        )
        print(
            f"    tools list   : {aggregate.get('mean_tools_prefill_latency_ms')} ms"
            f" ({aggregate.get('mean_tools_prefill_tokens')} tok)"
        )
    qualifier = " (user query only)" if split_active else ""
    print(f"  TTFT           : {aggregate.get('mean_ttft_ms')} ms{qualifier}")
    print(f"  prefill        : {aggregate.get('mean_prefill_latency_ms')} ms")
    print(f"  decode         : {aggregate.get('mean_decode_latency_ms')} ms")
    qualifier_e2e = " (user query + decode only)" if split_active else ""
    print(f"  E2E            : {aggregate.get('mean_e2e_latency_ms')} ms{qualifier_e2e}")
    print("\n--- Throughput ---")
    print(f"  prefill tok/s  : {aggregate.get('mean_prefill_tok_per_sec')}")
    print(f"  decode tok/s   : {aggregate.get('mean_decode_tok_per_sec')}")
    print("\n--- Memory ---")
    print(f"  model          : {weights_mb:.1f} MB (static, {variant_label})")
    print(f"  peak RAM       : {aggregate.get('peak_ram_mb')} MB")
    if (kv := aggregate.get("mean_kv_cache_kb")) is not None:
        print(f"  mean KV cache  : {kv} KB")
    if (gpu := aggregate.get("mean_peak_gpu_mb")) is not None:
        print(f"  mean peak GPU  : {gpu} MB")
