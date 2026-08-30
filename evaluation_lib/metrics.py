"""Aggregate latency/throughput/quality metrics over a benchmark run."""

from __future__ import annotations

import statistics

from evaluation_lib.system_info import peak_ram_mb


def _pct(values: list[float], p: float) -> float:
    """Return the *p*-th percentile of *values* (nearest-rank method)."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = min(int(len(sorted_v) * p / 100), len(sorted_v) - 1)
    return round(sorted_v[idx], 3)


def aggregate_metrics(per_example: list[dict]) -> dict:
    """Compute aggregate latency, throughput, and memory metrics."""

    def _mean(key: str) -> float | None:
        vals = [e[key] for e in per_example if e.get(key) is not None]
        return round(statistics.mean(vals), 3) if vals else None

    prefill_vals = [e["prefill_latency_ms"] for e in per_example]
    decode_vals = [e["decode_latency_ms"] for e in per_example]
    e2e_vals = [e["e2e_latency_ms"] for e in per_example]
    preprocessing_vals = [
        e["preprocessing_latency_ms"]
        for e in per_example
        if e.get("preprocessing_latency_ms") is not None
    ]
    system_prefill_vals = [
        e["system_prefill_latency_ms"]
        for e in per_example
        if e.get("system_prefill_latency_ms") is not None
    ]
    tools_prefill_vals = [
        e["tools_prefill_latency_ms"]
        for e in per_example
        if e.get("tools_prefill_latency_ms") is not None
    ]
    query_prefill_vals = [
        e["query_prefill_latency_ms"]
        for e in per_example
        if e.get("query_prefill_latency_ms") is not None
    ]

    return {
        "n_examples": len(per_example),
        "mean_prefill_latency_ms": _mean("prefill_latency_ms"),
        "p50_prefill_latency_ms": _pct(prefill_vals, 50),
        "p95_prefill_latency_ms": _pct(prefill_vals, 95),
        "mean_decode_latency_ms": _mean("decode_latency_ms"),
        "p50_decode_latency_ms": _pct(decode_vals, 50),
        "p95_decode_latency_ms": _pct(decode_vals, 95),
        "mean_e2e_latency_ms": _mean("e2e_latency_ms"),
        "p50_e2e_latency_ms": _pct(e2e_vals, 50),
        "p95_e2e_latency_ms": _pct(e2e_vals, 95),
        "mean_ttft_ms": _mean("ttft_ms"),
        "mean_prefill_tok_per_sec": _mean("prefill_tok_per_sec"),
        "mean_decode_tok_per_sec": _mean("decode_tok_per_sec"),
        "mean_prefill_tokens": _mean("n_input_tokens"),
        "mean_generated_tokens": _mean("n_generated_tokens"),
        "mean_preprocessing_latency_ms": _mean("preprocessing_latency_ms"),
        "p50_preprocessing_latency_ms": (
            _pct(preprocessing_vals, 50) if preprocessing_vals else None
        ),
        "p95_preprocessing_latency_ms": (
            _pct(preprocessing_vals, 95) if preprocessing_vals else None
        ),
        "mean_system_prefill_latency_ms": _mean("system_prefill_latency_ms"),
        "p50_system_prefill_latency_ms": (
            _pct(system_prefill_vals, 50) if system_prefill_vals else None
        ),
        "p95_system_prefill_latency_ms": (
            _pct(system_prefill_vals, 95) if system_prefill_vals else None
        ),
        "mean_system_prefill_tokens": _mean("system_prefill_tokens"),
        "mean_system_prefill_tok_per_sec": _mean("system_prefill_tok_per_sec"),
        "mean_tools_prefill_latency_ms": _mean("tools_prefill_latency_ms"),
        "p50_tools_prefill_latency_ms": (
            _pct(tools_prefill_vals, 50) if tools_prefill_vals else None
        ),
        "p95_tools_prefill_latency_ms": (
            _pct(tools_prefill_vals, 95) if tools_prefill_vals else None
        ),
        "mean_tools_prefill_tokens": _mean("tools_prefill_tokens"),
        "mean_tools_prefill_tok_per_sec": _mean("tools_prefill_tok_per_sec"),
        "mean_query_prefill_latency_ms": _mean("query_prefill_latency_ms"),
        "p50_query_prefill_latency_ms": (
            _pct(query_prefill_vals, 50) if query_prefill_vals else None
        ),
        "p95_query_prefill_latency_ms": (
            _pct(query_prefill_vals, 95) if query_prefill_vals else None
        ),
        "mean_query_prefill_tokens": _mean("query_prefill_tokens"),
        "mean_query_prefill_tok_per_sec": _mean("query_prefill_tok_per_sec"),
        "peak_ram_mb": round(peak_ram_mb(), 2),
        "mean_kv_cache_bytes": _mean("kv_cache_bytes"),
        "mean_kv_cache_kb": round((_mean("kv_cache_bytes") or 0) / 1024, 2),
        "mean_peak_gpu_mb": _mean("peak_gpu_mb"),
    }


def compute_quality(per_example: list[dict], dataset: list[dict], warmup: int) -> dict:
    """Compute tool-selection quality metrics over the measured examples."""
    measured = dataset[warmup:]
    total = len(per_example)
    if total == 0:
        return {}

    correct = sum(1 for e in per_example if e["correct"])

    none_pairs = [(e, ex) for e, ex in zip(per_example, measured) if ex["answer"] == "none"]
    none_correct = sum(1 for e, _ in none_pairs if e["predicted"] == "none")
    none_total = len(none_pairs)

    tool_name_sets = [{t["name"] for t in ex["available_tools"]} for ex in measured]
    invalid = sum(
        1
        for e, names in zip(per_example, tool_name_sets)
        if e["predicted"] != "none" and e["predicted"] not in names
    )

    return {
        "tool_accuracy": round(correct / total, 4),
        "exact_match_rate": round(correct / total, 4),
        "none_accuracy": round(none_correct / none_total, 4) if none_total else None,
        "invalid_tool_rate": round(invalid / total, 4),
        "n_correct": correct,
        "n_total": total,
        "n_none_examples": none_total,
        "n_invalid": invalid,
    }
