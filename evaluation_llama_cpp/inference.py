"""Core inference pass with TTFT measurement for the llama.cpp baseline.

Mirrors evaluation_baseline/inference.py's ``run_inference()`` contract and
returned-field names so ``evaluation_lib.metrics``/``plot_results.py`` logic
works unmodified across both benchmarks. Unlike the HF/PyTorch backend,
``llama_cpp``'s ``eval()``/``sample()`` calls are already synchronous (no
device queue to flush), so no ``synchronize()``-equivalent step is needed
around timers.
"""

from __future__ import annotations

import time
from typing import Any

from llama_cpp import Llama

from evaluation_lib.boundary import find_tools_query_boundary
from evaluation_lib.config import MAX_NEW_TOKENS
from evaluation_llama_cpp.cache import kv_cache_bytes, kv_cache_tokens, synchronize


def run_inference(
    llm: Llama,
    query_tokens: list[int],
    system_prefill_ms: float = 0.0,
    system_prefill_tokens: int = 0,
    tools_prefill_ms: float = 0.0,
    tools_prefill_tokens: int = 0,
    report_prefill_split: bool = False,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> dict:
    """Run one inference pass and return per-example timing and output data.

    Parameters
    ----------
    query_tokens:
        Token ids for JUST the user query -- the system-prompt and
        tools-list tokens were already ingested by the caller via
        ``ingest_prefix_segment`` and folded into *llm*'s KV cache.
    system_prefill_ms, system_prefill_tokens:
        Wall-clock time / token count already spent ingesting the static
        system-prompt prefix (prefill phase 1), measured by the caller
        before this function was invoked.
    tools_prefill_ms, tools_prefill_tokens:
        Wall-clock time / token count already spent ingesting the
        available-tools list (prefill phase 2), measured by the caller.
    report_prefill_split:
        When True, report the system/tools/query prefill split as separate
        fields. ``ttft_ms``/``prefill_latency_ms``/``e2e_latency_ms`` cover
        ONLY the user-query phase (+ decode for e2e) -- the system-prompt
        and tools-list ingestion are pre-processing, reported separately via
        ``preprocessing_latency_ms`` and the ``system_prefill_*`` /
        ``tools_prefill_*`` fields.
    """
    # Snapshot the cached prefix length *before* eval() mutates it in-place.
    cached_prefix_tokens = kv_cache_tokens(llm)

    synchronize(llm)
    t_start = time.perf_counter()
    llm.eval(query_tokens)
    synchronize(llm)
    t_query_done = time.perf_counter()
    query_prefill_ms = (t_query_done - t_start) * 1000

    eos_token = llm.token_eos()
    generated_tokens: list[int] = []
    for _ in range(max_new_tokens):
        token = llm.sample()
        if token == eos_token:
            break
        generated_tokens.append(token)
        llm.eval([token])
    # The final eval() above submits work for the next (unused) step; without
    # this barrier its async completion would leak past t_end, undercounting
    # decode_ms (mirrors the query-prefill sync above).
    synchronize(llm)
    t_end = time.perf_counter()
    decode_ms = max(0.0, (t_end - t_query_done) * 1000)

    preprocessing_ms = system_prefill_ms + tools_prefill_ms

    n_input = len(query_tokens) + cached_prefix_tokens
    n_generated = len(generated_tokens)

    prefill_tok_per_sec = (
        (len(query_tokens) / query_prefill_ms * 1000) if query_prefill_ms > 0 else None
    )
    decode_tok_per_sec = (
        ((n_generated - 1) / decode_ms * 1000) if decode_ms > 0 and n_generated > 1 else None
    )

    kv_bytes = kv_cache_bytes(llm)

    generated_text = llm.detokenize(generated_tokens).decode("utf-8", errors="ignore")

    output: dict[str, Any] = {
        "generated_text": generated_text.strip(),
        "n_input_tokens": n_input,
        "n_generated_tokens": n_generated,
        "prefill_latency_ms": round(query_prefill_ms, 3),
        "decode_latency_ms": round(decode_ms, 3),
        "e2e_latency_ms": round(query_prefill_ms + decode_ms, 3),
        "ttft_ms": round(query_prefill_ms, 3),
        "prefill_tok_per_sec": (
            round(prefill_tok_per_sec, 2) if prefill_tok_per_sec is not None else None
        ),
        "decode_tok_per_sec": (
            round(decode_tok_per_sec, 2) if decode_tok_per_sec is not None else None
        ),
        "kv_cache_bytes": kv_bytes,
        "peak_gpu_mb": None,  # llama.cpp/Metal exposes no cheap live-memory API
    }

    if report_prefill_split:
        query_prefill_tokens = len(query_tokens)
        system_tok_per_sec = (
            (system_prefill_tokens / system_prefill_ms * 1000) if system_prefill_ms > 0 else None
        )
        tools_tok_per_sec = (
            (tools_prefill_tokens / tools_prefill_ms * 1000) if tools_prefill_ms > 0 else None
        )
        query_tok_per_sec = (
            (query_prefill_tokens / query_prefill_ms * 1000) if query_prefill_ms > 0 else None
        )
        output.update(
            {
                "preprocessing_latency_ms": round(preprocessing_ms, 3),
                "system_prefill_latency_ms": round(system_prefill_ms, 3),
                "system_prefill_tokens": system_prefill_tokens,
                "system_prefill_tok_per_sec": (
                    round(system_tok_per_sec, 2) if system_tok_per_sec is not None else None
                ),
                "tools_prefill_latency_ms": round(tools_prefill_ms, 3),
                "tools_prefill_tokens": tools_prefill_tokens,
                "tools_prefill_tok_per_sec": (
                    round(tools_tok_per_sec, 2) if tools_tok_per_sec is not None else None
                ),
                "query_prefill_latency_ms": round(query_prefill_ms, 3),
                "query_prefill_tokens": query_prefill_tokens,
                "query_prefill_tok_per_sec": (
                    round(query_tok_per_sec, 2) if query_tok_per_sec is not None else None
                ),
            }
        )
    else:
        output.update(
            {
                "preprocessing_latency_ms": None,
                "system_prefill_latency_ms": None,
                "system_prefill_tokens": None,
                "system_prefill_tok_per_sec": None,
                "tools_prefill_latency_ms": None,
                "tools_prefill_tokens": None,
                "tools_prefill_tok_per_sec": None,
                "query_prefill_latency_ms": None,
                "query_prefill_tokens": None,
                "query_prefill_tok_per_sec": None,
            }
        )

    return output
