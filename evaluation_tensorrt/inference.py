"""Core inference pass with 3-phase prefill timing for the TensorRT-LLM baseline.

Mirrors evaluation_onnx/inference.py's ``run_inference()`` contract and all
returned-field names so ``evaluation_lib.metrics`` and ``plot_results.py``
work unmodified across benchmarks.

TensorRT-LLM execution notes
------------------------------
- ``ModelRunner.generate()`` is a synchronous call that blocks until all
  output tokens are produced.  TTFT cannot be extracted from it directly, so
  we split the prefill+first-token step from the remaining decode steps
  manually using ``max_new_tokens=1`` for the first call and
  ``max_new_tokens=remaining`` for the rest.
- The runner's internal KV cache is stateful across the prefill segments we
  feed manually (system prompt → tools list → user query), so we drive
  incremental feeding via separate ``generate()`` calls with
  ``return_dict=True`` and ``streaming=False``.
- ``torch.cuda.synchronize()`` is called around each timed section to ensure
  GPU work is complete before reading the clock.
"""

from __future__ import annotations

import time
from typing import Any

import torch

from evaluation_lib.config import MAX_NEW_TOKENS


def _cuda_sync() -> None:
    """Synchronize the default CUDA device if one is available."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def find_tools_query_boundary(
    full_token_ids: list[int], tools_only_token_ids: list[int]
) -> int:
    """Return the index where user-query tokens start within *full_token_ids*.

    Uses the longest common prefix between the full prompt and an
    empty-user-request variant (``build_tools_only_prompt``), mirroring the
    approach used in evaluation_baseline and evaluation_onnx.
    """
    n = min(len(full_token_ids), len(tools_only_token_ids))
    for i in range(n):
        if full_token_ids[i] != tools_only_token_ids[i]:
            return i
    return n


def _run_prefill_segment(
    runner: Any,
    token_ids: list[int],
) -> tuple[float, list[int]]:
    """Feed *token_ids* into the runner's KV cache and return (elapsed_ms, []).

    Uses ``max_new_tokens=0`` so TRT-LLM only runs the prefill step and
    returns no new tokens.  Timed with cuda.synchronize barriers.
    """
    input_tensor = torch.tensor([token_ids], dtype=torch.int32)

    _cuda_sync()
    t0 = time.perf_counter()
    runner.generate(
        batch_input_ids=[input_tensor],
        max_new_tokens=0,
        end_id=-1,  # no EOS -- we only want the prefill, not decode
        pad_id=0,
    )
    _cuda_sync()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return elapsed_ms, []


def run_inference(
    runner: Any,
    tokenizer: Any,
    query_token_ids: list[int],
    eos_token_ids: set[int],
    system_prefill_ms: float = 0.0,
    system_prefill_tokens: int = 0,
    tools_prefill_ms: float = 0.0,
    tools_prefill_tokens: int = 0,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> dict:
    """Run the user-query prefill + decode and return per-example timings.

    The system-prompt and tools-list tokens are assumed to already be baked
    into the runner's KV cache (ingested by the caller via
    ``_run_prefill_segment``).

    Returns
    -------
    dict with fields matching the schema produced by evaluation_baseline,
    evaluation_llama_cpp, and evaluation_onnx so that evaluation_lib.metrics
    and plot_results.py work unmodified.
    """
    input_tensor = torch.tensor([query_token_ids], dtype=torch.int32)
    n_query_tokens = len(query_token_ids)

    # ── Step 1: prefill query + first token (measures TTFT) ─────────────
    _cuda_sync()
    t_start = time.perf_counter()
    first_out = runner.generate(
        batch_input_ids=[input_tensor],
        max_new_tokens=1,
        end_id=min(eos_token_ids) if eos_token_ids else -1,
        pad_id=0,
        return_dict=True,
    )
    _cuda_sync()
    ttft_ms = (time.perf_counter() - t_start) * 1000

    # Extract the first generated token
    out_ids: list[int] = first_out["output_ids"][0][0].tolist()
    # output_ids includes the input tokens; strip them
    first_token_ids = out_ids[n_query_tokens:]
    generated_ids: list[int] = []
    hit_eos = False
    for tok_id in first_token_ids:
        if tok_id in eos_token_ids:
            hit_eos = True
            break
        generated_ids.append(tok_id)

    # ── Step 2: remaining decode tokens ─────────────────────────────────
    t_decode_start = time.perf_counter()
    if not hit_eos and len(generated_ids) < max_new_tokens:
        remaining = max_new_tokens - len(generated_ids)
        # Feed back the full context (query + generated so far) to continue
        continued_ids = query_token_ids + generated_ids
        cont_tensor = torch.tensor([continued_ids], dtype=torch.int32)

        _cuda_sync()
        cont_out = runner.generate(
            batch_input_ids=[cont_tensor],
            max_new_tokens=remaining,
            end_id=min(eos_token_ids) if eos_token_ids else -1,
            pad_id=0,
            return_dict=True,
        )
        _cuda_sync()

        cont_out_ids: list[int] = cont_out["output_ids"][0][0].tolist()
        new_ids = cont_out_ids[len(continued_ids) :]
        for tok_id in new_ids:
            if tok_id in eos_token_ids:
                break
            generated_ids.append(tok_id)

    decode_ms = (time.perf_counter() - t_decode_start) * 1000

    # ── Derived metrics ──────────────────────────────────────────────────
    n_generated = len(generated_ids)
    prefill_ms = ttft_ms  # TTFT = query prefill + first token
    e2e_ms = ttft_ms + decode_ms
    preprocessing_ms = system_prefill_ms + tools_prefill_ms

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    result: dict = {
        "generated_text": generated_text,
        "generated_tokens": n_generated,
        "prefill_latency_ms": round(prefill_ms, 3),
        "decode_latency_ms": round(decode_ms, 3),
        "e2e_latency_ms": round(e2e_ms, 3),
        "ttft_ms": round(ttft_ms, 3),
        "prefill_tok_per_sec": round(n_query_tokens / prefill_ms * 1000, 3)
        if prefill_ms > 0
        else None,
        "decode_tok_per_sec": round(n_generated / decode_ms * 1000, 3)
        if decode_ms > 0 and n_generated > 0
        else None,
        "prefill_tokens": n_query_tokens,
        "preprocessing_latency_ms": round(preprocessing_ms, 3),
        "system_prefill_latency_ms": round(system_prefill_ms, 3),
        "system_prefill_tokens": system_prefill_tokens,
        "system_prefill_tok_per_sec": round(
            system_prefill_tokens / system_prefill_ms * 1000, 3
        )
        if system_prefill_ms > 0
        else None,
        "tools_prefill_latency_ms": round(tools_prefill_ms, 3),
        "tools_prefill_tokens": tools_prefill_tokens,
        "tools_prefill_tok_per_sec": round(
            tools_prefill_tokens / tools_prefill_ms * 1000, 3
        )
        if tools_prefill_ms > 0
        else None,
        "query_prefill_latency_ms": round(prefill_ms, 3),
        "query_prefill_tokens": n_query_tokens,
        "query_prefill_tok_per_sec": round(n_query_tokens / prefill_ms * 1000, 3)
        if prefill_ms > 0
        else None,
        "peak_gpu_mb": _peak_gpu_mb(),
        "peak_ram_mb": None,  # populated by the caller after the loop
        "kv_cache_kb": None,  # TRT-LLM manages KV cache internally; not exposed
    }
    return result


def _peak_gpu_mb() -> float | None:
    """Return peak CUDA memory allocated in MB, or None if CUDA is unavailable."""
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / (1024**2), 2)
    return None
