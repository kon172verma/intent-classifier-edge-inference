"""Core inference pass with 3-phase prefill timing for the ONNX Runtime backend."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import onnxruntime as ort

from evaluation_lib.config import MAX_NEW_TOKENS
from evaluation_onnx.cache import Cache, kv_cache_bytes, kv_cache_tokens, run_segment


def run_inference(
    session: ort.InferenceSession,
    tokenizer: Any,
    query_ids: np.ndarray,
    cache: Cache,
    eos_token_ids: set[int],
    system_prefill_ms: float = 0.0,
    system_prefill_tokens: int = 0,
    tools_prefill_ms: float = 0.0,
    tools_prefill_tokens: int = 0,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> dict:
    """Run one inference pass (user query prefill + greedy decode) and return timing/output data.

    Parameters
    ----------
    query_ids:
        Tokenised user-query suffix only -- the system-prompt and
        tools-list tokens were already ingested by the caller via
        ``cache.run_segment`` and folded into *cache*.
    cache:
        KV cache already extended with the system-prompt + tools-list
        tokens (see ``evaluation_onnx/run.py``).
    system_prefill_ms, system_prefill_tokens, tools_prefill_ms, tools_prefill_tokens:
        Wall-clock time / token count already spent ingesting the static
        system-prompt prefix and tools list (prefill phases 1-2), measured
        by the caller before this function was invoked. Reported separately
        via ``preprocessing_latency_ms`` and the ``system_prefill_*`` /
        ``tools_prefill_*`` fields, matching ``evaluation_baseline`` and
        ``evaluation_llama_cpp``'s ``report_prefill_split=True`` contract.
    """
    cached_prefix_tokens = kv_cache_tokens(cache)

    t_start = time.perf_counter()  # noqa: F841
    cache, logits, query_prefill_ms = run_segment(session, query_ids, cache)
    t_query_done = time.perf_counter()
    # run_segment() already measures its own wall-clock time internally;
    # query_prefill_ms above IS that measurement (session.run() is
    # synchronous, so no separate barrier is needed -- see cache.py).

    next_token = int(np.argmax(logits[0, -1, :]))
    generated_tokens: list[int] = []
    hit_eos = next_token in eos_token_ids
    if not hit_eos:
        generated_tokens.append(next_token)

    # NOTE: must track "hit EOS" via an explicit flag, not by checking
    # `not generated_tokens` -- that expression is also true on the very
    # first iteration (nothing generated yet) as well as whenever EOS was
    # the very first predicted token, which would otherwise wrongly re-enter
    # the loop and feed the EOS token id back into the model as if it were
    # real content, causing runaway generation past the intended stop point.
    while not hit_eos and len(generated_tokens) < max_new_tokens:
        next_input = np.array([[next_token]], dtype=np.int64)
        cache, logits, _step_ms = run_segment(session, next_input, cache)
        next_token = int(np.argmax(logits[0, -1, :]))
        hit_eos = next_token in eos_token_ids
        if not hit_eos:
            generated_tokens.append(next_token)
    t_end = time.perf_counter()

    decode_ms = max(0.0, (t_end - t_query_done) * 1000)
    preprocessing_ms = system_prefill_ms + tools_prefill_ms

    n_input = query_ids.shape[1] + cached_prefix_tokens
    n_generated = len(generated_tokens)

    query_prefill_tokens = query_ids.shape[1]
    prefill_tok_per_sec = (
        (query_prefill_tokens / query_prefill_ms * 1000) if query_prefill_ms > 0 else None
    )
    decode_tok_per_sec = (
        ((n_generated - 1) / decode_ms * 1000) if decode_ms > 0 and n_generated > 1 else None
    )

    system_tok_per_sec = (
        (system_prefill_tokens / system_prefill_ms * 1000) if system_prefill_ms > 0 else None
    )
    tools_tok_per_sec = (
        (tools_prefill_tokens / tools_prefill_ms * 1000) if tools_prefill_ms > 0 else None
    )
    query_tok_per_sec = prefill_tok_per_sec

    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return {
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
        "kv_cache_bytes": kv_cache_bytes(cache),
        "peak_gpu_mb": None,  # No cheap CoreML/ORT live-memory query API.
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
