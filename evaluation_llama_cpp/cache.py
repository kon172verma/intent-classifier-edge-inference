"""KV-cache (llama.cpp state) inspection and prefix-cache pre-computation.

Mirrors evaluation_baseline/cache.py's role, adapted to llama-cpp-python's
stateful ``Llama`` object: instead of an explicit ``past_key_values`` tensor
that gets passed around and cloned, the KV cache lives inside the ``Llama``
instance itself (``llm.n_tokens`` / internal ctx state). ``llm.eval(tokens)``
extends that internal state exactly like a forward pass with
``past_key_values`` in the HF baseline. ``save_state()``/``load_state()``
snapshot/restore that internal state, standing in for ``clone_cache()``.
"""

from __future__ import annotations

import time

import llama_cpp.llama_cpp as llama_cpp_capi
from llama_cpp import Llama
from llama_cpp.llama import LlamaState


def synchronize(llm: Llama) -> None:
    """Block until all pending (possibly async, e.g. Metal) GPU work completes.

    llama.cpp's Metal backend submits compute asynchronously; without this
    barrier, wall-clock timing around ``eval()`` calls is unreliable -- work
    submitted by one call can silently bleed into the next call's timer
    (analogous to ``evaluation_lib.device.synchronize()`` for MPS/CUDA).
    """
    llama_cpp_capi.llama_synchronize(llm.ctx)


def ingest_prefix_segment(llm: Llama, tokens: list[int]) -> float:
    """Evaluate *tokens*, extending the model's KV cache. Returns elapsed ms.

    Generic building block used to time each phase of the 3-phase prefill
    split (system prompt -> tools list -> user query) separately, mirroring
    ``evaluation_baseline.cache.ingest_prefix_segment``.
    """
    if not tokens:
        return 0.0
    synchronize(llm)
    t0 = time.perf_counter()
    llm.eval(tokens)
    synchronize(llm)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return elapsed_ms


def kv_cache_tokens(llm: Llama) -> int:
    """Return the number of tokens currently held in the KV cache."""
    return llm.n_tokens


def kv_cache_bytes(llm: Llama) -> int:
    """Return the current KV-cache (+ state) size in bytes, without copying.

    Uses the low-level ``llama_state_get_size`` C API, which computes the
    serialized state size without actually performing the (expensive) copy
    that ``save_state()`` does -- safe to call after every example.
    """
    return int(llama_cpp_capi.llama_state_get_size(llm.ctx))


def compute_prefix_cache(llm: Llama, system_tokens: list[int]) -> tuple[LlamaState, int, float]:
    """Pre-compute and save the KV-cache state for the static system prompt.

    Returns
    -------
    state
        Saved ``LlamaState`` snapshot (pass to ``clone_prefix_cache`` before
        each example to restore the model to this point).
    prefix_len
        Number of tokens in the prefix.
    creation_ms
        Wall-clock time to compute the cache (ms). One-time cost.
    """
    llm.reset()
    synchronize(llm)
    t0 = time.perf_counter()
    llm.eval(system_tokens)
    synchronize(llm)
    creation_ms = (time.perf_counter() - t0) * 1000
    state = llm.save_state()
    return state, len(system_tokens), creation_ms


def clone_prefix_cache(llm: Llama, state: LlamaState) -> None:
    """Restore *llm*'s KV cache to the saved prefix *state* (in-place).

    Analogous to ``evaluation_baseline.cache.clone_cache`` -- this must be
    called before starting any per-example timers, since (like
    ``clone_cache``) its cost is deliberately NOT included in any reported
    timing bucket.
    """
    llm.load_state(state)
