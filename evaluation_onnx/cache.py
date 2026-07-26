"""KV-cache (as explicit numpy tensors) management for ONNX Runtime inference.

Unlike HF Transformers (``DynamicCache``) or llama.cpp (internal C++ state),
ONNX Runtime's decoder-with-past graphs are stateless functions: every
``session.run()`` call takes the *entire* KV cache as explicit
``past_key_values.<N>.key``/``.value`` input tensors and returns the extended
cache as ``present.<N>.key``/``.value`` output tensors. The caller is
responsible for holding onto and threading these numpy arrays between calls
-- this module provides that bookkeeping, mirroring the role
``evaluation_baseline/cache.py`` and ``evaluation_llama_cpp/cache.py`` play
for their respective backends.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import onnxruntime as ort

Cache = dict[str, np.ndarray]

_ORT_TO_NUMPY_DTYPE: dict[str, Any] = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
}


def _past_kv_specs(session: ort.InferenceSession) -> list[tuple[str, int, int, Any]]:
    """Return ``(input_name, num_heads, head_dim, numpy_dtype)`` for each KV input."""
    specs = []
    for inp in session.get_inputs():
        if not inp.name.startswith("past_key_values."):
            continue
        # shape: [batch_size, num_heads, past_sequence_length, head_dim]
        num_heads = inp.shape[1]
        head_dim = inp.shape[3]
        dtype = _ORT_TO_NUMPY_DTYPE[inp.type]
        specs.append((inp.name, num_heads, head_dim, dtype))
    return specs


def empty_cache(session: ort.InferenceSession) -> Cache:
    """Return a zero-length KV cache (batch_size=1, past_sequence_length=0)."""
    cache: Cache = {}
    for name, num_heads, head_dim, dtype in _past_kv_specs(session):
        cache[name] = np.zeros((1, num_heads, 0, head_dim), dtype=dtype)
    return cache


def kv_cache_tokens(cache: Cache) -> int:
    """Return the number of tokens stored in *cache*."""
    if not cache:
        return 0
    any_tensor = next(iter(cache.values()))
    return int(any_tensor.shape[2])


def kv_cache_bytes(cache: Cache) -> int:
    """Return the total number of bytes occupied by *cache*."""
    return sum(t.nbytes for t in cache.values())


def clone_cache(cache: Cache) -> Cache:
    """Return an independent deep copy of *cache*.

    Needed before reusing a prefix cache across examples: each
    ``session.run()`` call returns *new* arrays (ORT does not mutate inputs
    in-place), but the prefix cache dict itself is reused as a starting
    point for every example and must not be extended in place.
    """
    return {name: arr.copy() for name, arr in cache.items()}


def _present_to_past(outputs: dict[str, np.ndarray]) -> Cache:
    """Rename ``present.<N>.key/value`` outputs to ``past_key_values.<N>.key/value``
    so they can be fed directly back in as the next call's cache input.
    """
    return {
        name.replace("present.", "past_key_values.", 1): arr
        for name, arr in outputs.items()
        if name.startswith("present.")
    }


def run_segment(
    session: ort.InferenceSession,
    input_ids: np.ndarray,
    cache: Cache,
) -> tuple[Cache, np.ndarray, float]:
    """Run one forward pass over *input_ids*, extending *cache*.

    Generic building block used to time each phase of the 3-phase prefill
    split (system prompt -> tools list -> user query) separately, and for
    single-token decode steps.

    Parameters
    ----------
    input_ids:
        Shape ``(1, seq_len)`` int64 array. May be empty (``seq_len == 0``),
        in which case this is a no-op returning *cache* unchanged.

    Returns
    -------
    updated_cache
        *cache* extended with *input_ids* (or unchanged if *input_ids* is empty).
    logits
        Shape ``(1, seq_len, vocab_size)`` output logits.
    elapsed_ms
        Wall-clock time for this forward pass (0.0 if *input_ids* is empty).
        ONNX Runtime's ``session.run()`` is synchronous (blocks until the
        CPU/CoreML computation completes), so no explicit synchronization
        barrier is needed here (unlike llama.cpp's async Metal backend).
    """
    seq_len = input_ids.shape[1]
    if seq_len == 0:
        return cache, np.empty((1, 0, 0), dtype=np.float32), 0.0

    past_len = kv_cache_tokens(cache)
    total_len = past_len + seq_len
    feed: dict[str, np.ndarray] = {
        "input_ids": input_ids,
        "attention_mask": np.ones((1, total_len), dtype=np.int64),
        "position_ids": np.arange(past_len, total_len, dtype=np.int64)[None, :],
        **cache,
    }
    output_names = [o.name for o in session.get_outputs()]

    t0 = time.perf_counter()
    outputs = session.run(output_names, feed)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    output_map = dict(zip(output_names, outputs))
    logits = output_map["logits"]
    new_cache = _present_to_past(output_map)
    return new_cache, logits, elapsed_ms


def compute_prefix_cache(
    session: ort.InferenceSession, system_tokens: np.ndarray
) -> tuple[Cache, int, float]:
    """Pre-compute the KV cache for the static system-prompt prefix.

    Returns
    -------
    prefix_cache
        The computed KV cache (clone before each inference call).
    prefix_len
        Number of tokens in the prefix.
    creation_ms
        Wall-clock time to compute the cache (ms). One-time cost.
    """
    if system_tokens.shape[1] == 0:
        return empty_cache(session), 0, 0.0

    cache = empty_cache(session)
    new_cache, _logits, creation_ms = run_segment(session, system_tokens, cache)
    return new_cache, system_tokens.shape[1], creation_ms
