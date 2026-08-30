"""KV-cache inspection, cloning, and prefix-cache pre-computation."""

from __future__ import annotations

import time
from typing import Any

import torch

from evaluation_lib.compatibility import PromptSpec
from evaluation_lib.device import synchronize
from evaluation_lib.prompt import build_system_prefix_text

# ---------------------------------------------------------------------------
# KV-cache inspection
# ---------------------------------------------------------------------------


def kv_cache_bytes(cache: Any) -> int:
    """Return the total number of bytes occupied by *cache*."""
    if cache is None:
        return 0
    total = 0
    if hasattr(cache, "layers"):
        # Unified Cache API (transformers >= 4.56): list of CacheLayer objects,
        # each storing its own `.keys` / `.values` tensors.
        for layer in cache.layers:
            k = getattr(layer, "keys", None)
            v = getattr(layer, "values", None)
            if isinstance(k, torch.Tensor) and isinstance(v, torch.Tensor):
                total += k.nbytes + v.nbytes
    elif hasattr(cache, "key_cache"):
        # DynamicCache (transformers 4.38 - 4.55): parallel key/value lists
        for k, v in zip(cache.key_cache, cache.value_cache):
            if isinstance(k, torch.Tensor):
                total += k.nbytes + v.nbytes
    else:
        # Legacy tuple-of-tuples
        for layer_kv in cache:
            total += layer_kv[0].nbytes + layer_kv[1].nbytes
    return total


def kv_cache_tokens(cache: Any) -> int:
    """Return the number of tokens stored in *cache*."""
    if cache is None:
        return 0
    if hasattr(cache, "get_seq_length"):
        return cache.get_seq_length()
    if hasattr(cache, "_seen_tokens"):
        return cache._seen_tokens  # type: ignore[attr-defined]
    if cache and not hasattr(cache, "key_cache"):
        # Legacy tuple: shape is [batch, heads, seq, head_dim]
        return cache[0][0].shape[2]
    return 0


# ---------------------------------------------------------------------------
# Cache cloning
# ---------------------------------------------------------------------------


def clone_cache(cache: Any) -> Any:
    """Return a fully independent copy of *cache*.

    ``DynamicCache.update()`` mutates the cache in-place by appending new KV
    tensors.  Without cloning, the prefix cache would be corrupted after the
    first example.
    """
    if cache is None:
        return None
    if hasattr(cache, "layers"):
        # Unified Cache API (transformers >= 4.56): clone each CacheLayer's
        # `.keys` / `.values` tensors and mark the fresh layer as initialized
        # so `DynamicLayer.update()` appends to it instead of re-initializing.
        try:
            from transformers.cache_utils import DynamicCache, DynamicLayer
        except ImportError:
            from transformers import DynamicCache, DynamicLayer  # type: ignore[no-redef]
        fresh = DynamicCache()
        _f: Any = fresh
        cloned_layers: list[Any] = []
        for layer in cache.layers:
            new_layer = DynamicLayer()
            new_layer.keys = layer.keys.clone()
            new_layer.values = layer.values.clone()
            new_layer.dtype = layer.keys.dtype
            new_layer.device = layer.keys.device
            new_layer.is_initialized = True
            cloned_layers.append(new_layer)
        _f.layers = cloned_layers
        return fresh
    if hasattr(cache, "key_cache"):
        try:
            from transformers.cache_utils import DynamicCache
        except ImportError:
            from transformers import DynamicCache  # type: ignore[no-redef]
        _c: Any = cache
        fresh = DynamicCache()
        _fresh: Any = fresh
        _fresh.key_cache = [k.clone() for k in _c.key_cache]
        _fresh.value_cache = [v.clone() for v in _c.value_cache]
        if hasattr(_c, "_seen_tokens"):
            _fresh._seen_tokens = _c._seen_tokens
        return fresh
    # Legacy tuple-of-tuples
    return tuple((k.clone(), v.clone()) for k, v in cache)


# ---------------------------------------------------------------------------
# Prefix-cache pre-computation
# ---------------------------------------------------------------------------


def compute_prefix_cache(
    model: Any, tokenizer: Any, device: str, prompt_spec: PromptSpec | None = None
) -> tuple[Any, int, float]:
    """Pre-compute the KV cache for the static system-prompt prefix.

    Returns
    -------
    prefix_past_kv
        The computed KV cache (clone before each inference call).
    prefix_len
        Number of tokens in the prefix.
    creation_ms
        Wall-clock time to compute the cache (ms).  This is a one-time cost.
    """
    prefix_text = build_system_prefix_text(tokenizer, prompt_spec)
    if not prefix_text:
        print(
            "[prefix_cache] WARNING: could not build system prefix text. "
            "Prefix caching will have no effect."
        )
        return None, 0, 0.0

    prefix_ids = tokenizer(prefix_text, return_tensors="pt").input_ids.to(device)
    prefix_len = prefix_ids.shape[1]
    print(f"[prefix_cache] Computing KV cache for {prefix_len} system-prefix tokens …")

    synchronize(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=prefix_ids, use_cache=True)
    synchronize(device)
    creation_ms = (time.perf_counter() - t0) * 1000

    cache_kb = kv_cache_bytes(out.past_key_values) / 1024
    print(
        f"[prefix_cache] Done. creation_time={creation_ms:.1f} ms, cache_size={cache_kb:.1f} KB\n"
    )
    return out.past_key_values, prefix_len, creation_ms


# ---------------------------------------------------------------------------
# Tools-list prefix ingestion (prefill phase 1)
# ---------------------------------------------------------------------------
# Generic prefix-segment ingestion (used for both the system-prompt phase
# and the tools-list phase; the caller decides which token slice to pass).
# ---------------------------------------------------------------------------


def ingest_prefix_segment(
    model: Any,
    segment_ids: torch.Tensor,
    device: str,
    past_key_values: Any = None,
) -> tuple[Any, float]:
    """Run one forward pass over *segment_ids*, extending *past_key_values*.

    Generic building block used to time each phase of the 3-phase prefill
    split (system prompt -> tools list -> user query) separately: call this
    once per phase with that phase's token slice, chaining the returned
    cache into the next call.

    Returns
    -------
    updated_cache
        *past_key_values* extended with *segment_ids* (or unchanged if
        *segment_ids* is empty).
    elapsed_ms
        Wall-clock time for this forward pass (0.0 if *segment_ids* is empty).
    """
    if segment_ids.shape[1] == 0:
        return past_key_values, 0.0

    synchronize(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=segment_ids, past_key_values=past_key_values, use_cache=True)
    synchronize(device)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return out.past_key_values, elapsed_ms
