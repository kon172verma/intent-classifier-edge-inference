"""Core inference pass with TTFT measurement for the HF Transformers baseline."""

from __future__ import annotations

import time
from typing import Any

import torch
from transformers import LogitsProcessor, LogitsProcessorList

from evaluation_baseline.cache import kv_cache_bytes, kv_cache_tokens
from evaluation_lib.config import MAX_NEW_TOKENS
from evaluation_lib.device import synchronize
from evaluation_lib.system_info import peak_gpu_memory_mb, reset_peak_gpu_memory


class TTFTCapture(LogitsProcessor):
    """Capture time-to-first-token inside ``model.generate()``.

    ``LogitsProcessor.__call__`` is invoked once per decode step.  The first
    call occurs immediately after the prefill forward pass, so the elapsed
    time at that point equals the prefill latency / TTFT.
    """

    def __init__(self, device: str) -> None:
        self._device = device
        self._start: float | None = None
        self._fired: bool = False
        self.ttft_ms: float | None = None

    def arm(self) -> None:
        """Start the clock.  Call this immediately before ``model.generate()``."""
        synchronize(self._device)
        self._start = time.perf_counter()
        self._fired = False
        self.ttft_ms = None

    def __call__(self, input_ids: torch.Tensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if not self._fired and self._start is not None:
            synchronize(self._device)
            self.ttft_ms = (time.perf_counter() - self._start) * 1000
            self._fired = True
        return scores


def run_inference(
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    device: str,
    ttft_capture: TTFTCapture,
    past_key_values: Any = None,
    attention_mask: torch.Tensor | None = None,
    system_prefill_ms: float = 0.0,
    system_prefill_tokens: int = 0,
    tools_prefill_ms: float = 0.0,
    tools_prefill_tokens: int = 0,
    report_prefill_split: bool = False,
) -> dict:
    """Run one inference pass and return per-example timing and output data.

    Parameters
    ----------
    input_ids:
        Tokenised prompt (or the dynamic suffix when *past_key_values* provided).
        When *report_prefill_split* is True, this is just the user-query
        tokens — the system-prompt and tools-list tokens were already
        ingested by the caller via ``ingest_prefix_segment`` and folded into
        *past_key_values*.
    past_key_values:
        Pre-computed KV cache for the static prompt prefix.
    attention_mask:
        Full mask covering both the cached prefix and the current tokens.
    system_prefill_ms, system_prefill_tokens:
        Wall-clock time / token count already spent ingesting the static
        system-prompt prefix (prefill phase 1), measured by the caller
        before this function was invoked. It is zero after the initial
        one-time cache creation, which is reported separately in run metadata.
    tools_prefill_ms, tools_prefill_tokens:
        Wall-clock time / token count already spent ingesting the
        available-tools list (prefill phase 2), measured by the caller.
    report_prefill_split:
        When True, report the system/tools/query prefill split as separate
        fields. ``ttft_ms``/``prefill_latency_ms``/``e2e_latency_ms`` cover
        ONLY the user-query phase (+ decode for e2e) -- the system-prompt
        and tools-list ingestion are considered pre-processing that happens
        ahead of the "live" request and are reported separately via
        ``preprocessing_latency_ms`` and the ``system_prefill_*`` /
        ``tools_prefill_*`` fields.
    """
    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "use_cache": True,
        "do_sample": False,
        "logits_processor": LogitsProcessorList([ttft_capture]),
        "return_dict_in_generate": True,
    }
    if past_key_values is not None:
        generate_kwargs["past_key_values"] = past_key_values
    if attention_mask is not None:
        generate_kwargs["attention_mask"] = attention_mask

    # Snapshot the cached prefix length *before* generate() mutates the cache
    # in-place (DynamicLayer.update() appends new KV tensors as decoding
    # proceeds, so reading this after generate() would double-count tokens).
    cached_prefix_tokens = kv_cache_tokens(past_key_values)

    reset_peak_gpu_memory(device)
    ttft_capture.arm()
    synchronize(device)
    t_start = time.perf_counter()

    with torch.no_grad():
        result = model.generate(input_ids, **generate_kwargs)

    synchronize(device)
    t_end = time.perf_counter()
    gpu_peak_mb = peak_gpu_memory_mb(device)

    # call_e2e_ms/query_prefill_ms cover only *this* call (user-query
    # ingestion + decode). Pre-processing (system prompt + tools list,
    # ingested by the caller before this function was invoked) is
    # deliberately EXCLUDED from ttft_ms/prefill_latency_ms/e2e_latency_ms:
    # in a real deployment both are done ahead of time, so only the
    # user-query phase reflects true per-request latency.
    call_e2e_ms = (t_end - t_start) * 1000
    query_prefill_ms = ttft_capture.ttft_ms if ttft_capture.ttft_ms is not None else call_e2e_ms
    decode_ms = max(0.0, call_e2e_ms - query_prefill_ms)

    preprocessing_ms = system_prefill_ms + tools_prefill_ms

    # Effective context length includes cached prefix tokens.
    n_input = input_ids.shape[1] + cached_prefix_tokens
    n_generated = result.sequences.shape[1] - input_ids.shape[1]

    prefill_tok_per_sec = (
        (input_ids.shape[1] / query_prefill_ms * 1000) if query_prefill_ms > 0 else None
    )
    decode_tok_per_sec = (
        ((n_generated - 1) / decode_ms * 1000) if decode_ms > 0 and n_generated > 1 else None
    )

    final_cache = getattr(result, "past_key_values", None)
    kv_bytes = kv_cache_bytes(final_cache)

    new_ids = result.sequences[0, input_ids.shape[1] :].tolist()
    generated_text = tokenizer.decode([int(i) for i in new_ids], skip_special_tokens=True)

    output = {
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
        "peak_gpu_mb": gpu_peak_mb,
    }

    if report_prefill_split:
        query_prefill_tokens = input_ids.shape[1]
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
