"""Regression tests for Transformers prefix-cache generation."""

from types import SimpleNamespace
from typing import Any

import torch

from evaluation_baseline.inference import TTFTCapture, run_inference


class _RecordingModel:
    def __init__(self) -> None:
        self.input_ids: torch.Tensor | None = None

    def generate(self, input_ids: torch.Tensor, **_: Any) -> SimpleNamespace:
        self.input_ids = input_ids
        generated = torch.tensor([[99]], dtype=input_ids.dtype)
        return SimpleNamespace(
            sequences=torch.cat((input_ids, generated), dim=1), past_key_values=None
        )


class _Tokenizer:
    def decode(self, _: list[int], skip_special_tokens: bool) -> str:
        assert skip_special_tokens
        return "a"


def test_cached_generation_receives_full_prompt_and_reports_query_suffix() -> None:
    model = _RecordingModel()
    full_prompt = torch.tensor([[1, 2, 3, 4, 5]])
    prefix = ((torch.zeros((1, 1, 3, 1)), torch.zeros((1, 1, 3, 1))),)

    result = run_inference(
        model,
        _Tokenizer(),
        full_prompt,
        "cpu",
        TTFTCapture("cpu"),
        past_key_values=prefix,
        attention_mask=torch.ones_like(full_prompt),
        report_prefill_split=True,
    )

    assert torch.equal(model.input_ids, full_prompt)
    assert result["n_input_tokens"] == 5
    assert result["query_prefill_tokens"] == 2
