"""Regression tests for Transformers baseline model loading."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from evaluation_baseline.model_loader import load_model_and_tokenizer


def test_loader_passes_torch_dtype_to_transformers() -> None:
    """Avoid adding a non-serializable ``dtype`` field to the model config."""
    model_path = Path("/models/qwen")
    model = MagicMock()

    with (
        patch("evaluation_baseline.model_loader.AutoTokenizer.from_pretrained"),
        patch(
            "evaluation_baseline.model_loader.AutoModelForCausalLM.from_pretrained",
            return_value=model,
        ) as load_model,
    ):
        loaded_model, _ = load_model_and_tokenizer(
            "Qwen2.5-0.5B", "cpu", "float32", model_path
        )

    assert loaded_model is model
    load_model.assert_called_once_with(str(model_path), torch_dtype=torch.float32)
    model.eval.assert_called_once_with()
