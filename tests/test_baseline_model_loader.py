"""Regression tests for Transformers baseline model loading."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from transformers import AutoConfig, AutoTokenizer

from evaluation_baseline.model_loader import load_model_and_tokenizer


def test_transformers_runtime_recognizes_qwen3() -> None:
    """The pinned runtime must load the Qwen3 model type in the v2.1 manifest."""
    assert AutoConfig.for_model("qwen3").model_type == "qwen3"


def test_qwen3_tokenizer_metadata_loads() -> None:
    """Newer Qwen3 metadata stores extra special tokens as a list."""
    model_path = Path("models/v2.1/Qwen3-0.6B/transformers/merged")
    if not model_path.is_dir():
        return

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)

    assert type(tokenizer.extra_special_tokens) is list


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


def test_loader_attaches_packaged_chat_template_when_missing(tmp_path: Path) -> None:
    """Support checkpoints that package their template outside tokenizer_config.json."""
    template = "{{ messages }}"
    (tmp_path / "chat_template.jinja").write_text(template, encoding="utf-8")
    tokenizer = MagicMock()
    tokenizer.chat_template = None
    model = MagicMock()

    with (
        patch(
            "evaluation_baseline.model_loader.AutoTokenizer.from_pretrained",
            return_value=tokenizer,
        ),
        patch(
            "evaluation_baseline.model_loader.AutoModelForCausalLM.from_pretrained",
            return_value=model,
        ),
    ):
        load_model_and_tokenizer("Qwen2.5-0.5B", "cpu", "float32", tmp_path)

    assert tokenizer.chat_template == template
