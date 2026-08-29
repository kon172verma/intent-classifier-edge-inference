#!/usr/bin/env python3
"""Stage a checkpoint for a legacy Optimum ONNX export without altering it.

Optimum currently exports through Transformers 4.57.x in this project.  That
version expects Llama 3.1/3.2 RoPE settings in the legacy ``rope_scaling``
field, while the release checkpoint produced by Transformers 5 stores them in
``rope_parameters``.  Without the conversion, the exporter silently falls
back to the default RoPE theta (10,000) instead of Llama 3.2's 500,000 and
produces a numerically different model.

This utility creates a small staging directory: all checkpoint assets are
symlinked from the source directory and only ``config.json`` is written anew.
The original checkpoint therefore remains untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).parent.parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation_lib.config import MODEL_PATHS, MODEL_RUNS


def legacy_rope_compatible_config(config: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Return a config that Transformers 4.57 can use for Llama 3 RoPE.

    Transformers 4.57 reads Llama RoPE from the root-level ``rope_theta`` and
    ``rope_scaling`` fields.  Transformers 5 serializes the same information
    in ``rope_parameters`` instead.  Convert only Llama's scaled ``llama3``
    form; all other model configs are preserved byte-for-byte in meaning.
    """
    staged = deepcopy(config)
    rope_parameters = staged.get("rope_parameters")
    if not (
        staged.get("model_type") == "llama"
        and isinstance(rope_parameters, dict)
        and rope_parameters.get("rope_type") == "llama3"
    ):
        return staged, False

    legacy_scaling = deepcopy(rope_parameters)
    rope_theta = legacy_scaling.pop("rope_theta", staged.get("rope_theta", 10_000.0))
    staged["rope_theta"] = rope_theta
    staged["rope_scaling"] = legacy_scaling
    return staged, True


def legacy_tokenizer_compatible_config(config: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Return a tokenizer config readable by Transformers 4.57.

    Transformers 5 writes ``TokenizersBackend`` for a generic fast tokenizer.
    That class does not exist in Transformers 4.57, although its underlying
    ``tokenizer.json`` is fully compatible with ``PreTrainedTokenizerFast``.
    """
    staged = deepcopy(config)
    if staged.get("tokenizer_class") != "TokenizersBackend":
        return staged, False
    staged["tokenizer_class"] = "PreTrainedTokenizerFast"
    return staged, True


def stage_export_source(source_dir: Path, output_dir: Path) -> tuple[bool, bool]:
    """Create a symlinked export source and return whether RoPE was adapted."""
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {source_dir}")
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing staging directory: {output_dir}. "
            "Choose a new --output-dir or remove that generated directory first."
        )

    config_path = source_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Checkpoint has no config.json: {source_dir}")
    with config_path.open(encoding="utf-8") as f:
        config: dict[str, Any] = json.load(f)
    staged_config, rope_adapted = legacy_rope_compatible_config(config)
    tokenizer_config_path = source_dir / "tokenizer_config.json"
    staged_tokenizer_config: dict[str, Any] | None = None
    tokenizer_adapted = False
    if tokenizer_config_path.is_file():
        with tokenizer_config_path.open(encoding="utf-8") as f:
            tokenizer_config: dict[str, Any] = json.load(f)
        staged_tokenizer_config, tokenizer_adapted = legacy_tokenizer_compatible_config(
            tokenizer_config
        )

    output_dir.mkdir(parents=True)
    for source_item in source_dir.iterdir():
        if source_item.name == "config.json" or (
            tokenizer_adapted and source_item.name == "tokenizer_config.json"
        ):
            continue
        (output_dir / source_item.name).symlink_to(
            source_item, target_is_directory=source_item.is_dir()
        )
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(staged_config, f, indent=2, ensure_ascii=False)
        f.write("\n")
    if tokenizer_adapted and staged_tokenizer_config is not None:
        with (output_dir / "tokenizer_config.json").open("w", encoding="utf-8") as f:
            json.dump(staged_tokenizer_config, f, indent=2, ensure_ascii=False)
            f.write("\n")
    return rope_adapted, tokenizer_adapted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage a checkpoint with legacy-compatible Llama 3 RoPE for ONNX export."
    )
    parser.add_argument("--model", choices=list(MODEL_PATHS), required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Staging directory (default: models/_onnx_export_sources/<model>_legacy_rope)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (
        _REPO_ROOT / "models" / "_onnx_export_sources" / f"{MODEL_RUNS[args.model]}_legacy_rope"
    )
    rope_adapted, tokenizer_adapted = stage_export_source(MODEL_PATHS[args.model], output_dir)
    changes = []
    if rope_adapted:
        changes.append("adapted Llama 3 RoPE")
    if tokenizer_adapted:
        changes.append("adapted TokenizersBackend metadata")
    status = ", ".join(changes) if changes else "no compatibility adaptation required"
    print(f"[onnx-export] Created {output_dir} ({status}).")


if __name__ == "__main__":
    main()
