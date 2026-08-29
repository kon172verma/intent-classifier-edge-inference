#!/usr/bin/env python3
"""
release_scripts/merge_models.py
===============================
Download base models and adapters from HF_EXPERIMENTS_REPO, then export
merged (adapter-unloaded) checkpoints locally.

This is the "merge + unload" step of the release pipeline, now owned by
intent-classifier-inference.

Output feeds GGUF and ONNX export tooling in this repo and release.py's
upload step.

This script expects a Hugging Face token in environment variables as one of:
- HF_TOKEN
- HUGGINGFACE_TOKEN
- HUGGINGFACEHUB_API_TOKEN

Usage
-----
    # Default: merge the two v1.0 release runs (LoRA, config C, 1k)
    python release_scripts/merge_models.py

    # Explicit runs, optionally pinned to an exact experiment timestamp
    python release_scripts/merge_models.py --technique LoRA \
        --runs qwen3-0.6b_C_1k llama3.2-1b_C_1k:20260101-120000

Output layout
-------------
    models/<model_key>_<technique>_<config>_<dataset_size>_merged/safetensors/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, cast

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from release_scripts.release_common import (
    CURRENT_VERSION,
    FINETUNE_MODEL_REGISTRY,
    HF_EXPERIMENTS_REPO,
    REPO_ROOT,
    find_latest_experiment,
    hf_adapter_subfolder,
)

DEFAULT_OUTPUT_DIR = REPO_ROOT / "models"

# The two best v1.0 models (both LoRA, config C, 1k dataset — see RELEASES.md).
DEFAULT_RUNS = ["qwen3-0.6b_C_1k", "llama3.2-1b_C_1k"]
DEFAULT_TECHNIQUE = "LoRA"


def _read_hf_token() -> str:
    token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )
    if not token:
        raise RuntimeError("Missing Hugging Face token. Set HF_TOKEN (or HUGGINGFACE_TOKEN).")
    return token


def _parse_run(run: str) -> tuple[str, str, str, str | None]:
    """Parse "{model_key}_{lora_config}_{dataset_size}[:{timestamp}]"."""
    run_spec, _, timestamp = run.partition(":")
    parts = run_spec.rsplit("_", 2)
    if len(parts) != 3:
        raise ValueError(
            f"Cannot parse run '{run}'. Expected "
            "<model_key>_<config>_<size>[:<timestamp>], e.g. 'qwen3-0.6b_C_1k'."
        )
    model_key, lora_config, dataset_size = parts
    return model_key, lora_config, dataset_size, (timestamp or None)


def _resolve_adapter_subfolder(
    technique: str,
    model_key: str,
    lora_config: str,
    dataset_size: str,
    timestamp: str | None,
    version: str,
) -> str:
    if timestamp is None:
        entry = find_latest_experiment(
            technique=technique,
            model_key=model_key,
            lora_config=lora_config,
            dataset_size=dataset_size,
            version=version,
        )
        if entry is None:
            raise ValueError(
                "No experiment logged in EXPERIMENTS.jsonl for "
                f"{technique}/{model_key}_{lora_config}_{dataset_size} "
                f"(version={version}). Pass 'model_config_size:timestamp' explicitly."
            )
        timestamp = str(entry["timestamp"])
    return hf_adapter_subfolder(
        technique,
        model_key,
        lora_config,
        dataset_size,
        timestamp=timestamp,
        version=version,
    )


def merge_run(
    technique: str,
    run: str,
    version: str,
    output_root: Path,
    token: str,
) -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_key, lora_config, dataset_size, timestamp = _parse_run(run)
    if model_key not in FINETUNE_MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model_key '{model_key}'. Valid keys: {list(FINETUNE_MODEL_REGISTRY)}"
        )
    base_model_id = FINETUNE_MODEL_REGISTRY[model_key]
    adapter_subfolder = _resolve_adapter_subfolder(
        technique, model_key, lora_config, dataset_size, timestamp, version
    )

    run_tag = f"{model_key}_{technique}_{lora_config}_{dataset_size}"
    model_root = output_root / f"{run_tag}_merged"
    output_dir = model_root / "safetensors"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[+] Loading tokenizer: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=token, trust_remote_code=True)

    print(f"[+] Loading base model: {base_model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id, token=token, trust_remote_code=True, torch_dtype="auto"
    )

    print(f"[+] Loading adapter: {HF_EXPERIMENTS_REPO} ({adapter_subfolder})")
    peft_model = PeftModel.from_pretrained(
        base_model, HF_EXPERIMENTS_REPO, subfolder=adapter_subfolder, token=token
    )

    print("[+] Merging adapter with base model via merge_and_unload()")
    merged_model = cast(Any, peft_model).merge_and_unload()

    print(f"[+] Saving merged model to: {output_dir}")
    cast(Any, merged_model).save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    print(f"[ok] Completed: {model_root.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge and unload adapters from HF_EXPERIMENTS_REPO into local checkpoints."
    )
    parser.add_argument("--technique", default=DEFAULT_TECHNIQUE)
    parser.add_argument(
        "--runs",
        nargs="*",
        default=DEFAULT_RUNS,
        metavar="MODEL_CFG_SIZE[:TIMESTAMP]",
    )
    parser.add_argument("--version", default=CURRENT_VERSION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    # Load .env from inference repo root if present.
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        from dotenv import load_dotenv

        load_dotenv(env_file)
    token = _read_hf_token()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Output directory: {args.output_dir}")

    for run in args.runs:
        merge_run(args.technique, run, args.version, args.output_dir, token)

    print(f"\n[done] All requested merged models are available under {args.output_dir}")


if __name__ == "__main__":
    main()
