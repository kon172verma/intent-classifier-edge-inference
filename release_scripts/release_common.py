"""Shared constants/helpers for release orchestration in intent-classifier-inference."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT: Path = REPO_ROOT.parent

HF_EXPERIMENTS_REPO: str = "kon172verma/intent-classifier-experiments"
HF_RELEASE_REPO: str = "kon172verma/intent-classifier"

FINETUNE_MODEL_REGISTRY: dict[str, str] = {
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "llama3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
}


def _read_current_version() -> str:
    env_version = os.getenv("IC_RELEASE_VERSION")
    if env_version:
        return env_version

    version_file = REPO_ROOT / "VERSION"
    if version_file.exists():
        content = version_file.read_text(encoding="utf-8").strip()
        if content:
            return content
    return "v1.0"


CURRENT_VERSION: str = _read_current_version()


def experiments_registry_path() -> Path:
    env_path = os.getenv("IC_EXPERIMENTS_JSONL")
    if env_path:
        return Path(env_path).expanduser().resolve()

    return REPO_ROOT / "EXPERIMENTS.jsonl"


REGISTRY_PATH: Path = experiments_registry_path()


def generate_experiment_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def hf_adapter_subfolder(
    technique: str,
    model_key: str,
    lora_config: str,
    dataset_size: str,
    timestamp: str | None = None,
    version: str | None = None,
) -> str:
    version = version or CURRENT_VERSION
    timestamp = timestamp or generate_experiment_timestamp()
    return f"{version}/{model_key}_{technique}_{lora_config}_{dataset_size}_{timestamp}"


def hf_merged_subfolder(model_key: str) -> str:
    return f"{model_key}/safetensors"


def load_registry() -> list[dict[str, Any]]:
    if not REGISTRY_PATH.exists():
        return []

    rows: list[dict[str, Any]] = []
    for line in REGISTRY_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def find_latest_experiment(
    *,
    technique: str,
    model_key: str,
    lora_config: str,
    dataset_size: str,
    version: str | None = None,
) -> dict[str, Any] | None:
    matches = [
        e
        for e in load_registry()
        if e.get("technique") == technique
        and e.get("model_key") == model_key
        and e.get("lora_config") == lora_config
        and e.get("dataset_size") == dataset_size
        and (version is None or e.get("version") == version)
    ]
    if not matches:
        return None
    return max(matches, key=lambda e: str(e.get("timestamp", "")))
