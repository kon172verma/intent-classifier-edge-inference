#!/usr/bin/env python3
"""
release_scripts/release.py
==========================
Publish a version release from locally prepared artifacts.

Responsibilities of this script:
- Upload local safetensors artifacts to HF_RELEASE_REPO.
- Upload local gguf/onnx artifacts when present.
- Create local git tags (and optional HF tags).

This script does NOT merge adapters. Run merge_models.py first.

Expected local layout per run:
    models/<model_key>_<technique>_<config>_<dataset_size>_merged/
        safetensors/
        gguf/        (optional)
        onnx/        (optional)

Usage
-----
    # Upload models and create local git tag
    python release_scripts/release.py --version v1.0 --runs \
        qwen3-0.6b_LoRA_C_1k llama3.2-1b_LoRA_C_1k

    # Only create a local git tag (skip upload)
    python release_scripts/release.py --tag-only --version v1.0 --message "LoRA v1.0 release"
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from release_scripts.release_common import CURRENT_VERSION, HF_RELEASE_REPO, REPO_ROOT


def _parse_run(run: str) -> tuple[str, str, str, str]:
    """Parse "{model_key}_{technique}_{config}_{dataset_size}"."""
    parts = run.rsplit("_", 3)
    if len(parts) != 4:
        raise ValueError(
            f"Cannot parse run '{run}'. Expected format: "
            "<model_key>_<technique>_<config>_<dataset_size>, "
            "e.g. 'qwen3-0.6b_LoRA_C_1k'."
        )
    model_key, technique, lora_config, dataset_size = parts
    return model_key, technique, lora_config, dataset_size


def _run_root(run: str, models_root: Path) -> Path:
    model_key, technique, lora_config, dataset_size = _parse_run(run)
    return models_root / f"{model_key}_{technique}_{lora_config}_{dataset_size}_merged"


def _upload_folder(
    *,
    api: HfApi,
    local_dir: Path,
    remote_subdir: str,
    commit_message: str,
) -> None:
    if not local_dir.is_dir():
        raise FileNotFoundError(f"Local folder not found: {local_dir}")

    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=HF_RELEASE_REPO,
        path_in_repo=remote_subdir,
        repo_type="model",
        commit_message=commit_message,
    )


def _upload_one_run(run: str, version: str, models_root: Path, hf_token: str | None) -> None:
    model_key, technique, lora_config, dataset_size = _parse_run(run)
    local_root = _run_root(run, models_root)

    api = HfApi(token=hf_token)
    api.create_repo(repo_id=HF_RELEASE_REPO, repo_type="model", exist_ok=True, private=True)

    safetensors_dir = local_root / "safetensors"
    print(f"\nUploading safetensors for {run}: {safetensors_dir}")
    _upload_folder(
        api=api,
        local_dir=safetensors_dir,
        remote_subdir=f"{model_key}/safetensors",
        commit_message=(
            f"Add merged model {model_key} ({technique}-{lora_config}-{dataset_size}) [{version}]"
        ),
    )

    for format_name in ["gguf", "onnx"]:
        format_dir = local_root / format_name
        if not format_dir.is_dir():
            print(f"Skipping {format_name} for {run} (not found: {format_dir}).")
            continue

        print(f"Uploading {format_name} for {run}: {format_dir}")
        _upload_folder(
            api=api,
            local_dir=format_dir,
            remote_subdir=f"{model_key}/{format_name}",
            commit_message=(
                f"Add {format_name.upper()} for {model_key} "
                f"({technique}-{lora_config}-{dataset_size}) [{version}]"
            ),
        )


def _create_hf_tag(version: str, message: str, hf_token: str | None) -> None:
    api = HfApi(token=hf_token)
    print(f"\nCreating HF tag '{version}' on {HF_RELEASE_REPO}...")
    api.create_tag(
        repo_id=HF_RELEASE_REPO,
        repo_type="model",
        tag=version,
        tag_message=message,
        token=hf_token,
        exist_ok=True,
    )
    print(f"HF tag '{version}' created.")


def _create_github_tag(version: str, message: str) -> None:
    print(f"\nCreating local git tag '{version}'...")
    result = subprocess.run(
        ["git", "tag", "-a", version, "-m", message],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(f"WARNING: git tag failed: {result.stderr.strip()}")
        return
    print(f"Local tag '{version}' created.")
    print(f"Review it, then push with: git push origin {version}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload local release artifacts and tag the repository."
    )
    parser.add_argument("--version", default=CURRENT_VERSION, help="Version tag, e.g. v1.0")
    parser.add_argument(
        "--runs",
        nargs="*",
        default=[],
        metavar="MODEL_TECHNIQUE_CFG_SIZE",
        help=(
            "Runs to upload, e.g. qwen3-0.6b_LoRA_C_1k llama3.2-1b_LoRA_C_1k. "
            "Each run maps to models/<run>_merged/{safetensors,gguf,onnx}."
        ),
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=REPO_ROOT / "models",
        help="Root folder containing <run>_merged model directories.",
    )
    parser.add_argument(
        "--tag-only",
        action="store_true",
        help="Skip uploads; only create the version tag.",
    )
    parser.add_argument(
        "--message",
        default=None,
        help="Human-readable tag annotation. Defaults to the version string.",
    )
    parser.add_argument(
        "--no-tag",
        action="store_true",
        help="Upload artifacts but do NOT create a tag.",
    )
    parser.add_argument(
        "--hf-tag",
        action="store_true",
        help="Also create a tag on HF_RELEASE_REPO (default: local git tag only).",
    )
    args = parser.parse_args()

    if not args.tag_only and not args.runs:
        parser.error("Specify --runs unless using --tag-only.")

    hf_token: str | None = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("WARNING: HF_TOKEN not set. Pushes to private repos may fail.")

    if not args.tag_only:
        for run in args.runs:
            try:
                _upload_one_run(run, args.version, args.models_root, hf_token)
            except Exception as exc:
                print(f"ERROR uploading {run}: {exc}")
                print("Skipping this run; continuing with others.")

    if not args.no_tag:
        tag_message = args.message or args.version
        _create_github_tag(args.version, tag_message)
        if args.hf_tag:
            _create_hf_tag(args.version, tag_message, hf_token)
    else:
        print("\n--no-tag set; skipping tag creation.")

    print("\nDone.")


if __name__ == "__main__":
    main()
