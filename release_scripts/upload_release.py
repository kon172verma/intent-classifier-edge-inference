#!/usr/bin/env python3
"""Upload model folders to the Hugging Face release repo using HF Hub API.

No git required. Large files are uploaded directly; HF deduplicates on sha256 so
re-uploading the same binary a second time costs zero bandwidth.

Usage
-----
# Upload qwen3-0.6b first, then optionally delete local copy to free space:
python release_scripts/upload_release.py --model qwen3-0.6b
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from release_scripts.release_common import HF_RELEASE_REPO, WORKSPACE_ROOT

# Local clone of the HF release repo, next to this repository.
RELEASE_DIR = WORKSPACE_ROOT / "intent-classifier-release"


def _human_size(path: Path) -> str:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return f"{total / 1e9:.1f} GB"


def upload_model(model_key: str, api: HfApi) -> None:
    local_path = RELEASE_DIR / model_key
    if not local_path.is_dir():
        raise FileNotFoundError(f"Local folder not found: {local_path}")

    size = _human_size(local_path)
    print(f"Uploading {model_key}/ ({size}) -> {HF_RELEASE_REPO}/{model_key}/")
    print("  This may take a while for large files ...")

    api.upload_folder(
        folder_path=str(local_path),
        repo_id=HF_RELEASE_REPO,
        path_in_repo=model_key,
        repo_type="model",
        commit_message=f"Add {model_key}: safetensors, gguf, onnx [v1.0]",
    )
    print(f"[ok] {model_key}/ uploaded successfully.")
    print(f"     You can now delete the local copy to reclaim {size}:")
    print(f"     rm -rf {local_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload release models to Hugging Face.")
    parser.add_argument(
        "--model",
        choices=["qwen3-0.6b", "llama3.2-1b"],
        help="Which model folder to upload.",
    )
    args = parser.parse_args()

    if not args.model:
        parser.error("Specify --model.")

    api = HfApi()

    upload_model(args.model, api)


if __name__ == "__main__":
    main()
