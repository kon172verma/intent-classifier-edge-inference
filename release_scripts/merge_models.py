#!/usr/bin/env python3
"""Compatibility CLI for manifest-driven source acquisition and adapter merging.

The benchmark pipeline owns the implementation. This script remains for users
who previously invoked a release helper directly, but it no longer selects an
adapter from mutable run keys or writes to the legacy flat model layout.

Usage:
    python release_scripts/merge_models.py \
      --manifest manifests/v1.0.json --models all
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmark_pipeline.artifacts import ArtifactError, fetch_sources, merge_models
from benchmark_pipeline.manifest import ManifestError, load_manifest, select_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch manifest-pinned sources and merge selected adapters locally."
    )
    parser.add_argument(
        "--manifest", type=Path, required=True, help="Path to a version manifest JSON file."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        metavar="MODEL",
        help="Exact manifest model name(s), or the sole value 'all'.",
    )
    return parser.parse_args()


def _read_hf_token() -> str | None:
    return (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )


def main() -> int:
    args = parse_args()
    env_file = _REPO_ROOT / ".env"
    if env_file.exists():
        from dotenv import load_dotenv

        load_dotenv(env_file)

    try:
        manifest = load_manifest(args.manifest)
        models = select_models(manifest, args.models)
        fetch_results = fetch_sources(
            repo_root=_REPO_ROOT,
            manifest=manifest,
            models=models,
            token=_read_hf_token(),
        )
        merge_results = merge_models(repo_root=_REPO_ROOT, manifest=manifest, models=models)
    except (ArtifactError, ManifestError) as exc:
        print(f"merge_models: error: {exc}", file=sys.stderr)
        return 2

    for result in fetch_results:
        print(
            f"[fetch] {result['model']}: base={'created' if result['base_created'] else 'reused'}, "
            f"adapter={'created' if result['adapter_created'] else 'reused'}"
        )
    for result in merge_results:
        print(f"[merge] {result['model']}: {'created' if result['created'] else 'reused'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
