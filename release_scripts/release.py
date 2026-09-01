#!/usr/bin/env python3
"""Publish version-scoped, deployable model artifacts to Hugging Face.

Each selected manifest model receives its own Hub model repository. The merged
Transformers checkpoint is published at the repository root while GGUF and
ONNX artifacts remain in their format-specific subdirectories.

Example:
    python release_scripts/release.py --version v1.0 --models all --execute
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmark_pipeline.artifacts import _ARTIFACT_METADATA_NAME, model_root
from benchmark_pipeline.manifest import ManifestError, load_manifest, select_models


class ReleaseError(RuntimeError):
    """Raised when local artifacts cannot be safely assembled for publication."""


def _read_hf_token() -> str | None:
    """Load an optional local .env and return a supported Hub token variable."""
    env_file = _REPO_ROOT / ".env"
    if env_file.is_file():
        from dotenv import load_dotenv

        load_dotenv(env_file)
    return (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )


def _model_size_label(model_name: str) -> str:
    """Return the parameter-size suffix used in a release repository name."""
    matches = re.findall(r"(\d+(?:\.\d+)?)([BM])", model_name, flags=re.IGNORECASE)
    if not matches:
        raise ReleaseError(f"Cannot derive a parameter-size label from model name: {model_name}")
    number, unit = matches[-1]
    return f"{number}{unit.lower()}"


def release_repository_name(version: str, model_name: str) -> str:
    """Return the simple, version-and-size scoped Hub repository name."""
    return f"intent-classifier-{version}-{_model_size_label(model_name)}"


def release_repository_id(owner: str, version: str, model_name: str) -> str:
    """Return the fully qualified Hub repository ID for one release model."""
    return f"{owner}/{release_repository_name(version, model_name)}"


def _copy_directory(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise ReleaseError(f"Required artifact directory does not exist: {source}")
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(_ARTIFACT_METADATA_NAME),
    )


def _copy_directory_contents(source: Path, destination: Path) -> None:
    """Copy a checkpoint directory into the release repository root."""
    if not source.is_dir():
        raise ReleaseError(f"Merged Transformers checkpoint does not exist: {source}")
    for child in source.iterdir():
        if child.name == _ARTIFACT_METADATA_NAME:
            continue
        target = destination / child.name
        if child.is_dir():
            _copy_directory(child, target)
        else:
            shutil.copy2(child, target)


def _read_metadata(directory: Path) -> dict[str, Any] | None:
    path = directory / _ARTIFACT_METADATA_NAME
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReleaseError(f"Invalid artifact metadata: {path}") from exc
    if not isinstance(value, dict):
        raise ReleaseError(f"Artifact metadata must be a JSON object: {path}")
    return value


def _published_artifacts(local_model_root: Path) -> dict[str, Any]:
    """Return local provenance for the exact artifact directories being published."""
    artifacts: dict[str, Any] = {
        "transformers": _read_metadata(local_model_root / "transformers" / "merged")
    }
    for artifact_type in ("gguf", "onnx"):
        artifact_root = local_model_root / artifact_type
        if artifact_root.is_dir():
            artifacts[artifact_type] = {
                variant.name: _read_metadata(variant)
                for variant in sorted(artifact_root.iterdir())
                if variant.is_dir()
            }
    return artifacts


def _provenance(
    *, manifest: Mapping[str, Any], model: Mapping[str, Any], repo_id: str, local_model_root: Path
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "release_repository": repo_id,
        "manifest": {
            "version": manifest["version"],
            "experiments": manifest["experiments"],
            "dataset": manifest["dataset"],
            "prompt": manifest["prompt"],
        },
        "model": dict(model),
        "local_artifact_metadata": _published_artifacts(local_model_root),
    }


def _model_card(*, manifest: Mapping[str, Any], model: Mapping[str, Any], repo_id: str) -> str:
    prompt = manifest["prompt"]
    adapter = model["adapter"]
    return f"""---
library_name: transformers
base_model: {model["base_model_id"]}
tags:
- intent-classification
- tool-routing
- edge
---

# Intent Classifier {manifest["version"]} {model["name"]}

This is a merged, fine-tuned intent-classifier release. It is not the base
model named below.

## Model identity

| Field | Value |
| --- | --- |
| Release repository | `{repo_id}` |
| Release | `{manifest["version"]}` |
| Base model | `{model["base_model_id"]}` |
| Base revision | `{model["base_model_revision"]}` |
| Fine-tuning method | `{adapter["technique"]}`, configuration `{adapter["configuration"]}` |
| Adapter source | `{manifest["experiments"]["repository"]}` at `{manifest["experiments"]["revision"]}` |
| Adapter subfolder | `{adapter["subfolder"]}` |
| Dataset size | `{manifest["dataset"]["size"]}` |
| Prompt template | `{prompt["template_id"]}` |
| Native output | `{prompt["output_format"]}`; no-tool token `{prompt["model_no_tool_token"]}` |

## Artifacts

- **Transformers:** files at the repository root.
- **GGUF:** `gguf/<quant>/model.gguf` for llama.cpp.
- **ONNX:** `onnx/<variant>/` for ONNX Runtime. Keep all files in a variant
  directory together, including any external-data files referenced by
  `model.onnx`.

## Provenance

`benchmark_provenance.json` records the pinned manifest, source revisions, and
local artifact metadata used to produce this release.
"""


def release_plan(
    *,
    repo_root: Path,
    manifest: Mapping[str, Any],
    model: Mapping[str, Any],
    repo_id: str,
) -> dict[str, Any]:
    """Validate and describe the local files that will be published."""
    local_model_root = model_root(repo_root, manifest, model)
    merged = local_model_root / "transformers" / "merged"
    if not merged.is_dir():
        raise ReleaseError(f"Merged Transformers checkpoint does not exist: {merged}")
    artifact_paths: dict[str, Path] = {"transformers": merged}
    for artifact_type in ("gguf", "onnx"):
        artifact_root = local_model_root / artifact_type
        if artifact_root.is_dir():
            artifact_paths[artifact_type] = artifact_root
    return {
        "model": model["name"],
        "repo_id": repo_id,
        "local_model_root": local_model_root,
        "artifact_paths": artifact_paths,
    }


def assemble_release_tree(
    *,
    repo_root: Path,
    manifest: Mapping[str, Any],
    model: Mapping[str, Any],
    repo_id: str,
    destination: Path,
) -> dict[str, Any]:
    """Assemble one model's publishable files without changing local artifacts."""
    plan = release_plan(repo_root=repo_root, manifest=manifest, model=model, repo_id=repo_id)
    local_model_root = plan["local_model_root"]
    merged = plan["artifact_paths"]["transformers"]
    _copy_directory_contents(merged, destination)

    published_types = list(plan["artifact_paths"])
    for artifact_type in ("gguf", "onnx"):
        local_artifact_root = plan["artifact_paths"].get(artifact_type)
        if local_artifact_root is not None:
            _copy_directory(local_artifact_root, destination / artifact_type)

    provenance = _provenance(
        manifest=manifest, model=model, repo_id=repo_id, local_model_root=local_model_root
    )
    (destination / "benchmark_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (destination / "README.md").write_text(
        _model_card(manifest=manifest, model=model, repo_id=repo_id), encoding="utf-8"
    )
    return {
        "model": model["name"],
        "repo_id": repo_id,
        "local_model_root": str(local_model_root),
        "artifact_types": published_types,
    }


def _upload_release(
    *,
    api: HfApi,
    manifest: Mapping[str, Any],
    model: Mapping[str, Any],
    plan: Mapping[str, Any],
    private: bool,
    replace: bool,
) -> None:
    """Upload local artifact folders directly, without duplicating large weights."""
    repo_id = str(plan["repo_id"])
    artifact_paths = plan["artifact_paths"]
    local_model_root = Path(plan["local_model_root"])
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    ignore_patterns = [_ARTIFACT_METADATA_NAME, f"**/{_ARTIFACT_METADATA_NAME}"]

    for index, (artifact_type, artifact_path) in enumerate(artifact_paths.items()):
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(artifact_path),
            path_in_repo="." if artifact_type == "transformers" else artifact_type,
            ignore_patterns=ignore_patterns,
            delete_patterns="*" if replace and index == 0 else None,
            commit_message=f"Publish {manifest['version']} {model['name']} {artifact_type} artifacts",
        )

    provenance = _provenance(
        manifest=manifest,
        model=model,
        repo_id=repo_id,
        local_model_root=local_model_root,
    )
    api.upload_file(
        repo_id=repo_id,
        repo_type="model",
        path_in_repo="README.md",
        path_or_fileobj=_model_card(manifest=manifest, model=model, repo_id=repo_id).encode(),
        commit_message=f"Document {manifest['version']} {model['name']} release",
    )
    api.upload_file(
        repo_id=repo_id,
        repo_type="model",
        path_in_repo="benchmark_provenance.json",
        path_or_fileobj=(json.dumps(provenance, indent=2, sort_keys=True) + "\n").encode(),
        commit_message=f"Add {manifest['version']} {model['name']} provenance",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version", required=True, help="Manifest/release version, for example v1.0."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        metavar="MODEL",
        help="Exact manifest model name(s), or the sole value 'all'.",
    )
    parser.add_argument(
        "--owner",
        default="kon172verma",
        help="Hugging Face owner or organisation namespace (default: kon172verma).",
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=_REPO_ROOT / "models",
        help="Root containing the standard models/<version>/<model-name>/ layout.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create new Hub repositories as private. Existing repository visibility is unchanged.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete remote files absent from the assembled release before uploading. Requires --execute.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Create repositories and upload artifacts. Without this flag, print a no-write plan.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest_path = _REPO_ROOT / "manifests" / f"{args.version}.json"
    try:
        manifest = load_manifest(manifest_path)
        if manifest["version"] != args.version:
            raise ReleaseError(
                f"Manifest version {manifest['version']!r} does not match --version {args.version!r}"
            )
        models = select_models(manifest, args.models)
    except (ManifestError, ReleaseError) as exc:
        print(f"release: error: {exc}", file=sys.stderr)
        return 2

    repo_names = [release_repository_name(args.version, model["name"]) for model in models]
    if len(repo_names) != len(set(repo_names)):
        print(
            "release: error: selected models do not have unique version-and-size repository names",
            file=sys.stderr,
        )
        return 2
    if args.replace and not args.execute:
        print("release: error: --replace requires --execute", file=sys.stderr)
        return 2

    token = _read_hf_token() if args.execute else None
    if args.execute and not token:
        print("release: error: HF_TOKEN is required for --execute", file=sys.stderr)
        return 2

    api = HfApi(token=token) if args.execute else None
    for model in models:
        repo_id = release_repository_id(args.owner, args.version, model["name"])
        try:
            plan = release_plan(
                repo_root=args.models_root.parent, manifest=manifest, model=model, repo_id=repo_id
            )
        except ReleaseError as exc:
            print(f"release: error for {model['name']}: {exc}", file=sys.stderr)
            return 2

        print(f"[plan] {plan['model']} -> {plan['repo_id']} ({', '.join(plan['artifact_paths'])})")
        if api is None:
            continue
        _upload_release(
            api=api,
            manifest=manifest,
            model=model,
            plan=plan,
            private=args.private,
            replace=args.replace,
        )
        print(f"[uploaded] https://huggingface.co/{repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
