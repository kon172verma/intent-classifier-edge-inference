#!/usr/bin/env python3
"""Publish version-scoped, deployable model artifacts to Hugging Face.

All selected manifest models are published to one Hub model repository.  Each
model is isolated in a version-and-model subfolder, with its Transformers,
GGUF, and ONNX artifacts kept in their native directories.

Example:
    python release_scripts/release.py --version v1.0 --models all --execute
"""

from __future__ import annotations

import argparse
import json
import os
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


def release_model_folder(manifest: Mapping[str, Any], model: Mapping[str, Any]) -> str:
    """Return the versioned subfolder assigned to one model in the Hub repo."""
    version = manifest.get("version")
    slug = model.get("slug")
    if not isinstance(version, str) or not isinstance(slug, str) or not slug:
        raise ReleaseError("Manifest model must define a non-empty slug for release publication")
    return f"{version}-{slug}"


def _copy_directory(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise ReleaseError(f"Required artifact directory does not exist: {source}")
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(_ARTIFACT_METADATA_NAME),
    )


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
    *,
    manifest: Mapping[str, Any],
    model: Mapping[str, Any],
    repo_id: str,
    model_folder: str,
    local_model_root: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "release_repository": repo_id,
        "release_subfolder": model_folder,
        "manifest": {
            "version": manifest["version"],
            "experiments": manifest["experiments"],
            "dataset": manifest["dataset"],
            "prompt": manifest["prompt"],
        },
        "model": dict(model),
        "local_artifact_metadata": _published_artifacts(local_model_root),
    }


def _model_readme(
    *, manifest: Mapping[str, Any], model: Mapping[str, Any], repo_id: str, model_folder: str
) -> str:
    """Return nested model documentation, not a separate Hub model card.

    Hugging Face only interprets the repository-root README front matter as
    model-card metadata.  Root documentation is deliberately outside this
    publishing workflow, so this per-model README is ordinary Markdown.
    """
    prompt = manifest["prompt"]
    adapter = model["adapter"]
    return f"""# Intent Classifier {manifest["version"]} — {model["name"]}

This is a merged, fine-tuned intent-classifier release. It is not the base
model named below.

## Model identity

| Field | Value |
| --- | --- |
| Release repository | `{repo_id}` |
| Release subfolder | `{model_folder}` |
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

- **Transformers:** `transformers/`. Load with
  `subfolder="{model_folder}/transformers"` from `{repo_id}`.
- **GGUF:** `gguf/<quant>/model.gguf` for llama.cpp.
- **ONNX:** `onnx/<variant>/` for ONNX Runtime. Keep all files in a variant
  directory together, including any external-data files referenced by
  `model.onnx`.

## Provenance

`benchmark_provenance.json` records the pinned manifest, source revisions, and
local artifact metadata used to produce this model folder.
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
        "model_folder": release_model_folder(manifest, model),
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
    """Assemble one nested model folder without changing local artifacts."""
    plan = release_plan(repo_root=repo_root, manifest=manifest, model=model, repo_id=repo_id)
    local_model_root = plan["local_model_root"]
    model_folder = str(plan["model_folder"])
    model_destination = destination / model_folder
    model_destination.mkdir(parents=True, exist_ok=True)

    published_types = list(plan["artifact_paths"])
    for artifact_type in ("transformers", "gguf", "onnx"):
        local_artifact_root = plan["artifact_paths"].get(artifact_type)
        if local_artifact_root is not None:
            _copy_directory(local_artifact_root, model_destination / artifact_type)

    provenance = _provenance(
        manifest=manifest,
        model=model,
        repo_id=repo_id,
        model_folder=model_folder,
        local_model_root=local_model_root,
    )
    (model_destination / "benchmark_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (model_destination / "README.md").write_text(
        _model_readme(manifest=manifest, model=model, repo_id=repo_id, model_folder=model_folder),
        encoding="utf-8",
    )
    return {
        "model": model["name"],
        "repo_id": repo_id,
        "model_folder": model_folder,
        "local_model_root": str(local_model_root),
        "artifact_types": published_types,
    }


def _upload_release(
    *,
    api: HfApi,
    manifest: Mapping[str, Any],
    model: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> str:
    """Upload local artifact folders directly, without duplicating large weights."""
    repo_id = str(plan["repo_id"])
    artifact_paths = plan["artifact_paths"]
    local_model_root = Path(plan["local_model_root"])
    model_folder = str(plan["model_folder"])
    ignore_patterns = [_ARTIFACT_METADATA_NAME, f"**/{_ARTIFACT_METADATA_NAME}"]

    latest_commit: Any = None
    for artifact_type, artifact_path in artifact_paths.items():
        latest_commit = api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(artifact_path),
            path_in_repo=f"{model_folder}/{artifact_type}",
            ignore_patterns=ignore_patterns,
            commit_message=f"Publish {manifest['version']} {model['name']} {artifact_type} artifacts",
        )

    provenance = _provenance(
        manifest=manifest,
        model=model,
        repo_id=repo_id,
        model_folder=model_folder,
        local_model_root=local_model_root,
    )
    latest_commit = api.upload_file(
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=f"{model_folder}/README.md",
        path_or_fileobj=_model_readme(
            manifest=manifest, model=model, repo_id=repo_id, model_folder=model_folder
        ).encode(),
        commit_message=f"Document {manifest['version']} {model['name']} release",
    )
    latest_commit = api.upload_file(
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=f"{model_folder}/benchmark_provenance.json",
        path_or_fileobj=(json.dumps(provenance, indent=2, sort_keys=True) + "\n").encode(),
        commit_message=f"Add {manifest['version']} {model['name']} provenance",
    )
    revision = getattr(latest_commit, "oid", None)
    if not isinstance(revision, str) or len(revision) != 40:
        raise ReleaseError("Hugging Face did not return an immutable commit SHA for the upload")
    return revision


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
        "--repo-id",
        default="kon172verma/intent-classifier",
        help="Single Hugging Face model repository to receive release folders.",
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

    token = _read_hf_token() if args.execute else None
    if args.execute and not token:
        print("release: error: HF_TOKEN is required for --execute", file=sys.stderr)
        return 2

    plans: list[dict[str, Any]] = []
    for model in models:
        try:
            plan = release_plan(
                repo_root=args.models_root.parent,
                manifest=manifest,
                model=model,
                repo_id=args.repo_id,
            )
        except ReleaseError as exc:
            print(f"release: error for {model['name']}: {exc}", file=sys.stderr)
            return 2

        plans.append(plan)
        print(
            f"[plan] {plan['model']} -> {plan['repo_id']}/{plan['model_folder']} "
            f"({', '.join(plan['artifact_paths'])})"
        )

    if not args.execute:
        return 0

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    release_revision: str | None = None
    for model, plan in zip(models, plans, strict=True):
        release_revision = _upload_release(
            api=api,
            manifest=manifest,
            model=model,
            plan=plan,
        )
        print(f"[uploaded] https://huggingface.co/{args.repo_id}/tree/main/{plan['model_folder']}")
    assert release_revision is not None
    print(f"[pinned release revision] {release_revision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
