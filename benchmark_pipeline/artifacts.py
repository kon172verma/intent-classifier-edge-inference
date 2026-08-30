"""Manifest-driven acquisition and merge helpers for benchmark artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, cast


class ArtifactError(RuntimeError):
    """Raised when an artifact cannot be safely materialized or reused."""


SnapshotDownload = Callable[..., str]
_SOURCE_METADATA_NAME = ".benchmark_source.json"
_ARTIFACT_METADATA_NAME = ".benchmark_artifact.json"


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_inventory(root: Path) -> list[dict[str, Any]]:
    """Return deterministic checksums for the artifact files below ``root``."""
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in {_SOURCE_METADATA_NAME, _ARTIFACT_METADATA_NAME}:
            entries.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": _file_sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    return entries


def _read_metadata(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"Invalid artifact metadata: {path}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"Artifact metadata must be a JSON object: {path}")
    return value


def _write_metadata(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _require_reusable(target: Path, metadata_name: str, expected: Mapping[str, Any]) -> bool:
    """Return whether an existing target exactly matches its expected provenance."""
    if not target.exists():
        return False
    metadata = _read_metadata(target / metadata_name)
    if metadata is None:
        raise ArtifactError(
            f"Refusing to reuse non-empty artifact directory without metadata: {target}. "
            "Move it aside or use the agreed manifest-driven layout."
        )
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ArtifactError(
                f"Existing artifact provenance does not match the requested {key}: {target}"
            )
    return True


def _default_snapshot_download(**kwargs: Any) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ArtifactError(
            "Fetching model snapshots requires huggingface_hub. Install the project dependencies."
        ) from exc
    return str(snapshot_download(**kwargs))


def _validate_subfolder(subfolder: str) -> PurePosixPath:
    path = PurePosixPath(subfolder)
    if path.is_absolute() or ".." in path.parts or path == PurePosixPath("."):
        raise ArtifactError(f"Unsafe adapter subfolder: {subfolder!r}")
    return path


def materialize_source_snapshot(
    *,
    repository: str,
    revision: str,
    destination: Path,
    source_subfolder: str | None = None,
    token: str | None = None,
    snapshot_download_fn: SnapshotDownload | None = None,
) -> bool:
    """Create an immutable source snapshot, returning ``True`` when newly created.

    Hugging Face downloads into a temporary cache first. Only a completed, hashed
    snapshot is moved into the model layout, so interrupted downloads never look
    like reusable sources. ``source_subfolder`` strips the experiments-repository
    prefix while retaining just the selected adapter's files.
    """
    expected = {
        "kind": "source_snapshot",
        "repository": repository,
        "revision": revision,
        "source_subfolder": source_subfolder,
    }
    if _require_reusable(destination, _SOURCE_METADATA_NAME, expected):
        return False

    downloader = snapshot_download_fn or _default_snapshot_download
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="benchmark-source-", dir=destination.parent
    ) as temp_dir:
        staging_root = Path(temp_dir)
        download_kwargs: dict[str, Any] = {
            "repo_id": repository,
            "revision": revision,
            "cache_dir": staging_root / "hf-cache",
            "token": token,
        }
        if source_subfolder is not None:
            download_kwargs["allow_patterns"] = f"{source_subfolder}/**"
        try:
            snapshot_path = Path(downloader(**download_kwargs))
        except ArtifactError:
            raise
        except Exception as exc:
            raise ArtifactError(
                f"Unable to download {repository}@{revision}. "
                "Check network access, the pinned revision, and HF_TOKEN for gated models."
            ) from exc
        content_root = snapshot_path
        if source_subfolder is not None:
            content_root = snapshot_path / _validate_subfolder(source_subfolder)
        if not content_root.is_dir():
            raise ArtifactError(
                f"Snapshot {repository}@{revision} does not contain {source_subfolder!r}"
            )

        materialized = staging_root / "materialized"
        shutil.copytree(content_root, materialized)
        metadata = {
            **expected,
            "created_at": _utc_now(),
            "files": _file_inventory(materialized),
        }
        _write_metadata(materialized / _SOURCE_METADATA_NAME, metadata)
        materialized.replace(destination)
    return True


def model_root(repo_root: Path, manifest: Mapping[str, Any], model: Mapping[str, Any]) -> Path:
    """Return the agreed model-root directory for a manifest model."""
    return repo_root / "models" / str(manifest["version"]) / str(model["name"])


def fetch_model_sources(
    *,
    repo_root: Path,
    manifest: Mapping[str, Any],
    model: Mapping[str, Any],
    token: str | None = None,
    snapshot_download_fn: SnapshotDownload | None = None,
) -> dict[str, Any]:
    """Fetch the exact base and adapter snapshots required by one manifest model."""
    root = model_root(repo_root, manifest, model)
    experiments = manifest["experiments"]
    adapter = model["adapter"]
    base_created = materialize_source_snapshot(
        repository=str(model["base_model_id"]),
        revision=str(model["base_model_revision"]),
        destination=root / "source" / "base",
        token=token,
        snapshot_download_fn=snapshot_download_fn,
    )
    adapter_created = materialize_source_snapshot(
        repository=str(experiments["repository"]),
        revision=str(experiments["revision"]),
        destination=root / "source" / "adapter",
        source_subfolder=str(adapter["subfolder"]),
        token=token,
        snapshot_download_fn=snapshot_download_fn,
    )
    return {
        "model": model["name"],
        "base": str(root / "source" / "base"),
        "adapter": str(root / "source" / "adapter"),
        "base_created": base_created,
        "adapter_created": adapter_created,
    }


def fetch_sources(
    *,
    repo_root: Path,
    manifest: Mapping[str, Any],
    models: Iterable[Mapping[str, Any]],
    token: str | None = None,
    snapshot_download_fn: SnapshotDownload | None = None,
) -> list[dict[str, Any]]:
    """Fetch source snapshots for the selected models in manifest order."""
    return [
        fetch_model_sources(
            repo_root=repo_root,
            manifest=manifest,
            model=model,
            token=token,
            snapshot_download_fn=snapshot_download_fn,
        )
        for model in models
    ]


def _source_provenance(path: Path) -> dict[str, Any]:
    metadata = _read_metadata(path / _SOURCE_METADATA_NAME)
    if metadata is None:
        raise ArtifactError(f"Missing source snapshot metadata: {path}")
    return metadata


def _builder_versions() -> dict[str, str]:
    try:
        import peft
        import torch
        import transformers
    except ImportError as exc:
        raise ArtifactError("Merging requires peft, torch, and transformers.") from exc
    return {
        "peft": str(peft.__version__),
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "transformers": str(transformers.__version__),
    }


def merge_model(*, repo_root: Path, manifest: Mapping[str, Any], model: Mapping[str, Any]) -> bool:
    """Merge one locally snapshotted adapter and return whether a checkpoint was created."""
    root = model_root(repo_root, manifest, model)
    base_dir = root / "source" / "base"
    adapter_dir = root / "source" / "adapter"
    merged_dir = root / "transformers" / "merged"
    inputs = {"base": _source_provenance(base_dir), "adapter": _source_provenance(adapter_dir)}
    expected = {"kind": "merged_transformers", "inputs": inputs}
    if _require_reusable(merged_dir, _ARTIFACT_METADATA_NAME, expected):
        return False

    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ArtifactError("Merging requires peft and transformers.") from exc

    merged_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="benchmark-merged-", dir=merged_dir.parent) as temp_dir:
        staging_dir = Path(temp_dir) / "merged"
        tokenizer = AutoTokenizer.from_pretrained(
            base_dir, local_files_only=True, trust_remote_code=True
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_dir,
            local_files_only=True,
            torch_dtype="auto",
            trust_remote_code=True,
        )
        peft_model = PeftModel.from_pretrained(base_model, adapter_dir, local_files_only=True)
        # PEFT exposes merge_and_unload() on concrete LoRA/DoRA model wrappers,
        # but it is absent from the generic PeftModel type annotation.
        merged_model = cast(Any, peft_model).merge_and_unload()
        merged_model.save_pretrained(staging_dir, safe_serialization=True)
        tokenizer.save_pretrained(staging_dir)
        metadata = {
            **expected,
            "builder": _builder_versions(),
            "created_at": _utc_now(),
            "files": _file_inventory(staging_dir),
        }
        _write_metadata(staging_dir / _ARTIFACT_METADATA_NAME, metadata)
        staging_dir.replace(merged_dir)
    return True


def merge_models(
    *, repo_root: Path, manifest: Mapping[str, Any], models: Iterable[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Merge all selected models, requiring snapshots created by :func:`fetch_sources`."""
    return [
        {
            "model": model["name"],
            "created": merge_model(repo_root=repo_root, manifest=manifest, model=model),
        }
        for model in models
    ]
