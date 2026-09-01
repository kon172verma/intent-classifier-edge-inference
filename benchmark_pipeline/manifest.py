"""Load, validate, and resolve version-scoped benchmark manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ManifestError(ValueError):
    """Raised when a benchmark manifest is invalid or cannot be resolved."""


_STAGES = ("fetch", "merge", "build-artifacts", "download-release", "evaluate", "plot")
_DEFAULT_STAGES = ("fetch", "merge", "build-artifacts", "evaluate", "plot")
_OUTPUT_FORMATS = frozenset({"tool_name", "positional_id"})
_ARTIFACT_TYPES = frozenset({"transformers", "gguf", "onnx", "tensorrt"})


def _require_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"{field} must be an object")
    return value


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{field} must be a non-empty string")
    return value


def _require_string_list(value: Any, field: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise ManifestError(f"{field} must be a non-empty list of strings")
    return value


def load_manifest(path: Path) -> dict[str, Any]:
    """Read and validate a JSON benchmark manifest."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestError(f"Manifest does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"Manifest is not valid JSON: {path}: {exc}") from exc

    manifest = _require_mapping(raw, "manifest")
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Validate the project schema without depending on third-party packages."""
    if manifest.get("schema_version") != 1:
        raise ManifestError("schema_version must be 1")
    version = _require_string(manifest.get("version"), "version")

    experiments = _require_mapping(manifest.get("experiments"), "experiments")
    _require_string(experiments.get("repository"), "experiments.repository")
    revision = _require_string(experiments.get("revision"), "experiments.revision")
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise ManifestError("experiments.revision must be a 40-character lowercase Git SHA")

    release: dict[str, Any] | None = None
    if "release" in manifest:
        release = _require_mapping(manifest.get("release"), "release")
        _require_string(release.get("repository"), "release.repository")
        release_revision = _require_string(release.get("revision"), "release.revision")
        if len(release_revision) != 40 or any(
            char not in "0123456789abcdef" for char in release_revision
        ):
            raise ManifestError("release.revision must be a 40-character lowercase Git SHA")

    dataset = _require_mapping(manifest.get("dataset"), "dataset")
    if dataset.get("size") not in {"1k", "10k"}:
        raise ManifestError("dataset.size must be '1k' or '10k'")
    _require_string(dataset.get("split_root"), "dataset.split_root")
    splits = _require_mapping(dataset.get("splits"), "dataset.splits")
    for split_name in ("test_anchor", "calibration", "final_selection_test"):
        _require_string(splits.get(split_name), f"dataset.splits.{split_name}")
    if splits["test_anchor"] != "test_anchor.json":
        raise ManifestError("dataset.splits.test_anchor must be test_anchor.json")
    if splits["calibration"] != "calibration.json":
        raise ManifestError("dataset.splits.calibration must be calibration.json")
    if splits["final_selection_test"] != "test.json":
        raise ManifestError("dataset.splits.final_selection_test must be test.json")

    prompt = _require_mapping(manifest.get("prompt"), "prompt")
    _require_string(prompt.get("template_id"), "prompt.template_id")
    _require_string(prompt.get("system_prompt"), "prompt.system_prompt")
    output_format = _require_string(prompt.get("output_format"), "prompt.output_format")
    if output_format not in _OUTPUT_FORMATS:
        raise ManifestError(f"prompt.output_format must be one of {sorted(_OUTPUT_FORMATS)}")
    _require_string(prompt.get("model_no_tool_token"), "prompt.model_no_tool_token")
    _require_string(prompt.get("canonical_no_tool_value"), "prompt.canonical_no_tool_value")
    if output_format == "positional_id":
        positional_ids = _require_mapping(prompt.get("positional_ids"), "prompt.positional_ids")
        if positional_ids.get("ordering") != "a-z_then_A-Z":
            raise ManifestError("prompt.positional_ids.ordering must be a-z_then_A-Z")

    models = manifest.get("models")
    if not isinstance(models, list) or not models:
        raise ManifestError("models must be a non-empty list")
    names: set[str] = set()
    for index, model_value in enumerate(models):
        model = _require_mapping(model_value, f"models[{index}]")
        name = _require_string(model.get("name"), f"models[{index}].name")
        if name in names:
            raise ManifestError(f"models contains duplicate name: {name}")
        names.add(name)
        _require_string(model.get("slug"), f"models[{index}].slug")
        if release is not None:
            release_subfolder = _require_string(
                model.get("release_subfolder"), f"models[{index}].release_subfolder"
            )
            if release_subfolder.startswith("/") or ".." in Path(release_subfolder).parts:
                raise ManifestError(
                    f"models[{index}].release_subfolder must be a safe relative path"
                )
        _require_string(model.get("base_model_id"), f"models[{index}].base_model_id")
        base_model_revision = _require_string(
            model.get("base_model_revision"), f"models[{index}].base_model_revision"
        )
        if len(base_model_revision) != 40 or any(
            char not in "0123456789abcdef" for char in base_model_revision
        ):
            raise ManifestError(
                f"models[{index}].base_model_revision must be a 40-character lowercase Git SHA"
            )
        adapter = _require_mapping(model.get("adapter"), f"models[{index}].adapter")
        subfolder = _require_string(adapter.get("subfolder"), f"models[{index}].adapter.subfolder")
        if not subfolder.startswith(f"{version}/"):
            raise ManifestError(f"models[{index}].adapter.subfolder must begin with '{version}/'")
        _require_string(adapter.get("technique"), f"models[{index}].adapter.technique")
        _require_string(adapter.get("configuration"), f"models[{index}].adapter.configuration")

    profiles = manifest.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ManifestError("profiles must be a non-empty list")
    profile_ids: set[str] = set()
    target_compute: set[tuple[str, str]] = set()
    for index, profile_value in enumerate(profiles):
        profile = _require_mapping(profile_value, f"profiles[{index}]")
        profile_id = _require_string(profile.get("id"), f"profiles[{index}].id")
        if profile_id in profile_ids:
            raise ManifestError(f"profiles contains duplicate id: {profile_id}")
        profile_ids.add(profile_id)
        target = _require_string(profile.get("target"), f"profiles[{index}].target")
        compute = _require_string(profile.get("compute"), f"profiles[{index}].compute")
        if (target, compute) in target_compute:
            raise ManifestError(f"profiles contains duplicate target/compute: {target}/{compute}")
        target_compute.add((target, compute))
        engines = profile.get("engines")
        if not isinstance(engines, list) or not engines:
            raise ManifestError(f"profiles[{index}].engines must be a non-empty list")
        engine_names: set[str] = set()
        for engine_index, engine_value in enumerate(engines):
            engine = _require_mapping(engine_value, f"profiles[{index}].engines[{engine_index}]")
            engine_name = _require_string(engine.get("name"), "engine.name")
            if engine_name in engine_names:
                raise ManifestError(
                    f"profile {profile_id} contains duplicate engine: {engine_name}"
                )
            engine_names.add(engine_name)
            _require_string(engine.get("runtime_device"), "engine.runtime_device")
            artifact = _require_mapping(engine.get("artifact"), "engine.artifact")
            artifact_type = _require_string(artifact.get("type"), "engine.artifact.type")
            if artifact_type not in _ARTIFACT_TYPES:
                raise ManifestError(
                    f"engine.artifact.type must be one of {sorted(_ARTIFACT_TYPES)}"
                )
            _require_string_list(artifact.get("variants"), "engine.artifact.variants")
            _require_string_list(engine.get("evaluate_variants"), "engine.evaluate_variants")
            cache_modes = _require_string_list(engine.get("cache_modes"), "engine.cache_modes")
            if set(cache_modes) - {"prefix_cache"}:
                raise ManifestError("engine.cache_modes may only contain prefix_cache")


def resolve_profile(manifest: dict[str, Any], target: str, compute: str) -> dict[str, Any]:
    """Return the single profile matching a target and compute selection."""
    matches = [
        profile
        for profile in manifest["profiles"]
        if profile["target"] == target and profile["compute"] == compute
    ]
    if not matches:
        available = ", ".join(
            f"{profile['target']}/{profile['compute']}" for profile in manifest["profiles"]
        )
        raise ManifestError(
            f"No profile for target={target!r}, compute={compute!r}. Available: {available}"
        )
    return matches[0]


def select_models(manifest: dict[str, Any], requested_models: list[str]) -> list[dict[str, Any]]:
    """Select exact display names, or all manifest models in manifest order."""
    if not requested_models:
        raise ManifestError("At least one --models value is required")
    if "all" in requested_models:
        if requested_models != ["all"]:
            raise ManifestError("--models all cannot be combined with explicit model names")
        return list(manifest["models"])

    by_name = {model["name"]: model for model in manifest["models"]}
    unknown = [name for name in requested_models if name not in by_name]
    if unknown:
        available = ", ".join(by_name)
        raise ManifestError(f"Unknown model(s): {', '.join(unknown)}. Available: {available}")
    return [by_name[name] for name in requested_models]


def select_engines(
    profile: dict[str, Any], requested_engines: list[str] | None
) -> list[dict[str, Any]]:
    """Return selected engines, preserving the profile's declared ordering."""
    if not requested_engines or requested_engines == ["all"]:
        return list(profile["engines"])
    if "all" in requested_engines:
        raise ManifestError("--engines all cannot be combined with explicit engine names")
    by_name = {engine["name"]: engine for engine in profile["engines"]}
    unknown = [name for name in requested_engines if name not in by_name]
    if unknown:
        available = ", ".join(by_name)
        raise ManifestError(f"Unknown engine(s): {', '.join(unknown)}. Available: {available}")
    return [by_name[name] for name in requested_engines]


def expand_stages(requested_stages: list[str]) -> list[str]:
    """Expand 'all' and validate the ordered set of requested pipeline stages."""
    if not requested_stages or requested_stages == ["all"]:
        return list(_DEFAULT_STAGES)
    if "all" in requested_stages:
        raise ManifestError("--stages all cannot be combined with named stages")
    unknown = [stage for stage in requested_stages if stage not in _STAGES]
    if unknown:
        raise ManifestError(f"Unknown stage(s): {', '.join(unknown)}")
    stages = [stage for stage in _STAGES if stage in requested_stages]
    if "download-release" in stages and set(stages) & {"fetch", "merge", "build-artifacts"}:
        raise ManifestError(
            "download-release is an alternative to fetch, merge, and build-artifacts; "
            "do not select them together"
        )
    return stages


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def resolve_plan(
    manifest: dict[str, Any],
    *,
    target: str,
    compute: str,
    requested_models: list[str],
    requested_engines: list[str] | None,
    requested_stages: list[str],
    repo_root: Path,
) -> dict[str, Any]:
    """Resolve a no-side-effect plan for one profile and model selection."""
    profile = resolve_profile(manifest, target, compute)
    models = select_models(manifest, requested_models)
    engines = select_engines(profile, requested_engines)
    stages = expand_stages(requested_stages)
    if "download-release" in stages and not isinstance(manifest.get("release"), dict):
        raise ManifestError(
            "download-release requires a pinned release.repository and release.revision in the manifest"
        )
    split_root = repo_root / manifest["dataset"]["split_root"]
    dataset = manifest["dataset"]

    resolved_models = []
    for model in models:
        model_root = repo_root / "models" / manifest["version"] / model["name"]
        engine_requirements = []
        for engine in engines:
            artifact_type = engine["artifact"]["type"]
            artifact_paths = [
                model_root / artifact_type / variant for variant in engine["artifact"]["variants"]
            ]
            engine_requirements.append(
                {
                    "name": engine["name"],
                    "runtime_device": engine["runtime_device"],
                    "artifact_type": artifact_type,
                    "artifact_paths": [_repo_relative(path, repo_root) for path in artifact_paths],
                    "evaluate_variants": engine["evaluate_variants"],
                    "cache_modes": engine["cache_modes"],
                }
            )
        resolved_models.append(
            {
                "name": model["name"],
                "slug": model["slug"],
                "base_model_id": model["base_model_id"],
                "base_model_revision": model["base_model_revision"],
                "adapter": model["adapter"],
                "release_subfolder": model.get("release_subfolder"),
                "source_paths": {
                    "base": _repo_relative(model_root / "source" / "base", repo_root),
                    "adapter": _repo_relative(model_root / "source" / "adapter", repo_root),
                    "merged": _repo_relative(model_root / "transformers" / "merged", repo_root),
                },
                "engines": engine_requirements,
            }
        )

    return {
        "manifest": {
            "version": manifest["version"],
            "experiments_repository": manifest["experiments"]["repository"],
            "experiments_revision": manifest["experiments"]["revision"],
            "release": manifest.get("release"),
        },
        "profile": {"id": profile["id"], "target": target, "compute": compute},
        "stages": stages,
        "datasets": {
            "size": dataset["size"],
            "test_anchor": _repo_relative(split_root / dataset["splits"]["test_anchor"], repo_root),
            "calibration": _repo_relative(split_root / dataset["splits"]["calibration"], repo_root),
            "final_selection_test": _repo_relative(
                split_root / dataset["splits"]["final_selection_test"], repo_root
            ),
        },
        "prompt": {
            "template_id": manifest["prompt"]["template_id"],
            "output_format": manifest["prompt"]["output_format"],
        },
        "models": resolved_models,
    }
