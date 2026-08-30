"""Engine-specific artifact builders orchestrated by the benchmark pipeline."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from benchmark_pipeline.artifacts import (
    _ARTIFACT_METADATA_NAME,
    ArtifactError,
    _file_inventory,
    _read_metadata,
    _require_reusable,
    _utc_now,
    _write_metadata,
    model_root,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _merged_provenance(merged_dir: Path) -> dict[str, Any]:
    metadata = _read_metadata(merged_dir / _ARTIFACT_METADATA_NAME)
    if metadata is None or metadata.get("kind") != "merged_transformers":
        raise ArtifactError(
            f"Missing manifest-driven merged checkpoint metadata: {merged_dir}. "
            "Run --stages fetch merge --execute first."
        )
    return metadata


def _artifact_expected(kind: str, variant: str, merged: Mapping[str, Any]) -> dict[str, Any]:
    return {"kind": kind, "variant": variant, "input": dict(merged)}


def _write_artifact_metadata(
    *,
    destination: Path,
    expected: Mapping[str, Any],
    builder: Mapping[str, Any],
) -> None:
    metadata = {
        **expected,
        "builder": dict(builder),
        "created_at": _utc_now(),
        "files": _file_inventory(destination),
    }
    _write_metadata(destination / _ARTIFACT_METADATA_NAME, metadata)


def _move_variant(
    *,
    source: Path,
    destination: Path,
    expected: Mapping[str, Any],
    builder: Mapping[str, Any],
) -> None:
    if not source.is_dir():
        raise ArtifactError(f"Builder did not create expected artifact directory: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_artifact_metadata(destination=source, expected=expected, builder=builder)
    source.replace(destination)


def _run(command: list[str]) -> None:
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise ArtifactError(f"Required builder command is unavailable: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise ArtifactError(
            f"Builder command failed with exit code {exc.returncode}: {command[0]}"
        ) from exc


def _build_gguf(
    *,
    merged_dir: Path,
    model_dir: Path,
    variants: list[str],
    merged: Mapping[str, Any],
) -> list[dict[str, Any]]:
    required = [
        variant
        for variant in variants
        if not _require_reusable(
            model_dir / "gguf" / variant,
            _ARTIFACT_METADATA_NAME,
            _artifact_expected("gguf", variant, merged),
        )
    ]
    if not required:
        return [{"type": "gguf", "variant": variant, "created": False} for variant in variants]

    script = _REPO_ROOT / "scripts" / "build_gguf.py"
    conversion_python = os.getenv("BENCHMARK_GGUF_PYTHON", sys.executable)
    with tempfile.TemporaryDirectory(prefix="benchmark-build-gguf-", dir=model_dir) as temp_dir:
        staging_root = Path(temp_dir) / "gguf"
        command = [
            sys.executable,
            str(script),
            "--input-dir",
            str(merged_dir),
            "--output-root",
            str(staging_root),
            "--variants",
            *required,
            "--python",
            conversion_python,
        ]
        _run(command)
        builder = {
            "command": command,
            "name": "scripts/build_gguf.py",
            "python": sys.version.split()[0],
        }
        for variant in required:
            _move_variant(
                source=staging_root / variant,
                destination=model_dir / "gguf" / variant,
                expected=_artifact_expected("gguf", variant, merged),
                builder=builder,
            )
    return [
        {"type": "gguf", "variant": variant, "created": variant in required} for variant in variants
    ]


def _stage_export_source(source_dir: Path, destination: Path) -> None:
    from scripts.prepare_onnx_export_source import stage_export_source

    try:
        stage_export_source(source_dir, destination)
    except (FileNotFoundError, FileExistsError, OSError) as exc:
        raise ArtifactError(f"Unable to stage the ONNX export source: {exc}") from exc


def _build_onnx(
    *,
    merged_dir: Path,
    model_dir: Path,
    variants: list[str],
    merged: Mapping[str, Any],
    calibration_data: Path,
    system_prompt: str,
) -> list[dict[str, Any]]:
    expected_by_variant = {
        variant: _artifact_expected("onnx", variant, merged) for variant in variants
    }
    required = [
        variant
        for variant in variants
        if not _require_reusable(
            model_dir / "onnx" / variant, _ARTIFACT_METADATA_NAME, expected_by_variant[variant]
        )
    ]
    if not required:
        return [{"type": "onnx", "variant": variant, "created": False} for variant in variants]
    if not calibration_data.is_file() and "static-int8" in required:
        raise ArtifactError(f"Static INT8 calibration split does not exist: {calibration_data}")

    need_fp32 = "fp32" in required or any(
        variant in {"dynamic-int8", "static-int8"} for variant in required
    )
    fp32_destination = model_dir / "onnx" / "fp32"
    fp32_expected = _artifact_expected("onnx", "fp32", merged)
    fp32_exists = _require_reusable(fp32_destination, _ARTIFACT_METADATA_NAME, fp32_expected)
    export_variants = (["fp32"] if need_fp32 and not fp32_exists else []) + (
        ["fp16"] if "fp16" in required else []
    )

    with tempfile.TemporaryDirectory(prefix="benchmark-build-onnx-", dir=model_dir) as temp_dir:
        staging = Path(temp_dir)
        export_source = staging / "source"
        _stage_export_source(merged_dir, export_source)
        optimum_cli = os.getenv("BENCHMARK_ONNX_OPTIMUM_CLI", "optimum-cli")
        commands: list[list[str]] = []
        for variant in export_variants:
            command = [
                optimum_cli,
                "export",
                "onnx",
                "-m",
                str(export_source),
                "--task",
                "text-generation-with-past",
                "--dtype",
                variant,
                str(staging / variant),
            ]
            _run(command)
            commands.append(command)

        fp32_model = (
            staging / "fp32" / "model.onnx"
            if "fp32" in export_variants
            else fp32_destination / "model.onnx"
        )
        quantized = [variant for variant in required if variant in {"dynamic-int8", "static-int8"}]
        if quantized:
            try:
                from scripts.quantize_onnx import quantize_resolved_model
            except ImportError as exc:
                raise ArtifactError(
                    "ONNX quantization requires onnxruntime and its quantization extras."
                ) from exc
            quantize_resolved_model(
                checkpoint_dir=merged_dir,
                fp32_model=fp32_model,
                output_root=staging / "quantized",
                variants=quantized,
                calibration_data=calibration_data if "static-int8" in quantized else None,
                system_prompt=system_prompt if "static-int8" in quantized else None,
            )

        builder = {
            "commands": commands,
            "name": "optimum-cli + scripts/quantize_onnx.py",
            "python": sys.version.split()[0],
        }
        for variant in required:
            source = (
                staging / variant if variant in export_variants else staging / "quantized" / variant
            )
            _move_variant(
                source=source,
                destination=model_dir / "onnx" / variant,
                expected=expected_by_variant[variant],
                builder=builder,
            )
        if "fp32" in export_variants and "fp32" not in required:
            # The FP32 export is a prerequisite for quantization and must retain provenance too.
            _move_variant(
                source=staging / "fp32",
                destination=fp32_destination,
                expected=fp32_expected,
                builder=builder,
            )
    return [
        {"type": "onnx", "variant": variant, "created": variant in required} for variant in variants
    ]


def build_artifacts(
    *,
    repo_root: Path,
    manifest: Mapping[str, Any],
    models: Iterable[Mapping[str, Any]],
    engines: Iterable[Mapping[str, Any]],
    calibration_data: Path,
) -> list[dict[str, Any]]:
    """Build just the artifacts required by selected profile engines."""
    variants_by_type: dict[str, list[str]] = {}
    for engine in engines:
        artifact = engine["artifact"]
        artifact_type = str(artifact["type"])
        if artifact_type == "transformers":
            continue
        if artifact_type == "tensorrt":
            raise ArtifactError(
                "TensorRT-LLM is intentionally outside the Xavier pipeline; "
                "bare TensorRT integration is a separate future phase."
            )
        variants_by_type.setdefault(artifact_type, [])
        for variant in artifact["variants"]:
            if variant not in variants_by_type[artifact_type]:
                variants_by_type[artifact_type].append(variant)

    results: list[dict[str, Any]] = []
    for model in models:
        model_dir = model_root(repo_root, manifest, model)
        merged_dir = model_dir / "transformers" / "merged"
        merged = _merged_provenance(merged_dir)
        artifacts: list[dict[str, Any]] = []
        if "gguf" in variants_by_type:
            artifacts.extend(
                _build_gguf(
                    merged_dir=merged_dir,
                    model_dir=model_dir,
                    variants=variants_by_type["gguf"],
                    merged=merged,
                )
            )
        if "onnx" in variants_by_type:
            artifacts.extend(
                _build_onnx(
                    merged_dir=merged_dir,
                    model_dir=model_dir,
                    variants=variants_by_type["onnx"],
                    merged=merged,
                    calibration_data=calibration_data,
                    system_prompt=str(manifest["prompt"]["system_prompt"]),
                )
            )
        results.append({"model": model["name"], "artifacts": artifacts})
    return results
