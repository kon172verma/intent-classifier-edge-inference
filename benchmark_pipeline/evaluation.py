"""Execute selected engine runners into an isolated pipeline workspace."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from benchmark_pipeline.runs import RunWorkspaceError


class EvaluationError(RuntimeError):
    """Raised when a selected evaluator cannot produce its indexed report."""


_RUNNERS = {
    "transformers": "evaluation_baseline/run.py",
    "llama_cpp": "evaluation_llama_cpp/run.py",
    "onnx_runtime": "evaluation_onnx/run.py",
}


def _runner_device(engine: str, device: str) -> str:
    if engine == "llama_cpp" and device == "metal":
        return "mps"
    return device


def _artifact_path(
    repo_root: Path, model: dict[str, Any], engine: dict[str, Any], variant: str
) -> Path:
    root = repo_root / "models" / model["version"] / model["name"]
    artifact = engine["artifact_type"]
    if artifact == "transformers":
        return root / "transformers" / "merged"
    if artifact == "gguf":
        return root / "gguf" / variant / "model.gguf"
    if artifact == "onnx":
        return root / "onnx" / variant / "model.onnx"
    raise EvaluationError(f"No pipeline evaluator is implemented for artifact type {artifact!r}")


def evaluate_workspace(
    *,
    repo_root: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    plan: dict[str, Any],
    workspace: dict[str, Any],
) -> list[str]:
    """Run every locked report slot, failing before an accidental overwrite."""
    models = {model["name"]: {**model, "version": manifest["version"]} for model in plan["models"]}
    reports: list[str] = []
    for slot in workspace["lock"]["report_index"]:
        model = models[slot["model"]]
        engine = next(item for item in model["engines"] if item["name"] == slot["engine"])
        runner = _RUNNERS.get(engine["name"])
        if runner is None:
            raise EvaluationError(f"No evaluator is implemented for engine {engine['name']!r}")
        report_path = workspace["root"] / slot["report"]
        if report_path.exists():
            try:
                existing = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RunWorkspaceError(
                    f"Refusing to replace invalid report: {report_path}"
                ) from exc
            if existing.get("run_config", {}).get("run_id") != workspace["run_id"]:
                raise RunWorkspaceError(f"Refusing to overwrite indexed report: {report_path}")
            reports.append(slot["report"])
            continue
        report_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path = _artifact_path(repo_root, model, engine, slot["variant"])
        merged = repo_root / model["source_paths"]["merged"]
        command = [
            sys.executable,
            str(repo_root / runner),
            "--model",
            model["name"],
            "--machine",
            plan["profile"]["target"],
            "--device",
            _runner_device(engine["name"], engine["runtime_device"]),
            "--dataset",
            str(repo_root / plan["datasets"]["test_anchor"]),
            "--manifest",
            str(manifest_path),
            "--run-id",
            workspace["run_id"],
            "--output-dir",
            str(report_path.parent),
            "--output-file",
            str(report_path),
        ]
        if engine["name"] == "transformers":
            command += ["--model-path", str(artifact_path), "--dtype", slot["variant"]]
        elif engine["name"] == "llama_cpp":
            command += [
                "--gguf-path",
                str(artifact_path),
                "--tokenizer-path",
                str(merged),
                "--quant",
                slot["variant"],
            ]
        elif engine["name"] == "onnx_runtime":
            command += [
                "--onnx-path",
                str(artifact_path),
                "--tokenizer-path",
                str(merged),
                "--precision",
                slot["variant"],
            ]
        subprocess.run(command, cwd=repo_root, check=True)
        if not report_path.is_file():
            raise EvaluationError(f"Evaluator did not write indexed report: {report_path}")
        reports.append(slot["report"])
    return reports
