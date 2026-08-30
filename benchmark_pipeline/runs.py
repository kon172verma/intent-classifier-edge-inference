"""Create and validate isolated output workspaces for pipeline runs."""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class RunWorkspaceError(ValueError):
    """Raised when a pipeline run workspace cannot safely be used."""


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _slot_id(model: str, engine: str, variant: str) -> str:
    return f"{model}:{engine}:{variant}"


def create_run_workspace(
    *,
    repo_root: Path,
    manifest: dict[str, Any],
    plan: dict[str, Any],
    benchmark_scope: str = "standard",
) -> dict[str, Any]:
    """Create a run directory and immutable planned-report index before execution."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    base_id = f"{manifest['version']}_{stamp}"
    parent = repo_root / "run_results"
    run_root = parent / base_id
    suffix = 1
    while run_root.exists():
        suffix += 1
        run_root = parent / f"{base_id}_{suffix}"
    run_root.mkdir(parents=True)
    run_id = run_root.name

    report_index: list[dict[str, str]] = []
    for model in plan["models"]:
        for engine in model["engines"]:
            for variant in engine["evaluate_variants"]:
                root = run_root / f"{model['name']}_{variant}"
                report_dir = root / "reports" / engine["name"]
                analysis_dir = root / "analysis" / engine["name"]
                report_path = report_dir / "report.json"
                report_index.append(
                    {
                        "id": _slot_id(model["name"], engine["name"], variant),
                        "model": model["name"],
                        "engine": engine["name"],
                        "variant": variant,
                        "report": str(report_path.relative_to(run_root)),
                        "analysis_dir": str(analysis_dir.relative_to(run_root)),
                    }
                )

    lock = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "manifest": copy.deepcopy(manifest),
        "selection": copy.deepcopy(plan),
        "benchmark_scope": benchmark_scope,
        "report_index": report_index,
    }
    _write_json(run_root / "manifest.lock.json", lock)
    _write_json(
        run_root / "run_summary.json",
        {"schema_version": 1, "run_id": run_id, "reports": [], "plots": []},
    )
    return {"run_id": run_id, "root": run_root, "lock": lock}


def load_run_workspace(run_root: Path) -> dict[str, Any]:
    """Load an existing workspace without changing its locked plan."""
    lock_path = run_root / "manifest.lock.json"
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunWorkspaceError(f"Invalid pipeline run workspace: {run_root}") from exc
    if (
        lock.get("schema_version") != 1
        or not lock.get("run_id")
        or not isinstance(lock.get("report_index"), list)
    ):
        raise RunWorkspaceError(f"Invalid run lock file: {lock_path}")
    return {"run_id": lock["run_id"], "root": run_root, "lock": lock}


def validate_workspace_selection(workspace: dict[str, Any], plan: dict[str, Any]) -> None:
    """Ensure a resumed command cannot apply a different selection to a locked run."""
    expected = {
        _slot_id(model["name"], engine["name"], variant)
        for model in plan["models"]
        for engine in model["engines"]
        for variant in engine["evaluate_variants"]
    }
    indexed = {slot.get("id") for slot in workspace["lock"]["report_index"]}
    if indexed != expected:
        raise RunWorkspaceError(
            "The selected models, engines, or variants do not match the locked run. "
            "Use the original selection when resuming."
        )


def validate_workspace_scope(workspace: dict[str, Any], benchmark_scope: str) -> None:
    """Keep a smoke run separate from a candidate-selection benchmark."""
    if workspace["lock"].get("benchmark_scope", "standard") != benchmark_scope:
        raise RunWorkspaceError(
            "The requested benchmark scope does not match the locked run. "
            "Do not resume a smoke run as a standard benchmark, or vice versa."
        )


def write_summary(
    workspace: dict[str, Any], *, reports: list[str] | None, plots: list[str] | None
) -> None:
    """Record completed outputs separately from the immutable manifest lock."""
    path = workspace["root"] / "run_summary.json"
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        existing = {}
    _write_json(
        path,
        {
            "schema_version": 1,
            "run_id": workspace["run_id"],
            "reports": reports if reports is not None else existing.get("reports", []),
            "plots": plots if plots is not None else existing.get("plots", []),
        },
    )
