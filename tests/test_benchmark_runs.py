"""Tests for locked, isolated benchmark pipeline workspaces."""

from __future__ import annotations

import json
from pathlib import Path

from benchmark_pipeline.runs import create_run_workspace, load_run_workspace, write_summary
from evaluation_lib.plot_common import load_reports


def _manifest() -> dict:
    return {"version": "v2.1", "schema_version": 1}


def _plan() -> dict:
    return {
        "profile": {"id": "mac-cpu"},
        "models": [
            {
                "name": "Qwen3-0.6B",
                "engines": [
                    {"name": "transformers", "evaluate_variants": ["float32"]},
                    {"name": "onnx_runtime", "evaluate_variants": ["fp32", "static-int8"]},
                ],
            }
        ],
    }


def test_workspace_locks_all_planned_report_paths_before_execution(tmp_path: Path) -> None:
    workspace = create_run_workspace(repo_root=tmp_path, manifest=_manifest(), plan=_plan())

    assert workspace["root"].parent == tmp_path / "run_results"
    assert (workspace["root"] / "manifest.lock.json").is_file()
    assert len(workspace["lock"]["report_index"]) == 3
    for slot in workspace["lock"]["report_index"]:
        assert slot["report"].endswith("/report.json")
        assert not (workspace["root"] / slot["report"]).exists()

    loaded = load_run_workspace(workspace["root"])
    assert loaded["run_id"] == workspace["run_id"]
    write_summary(workspace, reports=["a/report.json"], plots=["a/summary.png"])
    summary = json.loads((workspace["root"] / "run_summary.json").read_text())
    assert summary["reports"] == ["a/report.json"]


def test_load_reports_can_filter_to_one_pipeline_run(tmp_path: Path) -> None:
    for name, run_id in (("selected.json", "run-a"), ("other.json", "run-b")):
        (tmp_path / name).write_text(json.dumps({"run_config": {"run_id": run_id}}))

    reports = load_reports(tmp_path, run_id="run-a")
    assert len(reports) == 1
    assert reports[0]["run_config"]["run_id"] == "run-a"
