"""Tests for the short, Mac-only pipeline smoke execution contract."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark_pipeline.evaluation import evaluate_workspace
from benchmark_pipeline.plotting import plot_workspace
from benchmark_pipeline.runs import create_run_workspace


class BenchmarkEvaluationTests(unittest.TestCase):
    """Verify command construction without loading models or invoking hardware runtimes."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.manifest = {"version": "v-test"}
        self.plan = {
            "profile": {"target": "mac", "compute": "cpu"},
            "datasets": {"test_anchor": "benchmark_data/10k/test_anchor.json"},
            "models": [
                {
                    "name": "Example-Model",
                    "source_paths": {"merged": "models/v-test/Example-Model/transformers/merged"},
                    "engines": [
                        {
                            "name": "transformers",
                            "runtime_device": "cpu",
                            "artifact_type": "transformers",
                            "evaluate_variants": ["float32"],
                        }
                    ],
                }
            ],
        }
        self.workspace = create_run_workspace(
            repo_root=self.root,
            manifest=self.manifest,
            plan=self.plan,
            benchmark_scope="smoke",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_smoke_passes_short_limit_and_scope_to_runner(self) -> None:
        commands: list[list[str]] = []

        def fake_run(command: list[str], **_: object) -> None:
            commands.append(command)
            output_path = Path(command[command.index("--output-file") + 1])
            output_path.write_text(
                json.dumps({"run_config": {"run_id": self.workspace["run_id"]}}),
                encoding="utf-8",
            )

        with patch("benchmark_pipeline.evaluation.subprocess.run", side_effect=fake_run):
            reports = evaluate_workspace(
                repo_root=self.root,
                manifest_path=self.root / "manifest.json",
                manifest=self.manifest,
                plan=self.plan,
                workspace=self.workspace,
                max_examples=3,
            )

        self.assertEqual(len(reports), 1)
        self.assertEqual(len(commands), 1)
        command = commands[0]
        self.assertEqual(command[command.index("--max-examples") + 1], "3")
        self.assertEqual(command[command.index("--warmup") + 1], "1")
        self.assertEqual(command[command.index("--benchmark-scope") + 1], "smoke")

    def test_plot_creates_one_model_chart_for_all_engine_variants(self) -> None:
        self.plan["models"][0]["engines"].append(
            {
                "name": "onnx_runtime",
                "runtime_device": "cpu",
                "artifact_type": "onnx",
                "evaluate_variants": ["fp32"],
            }
        )
        workspace = create_run_workspace(
            repo_root=self.root,
            manifest=self.manifest,
            plan=self.plan,
            benchmark_scope="smoke",
        )
        for slot in workspace["lock"]["report_index"]:
            report = workspace["root"] / slot["report"]
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(
                json.dumps(
                    {
                        "run_config": {"run_id": workspace["run_id"], "mode": "prefix_cache"},
                        "aggregate": {
                            "mean_preprocessing_latency_ms": 1.0,
                            "mean_e2e_latency_ms": 2.0,
                            "mean_ttft_ms": 1.0,
                            "mean_system_prefill_latency_ms": 1.0,
                            "mean_tools_prefill_latency_ms": 1.0,
                            "mean_query_prefill_latency_ms": 1.0,
                            "mean_decode_latency_ms": 1.0,
                            "peak_ram_mb": 10.0,
                            "mean_kv_cache_kb": 10.0,
                        },
                        "quality": {"tool_accuracy": 1.0},
                    }
                ),
                encoding="utf-8",
            )

        plots = plot_workspace(workspace)

        self.assertEqual(plots, ["analysis/Example-Model/mac_cpu.png"])
        self.assertTrue((workspace["root"] / plots[0]).is_file())


if __name__ == "__main__":
    unittest.main()
