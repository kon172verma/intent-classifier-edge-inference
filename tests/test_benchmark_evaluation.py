"""Tests for the short, Mac-only pipeline smoke execution contract."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark_pipeline.evaluation import evaluate_workspace
from benchmark_pipeline.runs import create_run_workspace


class BenchmarkEvaluationTests(unittest.TestCase):
    """Verify command construction without loading models or invoking hardware runtimes."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.manifest = {"version": "v-test"}
        self.plan = {
            "profile": {"target": "mac"},
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


if __name__ == "__main__":
    unittest.main()
