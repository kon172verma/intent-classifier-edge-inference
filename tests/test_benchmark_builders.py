"""Tests for Phase 3 artifact-builder orchestration."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark_pipeline.builders import build_artifacts


class BenchmarkBuilderTests(unittest.TestCase):
    """Exercise builder selection and artifact provenance without external tools."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.manifest = {"version": "v-test", "prompt": {"system_prompt": "Route tools."}}
        self.model = {"name": "Example-Model"}
        self.merged_dir = (
            self.root / "models" / "v-test" / "Example-Model" / "transformers" / "merged"
        )
        self.merged_dir.mkdir(parents=True)
        (self.merged_dir / ".benchmark_artifact.json").write_text(
            json.dumps({"kind": "merged_transformers", "inputs": {}}), encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_gguf_builder_creates_and_reuses_profile_variants(self) -> None:
        calls: list[list[str]] = []

        def fake_run(command: list[str]) -> None:
            calls.append(command)
            output_root = Path(command[command.index("--output-root") + 1])
            variants_start = command.index("--variants") + 1
            variants_end = command.index("--python")
            for variant in command[variants_start:variants_end]:
                destination = output_root / variant
                destination.mkdir(parents=True)
                (destination / "model.gguf").write_bytes(b"gguf")

        engines = [
            {
                "artifact": {
                    "type": "gguf",
                    "variants": ["Q4_K_M", "Q6_K"],
                }
            }
        ]
        with patch("benchmark_pipeline.builders._run", side_effect=fake_run):
            first = build_artifacts(
                repo_root=self.root,
                manifest=self.manifest,
                models=[self.model],
                engines=engines,
                calibration_data=self.root / "calibration.json",
            )
            second = build_artifacts(
                repo_root=self.root,
                manifest=self.manifest,
                models=[self.model],
                engines=engines,
                calibration_data=self.root / "calibration.json",
            )

        self.assertEqual(len(calls), 1)
        self.assertEqual([item["created"] for item in first[0]["artifacts"]], [True, True])
        self.assertEqual([item["created"] for item in second[0]["artifacts"]], [False, False])
        metadata = self.root / "models" / "v-test" / "Example-Model" / "gguf" / "Q4_K_M"
        self.assertTrue((metadata / ".benchmark_artifact.json").is_file())

    def test_transformers_engine_needs_no_derived_artifact(self) -> None:
        results = build_artifacts(
            repo_root=self.root,
            manifest=self.manifest,
            models=[self.model],
            engines=[{"artifact": {"type": "transformers", "variants": ["merged"]}}],
            calibration_data=self.root / "calibration.json",
        )
        self.assertEqual(results, [{"model": "Example-Model", "artifacts": []}])


if __name__ == "__main__":
    unittest.main()
