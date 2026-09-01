"""Tests for manifest-driven Hugging Face release staging."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from release_scripts.release import assemble_release_tree, release_repository_id


class ReleaseScriptTests(unittest.TestCase):
    """Ensure a release repository contains only deployable artifacts."""

    def test_release_tree_flattens_transformers_and_keeps_deployment_formats(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repo_root = Path(temporary_directory)
            manifest: dict[str, Any] = {
                "version": "v1.0",
                "experiments": {"repository": "owner/experiments", "revision": "a" * 40},
                "dataset": {"size": "1k"},
                "prompt": {
                    "template_id": "v1-tool-name",
                    "output_format": "tool_name",
                    "model_no_tool_token": "none",
                },
            }
            model: dict[str, Any] = {
                "name": "Qwen3-0.6B",
                "base_model_id": "Qwen/Qwen3-0.6B",
                "base_model_revision": "b" * 40,
                "adapter": {"technique": "LoRA", "configuration": "C", "subfolder": "v1.0/adapter"},
            }
            local_model = repo_root / "models" / "v1.0" / model["name"]
            merged = local_model / "transformers" / "merged"
            merged.mkdir(parents=True)
            (merged / "config.json").write_text("{}", encoding="utf-8")
            (merged / "model.safetensors").write_bytes(b"weights")
            (merged / ".benchmark_artifact.json").write_text("{}", encoding="utf-8")
            gguf = local_model / "gguf" / "Q4_K_M"
            gguf.mkdir(parents=True)
            (gguf / "model.gguf").write_bytes(b"gguf")
            onnx = local_model / "onnx" / "fp32"
            onnx.mkdir(parents=True)
            (onnx / "model.onnx").write_bytes(b"onnx")
            (local_model / "source" / "base").mkdir(parents=True)

            destination = repo_root / "staging"
            destination.mkdir()
            repo_id = release_repository_id("kon172verma", "v1.0", model["name"])
            result = assemble_release_tree(
                repo_root=repo_root,
                manifest=manifest,
                model=model,
                repo_id=repo_id,
                destination=destination,
            )

            self.assertEqual(result["repo_id"], "kon172verma/intent-classifier-v1.0-0.6b")
            self.assertTrue((destination / "config.json").is_file())
            self.assertTrue((destination / "model.safetensors").is_file())
            self.assertTrue((destination / "gguf" / "Q4_K_M" / "model.gguf").is_file())
            self.assertTrue((destination / "onnx" / "fp32" / "model.onnx").is_file())
            self.assertFalse((destination / "transformers").exists())
            self.assertFalse((destination / "source").exists())
            self.assertFalse((destination / ".benchmark_artifact.json").exists())
            provenance = json.loads((destination / "benchmark_provenance.json").read_text())
            self.assertEqual(provenance["release_repository"], repo_id)


if __name__ == "__main__":
    unittest.main()
