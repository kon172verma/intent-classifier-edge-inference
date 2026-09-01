"""Tests for manifest-driven Hugging Face release staging."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from release_scripts.release import assemble_release_tree, release_model_folder


class ReleaseScriptTests(unittest.TestCase):
    """Ensure a release repository contains only deployable artifacts."""

    def test_release_tree_keeps_each_format_in_a_versioned_model_folder(self) -> None:
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
                "slug": "qwen3-0.6b",
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
            (destination / "README.md").write_text("Manual root documentation", encoding="utf-8")
            (destination / "LICENSE").write_text("Manual root license", encoding="utf-8")
            repo_id = "kon172verma/intent-classifier"
            result = assemble_release_tree(
                repo_root=repo_root,
                manifest=manifest,
                model=model,
                repo_id=repo_id,
                destination=destination,
            )

            model_folder = release_model_folder(manifest, model)
            staged_model = destination / model_folder
            self.assertEqual(result["repo_id"], repo_id)
            self.assertEqual(result["model_folder"], "v1.0-qwen3-0.6b")
            self.assertTrue((staged_model / "transformers" / "config.json").is_file())
            self.assertTrue((staged_model / "transformers" / "model.safetensors").is_file())
            self.assertTrue((staged_model / "gguf" / "Q4_K_M" / "model.gguf").is_file())
            self.assertTrue((staged_model / "onnx" / "fp32" / "model.onnx").is_file())
            self.assertFalse((destination / "transformers").exists())
            self.assertFalse((destination / "source").exists())
            self.assertEqual((destination / "README.md").read_text(), "Manual root documentation")
            self.assertEqual((destination / "LICENSE").read_text(), "Manual root license")
            self.assertFalse((staged_model / "transformers" / ".benchmark_artifact.json").exists())
            self.assertFalse((staged_model / "README.md").read_text().startswith("---"))
            provenance = json.loads((staged_model / "benchmark_provenance.json").read_text())
            self.assertEqual(provenance["release_repository"], repo_id)
            self.assertEqual(provenance["release_subfolder"], model_folder)


if __name__ == "__main__":
    unittest.main()
