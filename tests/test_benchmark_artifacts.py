"""Tests for Phase 2 source-artifact safety and provenance."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from benchmark_pipeline.artifacts import (
    ArtifactError,
    download_model_release_artifacts,
    fetch_model_sources,
    materialize_source_snapshot,
)


class BenchmarkArtifactTests(unittest.TestCase):
    """Keep artifact acquisition deterministic without downloading model weights."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.calls: list[dict[str, Any]] = []

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def downloader(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        snapshot = Path(kwargs["cache_dir"]) / "snapshots" / kwargs["revision"]
        snapshot.mkdir(parents=True)
        subfolder = kwargs.get("allow_patterns")
        if subfolder:
            adapter = snapshot / subfolder.removesuffix("/**")
            adapter.mkdir(parents=True)
            (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
        else:
            (snapshot / "config.json").write_text("{}\n", encoding="utf-8")
        return str(snapshot)

    def test_snapshot_is_hashed_and_reused_only_with_matching_provenance(self) -> None:
        destination = self.root / "source" / "base"
        created = materialize_source_snapshot(
            repository="example/base",
            revision="a" * 40,
            destination=destination,
            snapshot_download_fn=self.downloader,
        )
        reused = materialize_source_snapshot(
            repository="example/base",
            revision="a" * 40,
            destination=destination,
            snapshot_download_fn=self.downloader,
        )

        self.assertTrue(created)
        self.assertFalse(reused)
        self.assertEqual(len(self.calls), 1)
        metadata = (destination / ".benchmark_source.json").read_text(encoding="utf-8")
        self.assertIn('"sha256"', metadata)

        with self.assertRaisesRegex(ArtifactError, "does not match"):
            materialize_source_snapshot(
                repository="example/base",
                revision="b" * 40,
                destination=destination,
                snapshot_download_fn=self.downloader,
            )

    def test_fetch_strips_the_adapter_subfolder_and_pins_both_sources(self) -> None:
        manifest = {
            "version": "v-test",
            "experiments": {"repository": "example/experiments", "revision": "b" * 40},
        }
        model = {
            "name": "Example-Model",
            "base_model_id": "example/base",
            "base_model_revision": "a" * 40,
            "adapter": {"subfolder": "v-test/adapter"},
        }

        result = fetch_model_sources(
            repo_root=self.root,
            manifest=manifest,
            model=model,
            snapshot_download_fn=self.downloader,
        )

        adapter_dir = self.root / "models" / "v-test" / "Example-Model" / "source" / "adapter"
        self.assertTrue(result["base_created"])
        self.assertTrue(result["adapter_created"])
        self.assertTrue((adapter_dir / "adapter_config.json").is_file())
        self.assertFalse((adapter_dir / "v-test").exists())
        self.assertEqual(self.calls[1]["allow_patterns"], "v-test/adapter/**")

    def test_release_download_materializes_profile_artifacts_and_reuses_them(self) -> None:
        manifest: dict[str, Any] = {
            "version": "v-test",
            "experiments": {"repository": "example/experiments", "revision": "a" * 40},
            "dataset": {"size": "1k"},
            "prompt": {"template_id": "v2-positional-id", "system_prompt": "route"},
            "release": {"repository": "example/intent-classifier", "revision": "b" * 40},
        }
        model: dict[str, Any] = {
            "name": "Example-Model",
            "slug": "example-model",
            "release_subfolder": "v-test-example-model",
            "base_model_id": "example/base",
            "base_model_revision": "c" * 40,
            "adapter": {"subfolder": "v-test/adapter", "technique": "LoRA", "configuration": "A"},
        }
        engines = [{"artifact": {"type": "onnx", "variants": ["static-int8"]}}]

        def release_downloader(**kwargs: Any) -> str:
            self.calls.append(kwargs)
            snapshot = Path(kwargs["cache_dir"]) / "snapshots" / kwargs["revision"]
            release_root = snapshot / model["release_subfolder"]
            release_root.mkdir(parents=True)
            provenance = {
                "release_repository": manifest["release"]["repository"],
                "release_subfolder": model["release_subfolder"],
                "manifest": {
                    "version": manifest["version"],
                    "experiments": manifest["experiments"],
                    "dataset": manifest["dataset"],
                    "prompt": manifest["prompt"],
                },
                "model": model,
            }
            (release_root / "benchmark_provenance.json").write_text(
                json.dumps(provenance), encoding="utf-8"
            )
            transformers = release_root / "transformers"
            transformers.mkdir()
            (transformers / "tokenizer.json").write_text("{}", encoding="utf-8")
            onnx = release_root / "onnx" / "static-int8"
            onnx.mkdir(parents=True)
            (onnx / "model.onnx").write_bytes(b"onnx")
            return str(snapshot)

        first = download_model_release_artifacts(
            repo_root=self.root,
            manifest=manifest,
            model=model,
            engines=engines,
            snapshot_download_fn=release_downloader,
        )
        second = download_model_release_artifacts(
            repo_root=self.root,
            manifest=manifest,
            model=model,
            engines=engines,
            snapshot_download_fn=release_downloader,
        )

        local_model = self.root / "models" / "v-test" / "Example-Model"
        self.assertEqual(len(self.calls), 1)
        self.assertTrue((local_model / "onnx" / "static-int8" / "model.onnx").is_file())
        self.assertTrue((local_model / "transformers" / "merged" / "tokenizer.json").is_file())
        self.assertTrue(all(artifact["created"] for artifact in first["artifacts"]))
        self.assertTrue(all(not artifact["created"] for artifact in second["artifacts"]))
        self.assertEqual(
            self.calls[0]["allow_patterns"],
            [
                "v-test-example-model/benchmark_provenance.json",
                "v-test-example-model/onnx/static-int8/**",
                "v-test-example-model/transformers/**",
            ],
        )

        manifest["release"] = {"repository": "example/intent-classifier", "revision": "d" * 40}
        with self.assertRaisesRegex(ArtifactError, "does not match"):
            download_model_release_artifacts(
                repo_root=self.root,
                manifest=manifest,
                model=model,
                engines=engines,
                snapshot_download_fn=release_downloader,
            )


if __name__ == "__main__":
    unittest.main()
