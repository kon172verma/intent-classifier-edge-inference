"""Regression tests for version manifests and dry-run resolution."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path

from benchmark_pipeline.manifest import (
    ManifestError,
    load_manifest,
    resolve_plan,
    validate_manifest,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFESTS_DIR = REPO_ROOT / "manifests"


class BenchmarkManifestTests(unittest.TestCase):
    """Keep version catalogues and profile resolution deterministic."""

    def test_all_version_manifests_validate(self) -> None:
        for version in ("v1.0", "v2.0", "v2.1"):
            manifest = load_manifest(MANIFESTS_DIR / f"{version}.json")
            self.assertEqual(manifest["version"], version)

    def test_v21_rpi_all_uses_anchor_calibration_and_exact_models(self) -> None:
        manifest = load_manifest(MANIFESTS_DIR / "v2.1.json")
        plan = resolve_plan(
            manifest,
            target="rpi",
            compute="cpu",
            requested_models=["all"],
            requested_engines=["all"],
            requested_stages=["all"],
            repo_root=REPO_ROOT,
        )

        self.assertEqual(plan["profile"]["id"], "rpi-cpu")
        self.assertEqual(plan["datasets"]["test_anchor"], "benchmark_data/10k/test_anchor.json")
        self.assertEqual(plan["datasets"]["calibration"], "benchmark_data/10k/calibration.json")
        self.assertEqual(
            [model["name"] for model in plan["models"]],
            ["Qwen2.5-0.5B", "Qwen3-0.6B", "SmolLM2-360M"],
        )
        self.assertEqual(
            [engine["name"] for engine in plan["models"][0]["engines"]],
            ["transformers", "llama_cpp", "onnx_runtime"],
        )
        self.assertEqual(plan["models"][0]["engines"][0]["cache_modes"], ["prefix_cache"])
        self.assertEqual(
            plan["models"][0]["base_model_revision"], "2b01de6d1108f9b2b5e46a726aa678a359b6c03b"
        )

    def test_explicit_model_and_engine_selection(self) -> None:
        manifest = load_manifest(MANIFESTS_DIR / "v2.0.json")
        plan = resolve_plan(
            manifest,
            target="mac",
            compute="gpu",
            requested_models=["Qwen3-0.6B"],
            requested_engines=["llama_cpp"],
            requested_stages=["build-artifacts", "evaluate"],
            repo_root=REPO_ROOT,
        )

        self.assertEqual(plan["stages"], ["build-artifacts", "evaluate"])
        self.assertEqual([model["name"] for model in plan["models"]], ["Qwen3-0.6B"])
        self.assertEqual(plan["models"][0]["engines"][0]["runtime_device"], "metal")

    def test_all_cannot_be_combined_with_an_explicit_model(self) -> None:
        manifest = load_manifest(MANIFESTS_DIR / "v1.0.json")
        with self.assertRaisesRegex(ManifestError, "cannot be combined"):
            resolve_plan(
                manifest,
                target="mac",
                compute="cpu",
                requested_models=["all", "Qwen3-0.6B"],
                requested_engines=["all"],
                requested_stages=["all"],
                repo_root=REPO_ROOT,
            )

    def test_xavier_profile_excludes_tensorrt_llm(self) -> None:
        manifest = load_manifest(MANIFESTS_DIR / "v2.1.json")
        profile = next(profile for profile in manifest["profiles"] if profile["id"] == "jetson-gpu")
        self.assertNotIn("tensorrt", [engine["name"] for engine in profile["engines"]])

    def test_download_release_requires_a_pinned_release_and_is_exclusive(self) -> None:
        manifest = load_manifest(MANIFESTS_DIR / "v1.0.json")
        with self.assertRaisesRegex(ManifestError, "requires a pinned release"):
            resolve_plan(
                manifest,
                target="rpi",
                compute="cpu",
                requested_models=["Qwen3-0.6B"],
                requested_engines=["all"],
                requested_stages=["download-release", "evaluate"],
                repo_root=REPO_ROOT,
            )

        released = copy.deepcopy(manifest)
        released["release"] = {"repository": "owner/release", "revision": "a" * 40}
        for model in released["models"]:
            model["release_subfolder"] = f"{released['version']}-{model['slug']}"
        validate_manifest(released)
        plan = resolve_plan(
            released,
            target="rpi",
            compute="cpu",
            requested_models=["Qwen3-0.6B"],
            requested_engines=["all"],
            requested_stages=["download-release", "evaluate", "plot"],
            repo_root=REPO_ROOT,
        )
        self.assertEqual(plan["stages"], ["download-release", "evaluate", "plot"])
        self.assertEqual(plan["models"][0]["release_subfolder"], "v1.0-qwen3-0.6b")

        with self.assertRaisesRegex(ManifestError, "alternative"):
            resolve_plan(
                released,
                target="rpi",
                compute="cpu",
                requested_models=["Qwen3-0.6B"],
                requested_engines=["all"],
                requested_stages=["fetch", "download-release"],
                repo_root=REPO_ROOT,
            )


if __name__ == "__main__":
    unittest.main()
