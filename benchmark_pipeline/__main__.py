"""Resolve or explicitly execute supported stages of the benchmark pipeline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from benchmark_pipeline.artifacts import ArtifactError, fetch_sources, merge_models
from benchmark_pipeline.manifest import (
    ManifestError,
    load_manifest,
    resolve_plan,
    select_models,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve a benchmark manifest into a no-side-effect execution plan."
    )
    parser.add_argument(
        "--manifest", type=Path, required=True, help="Path to a version manifest JSON file."
    )
    parser.add_argument(
        "--target", required=True, help="Target device family, such as mac, rpi, or jetson."
    )
    parser.add_argument("--compute", choices=["cpu", "gpu"], required=True, help="Compute path.")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        metavar="MODEL",
        help="Exact manifest model name(s), or the sole value 'all'.",
    )
    parser.add_argument(
        "--engines",
        nargs="+",
        default=["all"],
        metavar="ENGINE",
        help="Optional exact engine name(s), or the sole value 'all'.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["all"],
        metavar="STAGE",
        help="fetch, merge, build-artifacts, evaluate, plot, or the sole value 'all'.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the resolved plan as JSON.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute supported selected stages. Without this flag, the command is a dry run.",
    )
    return parser.parse_args()


def print_plan(plan: dict[str, Any]) -> None:
    """Render the resolved plan for an interactive terminal."""
    manifest = plan["manifest"]
    profile = plan["profile"]
    datasets = plan["datasets"]
    print("=== Benchmark Pipeline Dry Run ===")
    print(f"Version:  {manifest['version']}")
    print(f"Revision: {manifest['experiments_revision']}")
    print(f"Profile:  {profile['id']} ({profile['target']}/{profile['compute']})")
    print(f"Stages:   {', '.join(plan['stages'])}")
    print(f"Prompt:   {plan['prompt']['template_id']} ({plan['prompt']['output_format']})")
    print("Datasets:")
    print(f"  evaluation       {datasets['test_anchor']}")
    print(f"  calibration      {datasets['calibration']}")
    print(f"  final selection  {datasets['final_selection_test']}")

    for model in plan["models"]:
        print(
            f"\nModel: {model['name']} ({model['base_model_id']} @ {model['base_model_revision']})"
        )
        print(f"  adapter: {model['adapter']['subfolder']}")
        print(f"  merged:  {model['source_paths']['merged']}")
        for engine in model["engines"]:
            print(
                f"  {engine['name']} [{engine['runtime_device']}]: "
                f"evaluate {', '.join(engine['evaluate_variants'])}; "
                f"cache {', '.join(engine['cache_modes'])}"
            )
            for artifact_path in engine["artifact_paths"]:
                print(f"    artifact: {artifact_path}")


def _read_hf_token() -> str | None:
    """Use a token when configured; public snapshots do not require one."""
    return (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )


def execute_phase_two(
    *, manifest: dict[str, Any], requested_models: list[str], stages: list[str]
) -> dict[str, Any]:
    """Execute the acquisition and merge stages introduced in Phase 2."""
    unsupported = [stage for stage in stages if stage not in {"fetch", "merge"}]
    if unsupported:
        names = ", ".join(unsupported)
        raise ArtifactError(
            f"Execution is not implemented yet for stage(s): {names}. "
            "Phase 2 supports --stages fetch merge."
        )

    models = select_models(manifest, requested_models)
    token = _read_hf_token()
    execution: dict[str, Any] = {"stages": stages, "models": [model["name"] for model in models]}
    if "fetch" in stages:
        execution["fetch"] = fetch_sources(
            repo_root=REPO_ROOT,
            manifest=manifest,
            models=models,
            token=token,
        )
    if "merge" in stages:
        execution["merge"] = merge_models(repo_root=REPO_ROOT, manifest=manifest, models=models)
    return execution


def main() -> int:
    args = parse_args()
    try:
        manifest = load_manifest(args.manifest)
        plan = resolve_plan(
            manifest,
            target=args.target,
            compute=args.compute,
            requested_models=args.models,
            requested_engines=args.engines,
            requested_stages=args.stages,
            repo_root=REPO_ROOT,
        )
    except ManifestError as exc:
        print(f"benchmark_pipeline: error: {exc}", file=sys.stderr)
        return 2

    if not args.execute:
        if args.json:
            print(json.dumps(plan, indent=2))
        else:
            print_plan(plan)
        return 0

    try:
        execution = execute_phase_two(
            manifest=manifest,
            requested_models=args.models,
            stages=plan["stages"],
        )
    except (ArtifactError, ManifestError) as exc:
        print(f"benchmark_pipeline: error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps({"plan": plan, "execution": execution}, indent=2))
    else:
        print_plan(plan)
        print("\n=== Phase 2 Execution ===")
        for stage in execution["stages"]:
            print(f"Completed: {stage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
