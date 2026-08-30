"""Resolve a manifest-driven benchmark plan without downloading or executing work."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from benchmark_pipeline.manifest import ManifestError, load_manifest, resolve_plan

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
        print(f"\nModel: {model['name']} ({model['base_model_id']})")
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

    if args.json:
        print(json.dumps(plan, indent=2))
    else:
        print_plan(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
