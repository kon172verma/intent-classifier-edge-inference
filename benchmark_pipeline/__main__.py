"""Resolve or explicitly execute supported stages of the benchmark pipeline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from benchmark_pipeline.artifacts import (
    ArtifactError,
    download_release_artifacts,
    fetch_sources,
    merge_models,
)
from benchmark_pipeline.builders import build_artifacts
from benchmark_pipeline.manifest import (
    ManifestError,
    load_manifest,
    resolve_plan,
    resolve_profile,
    select_engines,
    select_models,
)
from benchmark_pipeline.runs import (
    RunWorkspaceError,
    create_run_workspace,
    load_run_workspace,
    validate_workspace_scope,
    validate_workspace_selection,
    write_summary,
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
        "--run-dir",
        type=Path,
        default=None,
        help="Existing run_results directory to resume for evaluate or plot.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a short Mac-only verification (three anchor examples, one warm-up).",
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
        help="fetch, merge, build-artifacts, download-release, evaluate, plot, or the sole value 'all'.",
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
    if manifest["release"] is not None:
        print(f"Release:  {manifest['release']['repository']} @ {manifest['release']['revision']}")
    print("Datasets:")
    print(f"  evaluation       {datasets['test_anchor']}")
    print(f"  calibration      {datasets['calibration']}")
    print(f"  final selection  {datasets['final_selection_test']}")

    for model in plan["models"]:
        print(
            f"\nModel: {model['name']} ({model['base_model_id']} @ {model['base_model_revision']})"
        )
        print(f"  adapter: {model['adapter']['subfolder']}")
        if model["release_subfolder"] is not None:
            print(f"  release: {model['release_subfolder']}")
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


def execute_pipeline(
    *,
    manifest: dict[str, Any],
    target: str,
    compute: str,
    requested_models: list[str],
    requested_engines: list[str],
    stages: list[str],
    manifest_path: Path,
    plan: dict[str, Any],
    run_dir: Path | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    """Execute selected stages, keeping evaluate/plot outputs in a locked workspace."""

    if smoke and target != "mac":
        raise ManifestError("--smoke is currently supported only for --target mac")
    if smoke and "evaluate" not in stages:
        raise ManifestError("--smoke requires the evaluate stage")

    models = select_models(manifest, requested_models)
    profile = resolve_profile(manifest, target, compute)
    engines = select_engines(profile, requested_engines)
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
    if "build-artifacts" in stages:
        execution["build-artifacts"] = build_artifacts(
            repo_root=REPO_ROOT,
            manifest=manifest,
            models=models,
            engines=engines,
            calibration_data=REPO_ROOT
            / manifest["dataset"]["split_root"]
            / manifest["dataset"]["splits"]["calibration"],
        )
    if "download-release" in stages:
        execution["download-release"] = download_release_artifacts(
            repo_root=REPO_ROOT,
            manifest=manifest,
            models=models,
            engines=engines,
            token=token,
        )
    workspace: dict[str, Any] | None = None
    if "evaluate" in stages or "plot" in stages:
        if run_dir is not None:
            workspace = load_run_workspace(run_dir)
            validate_workspace_selection(workspace, plan)
            validate_workspace_scope(workspace, "smoke" if smoke else "standard")
        elif "evaluate" in stages:
            workspace = create_run_workspace(
                repo_root=REPO_ROOT,
                manifest=manifest,
                plan=plan,
                benchmark_scope="smoke" if smoke else "standard",
            )
        else:
            raise RunWorkspaceError(
                "--stages plot requires --run-dir so it can use a locked report index"
            )
        execution["run_id"] = workspace["run_id"]
        execution["run_dir"] = str(workspace["root"])

    reports: list[str] = []
    if "evaluate" in stages:
        from benchmark_pipeline.evaluation import evaluate_workspace

        assert workspace is not None
        reports = evaluate_workspace(
            repo_root=REPO_ROOT,
            manifest_path=manifest_path,
            manifest=manifest,
            plan=plan,
            workspace=workspace,
            max_examples=3 if smoke else None,
        )
        execution["evaluate"] = reports
    plots: list[str] = []
    if "plot" in stages:
        from benchmark_pipeline.plotting import plot_workspace

        assert workspace is not None
        plots = plot_workspace(workspace)
        execution["plot"] = plots
    if workspace is not None:
        write_summary(
            workspace,
            reports=reports if "evaluate" in stages else None,
            plots=plots if "plot" in stages else None,
        )
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

    if args.smoke and args.target != "mac":
        print(
            "benchmark_pipeline: error: --smoke is currently supported only for --target mac",
            file=sys.stderr,
        )
        return 2
    if args.smoke and "evaluate" not in plan["stages"]:
        print("benchmark_pipeline: error: --smoke requires the evaluate stage", file=sys.stderr)
        return 2

    if not args.execute:
        if args.json:
            print(json.dumps(plan, indent=2))
        else:
            print_plan(plan)
        return 0

    try:
        execution = execute_pipeline(
            manifest=manifest,
            target=args.target,
            compute=args.compute,
            requested_models=args.models,
            requested_engines=args.engines,
            stages=plan["stages"],
            manifest_path=args.manifest,
            plan=plan,
            run_dir=args.run_dir,
            smoke=args.smoke,
        )
    except (
        ArtifactError,
        ManifestError,
        RunWorkspaceError,
        RuntimeError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"benchmark_pipeline: error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps({"plan": plan, "execution": execution}, indent=2))
    else:
        print_plan(plan)
        print("\n=== Pipeline Execution ===")
        for stage in execution["stages"]:
            print(f"Completed: {stage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
