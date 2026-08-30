#!/usr/bin/env python3
"""Migrate locked pipeline runs to the flat, model-centric report layout.

The migration is intentionally explicit: pass one or more existing run
directories. It refuses to overwrite any destination report or chart.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON file: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _report_destination(run_dir: Path, slot: dict[str, Any]) -> Path:
    return run_dir / str(slot["model"]) / f"{slot['engine']}_{slot['variant']}.json"


def migrate_run(run_dir: Path) -> None:
    """Migrate one run directory without changing its selected reports or run ID."""
    run_dir = run_dir.resolve()
    lock_path = run_dir / "manifest.lock.json"
    summary_path = run_dir / "run_summary.json"
    lock = _read_json(lock_path)
    summary = _read_json(summary_path)
    slots = lock.get("report_index")
    if not isinstance(slots, list):
        raise ValueError(f"Run lock has no report index: {lock_path}")

    moves: list[tuple[Path, Path]] = []
    old_to_new: dict[str, str] = {}
    for slot in slots:
        if not isinstance(slot, dict) or not all(
            isinstance(slot.get(field), str) for field in ("model", "engine", "variant", "report")
        ):
            raise ValueError(f"Run lock has an invalid report slot: {lock_path}")
        source = run_dir / slot["report"]
        destination = _report_destination(run_dir, slot)
        if not source.is_file() and source != destination:
            raise FileNotFoundError(f"Indexed report is missing: {source}")
        if destination.exists() and source != destination:
            raise FileExistsError(f"Refusing to overwrite report: {destination}")
        old_to_new[slot["report"]] = str(destination.relative_to(run_dir))
        moves.append((source, destination))

    profile = lock.get("selection", {}).get("profile", {})
    target = profile.get("target")
    compute = profile.get("compute")
    if not isinstance(target, str) or not isinstance(compute, str):
        raise ValueError(f"Run lock has no target/compute profile: {lock_path}")
    chart_moves: list[tuple[Path, Path]] = []
    for model in {str(slot["model"]) for slot in slots}:
        old_chart = run_dir / "analysis" / model / f"{target}_{compute}.png"
        new_chart = run_dir / model / "comparison.png"
        if old_chart.is_file():
            if new_chart.exists() and old_chart != new_chart:
                raise FileExistsError(f"Refusing to overwrite chart: {new_chart}")
            chart_moves.append((old_chart, new_chart))

    for source, destination in moves + chart_moves:
        if source == destination:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.replace(destination)

    for slot in slots:
        slot["report"] = old_to_new[slot["report"]]
        slot.pop("analysis_dir", None)
    lock["report_index"] = slots
    summary["reports"] = [old_to_new.get(path, path) for path in summary.get("reports", [])]
    old_chart_paths = {
        str(source.relative_to(run_dir)): str(destination.relative_to(run_dir))
        for source, destination in chart_moves
    }
    summary["plots"] = [old_chart_paths.get(path, path) for path in summary.get("plots", [])]
    _write_json(lock_path, lock)
    _write_json(summary_path, summary)

    for path in sorted(run_dir.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass
    print(f"Migrated {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path, help="Existing run_results directories")
    args = parser.parse_args()
    for run_dir in args.run_dirs:
        migrate_run(run_dir)


if __name__ == "__main__":
    main()
