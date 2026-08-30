"""Safe output-path handling shared by direct and pipeline evaluators."""

from __future__ import annotations

from pathlib import Path


def resolve_report_path(output_dir: Path, output_file: Path | None, filename: str) -> Path:
    """Return a report target and prevent an evaluator from replacing a prior report."""
    path = output_file if output_file is not None else output_dir / filename
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
