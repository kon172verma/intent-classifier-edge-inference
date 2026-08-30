"""Render pipeline charts strictly from reports named by a locked run index."""

from __future__ import annotations

import json
from typing import Any

from evaluation_lib.plot_common import plot_device_model


class PlotError(RuntimeError):
    """Raised when a locked report set cannot be plotted safely."""


def plot_workspace(workspace: dict[str, Any]) -> list[str]:
    """Create one all-engine comparison chart for every model in this locked run."""
    reports_by_model: dict[str, dict[str, dict]] = {}
    labels_by_model: dict[str, dict[str, str]] = {}
    for slot in workspace["lock"]["report_index"]:
        report_path = workspace["root"] / slot["report"]
        if not report_path.is_file():
            raise PlotError(f"Indexed report is missing: {report_path}")
        try:
            document = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise PlotError(f"Indexed report is invalid JSON: {report_path}") from exc
        if document.get("run_config", {}).get("run_id") != workspace["run_id"]:
            raise PlotError(f"Report belongs to a different run: {report_path}")
        key = f"{slot['engine']}:{slot['variant']}"
        reports_by_model.setdefault(slot["model"], {})[key] = document
        labels_by_model.setdefault(slot["model"], {})[key] = f"{slot['engine']}\n{slot['variant']}"

    profile = workspace["lock"]["selection"]["profile"]
    target = profile["target"]
    compute = profile["compute"]
    output: list[str] = []
    for model, variant_docs in reports_by_model.items():
        variant_order = list(variant_docs)
        out_path = workspace["root"] / model / "comparison.png"
        plot_device_model(
            target,
            compute,
            model,
            variant_docs,
            variant_order,
            labels_by_model[model],
            out_path,
            suptitle=f"{model} — {target} {compute} ({workspace['run_id']})",
        )
        output.append(str(out_path.relative_to(workspace["root"])))
    return output
