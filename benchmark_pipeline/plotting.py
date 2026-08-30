"""Render pipeline charts strictly from reports named by a locked run index."""

from __future__ import annotations

import json
from typing import Any

from evaluation_lib.plot_common import plot_device_model


class PlotError(RuntimeError):
    """Raised when a locked report set cannot be plotted safely."""


def plot_workspace(workspace: dict[str, Any]) -> list[str]:
    """Create one chart per model/engine from reports in this run's lock only."""
    reports: list[tuple[dict[str, str], dict]] = []
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
        reports.append((slot, document))

    output: list[str] = []
    for slot, document in reports:
        out_path = workspace["root"] / slot["analysis_dir"] / "summary.png"
        variant = slot["variant"]
        plot_device_model(
            slot["model"],
            slot["engine"],
            slot["model"],
            {variant: document},
            [variant],
            {variant: variant},
            out_path,
            suptitle=(f"{slot['model']} — {slot['engine']} — {variant} ({workspace['run_id']})"),
        )
        output.append(str(out_path.relative_to(workspace["root"])))
    return output
