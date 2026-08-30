"""Shared chart-rendering utilities for all evaluation backend plot_results.py files."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from evaluation_lib.config import MODEL_DISPLAY_NAMES  # noqa: F401 (re-exported)

_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

_PANELS = [
    (
        "Preprocessing / Processing / TTFT (ms)",
        ["Preprocessing", "Total Processing", "TTFT"],
        False,
    ),
    (
        "Prefill Phase Breakdown (ms)",
        ["System Prompt", "Tools List", "User Query", "Decode"],
        False,
    ),
    (
        "Quality & Memory (log scale)",
        ["Accuracy %", "Peak RAM MB", "KV Cache MB", "Peak GPU MB"],
        True,
    ),
]


def _annotate_bars(ax, bars, fmts: list[str]) -> None:
    for bar, fmt in zip(bars, fmts):
        ax.annotate(
            fmt,
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
        )


def _render_panel(
    ax, labels: list[str], values: list[float], fmts: list[str], log: bool, ylim: tuple
) -> None:
    bars = ax.bar(labels, values, color=_COLORS[: len(labels)])
    if log:
        ax.set_yscale("log")
    ax.set_ylim(*ylim)
    _annotate_bars(ax, bars, fmts)
    ax.set_xticklabels([])


def _values_preprocessing(aggregate: dict, run_config: dict) -> tuple[list, list]:
    preprocessing_ms = aggregate.get("mean_preprocessing_latency_ms") or 0.0
    if (aggregate.get("mean_system_prefill_latency_ms") or 0.0) == 0.0 and run_config.get(
        "mode"
    ) == "prefix_cache":
        cache_info = run_config.get("prefix_cache_info") or {}
        # baseline/llama_cpp use "cache_creation_ms"; onnx/tensorrt use "creation_time_ms"
        preprocessing_ms += cache_info.get(
            "cache_creation_ms", cache_info.get("creation_time_ms", 0.0)
        )
    values = [
        preprocessing_ms,
        aggregate.get("mean_e2e_latency_ms") or 0.0,
        aggregate.get("mean_ttft_ms") or 0.0,
    ]
    fmts = [f"{v:.0f} ms" for v in values]
    return values, fmts


def _values_phase_breakdown(aggregate: dict, run_config: dict) -> tuple[list, list]:
    system_ms = aggregate.get("mean_system_prefill_latency_ms") or 0.0
    if system_ms == 0.0 and run_config.get("mode") == "prefix_cache":
        cache_info = run_config.get("prefix_cache_info") or {}
        system_ms = cache_info.get("cache_creation_ms", cache_info.get("creation_time_ms", 0.0))
    values = [
        system_ms,
        aggregate.get("mean_tools_prefill_latency_ms") or 0.0,
        aggregate.get("mean_query_prefill_latency_ms") or 0.0,
        aggregate.get("mean_decode_latency_ms") or 0.0,
    ]
    fmts = [f"{v:.0f} ms" for v in values]
    return values, fmts


def _values_quality_memory(aggregate: dict, quality: dict) -> tuple[list, list]:
    accuracy_pct = (quality.get("tool_accuracy") or 0.0) * 100
    peak_ram = aggregate.get("peak_ram_mb") or 0.0
    kv_cache_mb = (aggregate.get("mean_kv_cache_kb") or 0.0) / 1024
    peak_gpu = aggregate.get("mean_peak_gpu_mb")
    # Log scale can't render a true 0; use a small placeholder so the "N/A" label shows up.
    peak_gpu_val = peak_gpu if peak_gpu is not None else 0.15
    values = [accuracy_pct, peak_ram, kv_cache_mb, peak_gpu_val]
    fmts = [
        f"{accuracy_pct:.1f}%",
        f"{peak_ram:.0f}",
        f"{kv_cache_mb:.1f}",
        "N/A" if peak_gpu is None else f"{peak_gpu:.0f}",
    ]
    return values, fmts


def load_reports(
    reports_dir: Path, mode: str | None = None, run_id: str | None = None
) -> list[dict]:
    """Load reports, optionally filtering by active mode and exact pipeline run ID."""
    reports = []
    for path in sorted(reports_dir.glob("*.json")):
        try:
            with open(path) as f:
                doc = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if mode is not None and doc.get("run_config", {}).get("mode") != mode:
            continue
        if run_id is not None and doc.get("run_config", {}).get("run_id") != run_id:
            continue
        reports.append(doc)
    return reports


def group_reports(reports: list[dict], variant_key: str) -> dict:
    """Group reports by (machine, device) -> model_key -> variant -> latest report."""
    grouped: dict[tuple[str, str], dict[str, dict[str, dict]]] = {}
    for doc in reports:
        rc = doc["run_config"]
        machine = rc.get("machine", "unknown")
        device = rc.get("device", "unknown")
        model_key = rc.get("model_key", "unknown")
        variant = rc.get(variant_key, "unknown")
        ts = rc.get("timestamp_utc", "")

        device_group = (machine, device)
        grouped.setdefault(device_group, {}).setdefault(model_key, {})
        existing = grouped[device_group][model_key].get(variant)
        if existing is None or ts > existing["run_config"].get("timestamp_utc", ""):
            grouped[device_group][model_key][variant] = doc
    return grouped


def plot_device_model(
    machine: str,
    device: str,
    model_key: str,
    variant_docs: dict[str, dict],
    variant_order: list[str],
    variant_labels: dict[str, str],
    out_path: Path,
    suptitle: str,
) -> None:
    """Render one chart PNG for a single (machine, device, model, variant set)."""
    available = [v for v in variant_order if v in variant_docs]
    if not available:
        print(
            f"[plot] ERROR: no variants available for "
            f"machine={machine} device={device} model={model_key}. Skipping."
        )
        return

    n_rows = len(available)
    fig = plt.figure(figsize=(15, 3.2 * n_rows + 1.4))
    gs = fig.add_gridspec(
        n_rows + 1, 3, height_ratios=[0.45] + [1] * n_rows, hspace=0.55, wspace=0.35
    )
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=0.995)

    for col, (title, labels, _) in enumerate(_PANELS):
        legend_ax = fig.add_subplot(gs[0, col])
        legend_ax.axis("off")
        handles = [Rectangle((0, 0), 1, 1, color=_COLORS[i]) for i in range(len(labels))]
        legend_ax.legend(
            handles,
            labels,
            loc="center",
            ncol=2,
            frameon=False,
            fontsize=8,
            title=title,
            title_fontsize=9,
        )

    row_values: list = []
    for variant in available:
        doc = variant_docs[variant]
        aggregate = doc.get("aggregate", {})
        quality = doc.get("quality", {})
        doc_run_config = doc.get("run_config", {})
        row_values.append(
            [
                _values_preprocessing(aggregate, doc_run_config),
                _values_phase_breakdown(aggregate, doc_run_config),
                _values_quality_memory(aggregate, quality),
            ]
        )

    column_ylims: list[tuple] = []
    for col, (_, _, log) in enumerate(_PANELS):
        col_values = [v for row in row_values for v in row[col][0]]
        col_max = max(col_values) if col_values else 1.0
        column_ylims.append((0.1, col_max * 3) if log else (0.0, col_max * 1.15))

    for row, variant in enumerate(available, start=1):
        for col, (_, labels, log) in enumerate(_PANELS):
            ax = fig.add_subplot(gs[row, col])
            values, fmts = row_values[row - 1][col]
            _render_panel(ax, labels, values, fmts, log, column_ylims[col])
            if col == 0:
                ax.set_ylabel(variant_labels.get(variant, variant), fontsize=12, fontweight="bold")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Wrote {out_path}")
