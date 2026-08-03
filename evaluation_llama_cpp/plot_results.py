#!/usr/bin/env python3
"""
Plot latency/throughput/quality charts from evaluation_llama_cpp JSON reports.

Mirrors evaluation_baseline/plot_results.py exactly, with "dtype" replaced by
"quant" (GGUF quantization level: Q8_0/Q6_K/Q4_K_M). For every (machine,
device) combination found in the results directory and for every model, this
produces one PNG with a grid of bar-chart panels:

    rows    = available quant levels, in order Q8_0 -> Q6_K -> Q4_K_M
              (only quants that actually have a matching result file are
              plotted; a device+model combination with zero matching files
              is skipped with an error message)
    columns = 3 metric panels, identical across all rows:
        1. Preprocessing / Total processing / TTFT   (3 bars, ms)
        2. System prompt / Tools list / User query / Decode  (4 bars, ms)
        3. Accuracy / Peak RAM / KV cache / Peak GPU  (4 bars, log scale)

All values are the *mean* aggregates from each run's JSON report. A shared
legend for each column is rendered once in a dedicated row at the top of the
figure so it never overlaps the chart area.

Peak GPU MB is always "N/A" here -- llama.cpp/Metal exposes no cheap live
GPU-memory query API (see evaluation_llama_cpp/inference.py).

Usage
------
    python evaluation_llama_cpp/plot_results.py
    python evaluation_llama_cpp/plot_results.py --mode kv_cache
    python evaluation_llama_cpp/plot_results.py --results-dir path/to/results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation_lib.config import MODEL_DISPLAY_NAMES, QUANT_LEVELS

_RESULTS_DIR = _REPO_ROOT / "evaluation_llama_cpp" / "results"
_CHARTS_DIR = _RESULTS_DIR / "charts"

# Row order requested: Q8_0 first (highest precision), then Q6_K, then Q4_K_M.
QUANT_ORDER = QUANT_LEVELS
QUANT_LABELS = {"Q8_0": "Q8_0", "Q6_K": "Q6_K", "Q4_K_M": "Q4_K_M"}

_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot charts from evaluation_llama_cpp JSON reports",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=_RESULTS_DIR,
        help="Directory containing run JSON reports",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_CHARTS_DIR,
        help="Directory to write PNG charts",
    )
    p.add_argument(
        "--mode",
        choices=["kv_cache", "prefix_cache"],
        default="prefix_cache",
        help=(
            "Which mode's reports to chart (only kv_cache/prefix_cache carry "
            "the system-prompt/tools-list/user-query prefill split)"
        ),
    )
    return p.parse_args()


def _load_reports(results_dir: Path, mode: str) -> list[dict]:
    """Load all JSON reports under *results_dir* matching *mode*."""
    reports = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            with open(path) as f:
                doc = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        run_config = doc.get("run_config", {})
        if run_config.get("mode") != mode:
            continue
        reports.append(doc)
    return reports


def _group_reports(reports: list[dict]) -> dict:
    """Group reports by (machine, device) -> model_key -> quant -> latest report."""
    grouped: dict[tuple[str, str], dict[str, dict[str, dict]]] = {}
    for doc in reports:
        rc = doc["run_config"]
        machine = rc.get("machine", "unknown")
        device = rc.get("device", "unknown")
        model_key = rc.get("model_key", "unknown")
        quant = rc.get("quant", "unknown")
        ts = rc.get("timestamp_utc", "")

        device_group = (machine, device)
        grouped.setdefault(device_group, {}).setdefault(model_key, {})
        existing = grouped[device_group][model_key].get(quant)
        if existing is None or ts > existing["run_config"].get("timestamp_utc", ""):
            grouped[device_group][model_key][quant] = doc
    return grouped


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
    """Render one bar panel with a pre-computed, figure-wide shared y-axis range.

    *ylim* is computed across ALL quant rows for this column beforehand (see
    _plot_device_model) so that bar heights are visually comparable across
    rows instead of each row silently rescaling to its own data range.
    """
    bars = ax.bar(labels, values, color=_COLORS[: len(labels)])
    if log:
        ax.set_yscale("log")
    ax.set_ylim(*ylim)
    _annotate_bars(ax, bars, fmts)
    ax.set_xticklabels([])


def _values_preprocessing(aggregate: dict, run_config: dict) -> tuple[list, list]:
    preprocessing_ms = aggregate.get("mean_preprocessing_latency_ms") or 0.0
    # Same one-time-cost substitution as _values_phase_breakdown: fold in the
    # real system-prompt cache-creation cost for prefix_cache mode instead
    # of the (correctly) zeroed-out per-example system_prefill_latency_ms.
    if (aggregate.get("mean_system_prefill_latency_ms") or 0.0) == 0.0 and run_config.get(
        "mode"
    ) == "prefix_cache":
        cache_info = run_config.get("prefix_cache_info") or {}
        preprocessing_ms += cache_info.get("cache_creation_ms", 0.0)
    values = [
        preprocessing_ms,
        aggregate.get("mean_e2e_latency_ms") or 0.0,
        aggregate.get("mean_ttft_ms") or 0.0,
    ]
    fmts = [f"{v:.0f} ms" for v in values]
    return values, fmts


def _values_phase_breakdown(aggregate: dict, run_config: dict) -> tuple[list, list]:
    system_ms = aggregate.get("mean_system_prefill_latency_ms") or 0.0
    # In prefix_cache mode the system prompt is ingested ONCE outside the
    # per-example loop, so its per-example cost is correctly 0 in the
    # aggregate -- the real (one-time, amortised) cost lives in
    # run_config.prefix_cache_info.cache_creation_ms instead. Show that
    # instead of a misleading 0 bar.
    if system_ms == 0.0 and run_config.get("mode") == "prefix_cache":
        cache_info = run_config.get("prefix_cache_info") or {}
        if "cache_creation_ms" in cache_info:
            system_ms = cache_info["cache_creation_ms"]
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
    # llama.cpp/Metal exposes no cheap live GPU-memory query API, so this is
    # always None here. Log scale can't render a true 0, so use a small
    # visible placeholder height so the "N/A" label still shows up.
    peak_gpu_val = peak_gpu if peak_gpu is not None else 0.15

    values = [accuracy_pct, peak_ram, kv_cache_mb, peak_gpu_val]
    fmts = [
        f"{accuracy_pct:.1f}%",
        f"{peak_ram:.0f}",
        f"{kv_cache_mb:.1f}",
        "N/A" if peak_gpu is None else f"{peak_gpu:.0f}",
    ]
    return values, fmts


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


def _plot_device_model(
    machine: str,
    device: str,
    model_key: str,
    quant_docs: dict[str, dict],
    mode: str,
    output_dir: Path,
) -> None:
    available_quants = [q for q in QUANT_ORDER if q in quant_docs]
    if not available_quants:
        print(
            f"[plot] ERROR: no precisions available for "
            f"machine={machine} device={device} model={model_key} mode={mode}. Skipping."
        )
        return

    n_rows = len(available_quants)
    fig = plt.figure(figsize=(15, 3.2 * n_rows + 1.4))
    gs = fig.add_gridspec(
        n_rows + 1, 3, height_ratios=[0.45] + [1] * n_rows, hspace=0.55, wspace=0.35
    )

    model_name = MODEL_DISPLAY_NAMES.get(model_key, model_key)
    fig.suptitle(
        f"{model_name} -- machine={machine}, device={device}, mode={mode}",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )

    # Dedicated legend row at the top -- one mini-legend per column, shared
    # across all quant rows below it, so it never overlaps the chart area.
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
    for _row, quant in enumerate(available_quants, start=1):
        doc = quant_docs[quant]
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

    # Compute one shared y-axis range PER COLUMN across all quant rows, so
    # bar heights are visually comparable instead of each row silently
    # rescaling to its own data range.
    column_ylims: list[tuple] = []
    for col, (_, _, log) in enumerate(_PANELS):
        col_values = [v for row in row_values for v in row[col][0]]
        col_max = max(col_values) if col_values else 1.0
        if log:
            column_ylims.append((0.1, col_max * 3))
        else:
            column_ylims.append((0.0, col_max * 1.15))

    for row, quant in enumerate(available_quants, start=1):
        for col, (_, labels, log) in enumerate(_PANELS):
            ax = fig.add_subplot(gs[row, col])
            values, fmts = row_values[row - 1][col]
            _render_panel(ax, labels, values, fmts, log, column_ylims[col])
            if col == 0:
                ax.set_ylabel(QUANT_LABELS.get(quant, quant), fontsize=12, fontweight="bold")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{model_key}_{machine}_{device}_{mode}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Wrote {out_path}")


def main() -> None:
    args = parse_args()

    reports = _load_reports(args.results_dir, args.mode)
    if not reports:
        print(f"[plot] ERROR: no reports found for mode={args.mode} in {args.results_dir}.")
        return

    grouped = _group_reports(reports)

    for (machine, device), by_model in grouped.items():
        for model_key in MODEL_DISPLAY_NAMES:
            quant_docs = by_model.get(model_key, {})
            _plot_device_model(machine, device, model_key, quant_docs, args.mode, args.output_dir)


if __name__ == "__main__":
    main()
