"""Plot latency/throughput/quality charts from evaluation_tensorrt JSON reports.

Mirrors evaluation_onnx/plot_results.py exactly, replacing "precision" with
"dtype" (TensorRT-LLM dtype: fp16 / bf16 / int8 / int4).

For every (machine, device) combination found in the results directory and for
every model this produces one PNG with a grid of bar-chart panels:

    rows    = available dtypes, in TENSORRT_DTYPES order (only dtypes that
              actually have a matching result file are plotted)
    columns = 3 metric panels:
        1. Preprocessing / Total processing / TTFT  (ms)
        2. System prompt / Tools list / User query / Decode  (ms)
        3. Accuracy % / Peak RAM MB / KV Cache MB / Peak GPU MB  (log scale)

Usage
------
    python evaluation_tensorrt/plot_results.py
    python evaluation_tensorrt/plot_results.py --results-dir path/to/results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation_lib.config import MODEL_DISPLAY_NAMES, TENSORRT_DTYPES

_RESULTS_DIR = _REPO_ROOT / "evaluation_tensorrt" / "results"
_CHARTS_DIR = _RESULTS_DIR / "charts"

DTYPE_ORDER = TENSORRT_DTYPES
DTYPE_LABELS = {
    "fp16": "FP16",
    "bf16": "BF16",
    "int8": "INT8 (SmoothQuant)",
    "int4": "INT4 (AWQ)",
}

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot charts from evaluation_tensorrt JSON reports",
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
    return p.parse_args()


def _load_reports(results_dir: Path) -> list[dict]:
    reports = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            with open(path) as f:
                doc = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        reports.append(doc)
    return reports


def _group_reports(reports: list[dict]) -> dict:
    """Group reports by (machine, device) -> model_key -> dtype -> latest report."""
    grouped: dict[tuple[str, str], dict[str, dict[str, dict]]] = {}
    for doc in reports:
        rc = doc["run_config"]
        machine = rc.get("machine", "unknown")
        device = rc.get("device", "cuda")
        model_key = rc.get("model_key", "unknown")
        dtype = rc.get("dtype", "unknown")
        ts = rc.get("timestamp_utc", "")

        device_group = (machine, device)
        grouped.setdefault(device_group, {}).setdefault(model_key, {})
        existing = grouped[device_group][model_key].get(dtype)
        if existing is None or ts > existing["run_config"].get("timestamp_utc", ""):
            grouped[device_group][model_key][dtype] = doc
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


def _render_panel(ax, labels, values, fmts, log: bool, ylim: tuple) -> None:
    bars = ax.bar(labels, values, color=_COLORS[: len(labels)])
    if log:
        ax.set_yscale("log")
    ax.set_ylim(*ylim)
    _annotate_bars(ax, bars, fmts)
    ax.set_xticklabels([])


def _values_preprocessing(aggregate: dict, run_config: dict) -> tuple[list, list]:
    preprocessing_ms = aggregate.get("mean_preprocessing_latency_ms") or 0.0
    if (
        aggregate.get("mean_system_prefill_latency_ms") or 0.0
    ) == 0.0 and run_config.get("mode") == "prefix_cache":
        cache_info = run_config.get("prefix_cache_info") or {}
        preprocessing_ms += cache_info.get("creation_time_ms", 0.0)
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
        if "creation_time_ms" in cache_info:
            system_ms = cache_info["creation_time_ms"]
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
    peak_gpu = aggregate.get("peak_gpu_mb")
    peak_gpu_val = peak_gpu if peak_gpu is not None else 0.15
    values = [accuracy_pct, peak_ram, kv_cache_mb, peak_gpu_val]
    fmts = [
        f"{accuracy_pct:.1f}%",
        f"{peak_ram:.0f}",
        f"{kv_cache_mb:.1f}",
        "N/A" if peak_gpu is None else f"{peak_gpu:.0f}",
    ]
    return values, fmts


def _plot_device_model(
    machine: str,
    device: str,
    model_key: str,
    dtype_docs: dict[str, dict],
    output_dir: Path,
) -> None:
    available_dtypes = [d for d in DTYPE_ORDER if d in dtype_docs]
    if not available_dtypes:
        print(
            f"[plot] ERROR: no dtypes available for "
            f"machine={machine} device={device} model={model_key}. Skipping."
        )
        return

    n_rows = len(available_dtypes)
    fig = plt.figure(figsize=(15, 3.2 * n_rows + 1.4))
    gs = fig.add_gridspec(
        n_rows + 1, 3, height_ratios=[0.45] + [1] * n_rows, hspace=0.55, wspace=0.35
    )

    model_name = MODEL_DISPLAY_NAMES.get(model_key, model_key)
    fig.suptitle(
        f"{model_name} -- machine={machine}, device={device}",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )

    for col, (title, labels, _) in enumerate(_PANELS):
        legend_ax = fig.add_subplot(gs[0, col])
        legend_ax.axis("off")
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=_COLORS[i]) for i in range(len(labels))
        ]
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
    for dtype in available_dtypes:
        doc = dtype_docs[dtype]
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
        if log:
            column_ylims.append((0.1, col_max * 3))
        else:
            column_ylims.append((0.0, col_max * 1.15))

    for row, dtype in enumerate(available_dtypes, start=1):
        for col, (_, labels, log) in enumerate(_PANELS):
            ax = fig.add_subplot(gs[row, col])
            values, fmts = row_values[row - 1][col]
            _render_panel(ax, labels, values, fmts, log, column_ylims[col])
            if col == 0:
                ax.set_ylabel(
                    DTYPE_LABELS.get(dtype, dtype),
                    fontsize=12,
                    fontweight="bold",
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{model_key}_{machine}_{device}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Wrote {out_path}")


def main() -> None:
    args = parse_args()
    reports = _load_reports(args.results_dir)
    if not reports:
        print(f"[plot] ERROR: no reports found in {args.results_dir}.")
        return
    grouped = _group_reports(reports)
    for (machine, device), by_model in grouped.items():
        for model_key in MODEL_DISPLAY_NAMES:
            dtype_docs = by_model.get(model_key, {})
            _plot_device_model(machine, device, model_key, dtype_docs, args.output_dir)


if __name__ == "__main__":
    main()
