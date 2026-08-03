#!/usr/bin/env python3
"""
Plot latency/throughput/quality charts from evaluation_baseline JSON reports.

For every (machine, device) combination found in the results directory (e.g.
"rpi+cpu", "mac+cpu", "mac+mps") and for every model, this produces one PNG
with a grid of bar-chart panels:

    rows    = available dtypes, in order float32 -> float16 -> bfloat16
              (only dtypes that actually have a matching result file are
              plotted; a device+model combination with zero matching files
              is skipped with an error message)
    columns = 3 metric panels, identical across all rows:
        1. Preprocessing / Total processing / TTFT   (3 bars, ms)
        2. System prompt / Tools list / User query / Decode  (4 bars, ms)
        3. Accuracy / Peak RAM / KV cache / Peak GPU  (4 bars, log scale)

All values are the *mean* aggregates from each run's JSON report. A shared
legend for each column is rendered once in a dedicated row at the top of the
figure so it never overlaps the chart area.

Only reports produced with ``--mode kv_cache`` or ``--mode prefix_cache`` have
the system-prompt/tools-list/user-query prefill split needed for columns 1-2
(see evaluation_baseline/run.py); "no_cache" reports are not usable here.

Usage
------
    python evaluation_baseline/plot_results.py
    python evaluation_baseline/plot_results.py --mode kv_cache
    python evaluation_baseline/plot_results.py --results-dir path/to/results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation_lib.config import MODEL_DISPLAY_NAMES
from evaluation_lib.plot_common import group_reports, load_reports, plot_device_model

_RESULTS_DIR = _REPO_ROOT / "evaluation_baseline" / "results"
_CHARTS_DIR = _RESULTS_DIR / "charts"

DTYPE_ORDER = ["float32", "float16", "bfloat16"]
DTYPE_LABELS = {"float32": "fp32", "float16": "fp16", "bfloat16": "bf16"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot charts from evaluation_baseline JSON reports",
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


def main() -> None:
    args = parse_args()
    reports = load_reports(args.results_dir, mode=args.mode)
    if not reports:
        print(f"[plot] ERROR: no reports found for mode={args.mode} in {args.results_dir}.")
        return
    grouped = group_reports(reports, "dtype")
    for (machine, device), by_model in grouped.items():
        for model_key in MODEL_DISPLAY_NAMES:
            variant_docs = by_model.get(model_key, {})
            model_name = MODEL_DISPLAY_NAMES.get(model_key, model_key)
            out_path = args.output_dir / f"{model_key}_{machine}_{device}_{args.mode}.png"
            plot_device_model(
                machine,
                device,
                model_key,
                variant_docs,
                DTYPE_ORDER,
                DTYPE_LABELS,
                out_path,
                suptitle=f"{model_name} -- machine={machine}, device={device}, mode={args.mode}",
            )


if __name__ == "__main__":
    main()
