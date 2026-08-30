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
    python evaluation_llama_cpp/plot_results.py --results-dir path/to/results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation_lib.config import MODEL_DISPLAY_NAMES, QUANT_LEVELS
from evaluation_lib.plot_common import group_reports, load_reports, plot_device_model

_RESULTS_DIR = _REPO_ROOT / "evaluation_llama_cpp" / "results"
_CHARTS_DIR = _RESULTS_DIR / "charts"

QUANT_ORDER = QUANT_LEVELS
QUANT_LABELS = {"Q8_0": "Q8_0", "Q6_K": "Q6_K", "Q4_K_M": "Q4_K_M"}


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
    return p.parse_args()


def main() -> None:
    args = parse_args()
    mode = "prefix_cache"
    reports = load_reports(args.results_dir, mode=mode)
    if not reports:
        print(f"[plot] ERROR: no reports found for mode={mode} in {args.results_dir}.")
        return
    grouped = group_reports(reports, "quant")
    for (machine, device), by_model in grouped.items():
        for model_key in MODEL_DISPLAY_NAMES:
            variant_docs = by_model.get(model_key, {})
            model_name = MODEL_DISPLAY_NAMES.get(model_key, model_key)
            out_path = args.output_dir / f"{model_key}_{machine}_{device}_{mode}.png"
            plot_device_model(
                machine,
                device,
                model_key,
                variant_docs,
                QUANT_ORDER,
                QUANT_LABELS,
                out_path,
                suptitle=f"{model_name} -- machine={machine}, device={device}, mode={mode}",
            )


if __name__ == "__main__":
    main()
