#!/usr/bin/env python3
"""
Plot latency/throughput/quality charts from evaluation_onnx JSON reports.

Mirrors evaluation_llama_cpp/plot_results.py exactly, with "quant" replaced
by "precision" (ONNX precision: fp32/fp16/dynamic-int8/static-int8). For
every (machine, device) combination found in the results directory and for
every model, this produces one PNG with a grid of bar-chart panels:

    rows    = available precisions, in the order defined by
              evaluation_lib.config.ONNX_PRECISIONS (only precisions that
              actually have a matching result file are plotted; a
              device+model combination with zero matching files is skipped
              with an error message)
    columns = 3 metric panels, identical across all rows:
        1. Preprocessing / Total processing / TTFT   (3 bars, ms)
        2. System prompt / Tools list / User query / Decode  (4 bars, ms)
        3. Accuracy / Peak RAM / KV cache / Peak GPU  (4 bars, log scale)

All values are the *mean* aggregates from each run's JSON report. A shared
legend for each column is rendered once in a dedicated row at the top of the
figure so it never overlaps the chart area.

Peak GPU MB is always "N/A" here -- ONNX Runtime/CoreML exposes no cheap
live GPU-memory query API (see evaluation_onnx/inference.py).

Usage
------
    python evaluation_onnx/plot_results.py
    python evaluation_onnx/plot_results.py --reports-dir path/to/reports
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation_lib.config import MODEL_DISPLAY_NAMES, ONNX_PRECISIONS
from evaluation_lib.plot_common import group_reports, load_reports, plot_device_model

_REPORTS_DIR = _REPO_ROOT / "evaluation_onnx" / "reports"
_ANALYSIS_DIR = _REPO_ROOT / "evaluation_onnx" / "analysis"

PRECISION_ORDER = ONNX_PRECISIONS
PRECISION_LABELS = {
    "fp32": "FP32",
    "fp16": "FP16",
    "dynamic-int8": "Dynamic INT8",
    "static-int8": "Static INT8",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot charts from evaluation_onnx JSON reports",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--reports-dir",
        type=Path,
        default=_REPORTS_DIR,
        help="Directory containing run JSON reports",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_ANALYSIS_DIR,
        help="Directory to write PNG charts",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    reports = load_reports(args.reports_dir)
    if not reports:
        print(f"[plot] ERROR: no reports found in {args.reports_dir}.")
        return
    grouped = group_reports(reports, "precision")
    for (machine, device), by_model in grouped.items():
        for model_key in MODEL_DISPLAY_NAMES:
            variant_docs = by_model.get(model_key, {})
            model_name = MODEL_DISPLAY_NAMES.get(model_key, model_key)
            out_path = args.output_dir / f"{model_key}_{machine}_{device}.png"
            plot_device_model(
                machine,
                device,
                model_key,
                variant_docs,
                PRECISION_ORDER,
                PRECISION_LABELS,
                out_path,
                suptitle=f"{model_name} -- machine={machine}, device={device}",
            )


if __name__ == "__main__":
    main()
