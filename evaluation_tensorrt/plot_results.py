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

_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation_lib.config import MODEL_DISPLAY_NAMES, TENSORRT_DTYPES
from evaluation_lib.plot_common import group_reports, load_reports, plot_device_model

_RESULTS_DIR = _REPO_ROOT / "evaluation_tensorrt" / "results"
_CHARTS_DIR = _RESULTS_DIR / "charts"

DTYPE_ORDER = TENSORRT_DTYPES
DTYPE_LABELS = {
    "fp16": "FP16",
    "bf16": "BF16",
    "int8": "INT8 (SmoothQuant)",
    "int4": "INT4 (AWQ)",
}


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


def main() -> None:
    args = parse_args()
    reports = load_reports(args.results_dir)
    if not reports:
        print(f"[plot] ERROR: no reports found in {args.results_dir}.")
        return
    grouped = group_reports(reports, "dtype")
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
                DTYPE_ORDER,
                DTYPE_LABELS,
                out_path,
                suptitle=f"{model_name} -- machine={machine}, device={device}",
            )


if __name__ == "__main__":
    main()
