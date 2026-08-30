#!/usr/bin/env python3
"""Prepare deterministic inference benchmark data splits from ``dataset_full``.

The output files are JSON arrays, matching the input format consumed by the
evaluation runners.  This script deliberately follows the training split
contract while adding a small, train-only static-quantization calibration set.

Split rules
-----------
1k mode (first 10 source files):

* train: ``sample_0002`` through ``sample_0009`` (800 examples)
* val: ``sample_0010`` (100 examples)
* test: ``sample_0001`` (100 examples)
* test_anchor: ``sample_0001`` (100 examples)
* calibration: ``sample_0002`` (100 examples; subset of train)

10k mode (all 100 source files):

* train: ``sample_0002`` through ``sample_0081`` (8,000 examples)
* val: ``sample_0082`` through ``sample_0091`` (1,000 examples)
* test: ``sample_0001`` plus ``sample_0092`` through ``sample_0100``
  (1,000 examples)
* test_anchor: ``sample_0001`` (100 examples)
* calibration: ``sample_0002`` (100 examples; subset of train)

``test_anchor`` is the only evaluation split intended for the edge benchmark
matrix. ``test`` is generated for one final-selection validation after a
device/engine/artifact variant has been selected; it is not scheduled for the
matrix sweep. ``calibration`` must never overlap with validation or either test
split.

Usage
-----
    python scripts/prepare_benchmark_splits.py --dataset-size 1k
    python scripts/prepare_benchmark_splits.py --dataset-size 10k
    python scripts/prepare_benchmark_splits.py --dataset-size 1k \
        --out-dir /path/to/benchmark_data
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "dataset_full"
DEFAULT_OUT_DIR = REPO_ROOT / "benchmark_data"


def sample_path(data_dir: Path, number: int) -> Path:
    """Return the path for the 1-based ``sample_XXXX.json`` source file."""
    return data_dir / f"sample_{number:04d}.json"


def sample_names(numbers: Iterable[int]) -> list[str]:
    """Return deterministic source-file names for a sequence of sample IDs."""
    return [f"sample_{number:04d}.json" for number in numbers]


def load_files(data_dir: Path, numbers: Iterable[int]) -> list[dict[str, Any]]:
    """Load and concatenate source JSON-array files, validating their shape."""
    examples: list[dict[str, Any]] = []
    for number in numbers:
        path = sample_path(data_dir, number)
        if not path.is_file():
            raise FileNotFoundError(f"Required source data file is missing: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list) or not all(isinstance(example, dict) for example in data):
            raise ValueError(f"Expected {path} to contain a JSON array of objects")
        examples.extend(data)
    return examples


def split_numbers(dataset_size: str) -> dict[str, list[int]]:
    """Return source-file indices for the stable 1k or 10k split contract."""
    anchor = [1]
    if dataset_size == "1k":
        train = list(range(2, 10))
        return {
            "train": train,
            "val": [10],
            "test": anchor,
            "test_anchor": anchor,
            "calibration": [2],
        }
    if dataset_size == "10k":
        train = list(range(2, 82))
        return {
            "train": train,
            "val": list(range(82, 92)),
            "test": anchor + list(range(92, 101)),
            "test_anchor": anchor,
            "calibration": [2],
        }
    raise ValueError(f"Unsupported dataset size: {dataset_size}")


def validate_split_contract(numbers_by_split: dict[str, list[int]]) -> None:
    """Ensure calibration is train-only and evaluation splits do not leak."""
    train = set(numbers_by_split["train"])
    val = set(numbers_by_split["val"])
    test = set(numbers_by_split["test"])
    anchor = set(numbers_by_split["test_anchor"])
    calibration = set(numbers_by_split["calibration"])

    if not calibration <= train:
        raise ValueError("Calibration files must be a subset of train files")
    if calibration & (val | test | anchor):
        raise ValueError("Calibration files must not overlap with val, test, or test_anchor")
    if not anchor <= test:
        raise ValueError("test_anchor files must be included in test")
    if train & (val | test) or val & test:
        raise ValueError("Train, validation, and test files must not overlap")


def write_json(examples: list[dict[str, Any]], output_path: Path) -> None:
    """Write one inference-compatible JSON array split."""
    output_path.write_text(
        json.dumps(examples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"  wrote {len(examples):>5} examples -> {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset-size",
        choices=["1k", "10k"],
        default="1k",
        help="1k uses source files 0001-0010; 10k uses 0001-0100.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing sample_XXXX.json source files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output root; split files are written to <out-dir>/<dataset-size>/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.out_dir.resolve() / args.dataset_size
    numbers_by_split = split_numbers(args.dataset_size)
    validate_split_contract(numbers_by_split)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Dataset size: {args.dataset_size}")
    print(f"Source dir:   {data_dir}")
    print(f"Output dir:   {output_dir}")

    split_counts: dict[str, int] = {}
    for split_name, numbers in numbers_by_split.items():
        examples = load_files(data_dir, numbers)
        write_json(examples, output_dir / f"{split_name}.json")
        split_counts[split_name] = len(examples)

    contract = {
        "dataset_size": args.dataset_size,
        "source_dir": str(data_dir),
        "format": "json-array",
        "edge_benchmark_evaluation_split": "test_anchor",
        "splits": {
            split_name: {
                "source_files": sample_names(numbers),
                "examples": split_counts[split_name],
            }
            for split_name, numbers in numbers_by_split.items()
        },
    }
    contract_path = output_dir / "split_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    print(f"  wrote split contract -> {contract_path}")


if __name__ == "__main__":
    main()
