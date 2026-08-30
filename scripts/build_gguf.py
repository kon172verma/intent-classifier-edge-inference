#!/usr/bin/env python3
"""Convert one merged Transformers checkpoint to requested GGUF variants.

The script does not infer model names or use the legacy flat model layout.
Callers supply the merged checkpoint and an empty output root explicitly.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SUPPORTED_VARIANTS = ("Q8_0", "Q6_K", "Q4_K_M")


def _require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    return path


def _resolve_binary(value: str) -> str:
    path = shutil.which(value)
    if path is None:
        raise FileNotFoundError(
            f"llama.cpp quantizer {value!r} is not on PATH. Install llama.cpp or pass --quantize."
        )
    return path


def build_gguf(
    *,
    input_dir: Path,
    output_root: Path,
    variants: list[str],
    converter: Path,
    quantize: str,
    python: str,
) -> list[list[str]]:
    """Build GGUF files in ``output_root/<variant>/model.gguf``.

    ``output_root`` must not exist. This makes the function suitable for a
    pipeline staging directory: a failed conversion cannot leave a partial
    reusable artifact in the final model directory.
    """
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Merged Transformers checkpoint does not exist: {input_dir}")
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite GGUF output root: {output_root}")
    unknown = sorted(set(variants) - set(_SUPPORTED_VARIANTS))
    if unknown:
        raise ValueError(f"Unsupported GGUF variant(s): {', '.join(unknown)}")

    converter = _require_file(converter, "llama.cpp conversion script")
    quantize_binary = _resolve_binary(quantize)
    output_root.mkdir(parents=True)
    commands: list[list[str]] = []
    with tempfile.TemporaryDirectory(prefix="benchmark-gguf-") as temp_dir:
        fp16_path = Path(temp_dir) / "model-f16.gguf"
        convert_command = [
            python,
            str(converter),
            str(input_dir),
            "--outfile",
            str(fp16_path),
            "--outtype",
            "f16",
        ]
        subprocess.run(convert_command, check=True)
        commands.append(convert_command)
        for variant in variants:
            destination = output_root / variant / "model.gguf"
            destination.parent.mkdir()
            quantize_command = [quantize_binary, str(fp16_path), str(destination), variant]
            subprocess.run(quantize_command, check=True)
            commands.append(quantize_command)
    return commands


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=_SUPPORTED_VARIANTS, required=True)
    parser.add_argument(
        "--converter",
        type=Path,
        default=_REPO_ROOT / "scripts" / "llama.cpp-src" / "convert_hf_to_gguf.py",
        help="Path to llama.cpp's convert_hf_to_gguf.py.",
    )
    parser.add_argument("--quantize", default="llama-quantize")
    parser.add_argument(
        "--python", default=sys.executable, help="Python with conversion dependencies."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        commands = build_gguf(
            input_dir=args.input_dir,
            output_root=args.output_root,
            variants=args.variants,
            converter=args.converter,
            quantize=args.quantize,
            python=args.python,
        )
    except (FileNotFoundError, FileExistsError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"build_gguf: error: {exc}", file=sys.stderr)
        return 2
    for command in commands:
        print("[gguf]", " ".join(command))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
