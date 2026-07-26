#!/usr/bin/env python3
"""
Quantize an FP32 ONNX export (see evaluation_onnx/readme.md for the export
step) to dynamic-INT8 and static-INT8 variants.

Run with the MAIN project venv (needs onnxruntime + transformers; unlike the
export step this does NOT need the isolated scripts/.venv-onnx, since
quantize_dynamic/quantize_static have no dependency on a pinned transformers
version):

    .venv/bin/python scripts/quantize_onnx.py --model qwen3
    .venv/bin/python scripts/quantize_onnx.py --model llama3

Produces
--------
    models/onnx/<stem>-dynamic-int8/model.onnx
    models/onnx/<stem>-static-int8/model.onnx

Static quantization calibration
--------------------------------
Static INT8 requires representative activation ranges. This script
calibrates using real (system prompt + tools list + user query) prompts from
the default dataset, shaped as **prefill-phase** inputs (full sequence in one
forward pass, empty KV cache) since that is the dominant compute cost. The
single-token decode-step forward passes (non-empty KV cache, seq_len=1) reuse
these same calibrated activation ranges -- a standard simplification for
static-quantizing decoder-with-past graphs that avoids having to calibrate
every possible KV cache length.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantType,
    quantize_dynamic,
    quantize_static,
)

_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation_lib.config import DATASET_DEFAULT, MODEL_PATHS
from evaluation_lib.prompt import build_full_prompt

ONNX_DIR = _REPO_ROOT / "models" / "onnx"
ONNX_STEMS = {"qwen3": "qwen3-0.6b", "llama3": "llama3.2-1b"}

N_CALIBRATION_SAMPLES = 32


def _non_weight_matmul_names(model_path: Path) -> list[str]:
    """Return MatMul node names that should be excluded from quantization.

    Two categories are excluded:

    1. Activation-times-activation matmuls -- e.g. each layer's
       attention-score (Q@K^T) and context (softmax@V) matmuls, and the
       rotary-embedding angle computation -- as opposed to genuine
       weight-bearing Linear projections (q_proj/k_proj/v_proj/o_proj/
       gate_proj/up_proj/down_proj, each MatMul(activation,
       weight_initializer)). Quantizing these to INT8 corrupts positional
       encoding and attention scores directly (there is no stable
       per-weight scale to calibrate -- both operands are dynamic
       activations); empirically this collapsed output to garbled tokens.

    2. ``lm_head`` -- the final vocab-projection layer. Empirically this
       remained highly sensitive to INT8 static quantization even after (1)
       was fixed: outputs became coherent English but lost tool-routing
       accuracy (over-verbose/hallucinated completions instead of the
       expected short tool name). Keeping it at full precision is a common
       practice for quantizing decoder LMs, since small logit errors here
       directly flip the argmax decision.

    3. ``mlp.down_proj`` in every layer -- the post-SwiGLU-activation
       projection. This is the layer most associated with the well-known
       "activation outlier" problem in LLM quantization literature (e.g.
       LLM.int8()/SmoothQuant): a small number of extreme-magnitude
       activation channels after the SiLU-gated elementwise product make a
       single per-tensor/per-channel INT8 scale a poor fit for the whole
       tensor. Empirically excluding this (in addition to (1) and (2)) was
       required to recover tool-routing accuracy with static (activation)
       quantization -- q/k/v/o_proj and gate_proj/up_proj quantized fine on
       their own.
    """
    model = onnx.load(str(model_path), load_external_data=False)
    initializer_names = {i.name for i in model.graph.initializer}
    excluded = [
        n.name
        for n in model.graph.node
        if n.op_type == "MatMul"
        and not any(inp in initializer_names for inp in n.input)
    ]
    excluded += [
        n.name
        for n in model.graph.node
        if n.op_type == "MatMul"
        and (n.name == "/lm_head/MatMul" or "down_proj" in n.name)
    ]
    return excluded


def _past_kv_input_names(model_path: Path) -> list[tuple[str, list[int]]]:
    """Introspect the ONNX graph for ``past_key_values.*`` input specs.

    Returns a list of ``(input_name, shape)`` pairs where dynamic dims
    (batch_size, past_sequence_length) are represented as ``0``.
    """
    model = onnx.load(str(model_path), load_external_data=False)
    specs = []
    for inp in model.graph.input:
        if not inp.name.startswith("past_key_values."):
            continue
        shape = [max(0, d.dim_value) for d in inp.type.tensor_type.shape.dim]
        specs.append((inp.name, shape))
    return specs


class PrefillCalibrationReader(CalibrationDataReader):
    """Feeds real dataset prompts shaped as prefill-phase (empty-cache) inputs.

    All samples are padded/truncated to the same fixed sequence length.
    Percentile/Entropy calibration (unlike MinMax) stacks every collected
    tensor into a single numpy array across all calibration samples, which
    requires identical shapes -- real prompts have varying token counts, so
    padding to a fixed length is required or ``quantize_static`` raises
    ``ValueError: setting an array element with a sequence`` (inhomogeneous
    shape) when using anything other than MinMax.
    """

    def __init__(self, model_key: str, model_path: Path, n_samples: int) -> None:
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            str(MODEL_PATHS[model_key]), clean_up_tokenization_spaces=False
        )
        with open(DATASET_DEFAULT) as f:
            dataset = json.load(f)
        self._prompts = [
            build_full_prompt(
                self._tokenizer, ex["user_request"], ex["available_tools"]
            )
            for ex in dataset[:n_samples]
        ]
        self._past_specs = _past_kv_input_names(model_path)
        self._idx = 0

        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        all_ids = [
            self._tokenizer(text, return_tensors="np").input_ids
            for text in self._prompts
        ]
        self._calib_seq_len = max(ids.shape[1] for ids in all_ids)

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._idx >= len(self._prompts):
            return None
        text = self._prompts[self._idx]
        self._idx += 1

        ids = self._tokenizer(text, return_tensors="np").input_ids.astype(np.int64)
        real_len = ids.shape[1]
        pad_len = self._calib_seq_len - real_len
        if pad_len > 0:
            pad_id = self._tokenizer.pad_token_id
            ids = np.concatenate(
                [ids, np.full((1, pad_len), pad_id, dtype=np.int64)], axis=1
            )
        seq_len = ids.shape[1]
        attention_mask = np.ones((1, seq_len), dtype=np.int64)
        if pad_len > 0:
            attention_mask[:, real_len:] = 0
        feed: dict[str, np.ndarray] = {
            "input_ids": ids,
            "attention_mask": attention_mask,
            "position_ids": np.arange(seq_len, dtype=np.int64)[None, :],
        }
        for name, shape in self._past_specs:
            # Empty KV cache: batch_size=1, past_sequence_length=0.
            concrete = [1 if i == 0 else max(0, d) for i, d in enumerate(shape)]
            feed[name] = np.zeros(concrete, dtype=np.float32)
        return feed


def quantize_model(model_key: str) -> None:
    stem = ONNX_STEMS[model_key]
    model_in = ONNX_DIR / f"{stem}-fp32" / "model.onnx"
    if not model_in.exists():
        raise FileNotFoundError(
            f"{model_in} not found -- export the FP32 ONNX model first "
            f"(see evaluation_onnx/readme.md)."
        )

    dyn_dir = ONNX_DIR / f"{stem}-dynamic-int8"
    dyn_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{model_key}] Dynamic INT8 quantization -> {dyn_dir}")
    exclude_nodes = _non_weight_matmul_names(model_in)
    quantize_dynamic(
        model_input=str(model_in),
        model_output=str(dyn_dir / "model.onnx"),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul"],
        nodes_to_exclude=exclude_nodes,
        per_channel=True,
        # The FP32 base model stores weights in an external .onnx_data file
        # (>2GB total); the quantized graph exceeds protobuf's 2GB
        # single-message limit without this.
        use_external_data_format=True,
    )

    static_dir = ONNX_DIR / f"{stem}-static-int8"
    static_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[{model_key}] Static INT8 quantization "
        f"(calibrating on {N_CALIBRATION_SAMPLES} samples) -> {static_dir}"
    )
    reader = PrefillCalibrationReader(model_key, model_in, N_CALIBRATION_SAMPLES)
    quantize_static(
        model_input=str(model_in),
        model_output=str(static_dir / "model.onnx"),
        calibration_data_reader=reader,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul"],
        nodes_to_exclude=exclude_nodes,
        per_channel=True,
        # The FP32 base model stores weights in an external .onnx_data file
        # (>2GB total); the calibrator's in-memory augmented graph exceeds
        # protobuf's 2GB single-message limit without this.
        use_external_data_format=True,
    )
    print(f"[{model_key}] Done.\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=list(ONNX_STEMS), required=True)
    args = p.parse_args()
    quantize_model(args.model)


if __name__ == "__main__":
    main()
