# evaluation_onnx

ONNX Runtime benchmark for the intent-classifier inference project — mirrors
`evaluation_baseline/` and `evaluation_llama_cpp/`'s architecture and JSON
schema, but runs the model via
[ONNX Runtime](https://onnxruntime.ai/) instead of HF Transformers/PyTorch
or llama.cpp. Compares an unquantized ONNX export against dynamic and
static INT8 post-training quantization, across CPU and CoreML (Apple
Silicon ANE/GPU) execution providers.

## Package Structure

```text
evaluation_onnx/
├── run.py           # CLI entrypoint — runs the full benchmark loop
├── model_loader.py  # Loads an ONNX Runtime session for a given precision/device
├── inference.py     # Single inference pass with 3-phase prefill timing
├── cache.py         # Explicit numpy KV-cache management, prefix-cache pre-computation
├── plot_results.py  # Renders comparison charts from JSON reports
├── reports/         # JSON output files written by run.py
└── analysis/        # Charts written by plot_results.py
```

Shared utilities (prompt construction, quality/latency aggregation, output
parsing, system RAM measurement) live in `evaluation_lib/` and are reused
unmodified from `evaluation_baseline/`.

## One-time setup: export + quantize the models

ONNX export requires a pinned `transformers`/`torch` version that would
conflict with this project's main venv, so it's done in an isolated venv
(same pattern as `evaluation_llama_cpp`'s GGUF conversion). Quantization,
however, has no such dependency and runs fine in the **main** venv.

```bash
# 1. Isolated venv for ONNX export
python3 -m venv scripts/.venv-onnx
scripts/.venv-onnx/bin/pip install --upgrade pip
scripts/.venv-onnx/bin/pip install "optimum[onnxruntime]" onnxruntime

# 2. Stage Llama 3.2 with a legacy-compatible RoPE config
#
# Optimum exports through Transformers 4.57.x, which reads the older
# `rope_scaling` field. The release checkpoint was produced by Transformers 5
# and uses `rope_parameters`; staging converts that configuration (and the
# Transformers-5-only `TokenizersBackend` metadata) without modifying the
# original checkpoint. Qwen3 does not need this step.
python scripts/prepare_onnx_export_source.py --model llama3

# 3. Export each HF checkpoint to ONNX (FP32 base + FP16 variant)
mkdir -p models/qwen3-0.6b_LoRA_C_1k_merged/onnx
mkdir -p models/llama3.2-1b_LoRA_C_1k_merged/onnx
for dtype in fp32 fp16; do
  scripts/.venv-onnx/bin/optimum-cli export onnx \
    -m models/qwen3-0.6b_LoRA_C_1k_merged/safetensors \
      --task text-generation-with-past --dtype ${dtype} \
    models/qwen3-0.6b_LoRA_C_1k_merged/onnx/${dtype}
  scripts/.venv-onnx/bin/optimum-cli export onnx \
    -m models/_onnx_export_sources/llama3.2-1b_LoRA_C_1k_legacy_rope \
      --task text-generation-with-past --dtype ${dtype} \
    models/llama3.2-1b_LoRA_C_1k_merged/onnx/${dtype}
done

# 4. Quantize (dynamic-int8 + static-int8), using the MAIN venv
.venv/bin/python scripts/quantize_onnx.py --model qwen3
.venv/bin/python scripts/quantize_onnx.py --model llama3
```

This produces:

- `models/qwen3-0.6b_LoRA_C_1k_merged/onnx/{fp32,fp16,dynamic-int8,static-int8}/model.onnx`
- `models/llama3.2-1b_LoRA_C_1k_merged/onnx/{fp32,fp16,dynamic-int8,static-int8}/model.onnx`

## Python bindings

`onnxruntime` (with CoreML support built in on macOS) goes in the **main**
project venv:

```bash
pip install onnxruntime
```

## Precisions

| Precision | What's quantized | Calibration | Notes |
| ----------- | ------------------- | ------------- | ------- |
| `fp32` | Nothing (baseline) | N/A | Unquantized reference |
| `fp16` | Nothing (cast only) | N/A | CoreML/ANE-friendly; NOT recommended on plain CPU (no fast FP16 SIMD path on ARM/x86) |
| `dynamic-int8` | Weights only (INT8); activations quantized on-the-fly at runtime | None needed | Fast to produce, moderate speedup |
| `static-int8` | Weights AND activations (INT8) | Calibrated offline on real dataset prompts | Full speedup, needs the calibration step |

### Recommended device × precision matrix

| Device | Precisions |
| -------- | ------------ |
| Mac CPU | `fp32`, `dynamic-int8`, `static-int8` |
| Mac CoreML | `fp16` only -- see CoreML limitation below |
| Raspberry Pi CPU | `fp32`, `dynamic-int8`, `static-int8` |

**CoreML + INT8 limitation:** `dynamic-int8`/`static-int8` currently fail on
the CoreML execution provider with:

```text
Input (past_key_values.0.key) has a dynamic shape ({-1,8,-1,128}) but the
runtime shape ({1,8,0,128}) has zero elements. This is not supported by the
CoreML EP. FP16 uses a one-token CPU bootstrap to turn the empty cache into a
non-empty KV cache before CoreML takes over; this is a one-time prefix-cache
creation cost, not part of request-time inference.
```

This is a known ONNX Runtime CoreML EP limitation -- it cannot handle the
zero-length KV cache tensor needed for the very first (empty-cache) prefill
call. `fp16` doesn't hit this (evidently routed differently internally), so
CoreML is validated for `fp16` only; use `--device cpu` for INT8 precisions.

## Important: quantization node exclusions

Naively quantizing every `MatMul` node in a decoder LLM graph corrupts the
model. `scripts/quantize_onnx.py` excludes three categories (see its
`_non_weight_matmul_names()` docstring for the full reasoning, empirically
derived while building this package):

1. **Activation×activation matmuls** (attention Q@K^T / softmax@V, rotary
   embedding angles) — no constant weight operand, quantizing these breaks
   positional encoding and attention entirely (garbled output).
2. **`lm_head`** (final vocab projection) — small logit errors here directly
   flip the argmax decision; stays highly sensitive to INT8 even after (1).
3. **`mlp.down_proj`** (post-SwiGLU-activation projection, every layer) —
   the classic "activation outlier" layer from the LLM quantization
   literature (LLM.int8()/SmoothQuant). Required specifically for
   **static** (activation) quantization to recover tool-routing accuracy;
   dynamic (weights-only) quantization did not need this exclusion.

## Caching

Only `prefix_cache`-style caching is implemented: the system prompt's KV
cache is pre-computed once and cloned (not re-ingested) for every example.
ONNX Runtime's decoder-with-past graphs are stateless functions — every
`session.run()` call takes the *entire* KV cache as explicit input tensors
and returns the extended cache as output tensors (see `cache.py`); there is
no in-process "no cache" mode analogous to HF's `use_cache=False` beyond
just starting from an empty cache dict every call, which isn't a
particularly interesting benchmark axis here.

## Usage

```bash
python evaluation_onnx/run.py --model qwen3 --precision static-int8 --device cpu
python evaluation_onnx/run.py --model llama3 --precision fp16 --device coreml

python evaluation_onnx/plot_results.py
```

## Implementation notes

- **`session.run()` is synchronous** on both CPU and CoreML execution
  providers — unlike llama.cpp's async Metal backend, no explicit
  synchronization barrier is needed around timed calls (see `cache.py`).
- **Tokenization** uses the original HF tokenizer directly (both for
  chat-template rendering AND the actual token ids fed to the ONNX graph) —
  unlike `evaluation_llama_cpp`, there's no separate GGUF-embedded
  tokenizer, since the ONNX export preserves the exact same vocabulary.
- **KV cache** is threaded explicitly as a dict of numpy arrays between
  calls; `present.<N>.key/value` outputs are renamed to
  `past_key_values.<N>.key/value` and fed back in as the next call's cache
  input (see `cache.py`'s `_present_to_past()`).
- **Peak GPU memory** is always reported as `null` — neither ONNX Runtime
  nor CoreML expose a cheap live-memory query API equivalent to
  `torch.mps.current_allocated_memory()`.
- **CoreML execution** falls back to CPU per-op for anything the CoreML
  backend can't run (ORT's built-in provider-fallback mechanism); the
  `run_config.onnxruntime_providers` field in each result JSON records the
  actual active provider list for the session.
- **Llama-3.2 RoPE compatibility:** the Optimum export environment currently
  uses Transformers 4.57.x, while the release checkpoint uses the
  Transformers 5 `rope_parameters` configuration format. Transformers 4.57
  instead reads Llama's `rope_theta` and `rope_scaling` fields. Exporting the
  checkpoint directly therefore falls back to a theta of 10,000 instead of
  Llama 3.2's 500,000 and severely degrades routing accuracy even for FP32.
  Always run `scripts/prepare_onnx_export_source.py --model llama3` before
  exporting; it creates a symlinked staging directory whose copied
  `config.json` maps the same Llama 3 RoPE values to the legacy fields and
  makes the tokenizer metadata readable by the exporter. It does not alter
  the release checkpoint.
