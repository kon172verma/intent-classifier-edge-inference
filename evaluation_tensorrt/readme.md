# evaluation_tensorrt

TensorRT-LLM benchmark for the intent-classifier inference project — mirrors
`evaluation_baseline/`, `evaluation_llama_cpp/`, and `evaluation_onnx/`'s
architecture and JSON schema, but runs the model via
[NVIDIA TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/) for
NVIDIA GPU hardware (primary target: **Jetson Orin**, cloud A100/H100).

## Package structure

```text
evaluation_tensorrt/
├── run.py           # CLI entrypoint — runs the full benchmark loop
├── model_loader.py  # Loads the compiled TRT-LLM engine + HF tokenizer
├── inference.py     # 3-phase prefill timing + decode loop
├── plot_results.py  # Renders comparison charts from JSON reports
└── results/         # JSON output files written by run.py
```

Shared utilities (prompt construction, quality/latency aggregation, output
parsing) live in `evaluation_lib/` and are reused unmodified.

The build helper lives in `scripts/build_trt_engine.py`.

---

## Hardware targets

| Device | JetPack / CUDA | SM | Recommended dtype |
|---|---|---|---|
| Jetson Orin AGX 64 GB | JetPack 6.x (CUDA 12.2+) | SM87 | `fp16` (no BF16 hardware) |
| Jetson Orin NX / Nano | JetPack 6.x | SM87 | `fp16` |
| NVIDIA A100 | CUDA 12.x | SM80 | `bf16` or `fp16` |
| NVIDIA H100 | CUDA 12.x | SM90 | `bf16` or `fp16` |

> **Jetson note**: BF16 is NOT supported on SM87.  Use `fp16` or `int8`.

---

## One-time setup

### 1. Install TensorRT-LLM on the target device

Follow the official installation guide for your platform:

- **Jetson Orin (JetPack 6)**:
  <https://nvidia.github.io/TensorRT-LLM/installation/jetson.html>
- **Linux x86 (data-centre GPU)**:
  <https://nvidia.github.io/TensorRT-LLM/installation/linux.html>

TensorRT-LLM cannot be installed on macOS (no CUDA).  The code in this
package can be read and reviewed on any machine, but must be executed on a
device with a compatible NVIDIA GPU.

> Recommended: create a dedicated virtualenv for TRT-LLM separate from the
> main project venv (TRT-LLM pins specific versions of `torch`, `pydantic`,
> etc.).

### 2. Copy the HF checkpoints to the target device

```bash
scp -r models/intent-classifier-qwen3-0.6b_C_1k_merged  jetson:/path/to/project/models/
scp -r models/intent-classifier-llama3.2-1b_C_1k_merged  jetson:/path/to/project/models/
```

### 3. Build the TensorRT engine

The `scripts/build_trt_engine.py` helper automates the two-step pipeline:

```
HF SafeTensors  ──convert_checkpoint.py──>  TRT-LLM checkpoint  ──trtllm-build──>  .engine
```

```bash
# Activate the TRT-LLM venv on the target device, then:

# Qwen3-0.6B, FP16 (Jetson primary)
python scripts/build_trt_engine.py --model qwen3 --dtype fp16

# Llama-3.2-1B, FP16
python scripts/build_trt_engine.py --model llama3 --dtype fp16

# INT8 SmoothQuant (W8A8)
python scripts/build_trt_engine.py --model qwen3 --dtype int8

# INT4 AWQ weight-only (W4A16)
python scripts/build_trt_engine.py --model qwen3 --dtype int4
```

Engines are written to `models/tensorrt/<stem>-<dtype>/`.  The build step
is hardware-specific: an engine built for SM87 (Jetson Orin) cannot run on
SM80 (A100).

---

## Dtypes

| `--dtype` | Weights | Activations | Method | Notes |
|---|---|---|---|---|
| `fp16` | FP16 | FP16 | — | **Jetson primary**; fastest safe option |
| `bf16` | BF16 | BF16 | — | Data-centre only; not available on SM87 |
| `int8` | INT8 | INT8 | SmoothQuant W8A8 | Requires `--smoothquant 0.5` at convert step |
| `int4` | INT4 | FP16 | AWQ W4A16 | Smallest model; quality penalty possible |

---

## Running the benchmark

```bash
# Activate the TRT-LLM venv on the target device, then:
python evaluation_tensorrt/run.py --model qwen3 --dtype fp16 --machine jetson
python evaluation_tensorrt/run.py --model llama3 --dtype fp16 --machine jetson
python evaluation_tensorrt/run.py --model qwen3 --dtype int8 --machine jetson
```

Results are written to `evaluation_tensorrt/results/`.

### CLI options

| Flag | Default | Description |
|---|---|---|
| `--model` | required | `qwen3` or `llama3` |
| `--dtype` | required | `fp16 \| bf16 \| int8 \| int4` |
| `--machine` | hostname | Label for the physical device (e.g. `jetson`) |
| `--dataset` | `dataset_full/sample_0001.json` | Dataset file |
| `--warmup` | 2 | Examples excluded from measurements |
| `--max-new-tokens` | 32 | Max tokens to generate per example |
| `--output-dir` | `evaluation_tensorrt/results/` | Where to write JSON |

---

## Output JSON schema

Identical to `evaluation_baseline/`, `evaluation_llama_cpp/`, and
`evaluation_onnx/` for cross-package `plot_results.py` comparisons:

```json
{
  "run_config": { "model_key", "dtype", "machine", "device": "cuda", ... },
  "aggregate":  { "mean_ttft_ms", "mean_decode_tok_per_sec", ... },
  "quality":    { "tool_accuracy", "invalid_tool_rate", ... },
  "per_example": [ { "id", "correct", "ttft_ms", "e2e_latency_ms", ... } ]
}
```

Key differences from `evaluation_onnx/`:
- `run_config.dtype` instead of `precision`
- `run_config.device` is always `"cuda"`
- `run_config.tensorrt_llm_version` instead of `onnxruntime_version`
- `per_example.kv_cache_kb` is `null` (TRT-LLM manages KV cache internally)

---

## Plotting charts

```bash
python evaluation_tensorrt/plot_results.py
```

Reads all JSON files from `evaluation_tensorrt/results/` and writes PNG
charts to `evaluation_tensorrt/results/charts/`.

---

## Known limitations

1. **No macOS support** — TensorRT-LLM requires CUDA.  The package can be
   read and developed on macOS but cannot be executed without an NVIDIA GPU.

2. **BF16 not available on Jetson Orin** — SM87 has no BF16 hardware unit.
   Use `fp16` or `int8` on Jetson.

3. **Engine is hardware-specific** — A compiled `.engine` file for SM87
   (Jetson Orin) will not run on SM80 (A100) and vice versa.  Rebuild when
   moving between GPU generations.

4. **KV cache size not reported** — TRT-LLM manages paged KV cache
   internally and does not expose a per-request size API.
   `kv_cache_kb` is `null` in all output records.

5. **TTFT measurement approximation** — TRT-LLM's `ModelRunner.generate()`
   is not designed for fine-grained token-level streaming timing.  TTFT is
   measured by issuing a separate `generate(max_new_tokens=1)` call for the
   user-query prefix — this adds a small overhead versus a native
   token-callback approach.  Relative comparisons across dtypes on the same
   hardware remain valid.
