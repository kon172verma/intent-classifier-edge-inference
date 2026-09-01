# Edge LLM Inference Plan

## Related Repositories

- **Training** — [kon172verma/intent-classifier](https://github.com/kon172verma/intent-classifier) (GitHub): fine-tuning code, configs, and reports.
- **Experiments** — [kon172verma/intent-classifier-experiments](https://huggingface.co/kon172verma/intent-classifier-experiments) (Hugging Face): every adapter produced during experimentation, organized by version.
- **Releases** — the [Hugging Face intent-classifier repository](https://huggingface.co/kon172verma/intent-classifier), with one versioned folder per
  fine-tuned model containing merged Transformers weights plus GGUF and ONNX
  exports.

## Purpose

This repository exists to evaluate inference for fine-tuned small language
models on edge hardware.

Primary target devices:

- Apple Silicon (development and reference environment)
- Raspberry Pi 5
- Qualcomm edge/mobile hardware
- NVIDIA Jetson Xavier

Primary target models:

- Qwen3 0.6B
- Llama 3.2 1B

## Model Source and Artifact Flow

Fine-tuning happens outside this repository.

- [kon172verma/intent-classifier](https://github.com/kon172verma/intent-classifier) (GitHub) is used for fine-tuning code and workflows.
- [kon172verma/intent-classifier-experiments](https://huggingface.co/kon172verma/intent-classifier-experiments) (Hugging Face) holds every adapter produced during experimentation, versioned by folder.
- The [Hugging Face release repository](https://huggingface.co/kon172verma/intent-classifier)
  holds version/model-scoped folders containing merged model weights and
  deployable artifacts. For example, `v1.0-qwen3-0.6b/` is the Qwen3 0.6B
  v1.0 release.

Merging and unloading adapters is handled by the manifest-driven benchmark
pipeline. `release_scripts/` only publishes completed release artifacts.

Inference flow in this repository:

1. Pull the merged release model (Transformers format) from its release
   folder.
2. Produce deployable artifacts for runtime comparison.
3. Publish completed artifacts with
   `release_scripts/release.py --version <version> --models all --execute`.

Artifact targets:

- Hugging Face Transformers (reference path)
- GGUF for llama.cpp
- ONNX for ONNX Runtime
- TensorRT-LLM path for Jetson Orin-or-newer and supported cloud GPUs
- vLLM where a suitable GPU-backed environment is available

## Manifest Pipeline

The pipeline resolves version manifests without side effects by default. It
validates a version manifest and prints the exact selected models, required
artifacts, supported engine variants, cache modes, and dataset inputs. It does
not download, merge, build, evaluate, or plot:

```bash
python scripts/prepare_benchmark_splits.py --dataset-size 10k

python -m benchmark_pipeline \
  --manifest manifests/v2.1.json \
  --target rpi --compute cpu \
  --models all \
  --stages all
```

Use exact manifest names for a subset, for example
`--models Qwen3-0.6B SmolLM2-360M`. Add `--json` for machine-readable output.

Phase 2 can fetch the pinned base-model and adapter snapshots, then merge the
adapter locally. This is deliberately opt-in because it downloads model
weights. It writes only to the versioned `models/` layout and refuses to reuse
a directory whose provenance does not match the manifest.

```bash
python -m benchmark_pipeline \
  --manifest manifests/v2.1.json \
  --target rpi --compute cpu \
  --models Qwen3-0.6B \
  --stages fetch merge \
  --execute
```

Set `HF_TOKEN` (or `HUGGINGFACE_TOKEN`) before selecting gated models such as
Llama. Phase 3 adds `build-artifacts` for GGUF and ONNX variants:

```bash
BENCHMARK_GGUF_PYTHON=scripts/.venv-convert/bin/python \
BENCHMARK_ONNX_OPTIMUM_CLI=scripts/.venv-onnx/bin/optimum-cli \
python -m benchmark_pipeline \
  --manifest manifests/v2.1.json \
  --target rpi --compute cpu \
  --models Qwen3-0.6B \
  --stages build-artifacts \
  --execute
```

The command requires a prior Phase 2 merged checkpoint. Static INT8 uses only
the manifest's `calibration.json`; it does not use `test_anchor` or full test
data. `evaluate` creates an isolated
`run_results/<version>_<target>_<compute>_<timestamp>/`
workspace and locks the report index before launching any evaluator; `plot`
uses only those indexed reports. A partial run can be resumed with `--run-dir`.

```bash
python -m benchmark_pipeline \
  --manifest manifests/v2.1.json \
  --target rpi --compute cpu \
  --models Qwen3-0.6B \
  --stages evaluate plot \
  --execute
```

TensorRT-LLM is excluded from the Jetson Xavier profile; bare TensorRT is a
separate future integration.

### Mac runtime smoke checks

After the relevant artifacts have been fetched, merged, and built, run the
short Mac checks below. They execute only the first three `test_anchor`
examples (one warm-up plus two measured examples), and the reports are marked
`benchmark_scope: "smoke"`; they are not candidate-selection results.

```bash
# Mac CPU: verifies the CPU runtimes and artifacts.
python -m benchmark_pipeline \
  --manifest manifests/v2.1.json \
  --target mac --compute cpu \
  --models Qwen3-0.6B \
  --stages evaluate plot --smoke --execute

# Mac GPU: verifies MPS, Metal llama.cpp, and CoreML ONNX Runtime.
python -m benchmark_pipeline \
  --manifest manifests/v2.1.json \
  --target mac --compute gpu \
  --models Qwen3-0.6B \
  --stages evaluate plot --smoke --execute
```

`--smoke` rejects RPi and Jetson targets by design. Those environments will be
verified manually once their hardware and runtimes are installed.

## Core Question

How much optimization is possible for tool-routing style inference on edge devices
when varying runtime, quantization, and caching strategy while keeping model and
workload fixed?

## Scope

In scope for this repository:

- Cross-device inference benchmarking
- Runtime comparison across device-compatible stacks
- Quantization evaluation
- KV cache and prompt prefix caching evaluation
- Accuracy versus latency and memory trade-off analysis

Out of scope for now:

- Architectural model changes such as MHA to GQA conversion
- Retraining and fine-tuning workflows
- Broad framework exploration beyond the primary runtime set

vLLM is worth keeping in the discussion, but it should be treated as a secondary
benchmark path rather than the first edge-focused target.

## Runtime Matrix

Planned runtime focus by platform:

| Device | Preferred runtimes |
| --- | --- |
| Apple Silicon | Transformers (MPS), llama.cpp (Metal), ONNX Runtime (CPU/CoreML EP) |
| Raspberry Pi 5 | llama.cpp (ARM CPU), ONNX Runtime (CPU) |
| Qualcomm | ONNX Runtime (CPU first, QNN when available) |
| Jetson Xavier | llama.cpp (CPU and CUDA offload), ONNX Runtime (CPU/CUDA/TensorRT EP), bare TensorRT |

TensorRT-LLM is a separate Jetson **Orin-or-newer** / data-centre path, not a
primary Xavier target: current upstream TensorRT-LLM support begins with
Ampere-class GPUs, while Xavier is Volta (SM72).  A bare TensorRT engine may
still be evaluated on Xavier, but it needs a distinct harness from the
TensorRT-LLM package in this repository.

Note: exact support depends on OS, SDK, driver, and runtime versions on each
physical device.

The device-by-runtime benchmark set and the kernel-based rationale are kept in
[devices_info.md](devices_info.md).  This table is the source of truth for
which variants are required, diagnostic-only, or intentionally omitted.

## Benchmark Method

### Caching Experiments

Benchmark prompt prefix caching for static context. The pipeline does not
schedule `kv_cache` or `no_cache` comparisons.

Workload pattern to emulate:

- Static: system prompt plus tool definitions
- Dynamic: user request
- Short output: usually a tool label

This makes TTFT and prefill behavior more important than long-form generation
throughput.

### Quantization Experiments

For GGUF and any other supported path, compare precision levels that are practical
for the device. At minimum, compare higher precision against several low-bit
variants and record quality impact.

## Metrics

Track the following for every test run:

- Time to first token
- Prefill latency
- Decode latency
- End-to-end latency
- Prefill tokens per second
- Decode tokens per second
- Peak memory usage (RAM and VRAM when relevant)
- KV cache memory footprint

Quality metrics for tool-routing:

- Tool-selection accuracy
- Exact-match rate
- Invalid tool rate
- None classification accuracy

A faster run is not a better run if accuracy degrades beyond acceptable limits.

## Execution Phases

### Readability and Onboarding Note

For initial understanding and faster onboarding, llama.cpp plus GGUF is easier
to reason about than ONNX plus ONNX Runtime.

Why llama.cpp plus GGUF is easier:

- One main artifact format and one primary runtime path
- Simpler mental model for quantization and local execution
- Fewer provider-specific configuration branches

Why ONNX plus ONNX Runtime feels harder at first:

- Requires export correctness and operator compatibility checks
- Behavior can differ by execution provider and device backend
- More moving parts for debugging across CPU, CUDA, TensorRT, and QNN

Practical guidance for this repository:

- Start with llama.cpp plus GGUF for baseline clarity and quick iteration.
- Add ONNX plus ONNX Runtime next for portability and hardware-provider testing.

### Phase 1: Apple Silicon Reference

- Download merged release models and verify parity with the reference reports.
- Build a stable benchmark harness.
- Record baseline metrics with Transformers.

### Phase 2: Caching Validation

- Add controlled prompt prefix caching experiments.
- Sweep static prefix size and compare with no-prefix-cache runs.

### Phase 3: GGUF Path

- Convert merged models to GGUF.
- Benchmark llama.cpp on Apple Silicon.
- Repeat on Raspberry Pi 5.

### Phase 4: ONNX Path

- Export to ONNX and benchmark ONNX Runtime on Apple Silicon.
- Run ONNX Runtime on Qualcomm CPU.
- Add QNN execution provider tests where available.

### Phase 5: Jetson Optimization

- Benchmark llama.cpp CPU and GPU offload modes.
- On Xavier, evaluate ONNX Runtime CUDA/TensorRT EP and bare TensorRT before
  considering a TensorRT-LLM port.
- Treat TensorRT-LLM as an Orin-or-newer / cloud path, subject to its support
  matrix and model-converter compatibility.
- Compare Jetson-specific acceleration against generic runtimes.

### Phase 6: Cross-Device Comparison

- Consolidate all measurements into common reporting format.
- Compare latency, memory, throughput, and quality across devices.
- Produce recommendation by deployment profile, not by a single global winner.

## Benchmark Dataset Requirements

Use a fixed and versioned dataset for tool-routing evaluation. Include:

- Straightforward tool calls
- Ambiguous user requests
- No-tool requests mapped to none
- Short and long tool lists
- Short and medium user inputs

Use the same dataset across all devices and runtime configurations.

## Dataset Layout and Split Contract

Current dataset folders:

- dataset_full
- dataset_sample

File equivalence used for testing continuity:

- dataset_full/sample_0001.json is exactly the same as
  dataset_sample/sample.json

Split that was use for fine-tuning and evaluation:

| Split | Files | Examples | Edge pipeline use |
| --- | --- | --- | --- |
| train | sample_0002.json to sample_0009.json | 800 | No |
| val | sample_0010.json | 100 | No |
| test | sample_0001.json | 100 | No |
| test_anchor | sample_0001.json | 100 | **Yes** |
| calibration | sample_0002.json (train-only) | 100 | Static quantization only |

Notes:

- test_anchor intentionally matches test in the current 1k setup.
- Edge matrix benchmarking always uses the 100-example test_anchor split.
  After selecting a final device/engine/variant configuration (for example,
  Raspberry Pi + llama.cpp + Q4_K_M), run its full version-specific test split
  once as final-selection validation; do not repeat that full test across the
  matrix.
- Static quantization calibration uses the small, deterministic train-only
  calibration split and must never use test_anchor or test examples.
- Generate inference-compatible JSON split files with:

  ```bash
  python scripts/prepare_benchmark_splits.py --dataset-size 1k
  python scripts/prepare_benchmark_splits.py --dataset-size 10k
  ```

## Reporting Format

Each benchmark report should include:

- Device and software stack details
- Model artifact and precision
- Runtime configuration
- Caching configuration
- Latency and throughput metrics
- Memory metrics
- Quality metrics
- Notes on failures or unsupported features

## Secondary Technology Backlog

The following technologies are worth investigating later, but are not part of the
current repository scope:

- vLLM
- ExecuTorch
- MLC-LLM
- LiteRT
- OpenVINO

They can be revisited after the primary runtime matrix is stable and benchmarked.

## Immediate Next Actions

1. Finalize benchmark harness and dataset schema on Apple Silicon.
2. Download merged release models and validate parity.
3. Run baseline and caching benchmarks in Transformers.
4. Add GGUF and llama.cpp path.
5. Add ONNX export and ONNX Runtime path.
6. Expand to Raspberry Pi, Qualcomm, and Jetson with the same workload.
7. Publish first cross-device report with reproducible configs.
