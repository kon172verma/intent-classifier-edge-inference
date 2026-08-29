# Edge LLM Inference Plan

## Related Repositories

- **Training** — [kon172verma/intent-classifier](https://github.com/kon172verma/intent-classifier) (GitHub): fine-tuning code, configs, and reports.
- **Experiments** — [kon172verma/intent-classifier-experiments](https://huggingface.co/kon172verma/intent-classifier-experiments) (Hugging Face): every adapter produced during experimentation, organized by version.
- **Release** — [kon172verma/intent-classifier](https://huggingface.co/kon172verma/intent-classifier) (Hugging Face): only the best merged models per release, plus GGUF and ONNX exports (this repo's benchmark input).

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
- [kon172verma/intent-classifier](https://huggingface.co/kon172verma/intent-classifier) (Hugging Face, the "release" repo) holds only the 2 best merged/unloaded models chosen per release, plus their GGUF and ONNX exports.

Merging and unloading adapters is handled in this repository under
`release_scripts/`.

Inference flow in this repository:

1. Pull the merged release model (Transformers format) from the release repo.
2. Produce deployable artifacts for runtime comparison.
3. Optionally push resulting artifacts back to the release repo via
  `release_scripts/release.py --runs <model_key>_<technique>_<config>_<size>`.

Artifact targets:

- Hugging Face Transformers (reference path)
- GGUF for llama.cpp
- ONNX for ONNX Runtime
- TensorRT-LLM path for Jetson where supported
- vLLM where a suitable GPU-backed environment is available

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
| Apple Silicon | Transformers, llama.cpp, ONNX Runtime |
| Raspberry Pi 5 | llama.cpp, ONNX Runtime |
| Qualcomm | ONNX Runtime (CPU first, QNN when available) |
| Jetson Xavier | llama.cpp (CPU and GPU offload), TensorRT-LLM, ONNX Runtime |

Note: exact support depends on OS, SDK, driver, and runtime versions on each
physical device.

## Benchmark Method

### Baseline

Measure baseline inference first for each device and runtime without prompt prefix
caching.

### Caching Experiments

Evaluate:

- Standard KV cache behavior during decode
- Prompt prefix caching for static context

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
- Evaluate TensorRT-LLM if toolchain compatibility is confirmed.
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

| Split | Files | Examples |
| --- | --- | --- |
| train | sample_0002.json to sample_0009.json | 800 |
| val | sample_0010.json | 100 |
| test | sample_0001.json | 100 |
| test_anchor | sample_0001.json | 100 |

Notes:

- test_anchor intentionally matches test in the current 1k setup.
- Keep this split stable when comparing runtimes and devices.

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
