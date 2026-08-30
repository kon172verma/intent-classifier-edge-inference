# Target Devices and Edge Kernel Matrix

This repository benchmarks the same released model and fixed workload across
edge devices.  A variant is included when the selected runtime has a credible
kernel path for it on that device; it is not included merely because a model
can be converted to that format.

Terminology matters here:

- **MPS** is PyTorch's Apple GPU backend and applies to the Transformers
  reference benchmark.
- **Metal** is llama.cpp's Apple GPU backend.  The llama.cpp CLI currently
  calls this `--device mps`, but reports and documentation should describe it
  as `metal` to avoid implying it uses PyTorch MPS.
- **CoreML EP**, not MPS, is ONNX Runtime's Apple accelerator provider.  It
  may partition a graph across CPU, GPU, and Apple Neural Engine (ANE).

## Benchmark status legend

- **Primary**: required comparison; optimized kernels are expected.
- **Anchor**: accuracy/memory reference; may be slower.
- **Diagnostic**: retain only to demonstrate a known poor or fallback path.
- **Omit**: do not schedule it in the standard matrix; record a failure only
  if compatibility changes.

## Edge device × runtime × artifact matrix

| Device / compute path | Runtime backend | Variants to benchmark | Status and kernel rationale |
| --- | --- | --- | --- |
| Apple Silicon CPU | Transformers / PyTorch CPU | FP32 | **Anchor**. Native CPU floating-point path; use to validate the reference model. |
| Apple Silicon CPU | Transformers / PyTorch CPU | FP16, BF16 | **Diagnostic**. These are not deployment candidates: PyTorch CPU low-precision matmul can be slower than FP32 on this workload. |
| Apple Silicon CPU | llama.cpp CPU | Q8_0, Q6_K, Q4_K_M | **Primary**. All are supported K-quant CPU paths; Q8_0 is the quality anchor, Q4_K_M is the constrained-memory target. |
| Apple Silicon CPU | ONNX Runtime CPU EP | FP32, dynamic INT8, static INT8 | **Primary**. INT8 uses the CPU quantized path. Dynamic INT8 is the first transformer PTQ variant; static INT8 is a calibrated comparison and must retain routing-accuracy checks. |
| Apple Silicon GPU | Transformers / PyTorch MPS | FP16, BF16 | **Primary**. These map to the intended Apple GPU low-precision path. Benchmark both rather than assuming one wins. |
| Apple Silicon GPU | Transformers / PyTorch MPS | FP32 | **Diagnostic**. Supported, but expected to be a poor baseline relative to FP16/BF16. |
| Apple Silicon GPU | llama.cpp Metal | Q8_0, Q6_K, Q4_K_M | **Primary**. llama.cpp supports K-quants on Metal; benchmark all-layer GPU offload separately from CPU-only runs. |
| Apple Silicon accelerator | ONNX Runtime CoreML EP | FP16 | **Primary**. CoreML EP is the correct ONNX accelerator path; retain provider assignment logs because unsupported nodes may fall back to CPU. |
| Apple Silicon accelerator | ONNX Runtime CoreML EP | FP32, dynamic INT8, static INT8 | **Omit** for the standard decoder-with-past suite. The current graph's empty, dynamic KV cache is incompatible with CoreML INT8; FP32 is not the useful low-precision CoreML comparison. Revisit only after export/provider compatibility changes. |
| Raspberry Pi 5 CPU | Transformers / PyTorch CPU | FP32 | **Diagnostic** only. Useful for parity but too slow for deployment comparison. |
| Raspberry Pi 5 CPU | Transformers / PyTorch CPU | FP16, BF16 | **Omit**. No worthwhile low-precision PyTorch CPU kernel path for this target. |
| Raspberry Pi 5 CPU | llama.cpp CPU (ARM NEON) | Q4_K_M, Q6_K, Q8_0 | **Primary**. K-quants have ARM-NEON support. Order experiments Q4_K_M → Q6_K → Q8_0; use the latter as the quality anchor, not an assumed speed winner. |
| Raspberry Pi 5 CPU | ONNX Runtime CPU EP | FP32, dynamic INT8, static INT8 | **Primary**. Compare both INT8 methods using the real prompt calibration set; dynamic INT8 is normally the first transformer candidate. |
| Jetson Xavier CPU (ARM) | llama.cpp CPU | Q4_K_M, Q6_K, Q8_0 | **Primary CPU baseline**. Same ARM K-quant rationale as Raspberry Pi; report power mode and clocks. |
| Jetson Xavier GPU (Volta SM72) | llama.cpp CUDA | Q4_K_M, Q6_K, Q8_0 | **Primary GPU comparison**. CUDA supports K-quants; include all-layer offload and record `n_gpu_layers`. |
| Jetson Xavier GPU (Volta SM72) | Transformers / PyTorch CUDA | FP16, FP32 | **FP16 primary; FP32 anchor**. Xavier has FP16/INT8 Tensor Cores, but no BF16 hardware path. |
| Jetson Xavier CPU/GPU | ONNX Runtime CPU, CUDA, or TensorRT EP | CPU: FP32/dynamic INT8/static INT8. GPU: FP16, then calibrated INT8 Q/DQ if the exported graph fully partitions. | **Primary portability path**. Treat CUDA and TensorRT EP as separate providers and record graph partitioning. Do not label either one "MPS". |
| Jetson Xavier GPU (Volta SM72) | Bare TensorRT | FP16 first; calibrated INT8 (W8A8) second | **Candidate, separate harness**. These are the Xavier-relevant TensorRT schemes. Do not start with INT4/AWQ/GPTQ here. |
| Jetson Xavier GPU (Volta SM72) | TensorRT-LLM | — | **Omit**. Current upstream TRT-LLM support is Ampere and newer; Xavier support would be unmaintained/community-only. |

### Kernel and artifact rules

1. GGUF labels (`Q8_0`, `Q6_K`, `Q4_K_M`) are llama.cpp tensor encodings,
   not generic INT8/INT4 models.  They must not be compared as if they were
   identical to ONNX INT8 or TensorRT W8A8.
2. `dynamic-int8` in this repo is weight-only with runtime activation
   quantization; `static-int8` uses offline activation calibration.  For
   transformer graphs, dynamic INT8 is the preferred first ONNX Runtime
   experiment, while static INT8 is retained as a calibrated alternative.
3. ONNX Runtime's CoreML EP must be logged with its actual provider list and
   partitioning.  Its API can use CPU, GPU, and ANE; it does not guarantee an
   all-ANE execution.
4. TensorRT's hardware capability is not TensorRT-LLM support.  Xavier can
   run appropriate TensorRT engines, but the existing `evaluation_tensorrt/`
   code is specifically a TensorRT-LLM harness and therefore targets Orin or
   newer NVIDIA GPUs.
5. Treat every low-bit result as a quality experiment.  The acceptance gate is
   the existing routing metrics (tool accuracy, invalid tool rate, and exact
   match), not model size or tokens/s alone.

## Apple M4 MacBook Air

- SoC: Apple M4 Macbook Air
- CPU: 10 cores
- GPU: 10-core integrated Apple GPU (Metal 4 support)
- Memory: 16 GB unified memory
- Storage: 500 GB internal SSD
- Role in this project: primary development and baseline reference platform

Why this device matters:

- Fast iteration for model conversion and benchmark harness validation
- Strong local baseline for latency and quality comparisons
- Useful reference before running constrained edge tests

### Transformers / PyTorch MPS precision support

| Precision / Method | MPS Support | Notes |
| --- | --- | --- |
| FP16 | ✅ | Fully supported, hardware-accelerated. Generally the fastest option on MPS. |
| BF16 | ✅ | Fully supported, hardware-accelerated. Slightly slower than FP16 today (less mature kernel coverage). |
| INT8 (`bitsandbytes` `load_in_8bit`) | ❌ | CUDA-only kernels; `device_map="mps"` raises an error rather than falling back. |
| INT8 (PyTorch native `quantize_dynamic`) | ❌ | Quantized int8 kernels (`qnnpack`/`fbgemm`) only register CPU backends. Works on CPU (arm64 uses `qnnpack`), not MPS. |
| INT8 (`torchao` weight-only/dynamic) | ⚠️ | More likely to dispatch to MPS ops than int4, but some paths assume CUDA fused kernels and may silently fall back to slow eager ops. |
| FP8 | ❌ | CUDA-only, typically Hopper-GPU-specific. |
| INT4 / NF4 (`bitsandbytes`) | ❌ | CUDA-only. |
| INT4 (`torchao` weight-only) | ⚠️ | Recent releases include MPS-specific packed-int4 kernels; support depends on installed version. |

## Raspberry Pi 5 Model B Rev 1.0

- Board: Raspberry Pi 5 Model B Rev 1.0
- CPU: 4-core ARM CPU
- Memory: about 8 GB system memory (7937 MiB reported)
- Storage: 127 GB local disk (mmcblk0)
- Network: Ethernet available
- Role in this project: constrained edge inference benchmark target

Why this device matters:

- Represents practical CPU-only edge deployment conditions
- Useful for quantized runtime comparisons under memory constraints
- Critical for TTFT, prefill latency, and RAM-footprint benchmarking

## Baseline Stack

- Apple M4 MacBook Air: Transformers MPS reference, then llama.cpp Metal and
  ONNX Runtime CPU/CoreML EP
- Raspberry Pi 5: llama.cpp plus GGUF first, then ONNX Runtime CPU
- Jetson Xavier: llama.cpp CPU/CUDA plus ONNX Runtime; investigate bare
  TensorRT FP16/W8A8 only after the portable paths are stable

## Sources and version-sensitive checks

- [llama.cpp feature matrix](https://github.com/ggml-org/llama.cpp/wiki/Feature-matrix)
  — confirms K-quant support on ARM NEON, Metal, and CUDA.
- [ONNX Runtime CoreML EP documentation](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)
  — defines the CoreML provider, compute-unit behaviour, and dynamic-shape
  caveats.
- [ONNX Runtime quantization documentation](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
  — describes the dynamic/static INT8 trade-off and recommends dynamic
  quantization first for transformer-based models.
- [TensorRT-LLM support matrix](https://nvidia.github.io/TensorRT-LLM/legacy/reference/support-matrix.html)
  — lists Ampere and newer as supported hardware; verify again whenever the
  JetPack, TensorRT, or TRT-LLM version changes.
