# evaluation_baseline

HF Transformers + PyTorch FP16 baseline benchmark for the intent-classifier
inference project. See [devices_info.md](../devices_info.md) for full
descriptions of the target hardware.

## Package Structure

```text
evaluation_baseline/
├── run.py           # CLI entrypoint — runs the full benchmark loop
├── model_loader.py  # Loads tokenizer + model (CPU first, then .to(device))
├── inference.py     # Single inference pass with TTFT measurement
├── cache.py         # KV-cache inspection, cloning, prefix-cache pre-computation
└── results/         # JSON output files written by run.py
```

Shared utilities live in `evaluation_lib/` (sibling package).

## Caching

All direct and pipeline runs use `prefix_cache`: the static system-prompt
prefix is pre-computed once and cloned per example.

## Usage

```bash
# Activate the project venv first
source .venv/bin/activate

python evaluation_baseline/run.py --model qwen3 --device mps
```

### CLI Options

| Flag | Default | Description |
| ------ | --------- | ------------- |
| `--model` | required | `qwen3` or `llama3` |
| `--device` | `auto` | `auto`, `mps`, `cpu`, or `cuda` |
| `--dataset` | `dataset_full/sample_0001.json` | Path to dataset JSON |
| `--output-dir` | `evaluation_baseline/results/` | Directory for JSON output |
| `--warmup` | `2` | Examples excluded from measurements (not recorded) |

## Output

Results are written to `results/<model>_<device>_prefix_cache_<timestamp>.json` with
three top-level sections:

- `run_config` — model, mode, device, dtype, dataset path, library versions,
  and model weight size in MB
- `aggregate` — mean/p50/p95 latencies, throughput, peak RAM, mean KV-cache
  size, and mean peak GPU memory
- `quality` — tool accuracy, exact-match rate, none-accuracy, invalid-tool rate
- `per_example` — full per-example timing and prediction detail

## Metrics Collected

### Latency

- `prefill_latency_ms` — time from prompt submission to first token (TTFT)
- `decode_latency_ms` — time spent generating remaining tokens
- `e2e_latency_ms` — total wall time including prefill and decode
- `ttft_ms` — time to first token (captured via `LogitsProcessor`)

### Throughput

- `prefill_tok_per_sec` — input tokens processed per second during prefill
- `decode_tok_per_sec` — output tokens generated per second during decode

### Memory

- `model_weights_mb` — static parameter + buffer size (FP16, device-agnostic)
- `peak_ram_mb` — peak system RSS across the full run
- `mean_kv_cache_kb` — mean KV-cache size per example
- `mean_peak_gpu_mb` — mean per-example GPU memory after inference
  (CUDA: true peak including activations; MPS: current allocated post-inference)
