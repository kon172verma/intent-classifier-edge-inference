# Development notes

Operational command reference for the manifest-driven benchmark pipeline and
the direct evaluator entry points. Run every command from the repository root
with the project virtual environment active unless the command says otherwise.

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Use the version manifest as the source of truth for model names, revisions,
prompt format, compatible engines, and artifact variants. Never substitute a
mutable Hugging Face `main` revision.

## Pipeline

### Prepare benchmark splits

```bash
python scripts/prepare_benchmark_splits.py --dataset-size 10k
```

This materialises the deterministic splits under `benchmark_data/10k/`:

- `test_anchor.json` — the 100-example edge benchmark and candidate-selection split.
- `calibration.json` — train-only data for static INT8 calibration.
- `test.json` — held for one final deployment validation only.

### Dry-run a plan first

```bash
python -m benchmark_pipeline \
  --manifest manifests/v2.1.json \
  --target rpi --compute cpu \
  --models Qwen3-0.6B \
  --engines llama_cpp \
  --stages all
```

Without `--execute`, the command only validates the manifest and prints the
resolved plan. Add `--json` when another tool needs the resolved plan as JSON.

| Flag | Meaning |
| --- | --- |
| `--manifest PATH` | Required version manifest, such as `manifests/v2.1.json`. |
| `--target` | Device family declared by the manifest: `mac`, `rpi`, or `jetson`. |
| `--compute` | Required device profile: `cpu` or `gpu`. |
| `--models` | Exact manifest model names, or the sole value `all`. |
| `--engines` | Exact profile engine names, or the sole value `all` (the default). |
| `--stages` | One or more stages, or the sole value `all`. |
| `--execute` | Performs the selected work. Omit it for a safe dry run. |
| `--run-dir PATH` | Resumes a locked pipeline workspace for `evaluate` or `plot`. |
| `--smoke` | Mac-only, three-example runtime verification; never candidate selection. |

The stages are:

- `fetch` — download pinned base and adapter snapshots.
- `merge` — merge the selected adapter into the base model.
- `build-artifacts` — build only variants selected by the resolved profile.
- `download-release` — download published, pinned artifacts instead of fetching,
  merging, or building locally.
- `evaluate` — evaluate `test_anchor` in a new or resumed locked workspace.
- `plot` — create charts using only reports indexed by that workspace.

`download-release` is mutually exclusive with `fetch`, `merge`, and
`build-artifacts`. The normal `--stages all` path means the local acquisition
path; use an explicit release-download command on constrained devices.

### Build from source artifacts

Set `HF_TOKEN` (or `HUGGINGFACE_TOKEN`) before fetching a gated model.

```bash
# Fetch the manifest-pinned source snapshots and merge one model.
python -m benchmark_pipeline \
  --manifest manifests/v2.1.json \
  --target rpi --compute cpu \
  --models Qwen3-0.6B \
  --stages fetch merge \
  --execute

# Build just the profile artifacts after a successful merge.
BENCHMARK_GGUF_PYTHON=scripts/.venv-convert/bin/python \
BENCHMARK_ONNX_OPTIMUM_CLI=scripts/.venv-onnx/bin/optimum-cli \
python -m benchmark_pipeline \
  --manifest manifests/v2.1.json \
  --target rpi --compute cpu \
  --models Qwen3-0.6B \
  --stages build-artifacts \
  --execute
```

`BENCHMARK_GGUF_PYTHON` selects the isolated GGUF conversion environment.
`BENCHMARK_ONNX_OPTIMUM_CLI` selects the isolated ONNX export CLI. Static INT8
calibration always uses the manifest's `calibration.json`, never an evaluation
or test split.

### Download an immutable published release

```bash
python -m benchmark_pipeline \
  --manifest manifests/v2.1.json \
  --target rpi --compute cpu \
  --models Qwen3-0.6B \
  --stages download-release evaluate plot \
  --execute
```

The release repository, 40-character commit SHA, and model subfolder must
already be present in the manifest. The pipeline writes the downloaded files
back to `models/<version>/<model-name>/` and records release provenance before
reusing them.

### Evaluate and plot a local build

```bash
python -m benchmark_pipeline \
  --manifest manifests/v2.1.json \
  --target rpi --compute cpu \
  --models Qwen3-0.6B \
  --stages evaluate plot \
  --execute
```

This creates `run_results/<version>_<target>_<compute>_<timestamp>/` with a
`manifest.lock.json`, `run_summary.json`, engine reports, and one comparison
chart per model. To finish a partial run, pass its exact directory:

```bash
python -m benchmark_pipeline \
  --manifest manifests/v2.1.json \
  --target rpi --compute cpu \
  --models Qwen3-0.6B \
  --stages evaluate plot \
  --run-dir run_results/v2.1_rpi_cpu_<timestamp> \
  --execute
```

Use `--stages plot --run-dir ... --execute` to regenerate charts from an
existing locked report index. Do not point a pipeline plot at historical direct
reports.

### Mac smoke verification

```bash
# Mac CPU
python -m benchmark_pipeline \
  --manifest manifests/v2.1.json \
  --target mac --compute cpu \
  --models Qwen3-0.6B \
  --stages evaluate plot --smoke --execute

# Mac GPU: MPS, Metal llama.cpp, and CoreML where configured
python -m benchmark_pipeline \
  --manifest manifests/v2.1.json \
  --target mac --compute gpu \
  --models Qwen3-0.6B \
  --stages evaluate plot --smoke --execute
```

Smoke runs use three anchor examples with one warm-up and are labelled
`smoke`. RPi and Jetson are deliberately rejected by `--smoke`; verify those
runtimes manually on their real hardware.

## Direct engine commands

Direct runs are useful for engine development and diagnostics. The
manifest-aware Transformers, llama.cpp, and ONNX runners write to their
engine-owned `evaluation_<engine>/reports/` directory by default; pipeline
runs must remain isolated in `run_results/`. Those active runners use
`prefix_cache` only—there is no `kv_cache` or `no_cache` option. TensorRT-LLM
is a separate legacy exception described below.

Use the explicit v2.1 paths below instead of legacy aliases such as `qwen3`.
The explicit paths ensure the model, tokenizer, prompt rules, and artifact are
the selected release. Replace `Qwen3-0.6B` and the artifact variant only with
values declared in the manifest.

### Transformers / PyTorch (`evaluation_baseline`)

```bash
# CPU reference run
python evaluation_baseline/run.py \
  --model Qwen3-0.6B \
  --model-path models/v2.1/Qwen3-0.6B/transformers/merged \
  --manifest manifests/v2.1.json \
  --device cpu --dtype float32 --machine rpi5 \
  --dataset benchmark_data/10k/test_anchor.json

# Apple Silicon MPS reference run
python evaluation_baseline/run.py \
  --model Qwen3-0.6B \
  --model-path models/v2.1/Qwen3-0.6B/transformers/merged \
  --manifest manifests/v2.1.json \
  --device mps --dtype float16 --machine mac \
  --dataset benchmark_data/10k/test_anchor.json
```

Key flags: `--device` is `auto`, `cpu`, `mps`, or `cuda`; `--dtype` is
`float32`, `bfloat16`, or `float16`; `--machine` labels the physical device;
`--output-dir` or `--output-file` changes the report location; `--warmup N`
excludes initial examples; `--max-examples N` makes a diagnostic prefix run and
must be greater than the warm-up count. Use `--benchmark-scope smoke` only for
a genuine short smoke run.

Plot direct reports only:

```bash
python evaluation_baseline/plot_results.py \
  --reports-dir evaluation_baseline/reports \
  --output-dir evaluation_baseline/analysis
```

### llama.cpp / GGUF (`evaluation_llama_cpp`)

```bash
python evaluation_llama_cpp/run.py \
  --model Qwen3-0.6B \
  --gguf-path models/v2.1/Qwen3-0.6B/gguf/Q4_K_M/model.gguf \
  --tokenizer-path models/v2.1/Qwen3-0.6B/transformers/merged \
  --quant Q4_K_M --manifest manifests/v2.1.json \
  --device cpu --machine rpi5 \
  --dataset benchmark_data/10k/test_anchor.json
```

Key flags: `--quant` must be `Q8_0`, `Q6_K`, or `Q4_K_M`; `--device` is
`auto`, `cpu`, `mps`, or `cuda`; `--n-ctx` must exceed the longest rendered
prompt; and `--tokenizer-path` supplies the chat-template renderer. The
remaining report, warm-up, smoke, and manifest flags have the same purpose as
the Transformers runner.

```bash
python evaluation_llama_cpp/plot_results.py \
  --reports-dir evaluation_llama_cpp/reports \
  --output-dir evaluation_llama_cpp/analysis
```

### ONNX Runtime (`evaluation_onnx`)

```bash
# CPU static INT8
python evaluation_onnx/run.py \
  --model Qwen3-0.6B \
  --onnx-path models/v2.1/Qwen3-0.6B/onnx/static-int8/model.onnx \
  --tokenizer-path models/v2.1/Qwen3-0.6B/transformers/merged \
  --precision static-int8 --manifest manifests/v2.1.json \
  --device cpu --machine rpi5 \
  --dataset benchmark_data/10k/test_anchor.json

# Apple CoreML
python evaluation_onnx/run.py \
  --model Qwen3-0.6B \
  --onnx-path models/v2.1/Qwen3-0.6B/onnx/fp16/model.onnx \
  --tokenizer-path models/v2.1/Qwen3-0.6B/transformers/merged \
  --precision fp16 --manifest manifests/v2.1.json \
  --device coreml --machine mac \
  --dataset benchmark_data/10k/test_anchor.json
```

Key flags: `--precision` is `fp32`, `fp16`, `dynamic-int8`, or `static-int8`;
`--device` is `auto`, `cpu`, `coreml`, `cuda`, or `qnn`. For Qualcomm QNN use
`--qnn-backend htp|gpu|cpu` and, when needed, `--qnn-lib-path PATH`. The
manifest profiles define which combinations are supported.

```bash
python evaluation_onnx/plot_results.py \
  --reports-dir evaluation_onnx/reports \
  --output-dir evaluation_onnx/analysis
```

### TensorRT-LLM (`evaluation_tensorrt`)

This is a legacy direct harness, not a current v2.x pipeline stage and not a
Jetson Xavier artifact. Its current CLI accepts legacy model keys and does not
accept `--manifest`, `--run-id`, or explicit versioned artifact paths. Use it
only with its compatible legacy engine and dataset contract until it is
integrated separately:

```bash
python evaluation_tensorrt/run.py \
  --model qwen3 --dtype fp16 --machine jetson \
  --dataset <legacy-compatible-dataset.json>
```

`--dtype` selects the prebuilt TensorRT-LLM engine precision (`fp16`, `bf16`,
`int8`, or `int4` where supported); `--max-new-tokens` caps generation; and
`--output-dir` controls legacy result placement. See
`evaluation_tensorrt/readme.md` for engine-build prerequisites. Do not mix its
results into a manifest-locked pipeline chart.

```bash
python evaluation_tensorrt/plot_results.py \
  --results-dir evaluation_tensorrt/results \
  --output-dir evaluation_tensorrt/results/charts
```

## Operational rules and troubleshooting

- Run `python -m benchmark_pipeline --help` or
  `python evaluation_<engine>/run.py --help` to inspect the installed CLI.
- Verify that requested model names and engine variants appear in the selected
  manifest profile; the pipeline rejects unsupported combinations.
- The standard edge split is always `test_anchor`. Do not use the full
  `test.json` for general matrix sweeps. A dedicated, correctly labelled
  final-selection command has not yet been added to the pipeline; add that
  workflow before recording the one permitted full-test validation.
- Ensure the target has enough disk space for model artifacts and enough RAM or
  VRAM for the selected variant before running `--execute`.
- For build failures, confirm that the isolated GGUF and ONNX environments
  exist and point `BENCHMARK_GGUF_PYTHON` and
  `BENCHMARK_ONNX_OPTIMUM_CLI` at their executables.
- The pipeline records provenance in source/artifact metadata and in
  `manifest.lock.json`. Do not manually overwrite files in a populated model
  artifact directory; provenance checks intentionally reject incompatible
  reuse.
