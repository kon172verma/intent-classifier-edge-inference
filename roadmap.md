# Manifest-Driven Edge Benchmark Pipeline

## Status

Phases 1 through 5 are implemented. The command remains a no-side-effect dry
run unless `--execute` is supplied. It supports `fetch`, `merge`,
`build-artifacts`, `evaluate`, and `plot`; evaluation and plots are isolated
in a locked run workspace. `AGENTS.md` defines the standards that subsequent
implementation must follow.

## Objective

Provide one resumable command that resolves a version manifest and a hardware
profile, then fetches source models/adapters, merges the selected adapters,
builds required inference artifacts, evaluates each compatible variant, and
generates charts from that run only.

Illustrative interface:

```bash
python -m benchmark_pipeline \
  --manifest manifests/v2.1.json \
  --target mac --compute gpu \
  --models Qwen3-0.6B SmolLM2-360M \
  --stages all

python -m benchmark_pipeline \
  --manifest manifests/v2.1.json \
  --target rpi --compute cpu \
  --models all \
  --stages all
```

`--models` accepts an exact manifest model name or `all`.
`--target` and `--compute` resolve a device profile, so only variants
supported by that profile are built and evaluated.

## Agreed layouts

### Model artifacts

```text
models/<version>/<model-name>/
  source/base/                 # pinned Hugging Face snapshot
  source/adapter/              # pinned experiment-adapter snapshot
  transformers/merged/         # adapter merged and unloaded
  gguf/Q8_0/
  gguf/Q6_K/
  gguf/Q4_K_M/
  onnx/fp32/
  onnx/fp16/
  onnx/dynamic-int8/
  onnx/static-int8/
  tensorrt/fp16/               # only for supported NVIDIA profiles
```

The local model directory is keyed by the manifest display name, not a
timestamped training-run name. Artifact metadata records the exact adapter
subfolder, HF commit SHA, source hashes, builder version, and command.

### Direct engine output

```text
evaluation_llama_cpp/reports/
evaluation_llama_cpp/analysis/
```

The same convention will apply to the other `evaluation_<engine>/` folders.
Direct commands retain this local, engine-owned output behavior.

### Pipeline output

```text
run_results/<version>_<target>_<compute>_<timestamp>/
  manifest.lock.json
  run_summary.json
  <model-name>/
    <engine>_<variant>.json
    comparison.png
```

Pipeline reports and charts must never be written into the direct-command
`evaluation_<engine>/reports` or `analysis` folders. Each pipeline chart
compares all selected engine/variant reports for one model and one locked
device profile, using only reports indexed by the run's `manifest.lock.json`.

## Version manifests

Create one manifest per version:

```text
manifests/v1.0.json
manifests/v2.0.json
manifests/v2.1.json
```

Each manifest will contain:

- `schema_version`, version ID, and the pinned experiments-repository commit.
- Exact model display name, base-model ID, adapter subfolder, technique,
  configuration, dataset split, and adapter revision for every selected model.
- Prompt template identity, system prompt text (or pinned external prompt
  file), output format, no-tool token, and expected-output mapping rule.
- Device profiles: `mac-cpu`, `mac-gpu`, `rpi-cpu`, `jetson-cpu`, and
  `jetson-gpu`; each states engines, artifact variants, cache modes, and
  unsupported combinations.
- Dataset path and its content hash, benchmark settings, warm-up count, and
  chart policy.

### Dataset split contract

`scripts/prepare_benchmark_splits.py` materializes the shared contract as JSON
arrays, which can be passed directly to the existing evaluation runners:

- `test_anchor.json`: `sample_0001.json` only (100 examples). This is the
  mandatory split for standard edge-pipeline evaluation and candidate
  selection.
- `calibration.json`: `sample_0002.json` only (100 examples from train). It
  is reserved for methods that require representative calibration data, such
  as ONNX Runtime static INT8.
- `train.json` and `val.json`: retained to document and verify the original
  1k/10k training contract, but never scheduled for inference validation.
- `test.json`: run exactly once for final-selection validation of a chosen
  deployment configuration, after the `test_anchor` benchmark has selected
  the candidate. It is not part of a full matrix sweep.

For 1k, `test` and `test_anchor` are both `sample_0001`. For 10k, `test` is
`sample_0001` plus `sample_0092` through `sample_0100`. This full test is
knowingly skipped during the edge matrix sweep, but is used once to validate
the final selected device/engine/variant combination. The generated
`split_contract.json` stores the exact input files and counts for provenance.

The v1.0, v2.0, and v2.1 manifests must each pin the exact adapter directories
already identified during planning. No implementation should infer a latest
adapter from the mutable Hugging Face `main` branch.

## Prompt/output compatibility plan

The dataset retains `answer` as the readable tool name. The prompt renderer
and output parser will become format-aware:

| Manifest output format | Prompt target | Canonical value used for quality |
| --- | --- | --- |
| `tool_name` (v1.0) | readable tool name or `none` | tool name or no-tool |
| `positional_id` (v2.0/v2.1) | dynamic ID (`a`-`z`, `A`-`Z`) or `-` | ID decoded through that example's displayed tool list, then tool name or no-tool |

Implementation will add a shared mapping module rather than a static mapping
script: tool order changes per example, so the ID-to-tool mapping must be
constructed from that example's `available_tools`. Reports will retain:

- raw generated text;
- parsed output token/name;
- canonical predicted tool name;
- canonical expected tool name;
- invalid-output reason, when applicable.

This keeps accuracy comparable across name-output and ID-output versions while
preserving the native format each model was trained to produce.

## Cache-mode decision

The pipeline and direct evaluators benchmark `prefix_cache` only: the static
prompt prefix is precomputed once and reused. Historical reports may retain
older cache labels, but they are no longer accepted evaluator modes.

## Historical reports and charts

Do **not** delete the existing `results/` and `analysis/` material yet. It is
useful baseline evidence and may be Git-tracked. The cleanup sequence is:

1. Add the new `reports/` and run-isolated pipeline layouts.
2. Reproduce a small selected set with the pipeline and validate report/chart
   parity.
3. Commit or tag the historical outputs, or move them to an explicit archive.
4. In a dedicated cleanup change, delete the obsolete `results/` directories
   and update documentation.

Deleting the old outputs is therefore appropriate later, but not before the
new pipeline has produced an auditable replacement.

## Implementation phases

1. **Manifest and planning layer**
   - [x] Add the three version manifests and schema validation.
   - [x] Add exact model-name and `all` selection.
   - [x] Implement dry-run profile resolution and dependency plan output.

2. **Artifact acquisition and merge**
   - [x] Add manifest-pinned base-model revisions, including Qwen2.5 and
     SmolLM2 support.
   - [x] Materialize immutable, checksummed base and adapter source snapshots
     in the agreed layout.
   - [x] Merge locally snapshotted adapters through the manifest-driven
     pipeline entry point and record builder/input provenance.
   - [x] Keep merging in the manifest-driven pipeline; release publication is
     intentionally separate from artifact construction.

3. **Artifact builders**
   - [x] Create `scripts/build_gguf.py`, which accepts explicit merged paths
     and produces only the profile-requested GGUF variants.
   - [x] Generalize ONNX export/quantization around resolved model paths and
     the manifest calibration split.
   - [x] Add provenance-checked `build-artifacts` orchestration for GGUF and
     ONNX artifacts.
   - [x] Keep TensorRT-LLM outside Xavier profiles; bare TensorRT remains a
     separate future integration.

4. **Evaluation compatibility**
   - [x] Make the Transformers, llama.cpp, and ONNX Runtime runners accept
     resolved model/artifact paths and manifest/run metadata.
   - [x] Add the format-aware prompt renderer, parser, and canonical mapping.
   - [x] Remove `kv_cache` and `no_cache` from direct evaluator and plotting
     interfaces; all active profiles use `prefix_cache` only.

5. **Reporting and plotting**
   - [x] Change direct defaults from `results/` to `reports/` and direct
     charts to `analysis/`.
   - [x] Add pipeline-specific output directories, deterministic report paths,
     and run-ID filtering.
   - [x] Write `manifest.lock.json` with the planned report index before
     evaluation or plotting; use `--run-dir` to resume a locked run.

6. **Verification and cleanup**
   - [x] Unit-test manifests, profile resolution, dynamic ID mapping, report
     isolation, and smoke-run command construction.
   - [x] Add a Mac-only `--smoke` path: it evaluates the first three
     `test_anchor` examples with one warm-up, labels the resulting reports as
     `smoke`, and rejects RPi/Jetson targets.
   - [ ] Run the real Mac CPU and Mac GPU smoke suites after their selected
     artifacts have been built. These short runs verify runtime integration,
     not candidate quality or latency.
   - [ ] Add RPi and Jetson verification only when those environments are set
     up manually; they remain intentionally out of the automated smoke path.
   - [ ] Archive and only then remove legacy output folders.

## Existing files expected to change when implementation is authorized

- `release_scripts/release.py` and its publication documentation
- `evaluation_lib/config.py`, `evaluation_lib/prompt.py`,
  `evaluation_lib/output_parser.py`, `evaluation_lib/metrics.py`, and
  `evaluation_lib/plot_common.py`
- `evaluation_baseline/`, `evaluation_llama_cpp/`, `evaluation_onnx/`, and
  `evaluation_tensorrt/` runners, loaders, and plotters
- `scripts/quantize_onnx.py`, `scripts/prepare_onnx_export_source.py`, and a
  new GGUF builder script
- Repository and engine README files plus new pipeline tests
