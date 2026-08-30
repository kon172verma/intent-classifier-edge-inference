# Manifest-Driven Edge Benchmark Pipeline

## Status

Planning only. No pipeline or evaluator changes are authorized by this
roadmap. `AGENTS.md` defines the standards that the implementation must follow.

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
run_results/<version>_<timestamp>/
  manifest.lock.json
  run_summary.json
  <model-name>_<variant>/
    reports/
      <engine>/
    analysis/
      <engine>/
```

Pipeline reports and charts must never be written into the direct-command
`evaluation_<engine>/reports` or `analysis` folders. Each plot reads only the
reports indexed by the pipeline run's `manifest.lock.json`.

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
| `tool_name` (v1.0) | readable tool name or `-` | tool name or no-tool |
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

Remove `no_cache` from the future benchmark surface. It is an artificial
autoregressive path and is already unavailable in llama.cpp.

Retain the useful comparison currently labelled `kv_cache`, but rename it to
`cold` during implementation:

- `cold`: standard decoder KV cache; static system/tool prefix is freshly
  ingested for every request.
- `prefix_cache`: the static prefix is precomputed once and reused.

This preserves the deployment-relevant measurement of prefix reuse without
confusing normal decode-time KV caching with the experiment name.

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

## Implementation phases (not started)

1. **Manifest and planning layer**
   - Add the three version manifests and schema validation.
   - Add exact model-name and `all` selection.
   - Implement `--dry-run` profile resolution and dependency plan output.

2. **Artifact acquisition and merge**
   - Refactor the existing merge helper to accept explicit manifest model
     specifications instead of hard-coded Qwen3/Llama3 run keys.
   - Add Qwen2.5 and SmolLM2 base-model support.
   - Materialize source snapshots and merged checkpoints in the agreed layout.

3. **Artifact builders**
   - Create a GGUF build entry point from the current documented commands.
   - Generalize ONNX export/quantization around resolved model paths.
   - Keep TensorRT-LLM outside Xavier profiles; treat bare TensorRT as a later,
     separate Xavier integration.

4. **Evaluation compatibility**
   - Make all engine runners accept resolved model/artifact paths and run
     metadata.
   - Add the format-aware prompt renderer, parser, and canonical mapping.
   - Rename `kv_cache` to `cold` and remove `no_cache`.

5. **Reporting and plotting**
   - Change direct defaults from `results/` to `reports/`.
   - Add pipeline-specific output directories and run-ID filtering.
   - Write a locked run index before plotting.

6. **Verification and cleanup**
   - Unit-test manifests, profile resolution, dynamic ID mapping, and report
     isolation.
   - Run a small Mac CPU/GPU smoke suite, then RPi and Jetson profiles.
   - Archive and only then remove legacy output folders.

## Existing files expected to change when implementation is authorized

- `release_scripts/merge_models.py` and `release_scripts/release_common.py`
- `evaluation_lib/config.py`, `evaluation_lib/prompt.py`,
  `evaluation_lib/output_parser.py`, `evaluation_lib/metrics.py`, and
  `evaluation_lib/plot_common.py`
- `evaluation_baseline/`, `evaluation_llama_cpp/`, `evaluation_onnx/`, and
  `evaluation_tensorrt/` runners, loaders, and plotters
- `scripts/quantize_onnx.py`, `scripts/prepare_onnx_export_source.py`, and a
  new GGUF builder script
- Repository and engine README files plus new pipeline tests
