# Repository Standards

These rules apply to all future changes in this repository.

## Benchmark configuration

- Keep one JSON manifest per experiment version in `manifests/`, for example
  `manifests/v1.0.json`, `manifests/v2.0.json`, and `manifests/v2.1.json`.
- A version manifest is the source of truth for its selected models, exact
  Hugging Face repository revision, adapter subfolder, base-model ID, prompt
  format, dataset, and device/engine quantization profiles.
- The `--models` CLI argument accepts exact manifest model names such as
  `SmolLM2-360M` or `Qwen3-0.6B`, or `all`. Do not introduce symbolic groups
  such as `v2.1-recommended`.
- Pin a Hugging Face commit SHA in every manifest and copy it into every
  artifact and report provenance record. Never benchmark an unpinned `main`
  revision by default.
- The edge benchmark evaluation split is always `test_anchor`, generated from
  `dataset_full/sample_0001.json` (100 examples). Full `test` data is not a
  standard pipeline stage on edge hardware.
- Run the full version-specific `test` split exactly once only for the final
  selected deployment configuration (device, engine, artifact variant, and
  cache mode). Record it as final-selection validation, not as a general
  benchmark result.
- Static-quantization calibration uses the deterministic train-only
  `calibration` split (`dataset_full/sample_0002.json`, 100 examples). It must
  never overlap with validation, `test`, or `test_anchor`.

## Artifact layout

Store all downloaded sources and derived artifacts below:

```text
models/<version>/<model-name>/
  source/base/
  source/adapter/
  transformers/merged/
  gguf/<quant>/
  onnx/<precision>/
  tensorrt/<precision>/
```

`source/base` and `source/adapter` are immutable snapshots. Derived artifacts
must record the input source revision and builder version. Do not use the old
flat `models/<run>_merged/` layout for new pipeline output.

TensorRT-LLM is not a Xavier pipeline artifact. Keep it out of the Jetson
Xavier profiles until bare TensorRT has a separately designed builder and
evaluator integration.

## Report and chart layout

Direct, engine-specific commands write to their engine folder:

```text
evaluation_<engine>/reports/
evaluation_<engine>/analysis/
```

Pipeline runs are isolated from direct runs and write to:

```text
run_results/<version>_<target>_<compute>_<timestamp>/
  manifest.lock.json
  run_summary.json
  <model-name>/
    <engine>_<variant>.json
    comparison.png
```

Every report and chart must include the run ID. A pipeline run creates one
comparison chart per selected model and device profile; it includes all of
that model's engine/variant reports for the run. A plot may only consume
reports belonging to that run ID; never scan and mix historical reports.

`python -m benchmark_pipeline --smoke` is a Mac-only runtime verification. It
uses three `test_anchor` examples with one warm-up and labels its reports as
`smoke`; never use those reports for candidate selection. RPi and Jetson
verification remains manual until those device environments are configured.

## Prompt and output compatibility

- v1.0 uses the name-output format: the expected and emitted label is the
  readable tool name, or `none` for no tool. Its `v1-tool-name` renderer must
  preserve the fine-tuning format: `Available Tools`, `Name`, `Description`,
  `User Request`, and the `Selected Tool:` completion cue.
- v2.0 and v2.1 use positional-ID output: IDs are assigned from the displayed
  tool order (`a`-`z`, then `A`-`Z`); `-` means no tool.
- Prompt text, prompt-template version, output format, and no-tool token must
  be declared in the version manifest. A separate prompt file is allowed only
  when the manifest pins its path and content SHA-256.
- Convert a positional-ID prediction to its corresponding readable tool name
  before computing quality metrics. Reports must retain both the raw model
  output and the canonical predicted tool name.

## Cache modes

- The standard benchmark mode is `prefix_cache`: it reuses the precomputed
  static prompt prefix.
- Do not schedule or expose `kv_cache` or `no_cache`. Historical reports may
  retain those labels, but all direct and pipeline evaluator interfaces use
  `prefix_cache` only.

## Change discipline

- Keep engine-specific artifact builders and evaluators separate; the pipeline
  orchestrates them and does not duplicate their inference implementations.
- Before deleting historical reports or charts, preserve them in Git history or
  an explicit archive and ensure the replacement pipeline can reproduce them.
- Add dry-run, manifest-validation, profile-resolution, and output-format
  compatibility tests before enabling a new pipeline stage by default.
