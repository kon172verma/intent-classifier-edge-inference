# release_scripts

Release workflow for intent-classifier-inference.

## Responsibilities

- `merge_models.py`: compatibility CLI for the manifest-driven source download and merge flow.
- `release.py`: upload already-prepared local artifacts (safetensors/gguf/onnx) and tag release.
- `upload_release.py`: upload a model folder from local `intent-classifier-release` clone.

## Manifest-Driven Model Layout

`merge_models.py` writes the standard, version-scoped layout:

```text
models/<version>/<model-name>/
  source/base/
  source/adapter/
  transformers/merged/
```

GGUF and ONNX artifact builders will add sibling directories beneath the same
model root in the next pipeline phase. The older release uploader still uses
the legacy release layout and is not invoked by `merge_models.py`.

## Commands

1. Merge adapters into local safetensors:

```bash
python release_scripts/merge_models.py --manifest manifests/v1.0.json --models all
```

2. Build GGUF and ONNX artifacts through the benchmark pipeline once Phase 3
   is implemented.

3. The legacy release uploader may still upload its existing release layout:

```bash
python release_scripts/release.py --version v1.0 --runs qwen3-0.6b_LoRA_C_1k llama3.2-1b_LoRA_C_1k
```

4. Optional: also create an HF tag:

```bash
python release_scripts/release.py --version v1.0 --runs qwen3-0.6b_LoRA_C_1k llama3.2-1b_LoRA_C_1k --hf-tag
```
