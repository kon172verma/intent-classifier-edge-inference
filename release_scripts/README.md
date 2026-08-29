# release_scripts

Release workflow for intent-classifier-inference.

## Responsibilities

- `merge_models.py`: download base model + adapter and write merged checkpoints to local `models/`.
- `release.py`: upload already-prepared local artifacts (safetensors/gguf/onnx) and tag release.
- `upload_release.py`: upload a model folder from local `intent-classifier-release` clone.

## Expected Local Model Layout

For each run:

`models/<model_key>_<technique>_<config>_<dataset_size>_merged/`

- `safetensors/`
- `gguf/` (optional)
- `onnx/` (optional)

Example:

`models/qwen3-0.6b_LoRA_C_1k_merged/safetensors/`

## Commands

1. Merge adapters into local safetensors:

```bash
python release_scripts/merge_models.py --technique LoRA --runs qwen3-0.6b_C_1k llama3.2-1b_C_1k
```

2. Prepare `gguf/` and `onnx/` under each `<run>_merged` folder (if needed).

3. Upload release artifacts and tag:

```bash
python release_scripts/release.py --version v1.0 --runs qwen3-0.6b_LoRA_C_1k llama3.2-1b_LoRA_C_1k
```

4. Optional: also create an HF tag:

```bash
python release_scripts/release.py --version v1.0 --runs qwen3-0.6b_LoRA_C_1k llama3.2-1b_LoRA_C_1k --hf-tag
```
