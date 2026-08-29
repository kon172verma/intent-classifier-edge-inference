# Releases

This file tracks finalized (and in-progress) releases. Each release picks the
2 best models from a version's experiments, merges/unloads them, and
publishes merged + GGUF + ONNX artifacts to the HF release repo
([kon172verma/intent-classifier](https://huggingface.co/kon172verma/intent-classifier)),
then tags this GitHub repo with the version.

Adapters for every release live in the HF experiments repo
([kon172verma/intent-classifier-experiments](https://huggingface.co/kon172verma/intent-classifier-experiments))
under the matching version folder — see `EXPERIMENTS.jsonl` for the full log.

## Next version in progress

`v1.1` — not yet started. Bump `VERSION` when experimentation begins.

## v1.0 (pending publish)

Status: candidates identified from local benchmarking
(intent-classifier-inference), not yet pushed to the release repo or tagged.

```yaml
version: v1.0
models:
  - name: qwen3-0.6b
    base_model: Qwen/Qwen3-0.6B
    training_method: LoRA
    lora_config: C
    dataset_size: 1k
    experiment_subfolder: v1.0/qwen3-0.6b_LoRA_C_1k_20260715-044041
    hf_release_path: LoRA_merged/qwen3-0.6b_C_1k
  - name: llama3.2-1b
    base_model: meta-llama/Llama-3.2-1B-Instruct
    training_method: LoRA
    lora_config: C
    dataset_size: 1k
    experiment_subfolder: v1.0/llama3.2-1b_LoRA_C_1k_20260715-052005
    hf_release_path: LoRA_merged/llama3.2-1b_C_1k
```

To publish this release:

```bash
python release.py --version v1.0 --technique LoRA \
    --runs qwen3-0.6b_C_1k:20260715-044041 llama3.2-1b_C_1k:20260715-052005 \
    --gguf-dir ../intent-classifier-inference/models/gguf \
    --onnx-dir ../intent-classifier-inference/models/onnx
```

This merges both adapters, pushes the merged models + GGUF + ONNX exports to
the release repo, and creates a local `v1.0` git tag (review, then
`git push origin v1.0`).
