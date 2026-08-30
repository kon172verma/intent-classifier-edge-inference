# evaluation_llama_cpp

llama.cpp (GGUF, quantized) benchmark for the intent-classifier inference
project — mirrors `evaluation_baseline/`'s architecture and JSON schema, but
runs the model via [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
instead of HF Transformers/PyTorch. See [devices_info.md](../devices_info.md)
for target hardware details.

## Package Structure

```text
evaluation_llama_cpp/
├── run.py           # CLI entrypoint — runs the full benchmark loop
├── model_loader.py  # Loads a quantized GGUF model via llama-cpp-python
├── inference.py     # Single inference pass with TTFT measurement
├── cache.py         # KV-state inspection, save/restore, prefix-cache pre-computation
├── plot_results.py  # Renders comparison charts from JSON reports
├── reports/         # JSON output files written by run.py
└── analysis/        # Charts written by plot_results.py
```

Shared utilities (prompt construction, quality/latency aggregation, output
parsing, system RAM measurement) live in `evaluation_lib/` and are reused
unmodified from `evaluation_baseline/`.

## One-time setup: convert + quantize the models

GGUF conversion requires a pinned `transformers`/`torch` version that
conflicts with this project's main venv, so it's done in an isolated venv.
The steps below only need to be run once (or whenever a model is retrained):

```bash
# 1. Install llama.cpp CLI tools (llama-quantize, llama-cli) via Homebrew
brew install llama.cpp

# 2. Get the HF -> GGUF conversion script
git clone --depth 1 https://github.com/ggml-org/llama.cpp scripts/llama.cpp-src

# 3. Isolated venv for conversion (avoids clashing with the main project's
#    transformers/torch versions)
python3 -m venv scripts/.venv-convert
source scripts/.venv-convert/bin/activate
pip install -r scripts/llama.cpp-src/requirements/requirements-convert_hf_to_gguf.txt
pip install -U transformers   # tokenizer_config.json format needs a recent version

# 4. Convert each HF checkpoint to GGUF (fp16 base, before quantization)
mkdir -p models/qwen3-0.6b_LoRA_C_1k_merged/gguf
mkdir -p models/llama3.2-1b_LoRA_C_1k_merged/gguf
python scripts/llama.cpp-src/convert_hf_to_gguf.py \
  models/qwen3-0.6b_LoRA_C_1k_merged/safetensors \
  --outfile models/qwen3-0.6b_LoRA_C_1k_merged/gguf/qwen3-0.6b-f16.gguf --outtype f16
python scripts/llama.cpp-src/convert_hf_to_gguf.py \
  models/llama3.2-1b_LoRA_C_1k_merged/safetensors \
  --outfile models/llama3.2-1b_LoRA_C_1k_merged/gguf/llama3.2-1b-f16.gguf --outtype f16

deactivate

# 5. Quantize to Q8_0 / Q6_K / Q4_K_M using the brew-installed llama-quantize
for q in Q8_0 Q6_K Q4_K_M; do
  llama-quantize \
    models/qwen3-0.6b_LoRA_C_1k_merged/gguf/qwen3-0.6b-f16.gguf \
    models/qwen3-0.6b_LoRA_C_1k_merged/gguf/qwen3-0.6b-${q}.gguf ${q}
  llama-quantize \
    models/llama3.2-1b_LoRA_C_1k_merged/gguf/llama3.2-1b-f16.gguf \
    models/llama3.2-1b_LoRA_C_1k_merged/gguf/llama3.2-1b-${q}.gguf ${q}
done
```

This produces:

- `models/qwen3-0.6b_LoRA_C_1k_merged/gguf/qwen3-0.6b-{Q8_0,Q6_K,Q4_K_M}.gguf`
- `models/llama3.2-1b_LoRA_C_1k_merged/gguf/llama3.2-1b-{Q8_0,Q6_K,Q4_K_M}.gguf`

## Python bindings

Install `llama-cpp-python` with Metal acceleration into the **main** project
venv (not the conversion venv):

```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python
```

## Quantization levels

| Level | ~Bits/weight | Notes |
| ------- | -------------- | ------- |
| `Q8_0` | 8.5 | Legacy simple quant, near-fp16 quality |
| `Q6_K` | 6.6 | k-quant, single mixture (no S/M/L variant exists) |
| `Q4_K_M` | 5.2 | k-quant, "Medium" mixture (bumps some layers to Q6_K for quality) |

## Caching

All direct and pipeline runs use `prefix_cache`: llama.cpp pre-computes the
static system-prompt state with `save_state()` and restores it per example.

## Usage

```bash
python evaluation_llama_cpp/run.py --model qwen3 --quant Q4_K_M --device mps
python evaluation_llama_cpp/run.py --model llama3 --quant Q8_0 --device cpu

python evaluation_llama_cpp/plot_results.py
```

## Implementation notes

- **Metal is asynchronous.** llama.cpp's Metal backend submits GPU work
  asynchronously; `evaluation_llama_cpp/cache.py`'s `synchronize()` (wrapping
  the low-level `llama_synchronize` C API) is called around every timed
  `eval()`/decode step — without it, wall-clock timing bleeds across calls
  (analogous to `torch.mps.synchronize()` in `evaluation_baseline`).
- **Prompt text** is rendered via the *original* HF tokenizer's
  `apply_chat_template()` (reusing `evaluation_lib.prompt` unmodified) so
  both benchmarks construct byte-identical prompt text; the resulting text is
  then re-tokenized with the GGUF model's own tokenizer (`Llama.tokenize()`)
  to get the token ids actually fed to the quantized model.
- **KV cache size** is read via the non-copying `llama_state_get_size` C API
  (cheap enough to call every example) rather than `save_state()` (which
  performs a full, expensive state copy).
- **Peak GPU memory** is always reported as `null` — llama.cpp/Metal exposes
  no cheap live-memory query API equivalent to
  `torch.mps.current_allocated_memory()`.
