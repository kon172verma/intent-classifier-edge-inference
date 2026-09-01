# Release publication

`release.py` publishes the locally built artifacts for one manifest version to
the single Hugging Face model repository `kon172verma/intent-classifier`.
Each release model receives a versioned subfolder:

```text
kon172verma/intent-classifier/
  v1.0-qwen3-0.6b/
    README.md
    benchmark_provenance.json
    transformers/
    gguf/
    onnx/
  v1.0-llama3.2-1b/
```

It reads `manifests/<version>.json` and `models/<version>/<model-name>/`.
All artifact formats retain their own directories. Source snapshots, adapters,
temporary files, artifact metadata, and pipeline run results are never
published.

The workflow never writes or deletes repository-root files. Add the root
`README.md`, `LICENSE`, and `THIRD_PARTY_LICENSES.txt` separately; the root
README is the only location where Hugging Face recognizes model-card metadata.
The nested model `README.md` files are ordinary per-model documentation.

## Commands

Preview the release folders and local artifacts to be published:

```bash
python release_scripts/release.py --version v1.0 --models all
```

Create the repository if needed and upload the release artifacts. `HF_TOKEN` is
loaded from `.env` when present:

```bash
python release_scripts/release.py --version v1.0 --models all --execute
```

The final line prints the immutable commit SHA containing the completed upload.
Copy that SHA into the version manifest's `release.revision`, then add each
model's `release_subfolder` (for example `v1.0-qwen3-0.6b`). Devices can then
use the benchmark pipeline's `download-release` stage without rebuilding the
artifacts locally.

Use `--private` only when creating a new private repository. To target a
different single repository, use `--repo-id`:

```bash
python release_scripts/release.py \
  --version v1.0 --models Qwen3-0.6B \
  --repo-id organisation/intent-classifier \
  --execute
```

The workflow intentionally has no remote-delete option. This protects the
manually maintained root documentation and licensing files. If the existing
Hub repository must be emptied before its first release, do that explicitly in
the Hugging Face UI before publishing.
