# Release publication

`release.py` publishes the locally built artifacts for one manifest version to
one Hugging Face model repository per release model:

```text
kon172verma/intent-classifier-v1.0-0.6b
kon172verma/intent-classifier-v1.0-1b
```

It reads `manifests/<version>.json` and `models/<version>/<model-name>/`.
The merged Transformers checkpoint is copied to the release-repository root;
GGUF and ONNX retain their `gguf/` and `onnx/` directories. Source snapshots,
adapters, temporary files, and pipeline run results are never published.

## Commands

Preview the repositories and local artifacts to be published:

```bash
python release_scripts/release.py --version v1.0 --models all
```

Create repositories as needed and upload the release artifacts. `HF_TOKEN` is
loaded from `.env` when present:

```bash
python release_scripts/release.py --version v1.0 --models all --execute
```

Use `--private` only when creating private release repositories. To replace an
existing release repository entirely, use the explicit destructive operation:

```bash
python release_scripts/release.py --version v1.0 --models all --replace --execute
```

`--replace` removes remote files that are not part of the newly assembled
release. It does not alter the local `models/` directory.
