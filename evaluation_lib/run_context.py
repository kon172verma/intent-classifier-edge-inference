"""Resolved evaluation inputs shared by direct runners and the pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evaluation_lib.compatibility import PromptSpec


def load_prompt_spec(manifest_path: Path | None) -> tuple[PromptSpec | None, dict[str, Any]]:
    """Load prompt and provenance fields from an optional version manifest."""
    if manifest_path is None:
        return None, {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Manifest does not exist: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Manifest is not valid JSON: {manifest_path}") from exc
    if not isinstance(manifest, dict) or not isinstance(manifest.get("prompt"), dict):
        raise ValueError(f"Manifest has no prompt object: {manifest_path}")
    provenance = {
        "manifest_path": str(manifest_path),
        "manifest_version": manifest.get("version"),
        "experiments_revision": manifest.get("experiments", {}).get("revision"),
    }
    return PromptSpec.from_manifest(manifest["prompt"]), provenance
