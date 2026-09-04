"""Production dependency wiring for the FastAPI gateway container."""

from __future__ import annotations

import os
from pathlib import Path

from service_edge_inference.domain import ServiceModelConfig
from service_edge_inference.gateway import GatewaySettings, create_app
from service_edge_inference.llama_client import LlamaServerClient


def create_runtime_app():
    """Create the gateway from local, provisioned assets only.

    Uvicorn invokes this factory with its factory option. The tokenizer is
    loaded with local_files_only=True so gateway startup cannot become an
    unpinned network download.
    """
    repository_root = Path(
        os.environ.get("SERVICE_REPOSITORY_ROOT", Path(__file__).resolve().parent.parent)
    )
    matrix_path = Path(
        os.environ.get(
            "SERVICE_MODEL_MATRIX",
            repository_root / "service_edge_inference" / "model_selection_matrix.json",
        )
    )
    manifest_path = Path(
        os.environ.get("SERVICE_MANIFEST", repository_root / "manifests" / "v2.1.json")
    )
    config = ServiceModelConfig.load(matrix_path, manifest_path)
    admin_token = os.environ.get("SERVICE_ADMIN_TOKEN")
    if not admin_token:
        raise RuntimeError("SERVICE_ADMIN_TOKEN must be configured")

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Gateway runtime requires the pinned Transformers dependency") from exc

    tokenizer_path = repository_root / config.tokenizer_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    llama_server = LlamaServerClient(
        os.environ.get("LLAMA_SERVER_URL", "http://llama-server:8080"),
        timeout_seconds=float(os.environ.get("LLAMA_TIMEOUT_SECONDS", "30")),
    )
    return create_app(
        model_config=config,
        tokenizer=tokenizer,
        llama_server=llama_server,
        settings=GatewaySettings(admin_token=admin_token),
    )
