# Edge tool-routing gateway

This package is the FastAPI gateway for the fixed deployment configuration in
the model-selection matrix:

- model: v2.1 Qwen2.5-0.5B, Q6_K GGUF;
- output: positional tool ID, mapped by the gateway to the submitted tool name;
- profiles: cpu and jetson_gpu.

The gateway never downloads a model or tokenizer. Before it starts, provision
the pinned GGUF artifact and the merged Transformers tokenizer snapshot into
the paths declared in the selection matrix. The gateway loads the tokenizer
with local_files_only=True.

## Development launch

Install the pinned gateway dependencies, provision the model/tokenizer assets,
and configure the internal llama-server address and admin credential:

    python -m pip install -r service_edge_inference/requirements.txt
    export SERVICE_ADMIN_TOKEN='replace-with-a-secret'
    export LLAMA_SERVER_URL='http://llama-server:8080'
    uvicorn service_edge_inference.runtime:create_runtime_app --factory

Only publish the gateway port. The llama-server endpoint must remain on the
internal container network.

## Current HTTP contract

- POST /v1/classify accepts a query object and returns the raw valid positional
  tool_id, canonical readable tool, and toolset_version.
- PUT /v1/toolset accepts the complete ordered JSON array of name/description
  objects. It requires X-Admin-Token, warms the candidate prefix, then
  atomically activates it.
- GET /health/live is process liveness.
- GET /health/ready returns 503 until llama-server is healthy and an active
  tool-set prefix has been warmed.
- GET /v1/toolset/status requires X-Admin-Token.

The gateway returns none for a malformed completion but returns 503 for
llama-server failures.
