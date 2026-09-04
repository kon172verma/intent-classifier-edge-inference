# Edge inference service roadmap

## Objective

Build a containerized, edge-deployable service that loads one manifest-pinned
GGUF artifact and routes a user request to one currently active tool. The
service owns the public contract, tool-set lifecycle, exact prompt rendering,
output validation, and observability. `llama-server` owns model execution and
KV-cache reuse.

The initial deployment mode is one active model instance and `--parallel 1`.
Tool-set changes may briefly queue requests while their prefix is warmed. This
keeps steady-state memory low and avoids adding a second KV-cache slot purely
for updates. A host-side deployment bootstrapper detects the target device and
starts the matching profile; the running gateway receives the resolved target
as immutable configuration.

## Scope and constraints

- Use a selected, manifest-pinned v2.x GGUF release artifact; do not download
  or build models inside the runtime request path.
- Render prompts through the release-aware shared prompt code. For v2.1 this
  means the v2 positional-ID user message inside the model's chat template,
  with Qwen thinking disabled.
- The model output is a positional ID (`a` through `z`, then `A` through `Z`)
  or `-`. The public canonical no-tool result is `none`.
- A tool's displayed order assigns its positional ID. Preserve submitted tool
  order; never sort tool definitions.
- Expose only `prefix_cache`; do not reintroduce `kv_cache` or `no_cache`.
- Keep the benchmark pipeline and its engine evaluators separate from service
  inference. The service may reuse shared prompt/output compatibility code but
  must not duplicate benchmark implementations.
- TensorRT-LLM is not part of the Jetson Xavier service path.

## Target architecture

```text
client/admin
    |
    +--> gateway container (FastAPI)
    |      - validate, version, and persist the active ordered tool set
    |      - render the exact release prompt
    |      - warm and use llama-server's prompt cache
    |      - constrain, validate, and map the generated positional ID
    |
    +--> llama-server container (internal network only)
           - pinned GGUF model, --parallel 1, prompt cache enabled
```

Mount model artifacts read-only. Publish only the gateway port. The inference
container should not be reachable from the LAN or the Internet directly.

## Public and administrative contract

Initial endpoints:

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/classify` | Classify `{ "query": "..." }` using the active tool set. |
| `PUT /v1/toolset` | Authenticated update of the complete ordered tool set. |
| `GET /health/live` | Gateway process is alive. |
| `GET /health/ready` | Model is healthy and one active tool-set prefix is warm. |
| `GET /v1/toolset/status` | Authenticated active version, count, and warm state. |

`POST /v1/classify` returns the raw model ID and canonical name:

```json
{
  "tool_id": "d",
  "tool": "call_handler",
  "toolset_version": "<content-hash>"
}
```

For no match it returns `{ "tool_id": "-", "tool": "none" }`. A malformed
model response must also resolve safely to `none`; an engine timeout or other
infrastructure failure must return `503`, not a false no-tool classification.

The tool-update endpoint accepts ordered `{name, description}` objects, not
client-provided model IDs. Reject duplicate names, empty fields, control
characters/newlines in names, and inputs that exceed the configured count or
token budget. Protect this endpoint with an administrative credential or a
private administrative network.

## Phases

### 0. Multi-target deployment contract

- Use `model_selection_matrix.json` as the initial service selection input.
  It pins v2.1 Qwen2.5-0.5B Q6_K for every supported profile. The only
  operator-selected deployment profiles are `cpu` (Mac CPU, Raspberry Pi, and
  Jetson CPU) and `jetson_gpu` (Jetson CUDA). There is no device auto-detection
  or model selection at runtime.
- Use positional-ID inference only. The gateway maps the valid raw model ID to
  the corresponding submitted tool name for its public response; `-` or a
  malformed output maps to canonical `none`. Do not deploy direct model
  `tool_name` output.
- Benchmark the same pinned model on every physical target profile using
  `test_anchor`. Target measurements validate the profile; they do not select
  a different model or quantization.
- Record the common model release commit and GGUF quantization, plus each
  profile's context limit, CPU thread/GPU-offload settings, and tested
  llama.cpp image digest in deployment configuration files.
- Establish product limits after tokenizing real worst-case inputs: maximum 52
  tools, maximum description/query tokens, and a context safety margin.

**Exit criteria:** the shared model configuration and both deployment profiles
are documented. Every published target has its own target-device benchmark
evidence. Run the full `test` split once only after each final target
configuration is validated and record it as final-selection validation.

### 1. Domain layer and prompt compatibility

- Create the service package and typed models for `Tool`, `ToolSet`,
  `ToolConfig`, and classification responses.
- Reuse the manifest prompt specification, positional-ID mapping, and output
  parser from `evaluation_lib`; extract a small shared helper only if required
  to avoid importing evaluator runtime code.
- Build the chat-template prompt with the same tokenizer files and rendering
  options used by the selected benchmark artifact.
- Derive a deterministic content hash from manifest version, prompt template,
  and the ordered normalized tools. Use it for idempotent updates, logging,
  and rollback metadata.
- Generate a request-specific GBNF grammar permitting exactly the active IDs
  plus `-`; retain application-side membership validation as the final guard.

**Exit criteria:** unit tests prove service prompt text/token IDs and ID-to-name
mapping match the v2.1 evaluator for representative data, including `none`.

### 2. llama-server integration and cache lifecycle

- Add a small asynchronous llama-server client with explicit timeouts and
  response-schema checks.
- Run the selected pinned llama.cpp server with a read-only model mount,
  `--parallel 1`, an explicit context limit, prompt caching, and metrics.
- On startup, restore the persisted active tool set and warm it before the
  readiness endpoint returns success.
- On `PUT /v1/toolset`: validate, render, and tokenize the candidate; acquire
  the shared inference lock; prefill the exact empty-query prompt with
  `cache_prompt: true` and `n_predict: 0`; publish the new configuration only
  after a successful warm-up; then release queued classifications.
- Classify using the active immutable config snapshot. Record cache/warm timing
  and tool-set version with every request.

**Exit criteria:** the first request after a successful update reuses the new
prefix; no request observes a partial update; a failed warm-up leaves the
previous tool set active.

### 3. Gateway safety and operations

- Limit request body size, concurrent/queued requests, update frequency, and
  inference timeouts. Return an explicit overload response when the queue is
  full.
- Keep user query text and tool descriptions out of default logs; log hashes,
  lengths, timing, model/artifact provenance, and outcome status instead.
- Implement liveness/readiness checks that combine gateway state with the
  internal llama-server health endpoint.
- Add structured logs and metrics for request count, queue depth, update and
  warm duration, cache reuse, inference duration, invalid outputs, and errors.
- Document an authenticated rollback path to a previously persisted tool-set
  version.

**Exit criteria:** fault-injection tests cover server startup delay, timeout,
malformed completion, failed warm-up, gateway restart, queue saturation, and
tool-set rollback.

### 4. Container packaging

- Add a gateway Dockerfile with pinned Python dependencies and a non-root user.
- Add a Compose deployment that starts the gateway only after llama-server is
  healthy, uses an internal network for inference, and mounts model/tokenizer
  assets read-only.
- Add a host-side bootstrap command that validates the operator-selected
  deployment profile, downloads the selected immutable release artifact before
  startup, and invokes the matching Compose profile. It must never download a
  model in the gateway request path.
- Provide separate pinned deployment profiles:
  - `cpu-multiarch` for Mac CPU and Raspberry Pi.
  - `cuda-arm64` matched to the Jetson's JetPack/L4T and CUDA compatibility.
- Do not use Docker-on-macOS results as a proxy for Metal deployment: Docker's
  Linux VM does not provide the native Metal runtime path.
- Apple GPU/MPS/Metal is not a service deployment profile. The service
  supports `mac_cpu` through the CPU-only multi-architecture container profile;
  the accelerated production profile is `cuda-arm64` (Jetson).
- Verify image architecture and llama.cpp build support on each target before
  publishing an image digest.

**Exit criteria:** the complete Compose bundle starts offline with provisioned
artifacts, returns ready only after warming, and exposes no public inference
port.

### 5. Verification and release

- Add unit tests for validation, hashing, grammar construction, prompt
  equivalence, parsing, cache-update atomicity, and endpoint error contracts.
- Add container integration tests with a controllable llama-server substitute,
  then run the same cases against the real selected GGUF artifact.
- Measure cold start, warm classification latency, p50/p95 latency, memory,
  tool-update interruption, and accuracy across tool-set sizes (5, 10, 20, 35,
  and 52) including close-match and no-tool requests.
- Publish an immutable deployment manifest containing image digests, the model
  release commit, artifact checksums, server flags, test results, and rollback
  instructions.

**Exit criteria:** target-device acceptance results meet the selected latency,
memory, and quality budgets and are reproducible from immutable provenance.

## Decisions to resolve before implementation

| Decision | Why it matters |
| --- | --- |
| First target device | Determines the CPU/CUDA image, runtime flags, and acceptance measurements. |
| Selected v2.1 model and GGUF quant | Must come from target-device `test_anchor` evidence, not historical reports. |
| Token/context limits | Bounds KV memory, warm-up time, and tool-set admission. |
| Tool-set persistence and admin auth | Determines restart behaviour and protects runtime routing configuration. |
| Latency and update-interruption budget | Determines queue size, timeout, and whether `--parallel 1` remains sufficient. |

## Acceptance checklist

- [ ] Exact v2.1 prompt/template compatibility is regression-tested.
- [ ] The service returns only an active tool name or canonical `none`.
- [ ] A tool-set update is validated, warmed, and atomically activated.
- [ ] A failure during update or inference is observable and never misreported
      as a successful tool routing decision.
- [ ] Images, model artifact, and server version are pinned and reproducible.
- [ ] Only the gateway is externally reachable; administrative updates are
      authenticated.
- [ ] The selected target has passed anchor benchmarking and one final
      full-test validation for the deployment configuration.
