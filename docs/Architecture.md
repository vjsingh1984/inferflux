# InferFlux Architecture

## System Overview
```
[Clients / SDKs]
      |
[HTTP Frontend] -- SSE --> streaming clients
      |  (thread pool: accept loop + N worker threads)
      |
[Auth Middleware] -- API-key / OIDC / rate limiter
      |
[Scheduler + Admission Control]
      |
      +--> [ModelRouter]  -- resolves model ID --> backend
      |       +--> [BackendManager] -- loads --> LlamaCPUBackend (llama.cpp)
      |
      +--> [Runtime Core]
              |- DeviceContext (CUDA/ROCm/MPS/CPU)
              |- Paged KV cache (LRU/Clock eviction, NVMe offload)
              |- Speculative decoding (draft + validator)
      |
[PolicyBackend] -- pluggable: INI store / OPA / Cedar / SQL
[Telemetry] -> Prometheus /metrics
[Admin API] -> guardrails, rate limits, API keys, models
[Security] -> OIDC provider / RBAC / Audit log
```

## Plugin Interfaces

InferFlux uses abstract interfaces at key boundaries to enable pluggable backends
without touching core server code. Each interface is a pure-virtual C++ class.

| Interface | Header | Purpose | Current Impl |
|-----------|--------|---------|--------------|
| `PolicyBackend` | `policy/policy_backend.h` | Policy storage and enforcement (API keys, guardrails, rate limits) | `PolicyStore` (encrypted INI) |
| `ModelRouter` | `scheduler/model_router.h` | Multi-model serving — list, load, unload, resolve models | *(interface only — wire-up pending)* |
| `DeviceContext` | `runtime/device_context.h` | Hardware abstraction — allocate, free device memory | `CPUDeviceContext` |

The `RequestBatch` struct (`scheduler/request_batch.h`) defines per-request state
(`InferenceRequest`) and batch grouping for continuous batching. It is the foundation
for replacing the current global-mutex scheduler with a batched pipeline.

## ModelRouter Activation Plan
### Current Gaps (Before)
- `ModelRouter` exists only as an interface (`scheduler/model_router.h`); the scheduler still owns a single `LlamaCPUBackend`.
- No persistence for model metadata, so APIs and CLI cannot list/load/unload models.
- Guardrails, rate limiter, and metrics are scoped to a single model ID, blocking per-model isolation and future workloads such as adapter hot reloads or A/B testing.

### Target Design (After)
- Scheduler depends only on `ModelRouter`, requesting model handles via `Resolve()` for every request. This swaps in multi-backend implementations (CPU, CUDA, MPS) without touching HTTP code.
- A default `SingleModelRouter` manages an in-process registry and reference-counted backend instances. Subsequent routers (e.g., `MultiModelRouter`, `RemoteRouter`) can extend the same interface.
- Model metadata (ID, backend type, KV footprint, health) is surfaced to HTTP/admin APIs and Prometheus metrics so autoscalers and operators can see routing decisions.
- CLI + config accept `models[].path/backend/role` arrays enabling declarative boot-time loading while keeping dynamic load/unload APIs for later.

### Execution Checklist
1. **Implement `SingleModelRouter`** — create `scheduler/single_model_router.{h,cpp}` that owns a map of `ModelInfo` → backend shared_ptr and satisfies the existing interface. Cover load/unload/list flows with unit tests.
2. **Scheduler integration (ARCH-4/5)** — refactor `scheduler/scheduler.{h,cpp}` so request admission calls `router->Resolve(request.model_id)` and batches carry resolved backend handles. Remove direct `LlamaCPUBackend` members.
3. **Model registry plumbing** — add a lightweight registry struct tracking path, backend hint, KV requirements, and readiness timestamps. Persist it in memory for now, but emit structured logs + metrics (`inferflux_model_routes_total`) for observability.
4. **Control-plane surfaces** — extend `server/http/http_server.cpp` admin routes plus `inferctl` to list models, trigger load/unload, and report errors. Respect scopes (`admin`) and guard concurrent loads via mutexes.
5. **Configuration + bootstrapping** — introduce `models` array in `config/server.yaml` and corresponding environment overrides so operators can pin a default model but opt into multi-model at startup.
6. **Testing + rollout** — unit tests for router operations, scheduler integration tests verifying multi-model routing, and CLI smoke tests. Update `/readyz` to cover “at least one ready model” semantics.

This plan unlocks PRD user stories that require per-tenant routing, enables Workstream A throughput milestones (multi-backend dispatch), and unblocks Workstream C (`inferctl pull`) by giving the CLI a first-class registry interface.

## RequestBatch Integration Plan
### Current Gaps (Before)
- The scheduler builds batches as plain vectors of `PendingRequest` and immediately executes them, so phases (`prefill`, `decode`) and token accounting are ad hoc counters instead of using `RequestBatch`.
- HTTP handlers now emit `InferenceRequest`, but batches still rely on bespoke `PendingRequest` bookkeeping and cannot share cancellation/streaming metadata with the executor.
- Prefill/decode overlap, fairness preemption, and GPU stream scheduling from the PRD/Roadmap depend on phase-aware batches, but the existing executor treats each request independently.

### Target Design (After)
- HTTP handlers materialize `InferenceRequest` objects (from `scheduler/request_batch.h`) for every call, including per-request scopes (priority, streaming hooks, token buffers) so the scheduler can operate solely on the portable struct.
- A `BatchBuilder` component groups requests into `RequestBatch` instances keyed by phase and token budgets, recording metrics/timestamps before handing them to the executor.
- The `BatchExecutor` consumes full batches, updates each `InferenceRequest` phase (`kPrefill` → `kDecode` → `kFinished`), and exposes hooks for streaming/cancellation plus future GPU dispatch.
- Scheduler queue + metrics emit per-batch stats (tokens, wait latency, first-token time) to unlock fairness, preemption, and Workstream A KPIs.

### Execution Checklist
1. **(Done)** InferenceRequest adoption — HTTP server, scheduler, and tests now pass `InferenceRequest` end-to-end so prompts, priorities, and streaming callbacks live in a single struct, with `InferenceResult` reported back to HTTP handlers.
2. **BatchBuilder abstraction** — split `BuildBatchLocked` into a dedicated helper that outputs `RequestBatch` objects with explicit token budgets, priority ordering, and phase tagging. Wire queue metrics to the struct instead of ad hoc loops.
3. **Prefill/decode bookkeeping** — teach `BatchExecutor` to iterate `RequestBatch` entries, adjust `RequestPhase`, and support staged execution (prefill pass populates KV cache, decode loops reuse cached context). This sets the stage for GPU stream handoffs.
4. **Streaming & cancellation plumbing** — connect `InferenceRequest.on_token` to HTTP streaming handlers so SSE/websocket updates originate from the scheduler, and expose cancellation flags for future `inferctl cancel` flows.
5. **Metrics + observability (Done)** — batch prefill/decode durations now feed `RecordPrefillDuration`/`RecordDecodeDuration` counters, exposed via `/metrics`.
6. **Streaming & cancellation (Done)** — HTTP SSE uses `InferenceRequest.on_token` callbacks and shared cancellation flags.
7. **Fairness & preemption (Next)** — priority aging feeds a preemption queue; add starvation tests and SSE cancellation fuzzing before GPU overlap work.

## Fairness & Preemption Plan
### Current Gaps (Before)
- Priority aging influences queue ordering but once a batch starts, there is no yield/preemption. Long prompts can still starve high-priority requests even on CPU/MPS hardware.
- Cancellation currently hinges on SSE disconnects; there is no fairness controller to requeue/abort inflight requests with policy awareness.

### Target Design (After)
- Introduce a `FairnessController` that scores pending requests by priority + wait time, enforces per-priority token budgets, and can signal `BatchExecutor` to yield after a timeslice.
- `RequestBatch` carries fairness metadata (priority level, service tokens consumed, cancellation flag). Yielded requests are reinserted with updated timestamps so aging continues.
- Metrics expose fairness data (`inferflux_preemptions_total`, per-priority token counters) so operators can see the impact of policy changes.

### Execution Checklist
1. **Fairness metadata** — extend `InferenceRequest` with `priority_level`, `service_tokens`, and `timeslice_tokens`. Add tracing/log hooks to show fairness decisions.
2. **Controller hook** — implement a fairness controller invoked after `BuildBatchLocked` that can drop/deferral requests and trim batches to respect per-priority quotas.
3. **Timeslice support** — teach `BatchExecutor` to stop after `timeslice_tokens` for any request marked as “yieldable”, re-queueing the request with remaining tokens.
4. **Cancellation tests** — extend `tests/integration/sse_cancel_test.py` and add scheduler unit tests that simulate cancellation during preemption to ensure no deadlocks.
5. **Metrics & observability** — add counters/histograms for preemptions, per-priority tokens, and fairness queue depth. Update `/metrics` documentation accordingly.
6. **Config + docs** — document new knobs (`scheduler.fairness.timeslice_ms`, `priority_weights`) in PRD/Roadmap, and add troubleshooting guidance for operators tuning fairness.

This plan completes ARCH-5, unlocks PRD Continuous Batching and Roadmap Workstream A deliverables (prefill/decode overlap, GPU scheduling), and lays the groundwork for Q4 fairness/SLO scheduling.

### Fairness Telemetry (OBS-1)
- `/metrics` now exports `inferflux_fairness_preemptions_total`, `inferflux_fairness_yields_total`, and `inferflux_fairness_resumes_total` so operators can trace when the scheduler swaps or slices requests.
- Per-priority token consumption is exposed via `inferflux_fairness_tokens_total{priority="<level>"}` alongside the existing queue-depth gauge (`inferflux_scheduler_queue_depth`), making it easy to verify SLO budgets across classes.
- Spans `scheduler.fairness.yield` and `scheduler.fairness.resume` wrap surrender/resume cycles with trace IDs, mirroring the new Prometheus counters for log-based investigations.

## Structured Output & Grammar
InferFlux now implements the PRD §Functional #8 requirements end-to-end:

1. **Request Parsing** — `server/http/http_server.cpp` accepts OpenAI-style `response_format` payloads (`json_object`, `json_schema`, or `grammar`) with a 16 KB size cap, normalizes schemas into strings, and persists capability flags on `InferenceRequest`.
2. **Schema Adaptation** — `runtime/structured_output/structured_output_adapter.*` feeds nlohmann JSON schemas through llama.cpp’s `json_schema_to_grammar` helper, producing backend-ready GBNF along with an optional “require JSON object” bit for downstream validation.
3. **Scheduler Plumbing** — `InferenceRequest` tracks `StructuredConstraint` data, and `Scheduler::ProcessBatch` rejects requests that target models without structured-output support before they enter the execution pipeline.
4. **Runtime Execution** — `BatchExecutor` uses the constraint to disable speculative decoding, attach llama grammar samplers via `LlamaCPUBackend::EnableGrammarConstraint`, and verify that any JSON-mode response is a valid object before returning it to HTTP clients.
5. **Sampling** — `LlamaCPUBackend` now wraps llama.cpp’s sampler chain (grammar + greedy) so decoding is constrained on-token rather than relying on best-effort post-processing. Grammar scopes are released after each request to keep shared backends safe.

Contract tests (`tests/unit/test_structured_output.cpp`) cover schema validation/grammar emission, and the existing SSE/tooling tests ensure the HTTP surface emits structured responses. Future backends (CUDA/MPS) and CLI docs reuse the same adapter interface, so extending structured output beyond CPU is now an incremental task instead of a rewrite.

### Adapter Pattern & Backend Capabilities
- **StructuredOutputAdapter** — Maintain OpenAI’s `response_format` as the public contract but introduce an internal adapter interface that validates payloads, enforces size limits, and emits backend-native constraints (e.g., llama.cpp GBNF via `json_schema_to_grammar`, regex/DSL for future engines).
- **Backend capability flags** — Extend `ModelRouter`/`BackendManager` to advertise whether a backend supports structured output and which adapter type it requires. Scheduler consults these flags so unsupported combinations fail fast.
- **Pluggable sampler hooks** — Wrap llama.cpp grammar sampler creation in a small `StructuredConstraint::Attach(BackendContext&)` helper so CUDA/ROCm/MPS backends (or entirely new runtimes) can implement constraint enforcement without touching HTTP or scheduler code.

## Modules
- **Runtime** (`runtime/`): `DeviceContext` abstraction, `CPUDeviceContext` implementation, `CudaDeviceContext` placeholder (FlashAttention-ready), Metal/MPS + BLAS autodetection (via `LLAMA_METAL`/`LLAMA_BLAS`) so hardware accelerators are toggled automatically, `PagedKVCache` with LRU/Clock eviction and async NVMe offload, speculative decoding (draft + validator), `BackendManager` for named model loading.
- **Prefix Cache** (`runtime/prefix_cache/`): LRU cache of prompt token prefixes so repeated prompts can skip generation; metrics track hit/miss rates to feed fairness work.
- **Constrained Decoder** (`scheduler/constrained_decoder.*`, planned): Grammar/JSON aware decoding path that consumes scheduler outputs before runtime execution to ensure schema compliance.
- **Prefix Cache** (`runtime/prefix_cache.*`, planned): Radix-tree cache of validated KV prefixes shared across tenants for agent workflows and speculative reuse.
- **Multimodal Adapter** (`runtime/multimodal/` planned): Pre/post-processing stage that converts base64 or URL-sourced images/audio into llama.cpp `libmtmd` tensors before scheduling.
- **Model** (`model/`): `SimpleTokenizer` (whitespace/punctuation splitter — stub for real tokenizer), GGUF loader (delegates to llama.cpp submodule).
- **Scheduler** (`scheduler/`): `Scheduler` with global mutex (continuous batching rewrite in progress via `RequestBatch`). `ModelRouter` interface for multi-model routing. Requests flow through `InferenceRequest`/`RequestBatch`, and results are returned as `InferenceResult` for HTTP compatibility.
- **Scheduler** (`scheduler/`): Request queue is priority-aware with aging (older requests automatically boost priority) to prevent starvation; batches capture queue wait time + execution time metrics.
- **Server** (`server/`): Multi-threaded `HttpServer` (accept loop + worker thread pool), SSE streaming via `InferenceRequest.on_token` callbacks, `ApiKeyAuth` (SHA-256 hashed), `OIDCValidator` (RS256 verification), `RateLimiter` (per-key sliding window), `Guardrail` (blocklist + OPA), `AuditLogger` (JSON lines), `MetricsRegistry` (Prometheus counters + batch histograms), and `server/tracing` for W3C trace propagation.
- **Policy** (`policy/`): `PolicyBackend` abstract interface, `PolicyStore` concrete implementation (encrypted INI with AES-GCM via OpenSSL). Admin APIs for CRUD on API keys, guardrails, and rate limits.
- **CLI** (`cli/`): `inferctl` with subcommands: `status`, `completion`, `chat` (interactive + streaming), `admin` (guardrails, rate-limit, api-keys).
- **Net** (`net/`): Shared `HttpClient` used by both CLI and OPA client.

### Scheduler & Continuous Batching
- HTTP handlers translate each request into an `InferenceRequest` (priority, enqueue timestamp, prompt tokens) and push it onto the scheduler queue.
- A background scheduler thread repeatedly builds `RequestBatch` objects by priority/age, bounded by token budget (`kMaxBatchTokens`) and max batch size. Each batch records timing/metrics and emits OpenTelemetry + Prometheus data (queue depth, batch size).
- Before executing, the scheduler reserves KV cache pages for the batch, resolves the target backend via `ModelRouter`, and transitions requests through `prefill`/`decode` phases.
- After generation, KV pages are released, metrics updated, and futures fulfilled so HTTP threads can stream completions. The batch structure keeps the door open for future prefill/decode overlap and fairness/preemption without touching HTTP code.
- **Upcoming GPU path**: The same batch abstraction will dispatch to CUDA/FlashAttention executors once the CUDA backend and FlashAttention kernels are in place (Workstream A). The scheduler already tags requests by phase and priority so GPU streams can overlap prefill/decode.

## Data Flow
1. Client sends HTTP request with prompt + parameters (optionally including base64/URL media descriptors).
2. Thread pool worker reads request (dynamic buffer, 16MB max, Content-Length aware).
3. CORS preflight (OPTIONS) handled immediately. Other requests proceed to auth.
4. Auth middleware validates API key (SHA-256 hash match) or OIDC JWT (RS256 signature + exp/nbf/iss/aud).
5. Rate limiter checks per-key token budget. Guardrail checks blocklist + optional OPA endpoint.
6. Multimodal Adapter (planned) resolves media payloads to tensors; Scheduler tokenizes text components.
7. Prefix Cache (planned) checks for matching KV prefixes; hits skip prefill and emit cache handles to runtime.
8. Constrained Decoder enforces grammar/JSON schemas before tokens are committed to the runtime backend.
9. Response sent as JSON (non-streaming) or SSE chunks (streaming).
10. Metrics counters updated. Audit logger writes JSON line if enabled.

## Health Probes
- `/healthz` — returns `{"status":"ok"}` when model loaded, `{"status":"degraded","reason":"model_not_loaded"}` otherwise. Always 200.
- `/readyz` — returns 200 when model is loaded and ready. Returns 503 `{"status":"not_ready"}` otherwise.
- `/livez` — always returns 200 `{"status":"alive"}`. Confirms the event loop is running.

## Deployment View
- **Standalone**: single binary, config YAML, runs on developer desktops (CPU/MPS).
- **GPU Node**: Docker image with CUDA runtime, NVMe-backed cache, Prometheus sidecar.
- **Kubernetes**: Helm chart provisions ConfigMaps/Secrets, Horizontal Pod Autoscaler based on queue depth. Use `/readyz` for readiness probe, `/livez` for liveness probe.

## Security & Operations
- API keys are SHA-256 hashed on write/read via `PolicyStore::SetApiKey` and stored in AES-GCM encrypted INI files when a passphrase is supplied.
- OIDC validator now fetches JWKS documents from `{issuer}/.well-known/jwks.json`, caches keys, and verifies RS256 signatures while enforcing issuer/audience/expiry.
- HttpServer can terminate TLS directly via OpenSSL when `server.tls.enabled` is set, or rely on upstream ingress when disabled.
- Health checks expose readiness (model loaded) vs liveness (event loop running).
- Audit logs hash prompts/responses by default via `AuditLogger::LogRequest` and only emit raw text when `debug_mode` is true.
- Guardrail hooks: blocklist-based content filtering and OPA policy evaluation.
