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
[Admin API] -> guardrails, rate limits, API keys
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

## Modules
- **Runtime** (`runtime/`): `DeviceContext` abstraction, `CPUDeviceContext` implementation, `CudaDeviceContext` placeholder (FlashAttention-ready), Metal/MPS + BLAS autodetection (via `LLAMA_METAL`/`LLAMA_BLAS`) so hardware accelerators are toggled automatically, `PagedKVCache` with LRU/Clock eviction and async NVMe offload, speculative decoding (draft + validator), `BackendManager` for named model loading.
- **Prefix Cache** (`runtime/prefix_cache/`): LRU cache of prompt token prefixes so repeated prompts can skip generation; metrics track hit/miss rates to feed fairness work.
- **Constrained Decoder** (`scheduler/constrained_decoder.*`, planned): Grammar/JSON aware decoding path that consumes scheduler outputs before runtime execution to ensure schema compliance.
- **Prefix Cache** (`runtime/prefix_cache.*`, planned): Radix-tree cache of validated KV prefixes shared across tenants for agent workflows and speculative reuse.
- **Multimodal Adapter** (`runtime/multimodal/` planned): Pre/post-processing stage that converts base64 or URL-sourced images/audio into llama.cpp `libmtmd` tensors before scheduling.
- **Model** (`model/`): `SimpleTokenizer` (whitespace/punctuation splitter — stub for real tokenizer), GGUF loader (delegates to llama.cpp submodule).
- **Scheduler** (`scheduler/`): `Scheduler` with global mutex (to be replaced by continuous batching via `RequestBatch`). `ModelRouter` interface for multi-model routing. `GenerateRequest`/`GenerateResponse` DTOs.
- **Server** (`server/`): Multi-threaded `HttpServer` (accept loop + worker thread pool), SSE streaming, `ApiKeyAuth` (SHA-256 hashed), `OIDCValidator` (RS256 verification), `RateLimiter` (per-key sliding window), `Guardrail` (blocklist + OPA), `AuditLogger` (JSON lines), `MetricsRegistry` (Prometheus counters).
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
