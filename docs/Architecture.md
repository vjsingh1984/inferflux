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

## Fairness & Preemption — **Implemented** (§2.9)

### Design
- `FairnessController` (`scheduler/fairness_controller.h/.cpp`) scores pending vs in-flight requests by priority + wait-time, enforces per-priority timeslice token budgets, and signals `BatchExecutor` to yield after a timeslice.
- `InferenceRequest` carries fairness fields: `priority_level`, `timeslice_tokens`, `remaining_decode_tokens`, `accumulated_output`, and `fairness_yielded`. Yielded requests are requeued with refreshed timestamps so priority aging continues.
- `Scheduler::ApplyFairness()` evaluates `FairnessDecision` after `BuildBatchLocked`; yielded requests are set `fairness_yielded=true` and reinserted into `pending_` without completing their futures.
- `Scheduler::UpdateFairnessConfig()` accepts live `FairnessConfig` updates so operators can tune thresholds without a restart.

### Configuration Knobs (`scheduler.fairness.*`)
| Knob | Env Var | Default | Effect |
| --- | --- | --- | --- |
| `enable_preemption` | `INFERFLUX_FAIRNESS_PREEMPT` | `false` | Allow preempting in-flight low-priority requests |
| `high_priority_threshold` | `INFERFLUX_FAIRNESS_HIGH_PRI` | `5` | Priority level at or above which requests are high-priority |
| `max_timeslice_tokens` | `INFERFLUX_FAIRNESS_TIMESLICE` | `0` (disabled) | Max decode tokens per scheduling slice for low-priority requests |

### Priority Queue Design
- Requests enter `pending_` with `priority` (integer, higher = more important) and `enqueue_time`.
- `BuildBatchLocked` sorts candidates by effective priority = `priority + age_boost`, where `age_boost` increments by 1 per 2 seconds of wait — preventing starvation.
- Batch is capped at `kMaxBatchSize=4` requests and `kMaxBatchTokens=8192` tokens.
- `FairnessController::ApplyTimeslice()` caps remaining decode length for low-priority requests when `max_timeslice_tokens > 0`.

### Metrics
- `inferflux_fairness_preemptions_total` — requests preempted (swapped out for higher-priority)
- `inferflux_fairness_yields_total` — requests yielded at timeslice boundary
- `inferflux_fairness_resumes_total` — yielded requests resumed after requeue
- `inferflux_fairness_tokens_total{priority=”<level>”}` — tokens generated per priority level
- `Span` events: `scheduler.fairness.yield` and `scheduler.fairness.resume` with trace IDs

### Implementation Checklist
1. [x] `InferenceRequest` extended with `priority_level`, `timeslice_tokens`, `remaining_decode_tokens`, `accumulated_output`, `fairness_yielded`.
2. [x] `FairnessController` implemented with `ApplyTimeslice()` and `FairnessDecision` output.
3. [x] `BatchExecutor` stops after `timeslice_tokens` and signals yield; `ProcessBatch` assembles `accumulated_output` across slices.
4. [x] 3 `[fairness]` unit tests in `test_scheduler.cpp` covering timeslice cap, yield/resume flow, and priority ordering.
5. [x] Fairness metrics (`RecordFairnessTokens`, `RecordFairnessPreemption`, `RecordFairnessYield`, `RecordFairnessResume`) wired in `MetricsRegistry`.
6. [x] `Span` hooks on yield and resume events for trace correlation.

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

## Phased Prefill/Decode — Implemented (§2.5 Option A)

InferFlux now runs prefill and decode as distinct phases within the same process, using
llama.cpp’s per-sequence KV cache to give each request an isolated state slot. This delivers
phase visibility, per-request KV lifetime control, and the hook structure needed for future
cross-process disaggregation — without requiring KV serialization or upstream llama.cpp changes.

### New LlamaCPUBackend Methods
| Method | Description |
| --- | --- |
| `Prefill(prompt, seq_id) → PrefillResult{n_past, ok}` | Clears slot `seq_id` via `llama_memory_seq_rm`, evaluates all prompt tokens, returns `n_past` (= prompt token count). Returns `{ok=false}` on error. |
| `Decode(n_past, seq_id, max_tokens, …) → string` | Autoregressive generation from `n_past` for slot `seq_id`. Supports streaming callbacks and cancellation. |
| `FreeSequence(seq_id)` | Releases KV slot `seq_id` via `llama_memory_seq_rm`. |
| `BatchAddSeq(batch, token, pos, seq_id, logits)` | Sets per-token `seq_id` on a `llama_batch`, letting concurrent requests share one `llama_context` without KV collision. |

### Sequence Slot Allocator
`Scheduler` maintains `seq_slots_free_` — a `std::vector<bool>` of `kMaxSequenceSlots=16` entries (all `true` at init). `AllocSeqSlot()` does a first-fit scan and marks the slot busy; returns `-1` when all are taken (request falls back to the legacy `Generate()` path). `FreeSeqSlot(slot)` is called in the `ProcessBatch` post-execution loop after full completion. This replaces the earlier `request_id % kMaxSequenceSlots` assignment, which caused KV corruption when concurrent requests shared a modulo slot.

### Per-Request Lifecycle
1. `BuildBatchLocked` tags new requests `RequestPhase::kPrefill`.
2. `ProcessBatch` kPrefill block: `AllocSeqSlot()` → `backend->Prefill(prompt, seq_id)` → store `n_past` + `sequence_id` on `InferenceRequest`.
3. Request promoted to `kDecode`; `BatchExecutor::ExecuteRequest` branches on `n_past >= 0` and calls `Decode(n_past, seq_id, decode_limit, …)` instead of `Generate()`.
4. After each slice, `n_past += completion_tokens` so the next fairness slice resumes from the correct KV position.
5. **Full completion**: `BatchExecutor` calls `backend->FreeSequence(seq_id)` (KV memory); `Scheduler::ProcessBatch` calls `FreeSeqSlot(slot)` (slot integer).
6. **Fairness yield**: KV slot preserved; `n_past` advanced; request requeues into `pending_decode_` for the next slice.

### KVChannel (Stub — Future Distributed Transport)
`runtime/disaggregated/kv_channel.*` provides `KVChannel` (thread-safe bounded queue) and `KVPacket` (carries `request_id`, `n_past`, `sequence_id`, `kv_blob`, metadata). The channel is only populated when `use_decode_workers_=true`; while decode workers are disabled (current default), the kPrefill block sets `enqueued=true` directly and skips the channel entirely, preventing fill-and-deadlock.

### Configuration
| Constant / Field | Value | Effect |
| --- | --- | --- |
| `kMaxSequenceSlots` | 16 | Free-list size; must exceed `kMaxBatchSize` (4) |
| `use_decode_workers_` | `false` (hard-coded) | When `true`, prefill→decode hand-off goes via `KVChannel` + separate decode threads |

### Implementation Checklist
1. [x] `LlamaCPUBackend::Prefill()` / `Decode()` / `FreeSequence()` / `BatchAddSeq()`
2. [x] `InferenceRequest.n_past{-1}` / `sequence_id{-1}` (phased state)
3. [x] `KVPacket.n_past` / `sequence_id` (future transport)
4. [x] Sequence slot free-list (`seq_slots_free_`, `AllocSeqSlot()`, `FreeSeqSlot()`)
5. [x] `ProcessBatch`: allocate slot, call `Prefill()`, store n_past+sequence_id; return slot on full completion
6. [x] `BatchExecutor::ExecuteRequest`: branch on `n_past>=0` for `Decode()`; advance n_past per slice; `FreeSequence()` on non-yield completion only
7. [x] KVChannel gate: only enqueue when `use_decode_workers_=true`
8. [x] 11 `[phased]` unit tests (`tests/unit/test_phased_execution.cpp`)
9. [x] Wire `DecodeWorkerLoop` + enable `use_decode_workers_=true` when `decode_pool_size > 0`
10. [x] `SerializeSequence` / `HydrateSequence` via `llama_state_seq_get_data` / `llama_state_seq_set_data`
11. [ ] SHM/RDMA transport layer (<5 ms transfer SLA)
12. [ ] Dual-pool `/readyz`, transfer-latency Prometheus histograms, Helm overlays for independent scaling

## MoE Expert Parallelism (§2.6)

### Current State
MoE model detection is implemented in the CPU backend and router layer.  Full
expert parallelism (sharding expert layers across GPU ranks) is a future work
item once multi-GPU topology discovery lands.

| Component | Status | Notes |
|-----------|--------|-------|
| `LlamaCPUBackend::IsMoE()` | Done | Reads `llm.expert_count` GGUF key |
| `LlamaCPUBackend::ExpertCount()` | Done | Returns n_experts (0 for non-MoE) |
| `LlamaCPUBackend::ActiveExperts()` | Done | Returns `llm.expert_used_count` |
| `ModelInfo.is_moe / n_experts / n_active_experts` | Done | Populated in `SingleModelRouter` after load |
| `RecordMoERequest()` Prometheus counter | Done | `inferflux_moe_requests_total` |
| `EPDispatch` stub interface | Done | `runtime/backends/ep_dispatch.h`; `LocalEPDispatch` owns all experts |
| Multi-GPU expert sharding | Pending | Requires NCCL/topology discovery |

### EPDispatch Interface
```cpp
// runtime/backends/ep_dispatch.h
class EPDispatch {
  virtual EPRank LocalRank() const = 0;   // rank, world_size, [expert_start, expert_end)
  virtual bool OwnsExpert(int expert_id) const = 0;
  virtual std::string Name() const = 0;
};
class LocalEPDispatch : public EPDispatch { ... }; // single-process default
```

### Implementation Checklist
1. [x] `IsMoE()`, `ExpertCount()`, `ActiveExperts()` on `LlamaCPUBackend`
2. [x] `ModelInfo`: `is_moe`, `n_experts`, `n_active_experts` (GGUF metadata)
3. [x] `SingleModelRouter::RegisterModel/LoadModel` populates MoE fields
4. [x] `RecordMoERequest()` metric + Prometheus output
5. [x] `EPDispatch` + `LocalEPDispatch` stub (`runtime/backends/ep_dispatch.h`)
6. [x] 9 `[moe]` unit tests (`tests/unit/test_moe_routing.cpp`)
7. [ ] Multi-GPU expert sharding via NCCL or shared-memory ring
8. [ ] EP-aware batch routing in `BatchExecutor`

### Adapter Pattern & Backend Capabilities
- **StructuredOutputAdapter** — Maintain OpenAI’s `response_format` as the public contract but introduce an internal adapter interface that validates payloads, enforces size limits, and emits backend-native constraints (e.g., llama.cpp GBNF via `json_schema_to_grammar`, regex/DSL for future engines).
- **Backend capability flags** — Extend `ModelRouter`/`BackendManager` to advertise whether a backend supports structured output and which adapter type it requires. Scheduler consults these flags so unsupported combinations fail fast.
- **Pluggable sampler hooks** — Wrap llama.cpp grammar sampler creation in a small `StructuredConstraint::Attach(BackendContext&)` helper so CUDA/ROCm/MPS backends (or entirely new runtimes) can implement constraint enforcement without touching HTTP or scheduler code.

## Modules
- **Runtime** (`runtime/`): `DeviceContext` abstraction, `CPUDeviceContext` implementation, `CudaDeviceContext` placeholder (FlashAttention-ready), Metal/MPS + BLAS autodetection (via `LLAMA_METAL`/`LLAMA_BLAS`) so hardware accelerators are toggled automatically, `PagedKVCache` with LRU/Clock eviction and async NVMe offload, speculative decoding (draft + validator), `BackendManager` for named model loading.
- **Prefix Cache** (`runtime/prefix_cache/`): `RadixPrefixCache` — a compressed trie (radix tree) over token ID sequences. `Lookup` returns the longest matching prefix plus a `matched_tokens` out-parameter for partial-hit metrics; `Insert` splits trie edges on mismatch. LRU eviction prunes the least-recently-used completion leaf when the tree exceeds capacity. Exact-match completions skip generation entirely; partial matches are tracked via `inferflux_prefix_matched_tokens_total` and `inferflux_prefix_partial_hits_total` Prometheus counters for future KV page reuse analysis. The original `PrefixCache` (flat hash-map) is retained for reference but is no longer used at runtime.
- **Constrained Decoder** (`scheduler/constrained_decoder.*`, planned): Grammar/JSON aware decoding path that consumes scheduler outputs before runtime execution to ensure schema compliance.
- **Multimodal Adapter** (`runtime/multimodal/`): `ImagePreprocessor` parses OpenAI `image_url` content arrays, decodes base64 data URIs / fetches HTTP URLs, generates SHA-256 image IDs, and injects `<__media__>` markers into the flattened prompt. `LlamaCPUBackend::LoadMmproj()` / `GenerateWithImages()` integrate libmtmd for actual vision inference (requires `-DENABLE_MTMD=ON`).
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
6. `ImagePreprocessor` extracts image_url parts from content arrays, decodes/fetches images, and injects `<__media__>` markers; Scheduler tokenizes the flattened text prompt.
7. `RadixPrefixCache` checks for exact-match completions (skip generation) and records partial prefix depth for future KV reuse metrics.
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
