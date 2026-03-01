# InferFlux Roadmap

## Q2 — MVP (complete)
- [x] CPU & MPS backends with SSE streaming via llama.cpp.
- [x] Policy store with RBAC scopes, guardrail/rate-limit admin APIs (AES-GCM encryption).
- [x] CLI interactive mode (`inferctl chat --interactive`), admin commands.
- [x] Prometheus metrics endpoint (`/metrics`).
- [x] Audit logging (JSON lines).
- [x] API-key auth (SHA-256 hashed) + OIDC JWT validation (RS256 + exp/nbf).
- [x] Adopted nlohmann/json (DAT-1) and Catch2 test framework (TST-2).
- [x] 44 unit tests across 9 modules (TST-1).
- [x] Dynamic HTTP buffer (INF-3), graceful shutdown (CQ-5).
- [x] Security fixes: OPA JSON injection (SEC-3), audit logger injection (SEC-4).
- [x] Thread safety: ApiKeyAuth, RateLimiter, MetricsRegistry, BackendManager.
- **KPIs**: 30 tok/s/request on CPU; policy updates <250 ms; admin CLI task success >=95%.

## H2 Workstreams Overview
Roadmap execution from Q3 onward is organized into themed workstreams. Each stream has an explicit Definition of Done (DoD) checklist referencing the Tech Debt tracker so that sign-off is tied to objective debt burndown.

### Workstream A — Throughput Foundation (Q3 focus)
Goal: reach competitive continuous batching throughput on CPU/MPS today while paving the way for future CUDA deployments once compatible hardware is available.

- **DoD**
  - [x] Continuous batching replaces global mutex; `RequestBatch` wired end-to-end (INF-2).
  - [x] Prefix cache online with metrics + eviction policies (INF-6/§2.4) — `RadixPrefixCache` (compressed trie, partial-match tracking, LRU eviction, 12 unit tests).
  - [ ] CUDA backend with FlashAttention-3 kernels validated on L40S (INF-2, §2.7 KPIs). Subtasks: enable llama.cpp CUDA build, add FlashAttention config knobs, implement BatchExecutor with prefill/decode overlap, wire GPU KV cache. **Hardware constraint:** hold execution until compatible CUDA hardware is available; keep design work ready.
- [ ] Priority-aware fairness scheduler on CPU/MPS (preemption + cancellation) so agents get SLO-backed latency even without CUDA. Leverages `RequestBatch` plan §7.2.
  - [ ] ModelRouter routes multi-model requests with hot load/unload (ARCH-4/5/6, see Architecture “ModelRouter Activation Plan”).
  - [ ] Priority fairness queue + preemption hooks (ARCH-5 follow-up); expose `runtime.fairness.*` knobs (enable_preemption, high_priority_threshold, max_timeslice_tokens) in config/env.
  - [ ] SSE cancellation regression tests (ctest target) kept green.
  - [ ] SimpleTokenizer metrics replaced with llama.cpp tokenizer to align TPS telemetry (INF-7).
  - [x] Latency histograms + queue-depth gauges emitted (OBS-1) to prove KPI gains; add fairness counters (preemptions, per-priority tokens).
  - [ ] Design scaffolding for Intel GPU + AMD ROCm backends (build flags, DeviceContext hooks) so hardware bring-up is unblocked once samples arrive.
  - **Exit KPIs**: ≥400 tok/s aggregate on L40S (future), prefix cache hit rate >60%, TTFT <250 ms with guardrails enabled, fairness tests demonstrate <5% variance across priorities on CPU/MPS.

### Workstream B — Enterprise Security & Observability (Q3 gating)
Goal: close SEC/OBS debts required for enterprise pilots and comply with PRD security caveats.

- **DoD**
- [x] Native TLS for HttpServer + HttpClient (SEC-5) with e2e tests.
  - [x] JWKS fetch + signature verification for OIDCValidator (SEC-1).
  - [ ] PolicyStore hashes API keys on write/read (SEC-2) with migration tooling.
  - [ ] Audit logger defaults to prompt hashing + configurable redaction (OBS-4).
  - [ ] OpenTelemetry traces cover tokenize→schedule→backend pipeline (OBS-2).
  - [ ] Guardrail verdict latency profiled and <500 ms P95 (NFR / KPI table).
  - **Exit KPIs**: Policy replication lag <30 s, zero plaintext secrets on disk, tracing coverage ≥90% of request path.

### Workstream C — Developer Experience & Multimodal (Q3 completion)
Goal: ship the customer-facing differentiators promised in the PRD.

- **DoD**
  - [ ] Structured output / JSON mode via llama.cpp grammar sampling with schema contract tests (PRD §Functional, TechDebt §2.1). Includes HTTP parser updates for `response_format`, adapter interface + backend capability flags, InferenceRequest plumbing, backend grammar hooks, and contract/integration tests.
  - [ ] Tool/function calling parity with OpenAI semantics (TechDebt §2.3).
  - [ ] Multimodal (vision) ingestion path via `libmtmd` including preprocessing metrics (TechDebt §2.2). Requires request parsing, tensor staging, and observability hooks.
  - [ ] Prefix cache APIs exposed to `inferctl` for agent workflows, plus CLI/docs showing cache warmers and status.
  - [ ] `inferctl pull` + model registry CLI with progress reporting (TechDebt §2.8).
  - [ ] Developer docs + examples updated for new params and guardrails.
  - **Exit KPIs**: 99.5% JSON schema conformance, multimodal preprocessing <80 ms/image on CUDA, CLI SUS ≥80.

### Workstream D — Distributed Ops & Fairness (Q4 focus)
Goal: unlock large-cluster deployments and SLO-aware scheduling.

- **DoD**
  - [ ] Disaggregated prefill/decode path with KV transfer latency <5 ms (TechDebt §2.5):
    - [ ] Split scheduler queues + metrics for prefill vs decode.
    - [ ] Implement `runtime/disaggregated/kv_channel` (SHM + RDMA adapters) with trace hooks.
    - [ ] Stand up dedicated prefill workers that emit KV tickets into the decode queue.
    - [ ] Extend decode workers/BatchExecutor to hydrate KV from the channel and stream tokens.
    - [ ] Wire `/readyz`, Prometheus gauges, and chaos tests for independent pool failures.
    - [ ] Publish Helm/docker overlays that scale pools independently; add CI smoke test.
  - [ ] Expert parallelism + tensor/pipeline parallel knobs exposed (TechDebt §2.6).
  - [ ] Request priority/fairness scheduling with starvation prevention (TechDebt §2.9).
  - [ ] Model registry with signed manifests + attestation (Roadmap Q4).
  - [ ] YAML parser replaced with supported config stack (DAT-2).
  - [ ] Web admin console surfaces queue depth, guardrail decisions, and live traces.
  - **Exit KPIs**: Guardrail verdict latency <500 ms, policy replication consistency 99.95%, admin UX SUS ≥80.

## Stretch Goals (post-Workstream D)
- Multi-region active/active with workload-aware routing and autoscaler hints (queue depth, KV fragmentation).
- Budget-aware autoscaling (cost per token) and GPU sharing controls.
- Frontend SDKs for major frameworks (LangChain, LlamaIndex).
- LoRA stacking and hot adapter reloads.
