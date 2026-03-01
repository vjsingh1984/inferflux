# Non-Functional Requirements

## Performance
- P99 prompt latency < 250 ms for 2k token prompts on A100 with CUDA backend.
- Streaming throughput >= 30 tok/s per request; aggregate >= 400 tok/s per GPU.
- CPU fallback must sustain 8 tok/s for demo workloads.

### Pipeline KPIs
| KPI | Target | Owner | Notes |
| --- | --- | --- | --- |
| Constrained decoding overhead | <5% delta vs. unconstrained for JSON schema workloads | Runtime | `StructuredOutputAdapter` (schema→GBNF) live; full constrained decoder with per-token histogram planned |
| Prefix cache hit rate (agent traces) | >60% for cached prefixes in benchmark corpus | Scheduler | `RadixPrefixCache` (compressed trie) live; KV page reuse requires llama.cpp multi-sequence API |
| Multimodal preprocessing latency | <80 ms per 1 MP image on CUDA, <150 ms on CPU | Runtime | Uses llama.cpp `libmtmd`; `ImagePreprocessor` live with `-DENABLE_MTMD=ON` |
| Guardrail verdict latency | <500 ms P95 including OPA calls | Policy | Blocks GA sign-off |
| Policy replication lag | <30 s across replicas | Policy | Mirrors PRD persona KPI |

## Scalability
- Horizontal scaling to 64 GPUs per cluster with shared KV cache metadata.

### Disaggregated Prefill/Decode Targets (§2.5)
Once decode workers and transport are enabled, the following SLOs apply:

| KPI | Target | Notes |
| --- | --- | --- |
| KV transfer latency (same-node SHM) | <5 ms P99 | Measured from `Prefill()` return to first `Decode()` token |
| KV transfer latency (multi-node RDMA) | <10 ms P99 | Single-hop InfiniBand / RoCEv2 |
| Prefill queue depth | Exposed via Prometheus gauge | Enables autoscaler to scale prefill pool independently |
| Decode queue depth | Exposed via Prometheus gauge | Enables autoscaler to scale decode pool independently |
| Blocked ticket rate | <1% of requests | Tickets rejected by a full `KVChannel` force prefill retry |
- Support for MIG partitioning and multi-node deployments using Redis/etcd for scheduler coordination.
- Multi-region active/active design with weighted routing, autoscaler hints (queue depth, cache pressure), and BYO object-storage for weights/adapters.

## Reliability
- Target 99.9% availability with graceful degradation when GPUs fail (automatic CPU fallback for small prompts).
- Hot-reload adapters/models in <5 s with no dropped connections.
- Persistent KV cache metadata to recover within 30 s after restart.

## Security
- API-key + OIDC auth (PKCE + workload identity), per-tenant rate limiting, TLS termination via ingress.
- Logs redact prompts by default; enable debug logs via config.
- Signed model packages (checksum verification) before loading weights.
- RBAC scopes (admin, read, generate, adapter-manager) enforced at gateway and CLI.
- Audit logging (structured JSON) shipped to SIEM/KMS-encrypted storage; adapter secrets encrypted with KMS key.
- Guardrail API to plug in policy engines (PII scrubbing, jailbreak detection, classifier enforcement).
- **TLS requirement (SEC-5)**: In-process TLS must be available for deployments that cannot front InferFlux with an ingress controller. Enable via `server.tls.enabled=true` + `server.tls.cert_path` + `server.tls.key_path` (env: `INFERFLUX_TLS_ENABLED`, `INFERFLUX_TLS_CERT_PATH`, `INFERFLUX_TLS_KEY_PATH`). `HttpClient` follows the same policy for outbound OIDC/JWKS calls. Status: **complete**.
- **Current caveats**: None — built-in TLS is implemented; clusters that terminate at an external ingress may leave it disabled.

## Operability
- Prometheus metrics for latency, queue depth, KV cache usage, GPU memory fragmentation, adapter hits, backend mode.
- OpenTelemetry traces for tokenizer→scheduler→backend pipeline.
- Configurable structured logging (JSON) with log rotation and log-sampling per tenant, plus `/metrics` and `/debug` endpoints (token traces, adapter status).
- CLI + admin UI for live streaming, SSE playback, interactive transcripts, and health dashboards.

### Required Prometheus Metrics (OBS-1)
The following histograms and gauges must be exported at `/metrics`:

| Metric | Type | Description |
| --- | --- | --- |
| `inferflux_request_latency_seconds` | Histogram | End-to-end request latency (P50/P95/P99) |
| `inferflux_queue_wait_seconds` | Histogram | Time from enqueue to batch execution start |
| `inferflux_batch_exec_seconds` | Histogram | Total batch execution time |
| `inferflux_prefill_duration_seconds` | Histogram | Prefill phase duration per request |
| `inferflux_decode_duration_seconds` | Histogram | Decode phase duration per request |
| `inferflux_scheduler_queue_depth` | Gauge | Pending requests in the scheduler queue |
| `inferflux_prefix_hits_total` | Counter | Exact-match prefix cache hits |
| `inferflux_prefix_misses_total` | Counter | Prefix cache misses |
| `inferflux_prefix_matched_tokens_total` | Counter | Tokens matched across partial prefix hits |
| `inferflux_prefix_partial_hits_total` | Counter | Lookups with shared prefix but no full hit |
| `inferflux_fairness_preemptions_total` | Counter | Requests preempted by the fairness controller |
| `inferflux_fairness_yields_total` | Counter | Requests yielded after timeslice expiry |
| `inferflux_fairness_resumes_total` | Counter | Resumed requests after fairness yield |
| `inferflux_fairness_tokens_total` | Counter (per priority) | Tokens generated per priority level |
| `inferflux_multimodal_images_total` | Counter | Images preprocessed across all requests |
| `inferflux_multimodal_requests_total` | Counter | Requests with ≥1 image input |

### Planned Module Deliverables
- **Constrained Decoder**: Adds histogram visibility for grammar token latency and schema success rate.
- **Prefix Cache**: Per-tenant cache quotas (quota enforcement not yet implemented).
- **Multimodal Adapter**: GPU tensor conversion time histogram (pending CUDA backend).

## Compliance & Testing
- Deterministic CPU integration tests for reproducibility.
- Performance regression suite replaying trace logs.
- Dependency license audit (MIT/BSD/Apache 2.0 only) and SBOM generation per build, with attestation/publishing via `gh release`.

### Unit Test Coverage Targets (TST-1)
Each module must maintain a minimum Catch2 test count; current baseline is 84 unit tests across 4 ctest targets:

| ctest Target | Scope | Minimum Tests |
| --- | --- | --- |
| `UnitTests` | All unit modules (auth, rate limiter, guardrail, audit, metrics, scheduler, policy, OIDC, tokenizer, structured output, prefix cache, multimodal, tracing) | 70 |
| `FairnessTests` | Fairness controller (timeslice, preemption, resume) | 3 |
| `StubIntegration` | HTTP-level integration without a loaded model | 17 |
| `SSECancel` | SSE streaming + in-flight cancellation | 1 end-to-end scenario |

New modules must ship with ≥5 unit tests before merging. Integration tests must cover the happy path and at least one auth/error path.
