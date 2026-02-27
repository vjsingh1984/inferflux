# InferFlux: Competitive Assessment, Tech Debt & Strategic Roadmap

This document tracks InferFlux's competitive positioning against best-in-class inference servers,
identifies critical gaps in the codebase, and maps required changes to specific files, docs, and
milestones. It is intended to persist across development sessions as a living tracker.

---

## 1. Competitive Ranking (as of early 2026)

### Landscape Summary

The inference server space has consolidated around **vLLM** and **SGLang** as the production
open-source standards. HuggingFace deprecated TGI (Dec 2025) in their favor. NVIDIA's
**TensorRT-LLM** remains peak-performance on NVIDIA hardware, now orchestrated by **Dynamo**.
**llama.cpp** dominates local/edge inference, powering both Ollama and LM Studio.

### InferFlux vs. Competitors — Honest Assessment

| Capability                    | vLLM | SGLang | TRT-LLM | llama.cpp | Ollama | InferFlux | Target Grade | Owner |
|-------------------------------|:----:|:------:|:-------:|:---------:|:------:|:---------:|:------------:|-------|
| Production throughput         |  A   |   A+   |   A+    |     C     |   D    |   **F**   | B (Q4) | Runtime |
| Continuous batching           |  A   |   A+   |   A     |    N/A    |  N/A   |   **F**   | B (Q3) | Scheduler |
| KV cache efficiency           |  B+  |   A+   |   A     |    N/A    |  N/A   |   **F**   | B (Q3) | Runtime |
| Speculative decoding          |  A   |   A    |   A     |    B+     |   B    |   **D**   | B (Q3) | Runtime |
| Structured output / JSON mode |  A   |   A+   |   B+    |    B+     |   B    |   **F**   | B+ (Q3) | Runtime |
| Multimodal / vision           |  A   |   A    |   B+    |    B+     |   B+   |   **F**   | C+ (Q3) | Runtime |
| Tool / function calling       |  A   |   A    |   B     |    C      |   B    |   **F**   | B (Q3) | Server |
| Quantization breadth          |  A+  |   B+   |   A     |    A+     |   A    |   **D**   | B (Q4) | Runtime |
| Hardware breadth              |  A+  |   B+   |   C     |    A+     |   B    |   **D**  | B (Q4) | Runtime |
| Disaggregated prefill/decode  |  A   |   A+   |   A     |    N/A    |  N/A   |   **F**   | C (Q4) | Distributed Runtime |
| Model parallelism (TP/PP/EP)  |  A   |   A+   |   A     |    C      |   C    |   **F**   | C (Q4) | Distributed Runtime |
| OpenAI API compatibility      |  A   |   A    |   B     |    B      |   A    |   **C**   | B (Q3) | Server |
| Enterprise auth & RBAC        |  B   |   C    |   B     |    F      |   F    |   **C+**  | B+ (Q3) | Policy |
| Observability                 |  A   |   B    |   A     |    D      |   D    |   **D+**  | B (Q3) | Observability |
| Ease of local setup           |  B+  |   B    |   C     |    C      |   A+   |   **C**   | B (Q3) | CLI |
| Model management UX           |  B   |   B    |   C     |    C      |   A+   |   **F**   | C+ (Q3) | CLI |
| Test coverage & CI maturity   |  A   |   A    |   A     |    A      |   B    |   **F**   | B (Q3) | QA |

**Overall grade: D- (early prototype)**

InferFlux has strong *architectural vision* (enterprise auth, policy store, multi-backend) but
the implementation is at stub/MVP stage. The competitive gap is largest in the core inference
pipeline (batching, KV cache, parallelism) and in table-stakes features that every 2026 server
must have (structured output, multimodal, tool calling).

### Where InferFlux Has Potential Differentiation

1. **Integrated policy engine with encrypted persistence** — No competitor has built-in RBAC +
   encrypted policy store + admin APIs. vLLM and SGLang rely on external auth layers.
2. **Single-binary multi-backend** — The vision of CPU/CUDA/ROCm/MPS from one binary is
   shared only with llama.cpp (which lacks enterprise features).
3. **Cloud-native from day one** — Helm/Docker/Prometheus built into the architecture, vs.
   bolted on in vLLM/SGLang.

---

## 2. Critical Features Missing from PRD/Requirements/Architecture

These are **table-stakes for any serious inference server in 2026** and must be incorporated
into the planning documents. Each item includes which doc(s) need updating.

### 2.1 Structured Output / Constrained Decoding

**Why critical:** Gartner predicts 95% of enterprise LLM deployments will use constrained
decoding by 2027. vLLM uses XGrammar; SGLang uses Compressed FSM. Both achieve near-zero
overhead (~50us/token).

- **What to add:** JSON Schema mode, regex constraints, GBNF grammar support (leverage
  llama.cpp's built-in grammar sampling as starting point).
- **Update:** `docs/PRD.md` §Functional Requirements — add structured output endpoint params.
- **Update:** `docs/Architecture.md` §Modules — add constrained decoding module between
  scheduler and backend.
- **Update:** `docs/NFR.md` §Performance — add latency target for constrained vs unconstrained.
- **Implement in:** `scheduler/scheduler.h` (grammar-aware token selection),
  `server/http/http_server.cpp` (parse `response_format` field).

### 2.2 Multimodal / Vision Model Support

**Why critical:** Every major competitor supports vision models. Ollama has drag-and-drop image
support. llama.cpp added `libmtmd` in April 2025.

- **What to add:** Image input for VLMs (LLaVA, Qwen-VL), base64 and URL image inputs in
  chat messages, leverage llama.cpp's `libmtmd`.
- **Update:** `docs/PRD.md` §Functional Requirements — add multimodal input support.
- **Update:** `docs/Architecture.md` §Data Flow — add image preprocessing stage.
- **Implement in:** `server/http/http_server.cpp` (parse image content parts),
  `model/` (new image encoder module), `scheduler/scheduler.h` (handle mixed modality batches).

### 2.3 Tool Calling / Function Calling

**Why critical:** Foundation for agent workflows. vLLM, SGLang, Ollama all support it.
OpenAI-compatible tool calling is expected by every SDK (LangChain, LlamaIndex).

- **What to add:** `tools` and `tool_choice` parameters in chat completions, streaming tool
  call deltas, model-specific chat templates for tool use.
- **Update:** `docs/PRD.md` §Functional Requirements, §User Stories (research scientist
  running agents).
- **Implement in:** `server/http/http_server.cpp` (parse tools array, emit tool_call in
  response), `model/tokenizer/` (chat template with tool definitions).

### 2.4 Prefix Caching / Automatic KV Cache Reuse

**Why critical:** SGLang's RadixAttention achieves 85-95% cache hit for few-shot workloads vs.
15-25% for naive paging. This is the single biggest throughput differentiator for multi-turn
and agent workloads.

- **What to add:** Radix-tree or hash-based prefix matching in the KV cache, automatic reuse
  across requests sharing common prefixes (system prompts, few-shot examples).
- **Update:** `docs/Architecture.md` §Runtime — add prefix cache subsystem.
- **Update:** `docs/NFR.md` §Performance — add cache hit rate KPIs.
- **Implement in:** `runtime/kv_cache/paged_kv_cache.h` (prefix tree index),
  `scheduler/scheduler.h` (prefix-aware request routing).

### 2.5 Disaggregated Prefill/Decode

**Why critical:** Now the default production architecture. vLLM deploys it at Meta, LinkedIn,
Mistral. SGLang tested it on 96 H100s. NVIDIA Dynamo provides orchestration for it.

- **What to add:** Separate prefill and decode worker pools, KV cache transfer via shared
  memory or RDMA, independent scaling of prefill vs decode instances.
- **Update:** `docs/Architecture.md` §Deployment View — add disaggregated topology.
- **Update:** `docs/Roadmap.md` — add as Q3 milestone (currently not mentioned).
- **Update:** `docs/NFR.md` §Scalability — add KV transfer latency targets.
- **Implement in:** New `runtime/disaggregated/` module, `scheduler/scheduler.h` (split
  prefill/decode queues into separate schedulable units).

### 2.6 Expert Parallelism for MoE Models

**Why critical:** DeepSeek-V3/R1 and Mixtral dominate open-source. SGLang achieves 5x
throughput over vanilla TP for MoE models using expert parallelism.

- **What to add:** Expert-parallel execution, MoE-aware batch scheduling, fused MoE kernels.
- **Update:** `docs/PRD.md` §Goals — mention MoE model support explicitly.
- **Update:** `docs/Architecture.md` §Runtime — add parallelism strategies section.
- **Implement in:** `runtime/backends/` (EP dispatch), `scheduler/scheduler.h` (MoE-aware
  batching).

### 2.7 Flash Attention Integration

**Why critical:** FlashAttention-3 achieves 1.5-2x over FA2, 75% GPU utilization on Hopper.
Every production GPU server uses it.

- **What to add:** FA3 kernels for CUDA path, leverage llama.cpp's Metal attention for MPS.
- **Update:** `docs/Architecture.md` §Runtime — document attention kernel strategy.
- **Implement in:** `runtime/backends/` (CUDA backend with FA3), link against
  flash-attention or cutlass.

### 2.8 Model Management UX (`inferctl pull`)

**Why critical:** Ollama's `ollama pull llama3` is the gold standard for local UX. The PRD
mentions `inferctl pull` but it is not implemented.

- **What to add:** Pull models from HuggingFace Hub, local model registry, model listing,
  progress indicators, quantization selection.
- **Update:** `docs/PRD.md` §Functional Requirements — detail the pull workflow.
- **Implement in:** `cli/main.cpp` (pull subcommand), new `model/registry/` module.

### 2.9 Request Priority & Fairness Scheduling

**Why critical:** Emerging as a differentiator. PROSERVE and FairBatching (published 2025)
show SLO-aware scheduling prevents starvation under load. vLLM has priority-aware preemption.

- **What to add:** Per-request priority levels, SLO-based admission control, preemption of
  low-priority requests, starvation prevention.
- **Update:** `docs/PRD.md` §Functional Requirements — add priority parameter.
- **Update:** `docs/Architecture.md` §Scheduler — add priority queue design.
- **Implement in:** `scheduler/scheduler.h` (priority queue, preemption logic).

---

## 3. Tech Debt Registry

Each item is tagged with severity, the file(s) affected, and a suggested fix. Items are
grouped by the system area and ordered by severity within each group.

### 3.1 Security — CRITICAL

| ID | Issue | File(s) | Fix | Affects Docs |
|----|-------|---------|-----|-------------|
| SEC-1 | **[FIXED]** OIDC validator fetches JWKS endpoints and verifies RS256 signatures while enforcing issuer/audience/temporal claims. | `server/auth/oidc_validator.cpp` | *(Fixed)* | `docs/Architecture.md` §Security |
| SEC-2 | **[FIXED]** PolicyStore now hashes API keys with SHA-256 before persisting and Admin APIs operate on hashed material. Existing plaintext files must be rotated via CLI. | `policy/policy_store.cpp` | *(Fixed)* | `docs/Architecture.md` §Security |
| SEC-3 | **[FIXED]** JSON injection in OPA client. The prompt is now escaped before being inserted into the JSON payload. | `policy/opa_client.cpp` | *(Fixed)* | `docs/Policy.md` |
| SEC-4 | **[FIXED]** Audit logger JSON injection. All fields are now escaped before being written to the log. | `server/logging/audit_logger.cpp` | *(Fixed)* | `docs/NFR.md` §Security |
| SEC-5 | **[FIXED]** HttpServer can terminate TLS via OpenSSL (configurable cert/key), and HttpClient supports HTTPS for control-plane requests. | `server/http/http_server.cpp`, `net/http_client.cpp` | *(Fixed)* | `docs/NFR.md` §Security |

### 3.2 Core Inference Pipeline — HIGH

| ID | Issue | File(s) | Fix | Affects Docs |
|----|-------|---------|-----|-------------|
| INF-1 | **[FIXED]** Multi-threaded HTTP server with configurable thread pool (accept loop + worker threads via condition variable queue). | `server/http/http_server.cpp` | *(Fixed)* | — |
| INF-2 | Queue-backed scheduler with RequestBatch, priority sorting, ModelRouter integration, and batching metrics is live; next step is true prefill/decode overlap, GPU-aware execution, and FlashAttention integration. | `scheduler/scheduler.cpp`, `runtime/backends/cuda/`, `cmake/` | **FA3 plan**: (1) enable llama.cpp CUDA/FlashAttention build flags and emit `INFERFLUX_HAS_CUDA` (**DONE**); (2) expose FlashAttention knobs in config → `LlamaBackendConfig` (**DONE**); (3) add BatchExecutor abstraction (prefill/decode stages) with CUDA stream hooks (**DONE** for CPU path); (4) implement GPU KV cache plumbing and FA3 call sites. Metal/MPS (`LLAMA_METAL`) and BLAS acceleration are now auto-enabled when available. Track completion across Workstream A milestones. | `docs/Architecture.md` §Scheduler, `docs/PRD.md` §Goals |
| INF-3 | **[FIXED]** Dynamic HTTP buffer with Content-Length awareness, 16MB max, 413 for oversized. | `server/http/http_server.cpp` | *(Fixed)* | — |
| INF-4 | **[FIXED]** `llama_backend_init()`/`free()` now reference-counted via static counter with mutex. | `runtime/backends/cpu/llama_backend.cpp` | *(Fixed)* | — |
| INF-5 | **[FIXED]** Stub fallback returns 503 Service Unavailable with clear error message. | `scheduler/scheduler.cpp`, `server/http/http_server.cpp` | *(Fixed)* | `docs/PRD.md` §Acceptance Criteria |
| INF-6 | **[FIXED]** Scheduler now reserves/releases KV cache pages per batch, preparing for future offload/prefetch work. | `scheduler/scheduler.cpp` | *(Fixed)* | — |
| INF-7 | SimpleTokenizer is a whitespace/punctuation splitter — token counts will not match llama.cpp's real tokenizer, affecting metrics. | `model/tokenizer/simple_tokenizer.cpp` | Use llama.cpp tokenizer for metrics when llama backend is loaded; keep SimpleTokenizer only as fallback | — |

### 3.3 Data Handling — MEDIUM

| ID | Issue | File(s) | Fix | Affects Docs |
|----|-------|---------|-----|-------------|
| DAT-1 | **[FIXED]** Adopted nlohmann/json v3.11.3 — replaced 4 hand-rolled parsers (~310 lines). CLI pending (CQ-1). | `external/nlohmann/json.hpp` | *(Fixed)* | — |
| DAT-2 | Hand-rolled YAML parser is fragile — no multi-level nesting, no anchors, no error reporting | `server/main.cpp:124-358` | Adopt yaml-cpp or similar; alternatively use a simpler format (TOML) | — |
| DAT-3 | GGUF loader is a complete stub — returns dummy tensor with file size | `model/gguf/gguf_loader.cpp` | Implement real GGUF parsing or document that it is intentionally delegated to llama.cpp | `docs/Architecture.md` §Model |
| DAT-4 | **[FIXED]** Added CORS headers (Access-Control-Allow-Origin: *) to all responses, OPTIONS preflight handling. | `server/http/http_server.cpp` | *(Fixed)* | — |

### 3.4 Thread Safety — MEDIUM

| ID | Issue | File(s) | Fix | Affects Docs |
|----|-------|---------|-----|-------------|
| THR-1 | **[FIXED]** `ApiKeyAuth` now uses `std::shared_mutex`. | `server/auth/api_key_auth.cpp` | *(Fixed)* | — |
| THR-2 | **[FIXED]** `RateLimiter` is now thread-safe using a `std::mutex`. | `server/auth/rate_limiter.cpp` | *(Fixed)* | — |
| THR-3 | **[FIXED]** `MetricsRegistry` uses a mutex for `backend_` and atomics for counters. | `server/metrics/metrics.cpp` | *(Fixed)* | — |
| THR-4 | **[FIXED]** `BackendManager` uses a `std::mutex` to protect its map. | `runtime/backends/backend_manager.cpp` | *(Fixed)* | — |

### 3.5 Observability Gaps — MEDIUM

| ID | Issue | File(s) | Fix | Affects Docs |
|----|-------|---------|-----|-------------|
| OBS-1 | **[FIXED]** MetricsRegistry exposes request latency histograms plus active connection and queue-depth gauges. | `server/metrics/metrics.cpp` | *(Fixed)* | `docs/NFR.md` §Operability |
| OBS-2 | No OpenTelemetry traces (described in Architecture.md and NFR.md) | — | Integrate opentelemetry-cpp SDK; add spans for tokenize → schedule → infer → stream stages | `docs/Architecture.md` §Observability |
| OBS-3 | **[FIXED]** Added `/readyz` (model-aware), `/livez` (always 200), updated `/healthz` to show degraded status. | `server/http/http_server.cpp` | *(Fixed)* | — |
| OBS-4 | **[FIXED]** AuditLogger hashes prompts/responses by default and emits raw text only when `debug_mode=true`. | `server/logging/audit_logger.cpp` | *(Fixed)* | `docs/Architecture.md` §Security |

### 3.6 Testing — MEDIUM

| ID | Issue | File(s) | Fix | Affects Docs |
|----|-------|---------|-----|-------------|
| TST-1 | **[FIXED]** 44 Catch2 tests across 9 files: ApiKeyAuth(6), RateLimiter(4), Guardrail(6), AuditLogger(3), Metrics(6), Scheduler(2), PolicyStore(4), OIDC(8), plus originals. | `tests/unit/` | *(Fixed)* | `docs/NFR.md` §Testing |
| TST-2 | **[FIXED]** Adopted Catch2 v3.7.1 amalgamated. All tests use TEST_CASE/REQUIRE. | `external/catch2/`, `CMakeLists.txt` | *(Fixed)* | — |
| TST-3 | Integration tests skip entirely when INFERFLUX_MODEL_PATH is not set (CI never runs them) | `CMakeLists.txt` | Add stub-mode integration tests that work without a model file | — |
| TST-4 | Integration test `http_get` sends no auth header — may pass/fail depending on auth state | `tests/integration/sse_metrics_test.py` | Fix test helpers to include API key header | — |

### 3.7 Code Quality — LOW

| ID | Issue | File(s) | Fix | Affects Docs |
|----|-------|---------|-----|-------------|
| CQ-1 | **[FIXED]** CLI refactored from 648→310 lines using shared HttpClient and nlohmann/json. | `cli/main.cpp`, `net/http_client.cpp` | *(Fixed)* | — |
| CQ-2 | **[FIXED]** `CPUDeviceContext` now has a destructor that frees all tracked allocations. | `runtime/backends/cpu/cpu_backend.cpp/.h` | *(Fixed)* | — |
| CQ-3 | **[VERIFIED INCORRECT]** `Guardrail::Enabled()` returns false when only OPA endpoint is configured (empty blocklist). The implementation correctly checks both `blocklist_.empty()` and `opa_endpoint_.empty()`. | `server/policy/guardrail.cpp:74` | *(Not a bug)* | — |
| CQ-4 | **[FIXED]** `HttpClient::send()` now loops on partial sends and throws on failure. Also added 30s socket timeouts. | `net/http_client.cpp` | *(Fixed)* | — |
| CQ-5 | **[FIXED]** `Stop()` shuts down listening socket via `shutdown(SHUT_RDWR)` to unblock `accept()`. | `server/http/http_server.cpp` | *(Fixed)* | — |
| CQ-6 | Config path defaults to relative `"config/server.yaml"` — depends on CWD | `server/main.cpp:65` | Resolve relative to executable path or require absolute path | — |
| CQ-7 | **[FIXED]** Default `max_tokens` raised from 64/32 to 256 in both scheduler and CLI. | `scheduler/scheduler.h`, `cli/main.cpp` | *(Fixed)* | — |

---

## 4. Strategic Feature Roadmap — Revised

This integrates competitive gaps into the existing quarterly milestones from `docs/Roadmap.md`.
Items marked **[NEW]** are not in the current roadmap. Items marked **[UPGRADE]** need their
scope expanded.

### Q2 — MVP (current milestone)

**Focus: Make what exists actually work correctly.**

| Priority | Item | Status | Tracking IDs | Key Files |
|----------|------|--------|-------------|-----------|
| P0 | Fix OIDC signature verification | **Done** — JWKS fetch + RS256 verify + claim validation | SEC-1 | `server/auth/oidc_validator.cpp` |
| P0 | Hash API keys at rest and in memory | **Partial** — ApiKeyAuth hashes with SHA-256; PolicyStore disk format remaining | SEC-2 | `server/auth/api_key_auth.cpp`, `policy/policy_store.cpp` |
| P1 | Adopt JSON library (nlohmann/json) | **Done** | DAT-1 | `external/nlohmann/json.hpp` |
| P1 | Add unit tests for auth, rate limiter, policy store, guardrail | **Done** | TST-1 | `tests/unit/` (9 files, 44 tests) |
| P2 | Dynamic receive buffer in HTTP server | **Done** | INF-3 | `server/http/http_server.cpp` |
| P2 | Add graceful shutdown | **Done** | CQ-5 | `server/http/http_server.cpp` |

**Existing Q2 items from `docs/Roadmap.md` remain:** CPU & MPS backends with SSE streaming,
policy store with RBAC, CLI interactive mode, Prometheus metrics, audit logging.

### Q3 — Performance & Modern Features

**Focus: Close the core inference gap, add table-stakes features.**

| Priority | Item | Status | Tracking IDs | Key Files |
|----------|------|--------|-------------|-----------|
| P0 | Multi-threaded HTTP server (epoll/kqueue) | **Done** | INF-1 | `server/http/http_server.cpp` |
| P0 | Continuous batching in scheduler | Not started | INF-2 | `scheduler/scheduler.cpp` |
| P0 | **[NEW]** Structured output / JSON mode | Not started | §2.1 | `scheduler/`, `server/http/` |
| P0 | **[NEW]** Tool calling / function calling | Not started | §2.3 | `server/http/http_server.cpp` |
| P0 | **[UPGRADE]** CUDA backend with Flash Attention | Not started | §2.7 | `runtime/backends/cuda/` (new) |
| P1 | **[NEW]** Prefix caching (radix-tree KV reuse) | Not started | §2.4 | `runtime/kv_cache/paged_kv_cache.h` |
| P1 | **[NEW]** Multimodal / vision model support | Not started | §2.2 | `model/`, `server/http/` |
| P1 | **[NEW]** Model pull from HuggingFace (`inferctl pull`) | Not started | §2.8 | `cli/main.cpp`, `model/registry/` (new) |
| P1 | TLS support (server + HttpClient) | Not started | SEC-5 | `server/http/`, `net/http_client.cpp` |
| P1 | Latency histograms and queue depth gauges in metrics | Not started | OBS-1 | `server/metrics/` |
| P1 | OpenTelemetry trace integration | Not started | OBS-2 | New dependency |
| P2 | Readiness vs liveness health probes | **Done** | OBS-3 | `server/http/http_server.cpp` |
| P2 | CORS headers | **Done** | DAT-4 | `server/http/http_server.cpp` |
| P2 | Refactor CLI to reuse HttpClient | **Done** | CQ-1 | `cli/main.cpp` |

**Existing Q3 items from `docs/Roadmap.md` remain:** CUDA/ROCm acceleration, NVMe-assisted KV
cache, speculative decoding, LoRA stacking, hot adapter reloads, autoscaler hints.

### Q4 — Enterprise, Scale & Differentiation

**Focus: Distributed operations, advanced scheduling, competitive differentiation.**

| Priority | Item | Status | Tracking IDs | Key Files |
|----------|------|--------|-------------|-----------|
| P0 | **[NEW]** Disaggregated prefill/decode | Not started | §2.5 | `runtime/disaggregated/` (new), `scheduler/` |
| P0 | **[NEW]** Expert parallelism for MoE models | Not started | §2.6 | `runtime/backends/`, `scheduler/` |
| P1 | **[NEW]** Request priority & fairness scheduling | Not started | §2.9 | `scheduler/scheduler.h` |
| P1 | **[UPGRADE]** Tensor + pipeline parallelism | Not started | — | `runtime/backends/` |
| P1 | Prompt/response hashing in audit logs | Not started | OBS-4 | `server/logging/audit_logger.cpp` |
| P2 | YAML parser replacement (yaml-cpp) | Not started | DAT-2 | `server/main.cpp` |
| P2 | **[NEW]** SBOM generation per build | Not started | — | `CMakeLists.txt`, CI |

**Existing Q4 items from `docs/Roadmap.md` remain:** Distributed scheduler, OPA/Cedar
integration, model registry + signed manifests, web admin console.

---

## 5. Doc Update Tracker

This maps which planning documents need updates to incorporate the competitive gaps
identified above. Check items off as they are addressed.

### `docs/PRD.md`

- [X] §Functional Requirements: Add structured output (`response_format` param) — §2.1
- [X] §Functional Requirements: Add tool calling (`tools`, `tool_choice` params) — §2.3
- [ ] §Functional Requirements: Add multimodal input (image content parts) — §2.2
- [X] §Functional Requirements: Add `inferctl pull` workflow — §2.8
- [ ] §Functional Requirements: Add request priority parameter — §2.9
- [ ] §Goals: Mention MoE model support explicitly — §2.6
- [X] §User Stories: Add agent workflow story (tool calling) — §2.3
- [ ] §User Stories: Add vision model story — §2.2
- [X] §Acceptance Criteria: Return 503 (not 200) when no model loaded — INF-5
- [X] §Milestones: Add structured output, tool calling to Q3 scope
- [X] §Vision: Updated vision statement per design review recommendations

### `docs/Architecture.md`

- [X] §Modules: Rewritten to match actual implementation (accurate module descriptions)
- [X] §Data Flow: Updated to reflect actual request flow (dynamic buffer, auth, rate limit, guardrail)
- [X] §Plugin Interfaces: Added new section documenting PolicyBackend, ModelRouter, DeviceContext, RequestBatch
- [X] §Health Probes: Added new section documenting /healthz, /readyz, /livez
- [X] §Security: Document OIDC signature verification — SEC-1
- [X] §Security: Document API key hashing — SEC-2
- [X] §Security: Document readiness vs liveness health checks — OBS-3
- [ ] §Modules: Add constrained decoding module — §2.1
- [ ] §Modules: Add parallelism strategies section (TP/PP/EP) — §2.6, §2.7
- [ ] §Data Flow: Add image preprocessing stage — §2.2
- [ ] §Deployment View: Add disaggregated prefill/decode topology — §2.5
- [ ] §Runtime: Add prefix cache subsystem — §2.4
- [ ] §Runtime: Document attention kernel strategy (FA3) — §2.7
- [ ] §Scheduler: Add priority queue design — §2.9
- [ ] §Observability: Document audit log redaction (hash by default) — OBS-4

### `docs/NFR.md`

- [ ] §Performance: Add latency target for constrained vs unconstrained decoding — §2.1
- [ ] §Performance: Add prefix cache hit rate KPIs — §2.4
- [ ] §Scalability: Add KV transfer latency targets for disaggregation — §2.5
- [ ] §Security: Require TLS support — SEC-5
- [X] §Security: Require JWT signature verification — SEC-1
- [ ] §Operability: Add histogram and gauge metric requirements — OBS-1
- [ ] §Testing: Require unit test coverage targets per module — TST-1

### `docs/Roadmap.md`

- [X] Q2: Updated to show all items complete with checkmarks
- [X] Q3: Add structured output, tool calling, multimodal as milestones — §2.1, §2.2, §2.3
- [X] Q3: Add prefix caching milestone — §2.4
- [X] Q3: Add `inferctl pull` milestone — §2.8
- [X] Q3: Marked completed items (INF-1, OBS-3, DAT-4, CQ-1, ARCH-1/2/3)
- [X] Q4: Add disaggregated prefill/decode — §2.5
- [X] Q4: Add expert parallelism — §2.6
- [X] Q4: Add request priority scheduling — §2.9

### `docs/Policy.md`

- [ ] §Capabilities: Document Guardrail::Enabled() fix for OPA-only configs — CQ-3
- [X] §Capabilities: Note JSON injection fix in OPA client — SEC-3

---

## 6. Governance & Review Cadence

- **Security & Observability Fortnightly Review**: Every other Monday the owners listed for SEC-1/2/5 and OBS-1/2/4 provide progress, blockers, and updated ETAs. Notes from each review append to this document so regressions remain visible until closure.
- **Capability Grade Tracking**: Workstream leads update the capability table after each sprint to confirm whether target grades (e.g., “B (Q3)”) remain achievable. Missed checkpoints automatically trigger roadmap/tech-debt reprioritization.
- **Tech Debt Sync**: Severity-A/B items that do not change status for two consecutive reviews escalate to the steering group for staffing or scope adjustment.

## 7. Vision Statement Gaps

The current PRD positions InferFlux as a "vLLM-inspired replacement for LM Studio and Ollama."
This positioning needs sharpening given the 2026 landscape:

### Recommended Vision Pivot

> **InferFlux: The enterprise-native inference server that runs anywhere.**
>
> While vLLM and SGLang optimize for raw throughput on GPU clusters, and Ollama optimizes for
> local simplicity, InferFlux is the only inference server built from the ground up with
> integrated enterprise controls (RBAC, encrypted policy, audit, guardrails) that runs
> identically on a developer's MacBook and a 64-GPU Kubernetes cluster.

### Key Narrative Shifts for PRD/README

1. **Stop claiming vLLM-level throughput as a near-term goal.** vLLM has hundreds of
   contributors and years of kernel optimization. Instead, position as "comparable throughput
   via llama.cpp integration with enterprise features that vLLM lacks."

2. **Lead with the policy engine.** No competitor has built-in RBAC + encrypted policy +
   admin APIs. This is the genuine differentiator. The PRD should promote this from a bullet
   point to the primary narrative.

3. **Embrace llama.cpp as the inference engine.** The GGUF loader stub and CPUDeviceContext
   stub suggest an intent to reimplement inference. This is impractical. The strategic move is
   to deeply integrate llama.cpp (which already supports CUDA, Metal, Vulkan, speculative
   decoding, grammar sampling, multimodal) and focus InferFlux's value-add on the serving
   layer above it.

4. **Add "agent-ready" to the USPs.** Structured output + tool calling + prefix caching
   together make a server "agent-ready." This is the buying criterion for 2026 platform
   engineers.

---

## 7. Foundational Design Abstractions

These interfaces were introduced to prevent architectural dead ends. They define the
plugin boundaries that future features (multi-model, continuous batching, alternative
policy stores) will implement. All are header-only with no runtime cost until used.

| ID | Interface | File | Purpose | Status |
|----|-----------|------|---------|--------|
| ARCH-1 | `PolicyBackend` | `policy/policy_backend.h` | Abstract interface for policy storage/enforcement. `PolicyStore` (INI) implements it. HttpServer now depends on the interface, not the concrete store. Enables future OPA/Cedar/SQL backends. | **Done** |
| ARCH-2 | `ModelRouter` | `scheduler/model_router.h` | Abstract interface for multi-model serving. Defines `ListModels`, `LoadModel`, `UnloadModel`, `Resolve`. Prevents single-model-per-server dead end. | **Done** (interface only) |
| ARCH-3 | `RequestBatch` | `scheduler/request_batch.h` | Per-request state (`InferenceRequest`) and batch grouping (`RequestBatch`) for continuous batching. Defines request phases, priority, timing, and streaming callbacks. | **Done** (interface only) |
| ARCH-4 | Wire `ModelRouter` into `Scheduler` | `scheduler/scheduler.h` | Replace direct `LlamaCPUBackend` dependency with `ModelRouter` abstraction. | Not started |
| ARCH-5 | Wire `RequestBatch` into `Scheduler` | `scheduler/scheduler.h` | Replace `GenerateRequest`/`GenerateResponse` with `InferenceRequest` flow through `RequestBatch`. | Not started |
| ARCH-6 | Create `SingleModelRouter` | `scheduler/single_model_router.cpp` | Default `ModelRouter` implementation wrapping a single backend. Drop-in replacement for current behavior. | Not started |

---

## 8. Cross-Session Tracking

This document should be consulted at the start of each development session. Use the tracking
IDs (SEC-1, INF-2, etc.) in commit messages and PR descriptions for traceability.

**Quick reference — what to work on next, by role:**

| If you are... | Start with |
|---------------|-----------|
| Fixing security | **Done** — SEC-2 (PolicyStore hashing) and SEC-5 (TLS) |
| Wiring abstractions | ARCH-4 (ModelRouter→Scheduler), ARCH-5 (RequestBatch→Scheduler), ARCH-6 (SingleModelRouter) |
| Improving core performance | INF-2 (continuous batching, Q3 P0) |
| Adding features | §2.1 structured output, §2.3 tool calling (Q3 P0) |
| Improving test coverage | TST-3 (stub-mode integration tests), TST-4 (auth headers) |
| Improving observability | OBS-1 (histograms), OBS-2 (OpenTelemetry), OBS-4 (audit hash) |
| Refactoring code quality | DAT-2 (YAML parser), CQ-6 (config path) |
| Updating docs | See §5 Doc Update Tracker |
