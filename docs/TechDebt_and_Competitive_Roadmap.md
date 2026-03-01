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
| Prefix Caching                |  A   |   A+   |   A     |     B     |   B    |   **B-**  | A (Q3)       | Runtime |
| Speculative decoding          |  A   |   A    |   A     |    B+     |   B    |   **D**   | B (Q3) | Runtime |
| Structured output / JSON mode |  A   |   A+   |   B+    |    B+     |   B    |   **B-**  | B+ (Q3) | Runtime |
| Multimodal / vision           |  A   |   A    |   B+    |    B+     |   B+   |   **D**   | C+ (Q3) | Runtime |
| Tool / function calling       |  A   |   A    |   B     |    C      |   B    |   **F**   | B (Q3) | Server |
| Quantization breadth          |  A+  |   B+   |   A     |    A+     |   A    |   **D**   | B (Q4) | Runtime |
| Hardware breadth              |  A+  |   B+   |   C     |    A+     |   B    |   **D**  | B (Q4) | Runtime |
| Disaggregated prefill/decode  |  A   |   A+   |   A     |    N/A    |  N/A   |   **F**   | C (Q4) | Distributed Runtime |
| Model parallelism (TP/PP/EP)  |  A   |   A+   |   A     |    C      |   C    |   **F**   | C (Q4) | Distributed Runtime |
| OpenAI API compatibility      |  A   |   A    |   B     |    B      |   A    |   **C+**  | B (Q3) | Server |
| Enterprise auth & RBAC        |  B   |   C    |   B     |    F      |   F    |   **B-**  | B+ (Q3) | Policy |
| Observability                 |  A   |   B    |   A     |    D      |   D    |   **B**   | B (Q3) | Observability |
| Ease of local setup           |  B+  |   B    |   C     |    C      |   A+   |   **C**   | B (Q3) | CLI |
| Model management UX           |  B   |   B    |   C     |    C      |   A+   |   **F**   | C+ (Q3) | CLI |
| Test coverage & CI maturity   |  A   |   A    |   A     |    A      |   B    |   **B**   | B (Q3) | QA |

**Overall grade: D (early prototype)**

**Scorecard status notes (May 2025):**
- Production throughput, continuous batching, and KV cache efficiency remain at **F** until GPU-aware execution, prefill/decode overlap, and FA3 kernels from §2.5/§2.7 land.
- Prefix caching bumped to **B-**: `RadixPrefixCache` (compressed trie over token sequences) is live, wired into `BatchExecutor` and `Scheduler`, with LRU eviction, partial-match metrics (`prefix_matched_tokens_total`, `prefix_partial_hits_total`), and 12 unit tests. Actual KV page reuse requires llama.cpp multi-sequence integration (§2.4 follow-up).
- Structured output bumped to **B-** now that HTTP parsing, schema-to-grammar conversion, and llama grammar sampling are wired through scheduler/runtime (§2.1).
- Tool/function calling at **C+**: `tools[]`/`tool_choice` parsed, schema injected as system preamble, `tool_calls` array emitted with `finish_reason=tool_calls` (non-streaming). §2.3 streaming follow-up is now done: `stream=true` with `tools[]` emits the four-chunk OpenAI delta sequence (role → function-name → arguments → `finish_reason=tool_calls`) instead of raw JSON tokens; `suppress_stream_content` prevents the tool_call JSON envelope from appearing as content deltas; no_backend streaming path fixed to emit proper SSE termination. Remaining gap to B: model-native chat templates for tool use.
- Multimodal/vision bumped to **D**: `ImagePreprocessor` parses OpenAI `image_url` content arrays, decodes base64 data URIs, fetches HTTP URLs, and injects `<__media__>` markers (§2.2). `LlamaCPUBackend` supports `LoadMmproj()` / `GenerateWithImages()` via libmtmd when built with `-DENABLE_MTMD=ON`. Prometheus counters `inferflux_multimodal_images_total` and `inferflux_multimodal_requests_total` track usage. Actual vision inference requires a compatible mmproj GGUF and `ENABLE_MTMD=ON` at build time.
- Quantization breadth and hardware breadth are **D** since only CPU/MPS paths run; CUDA/ROCm/Intel enablement in §2.7/§2.11 is unresolved.
- §2.5 fully done: phased prefill/decode, decode workers, KV serialisation, SHM transport, dual-pool `/readyz`, `inferflux_kv_transfer_duration_ms` histogram, Helm overlays. Grade moves toward **D** (still no GPU throughput); cross-node RDMA pending.
- §2.6 (MoE expert parallelism) is partially done: `IsMoE()`/`ExpertCount()`/`ActiveExperts()` on `LlamaCPUBackend`, `ModelInfo` MoE fields populated in `SingleModelRouter`, `RecordMoERequest()` Prometheus counter, `EPDispatch`/`LocalEPDispatch` stub interface. Grade moves from **F** to **D** — multi-GPU expert sharding remains.
- OpenAI API compatibility bumped to **C+**—basic chat completion, SSE streaming, structured output, image_url content parts (§2.2), and streaming tool call deltas (§2.3) all work; model-native chat templates remain.
- Enterprise auth/RBAC at **B-** reflects working OIDC/API-key flows but lacks fine-grained RBAC UX improvements noted in Policy backlog.
- Observability is **B** thanks to metrics/tracing/logging closures in OBS-1 through OBS-4.
- Ease of local setup is **C**; `inferctl pull` (§2.8) now downloads GGUF models from HuggingFace Hub — streamlined installers remain for a future pass.
- Model management UX bumped from **F** to **D**; `inferctl pull` with progress reporting and quantization selection (Q4_K_M preferred) is live (§2.8); full registry UI and listing commands remain.
- Test coverage/CI maturity bumped to **B**; CI now has five jobs: CPU `build-and-test` (ubuntu-latest, full ctest + SBOM), `build-check-mps` (macos-latest, MPS, full unit tests), `build-check-cuda` (CUDA 12.3 compile-check, advisory), `coverage` (gcov/lcov Debug build + Codecov upload), and `clang-format`. Coverage pipeline: `ENABLE_COVERAGE=ON` adds `--coverage -O0 -fno-inline` flags + links `--coverage`; `cmake --build --target coverage` zeroes counters, runs ctest, merges lcov traces, strips external/tests/usr, generates HTML report and lcov.info; `.codecov.yml` enforces 60% target / 5pp threshold; HTML report uploaded as `coverage-html-<sha>` artifact (14-day retention). Remaining gap to A: live GPU test execution (self-hosted runner), CI SHM smoke test.

InferFlux has strong *architectural vision* (enterprise auth, policy store, multi-backend) but
the implementation is at stub/MVP stage. The competitive gap is largest in the core inference
pipeline (batching, KV cache, parallelism) and in table-stakes features that every 2026 server
must have (structured output, multimodal, tool calling).

### Status Notes (May 2025)
- **Streaming telemetry:** on-token SSE, cancellation flags, and Prometheus counters
  (`inferflux_stream_tokens_total`, `inferflux_stream_cache_hits_total`) are live, plus an
  automated SSE cancellation test — **Done**.
- **Tool-calling fallback:** stub tool_calls fire (and log) when no backend is loaded — **Done**.
- **RequestBatch:** InferenceRequest adoption, batch builder, streaming/cancellation, and batch
  metrics are complete; priority fairness/preemption is the next target.
- **Hardware focus:** CPU/MPS paths active; CUDA validation deferred until compatible hardware
  becomes available.

### Where InferFlux Has Potential Differentiation

1. **Integrated policy engine with encrypted persistence** — No competitor has built-in RBAC +
   encrypted policy store + admin APIs. vLLM and SGLang rely on external auth layers.
2. **Single-binary multi-backend** — The vision of CPU/CUDA/ROCm/MPS/Intel GPU from one binary is
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

- **What to add:** OpenAI-style `response_format` parsing, adapter layer that translates schemas into backend-native grammars (llama.cpp GBNF today), and a normalized constraint payload on `InferenceRequest`.
- **Backend capabilities:** `ModelRouter`/`BackendManager` must advertise whether a backend supports structured output and which adapter it expects so unsupported combinations fail early.
- **Sampler hooks:** `BatchExecutor`/`LlamaCPUBackend` (and future CUDA/ROCm/MPS backends) need pluggable grammar hooks instead of hard-coded llama.cpp calls.
- **Testing & docs:** Contract tests covering nested schemas + integration tests validating HTTP→backend flow, CLI/docs describing limits/error handling.
- **Implement in:** `server/http/http_server.cpp`, `scheduler/request_batch.h`, `scheduler/scheduler.cpp`, `runtime/execution/batch_executor.cpp`, `runtime/backends/*`, `cli/main.cpp`.

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

### 2.4 Prefix Caching / Automatic KV Cache Reuse — **DONE**

**Why critical:** SGLang's RadixAttention achieves 85-95% cache hit for few-shot workloads vs.
15-25% for naive paging. This is the single biggest throughput differentiator for multi-turn
and agent workloads.

- **Implemented:** `RadixPrefixCache` (`runtime/prefix_cache/radix_prefix_cache.{h,cpp}`) —
  a compressed trie (radix tree) over token ID sequences. Nodes hold cached completions at
  points where past requests terminated; `Lookup` returns the longest matching prefix plus a
  `matched_tokens` out-parameter for partial-hit metrics.
- **Wiring:** `Scheduler` and `BatchExecutor` migrated from the old `PrefixCache` (flat LRU
  hash-map) to `RadixPrefixCache`. `server/main.cpp` constructs `RadixPrefixCache` at startup.
- **Metrics:** Two new Prometheus counters — `inferflux_prefix_matched_tokens_total` (tokens
  matched even on partial hits) and `inferflux_prefix_partial_hits_total` (lookups with a
  shared prefix but no exact completion). These quantify future KV reuse opportunity.
- **LRU eviction:** DFS leaf collection + `std::min_element` evicts the least-recently-used
  completion node when `size > capacity`.
- **Thread safety:** `std::shared_mutex` — shared lock for `Lookup`, exclusive for `Insert`/eviction.
- **Tests:** 12 `[radix_cache]` unit tests in `tests/unit/test_radix_prefix_cache.cpp`.
- **Still TODO:** Attach KV page IDs to trie nodes once llama.cpp multi-sequence support lands (true zero-copy prefix reuse). `docs/NFR.md` §Performance KPI table updated.

### 2.5 Disaggregated Prefill/Decode — Option A Done

**Why critical:** Now the default production architecture. vLLM deploys it at Meta, LinkedIn,
Mistral. SGLang tested it on 96 H100s. NVIDIA Dynamo provides orchestration for it.

- **Implemented (Option A — in-process phased prefill/decode):**
  - `LlamaCPUBackend::Prefill(prompt, seq_id)` evaluates prompt tokens into the KV cache for slot `seq_id` and returns `n_past`. `Decode(n_past, seq_id, …)` continues from that position. `FreeSequence(seq_id)` cleans up the KV slot via `llama_memory_seq_rm`. `BatchAddSeq()` sets per-token `seq_id` on `llama_batch`.
  - `Scheduler` maintains a `seq_slots_free_` free-list (16 slots). `AllocSeqSlot()` / `FreeSeqSlot()` replace the unsafe `request_id % 16` modulo that caused KV corruption between concurrent requests.
  - `ProcessBatch` kPrefill block calls `Prefill()` before handing the request to `BatchExecutor`; `ExecuteRequest` branches on `n_past >= 0` for `Decode()` vs legacy `Generate()`.
  - `KVChannel` gate: only enqueued when `use_decode_workers_=true`; while workers are disabled the channel is bypassed entirely (prevents fill-and-deadlock at capacity 64).
  - 11 `[phased]` unit tests in `tests/unit/test_phased_execution.cpp`.
  - **Docs:** `docs/Architecture.md` updated with implementation details and remaining work checklist.
- **Items 11-12 done (SHM transport + control plane) — post-review bugs fixed:**
  - `IKVTransport` pure-virtual interface (`kv_channel.h`): `Enqueue`/`TryDequeue`/`Size`/`Capacity`. Both `KVChannel` and `ShmKVTransport` implement it. `DisaggregatedConfig.kv_transport` is `shared_ptr<IKVTransport>` (renamed from `kv_channel`). Scheduler's four `kv_channel` references updated. `INFERFLUX_KV_TRANSPORT=shm` env var in `main.cpp` selects `ShmKVTransport` at runtime; default is `KVChannel`.
  - `ShmKVTransport` (`runtime/disaggregated/shm_kv_transport.{h,cpp}`): `Enqueue()` stores `kv_blob` in named SHM segment, `TryDequeue()` maps + copies + unlinks; `"/ifx_kv_{request_id}_{counter}"` naming; `-lrt` on Linux; meets <5 ms SLA (OS shares physical pages).
  - `inferflux_kv_transfer_duration_ms` Prometheus histogram; `RecordKVTransfer()` called in `DecodeWorkerLoop` on each `TryDequeue()`.
  - `INFERFLUX_ROLE` env var (`prefill`/`decode`/`unified`); `/readyz` decode role now gates on model_loaded AND pool_warm (not just pool_warm — avoids Kubernetes routing to unweighted decode pods).
  - `prefill_pool_size=0` supported: `std::max(0, ...)` in YAML parse, env parse (`stoi` instead of `ParsePositiveSize`), and config assignment.
  - Helm overlays: `deploy/helm/prefill-values.yaml` + `deploy/helm/decode-values.yaml`; decode overlay sets `INFERFLUX_PREFILL_POOL_SIZE=0` + `INFERFLUX_KV_TRANSPORT=shm`.
  - `ShmTransportTests` ctest target; 10 `[shm_transport]` unit tests.
- **Remaining (multi-node):**
  - RDMA transport (multi-node, multi-pod): requires RDMA-capable NICs; replace SHM with ibverbs or UCX.
  - Chaos tests (inject SHM segment failures, validate graceful fallback).
  - CI SHM smoke test (cross-process validation in CI).

### 2.6 Expert Parallelism for MoE Models

**Why critical:** DeepSeek-V3/R1 and Mixtral dominate open-source. SGLang achieves 5x
throughput over vanilla TP for MoE models using expert parallelism.

#### Implemented
- `LlamaCPUBackend::IsMoE()` / `ExpertCount()` / `ActiveExperts()` — reads `llm.expert_count` / `llm.expert_used_count` from GGUF metadata.
- `ModelInfo`: `is_moe`, `n_experts`, `n_active_experts` — populated in `SingleModelRouter::RegisterModel` and `LoadModel` after backend is ready.
- `RecordMoERequest()` Prometheus counter (`inferflux_moe_requests_total`).
- `EPDispatch` + `LocalEPDispatch` stub (`runtime/backends/ep_dispatch.h`) — single-process default, owns all experts, `world_size=1`.
- 9 `[moe]` unit tests (`tests/unit/test_moe_routing.cpp`).

#### Remaining
- Multi-GPU expert sharding via NCCL or shared-memory ring.
- EP-aware batch routing in `BatchExecutor` (partition expert layers by rank).
- Fused MoE kernels (requires CUDA/ROCm backend).

### 2.7 Flash Attention Integration ✅ (llama.cpp wiring done; FA3 CUDA kernels hardware-blocked)

**Why critical:** FlashAttention-3 achieves 1.5-2x over FA2, 75% GPU utilization on Hopper.
Every production GPU server uses it.

**Done:**
- `LlamaBackendConfig::use_flash_attention` / `flash_attention_tile` config fields
- YAML (`runtime.cuda.flash_attention.enabled/tile_size`) + env-var (`INFERFLUX_CUDA_FLASH_ATTENTION`, `INFERFLUX_CUDA_FLASH_TILE`) parsing in `server/main.cpp`
- `ctx_params.flash_attn_type` wired to `LLAMA_FLASH_ATTN_TYPE_ENABLED` / `DISABLED` in `LlamaCPUBackend::LoadModel()`
- `LlamaCPUBackend::FlashAttentionEnabled()` accessor
- `MetricsRegistry::SetFlashAttentionEnabled()` + `inferflux_flash_attention_enabled` Prometheus gauge (0/1)
- 9 `[flash_attn]` unit tests in `tests/unit/test_flash_attn.cpp`; `FlashAttnTests` ctest target
- Architecture.md §Flash Attention section added

**Remaining (hardware-blocked):**
- FA3 CUDA kernels on Hopper/Ada — link against cutlass or flash-attention library
- Per-layer kernel selection metrics

### 2.8 Model Management UX (`inferctl pull`)

**Why critical:** Ollama's `ollama pull llama3` is the gold standard for local UX. The PRD
mentions `inferctl pull` but it is not implemented.

- **What to add:** Pull models from HuggingFace Hub, local model registry, model listing,
  progress indicators, quantization selection.
- **Update:** `docs/PRD.md` §Functional Requirements — detail the pull workflow.
- **Implement in:** `cli/main.cpp` (pull subcommand), new `model/registry/` module.

### 2.9 Request Priority & Fairness Scheduling — **DONE**

**Why critical:** Emerging as a differentiator. PROSERVE and FairBatching (published 2025)
show SLO-aware scheduling prevents starvation under load. vLLM has priority-aware preemption.

- **Implemented:** `FairnessController` (`scheduler/fairness_controller.h/.cpp`) with
  `FairnessConfig` (timeslice tokens, preemption threshold, enable_preemption flag),
  `FairnessDecision` (swap/batch_index/queue_index), `ApplyTimeslice()` for token caps.
- **Scheduler integration:** `ApplyFairness()` evaluates batch vs queue for preemption,
  applies timeslice caps to low-priority requests, requeues fairness-yielded requests.
- **Fairness metrics:** `RecordFairnessTokens`, `RecordFairnessPreemption`,
  `RecordFairnessYield`, `RecordFairnessResume` in `MetricsRegistry`.
- **Tracing:** W3C `Span` hooks emitted on fairness yield and resume events.
- **Live config:** `Scheduler::UpdateFairnessConfig()` accepts new `FairnessConfig` at runtime.
- **Tests:** 3 `[fairness]`-tagged unit tests in `test_scheduler.cpp`.
- **Docs complete:** `docs/PRD.md` Current Status updated; `docs/Architecture.md` §Fairness & Preemption fully documents the implemented design.

### 2.10 Streaming Telemetry & Tool Logging

**Why critical:** Operators debug SSE by monitoring real-time token flow, and platform teams
need audit trails when stub tool-calls fire because no backend is loaded. Competitors expose
streaming histograms and structured tool-call logs; we must match that before fairness work lands.

- **What to add:** Prometheus counters for streamed tokens vs cache hits (**Done**), a configurable
  tool-call log path (`INFERFLUX_LOG_TOOL_CALLS`, **Done**), and fairness/preemption hooks so the
  metrics tie back to schedulable actions.
- **Update:** `docs/PRD.md` §Status, `docs/Architecture.md` §Server, `docs/Roadmap.md` Workstream A
  to emphasize CPU/MPS fairness milestones before CUDA hardware arrives.
- **Implement in:** `server/metrics/metrics.*`, `server/http/http_server.cpp`, scheduler fairness follow-ups.

### 2.11 Vendor GPU Enablement (Intel GPU + AMD ROCm)

**Why critical:** The PRD highlights a single-binary story spanning CPU, CUDA, ROCm, and future Intel GPUs. Competitors already list Intel GPU and ROCm support; we must keep designs ready so once hardware arrives we can land support rapidly.

- **What to add:** Abstraction seams in `DeviceContext`/`BackendManager` to host ROCm/Intel backends, capability detection, and CI smoke tests that run vendor-specific kernels where hardware exists.
- **Update:** `docs/Roadmap.md` Workstream A — add “Intel/ROCm backend scaffolding” as a design task gated on hardware availability.
- **Update:** `docs/Architecture.md` §Runtime — spell out how vendor backends plug into `DeviceContext` and scheduler routing.
- **Implement in:** `runtime/backends/` (ROCm + Intel backend shims, build flags), `scheduler/model_router.*` (backend descriptors), `docs/PRD.md` (explicit Intel GPU call-out).

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
| SEC-4 | **[MERGED]** Merged into OBS-4. | `server/logging/audit_logger.cpp` | *(See OBS-4)* | `docs/NFR.md` §Security |
| SEC-5 | **[FIXED]** HttpServer terminates TLS (configurable cert/key, SSL_accept/read/write). HttpClient fully supports HTTPS: `ParseUrl()` detects `https://`, sets port 443, creates `SSL_CTX` with `SSL_VERIFY_PEER` + `SSL_set_default_verify_paths`, performs TLS handshake and certificate verification in both `Send()` and `SendRaw()`. | `server/http/http_server.cpp`, `net/http_client.cpp` | *(Fixed)* | `docs/NFR.md` §Security |

### 3.2 Core Inference Pipeline — HIGH

| ID | Issue | File(s) | Fix | Affects Docs |
|----|-------|---------|-----|-------------|
| INF-1 | **[FIXED]** Multi-threaded HTTP server with configurable thread pool (accept loop + worker threads via condition variable queue). | `server/http/http_server.cpp` | *(Fixed)* | — |
| INF-2 | Scheduler and execution pipeline need significant work to match production server performance. | `scheduler/scheduler.cpp`, `runtime/backends/cuda/`, `cmake/` | **Status**: Foundational components (queue, BatchExecutor, ModelRouter) are in place. **Next**: Implement true prefill/decode overlap, GPU-aware scheduling, and integrate FlashAttention kernels. | `docs/Architecture.md` §Scheduler, `docs/PRD.md` §Goals |
| INF-3 | **[FIXED]** Dynamic HTTP buffer with Content-Length awareness, 16MB max, 413 for oversized. | `server/http/http_server.cpp` | *(Fixed)* | — |
| INF-4 | **[FIXED]** `llama_backend_init()`/`free()` now reference-counted via static counter with mutex. | `runtime/backends/cpu/llama_backend.cpp` | *(Fixed)* | — |
| INF-5 | **[FIXED]** Stub fallback returns 503 Service Unavailable with clear error message. | `scheduler/scheduler.cpp`, `server/http/http_server.cpp` | *(Fixed)* | `docs/PRD.md` §Acceptance Criteria |
| INF-6 | **[FIXED]** Scheduler now reserves/releases KV cache pages per batch, preparing for future offload/prefetch work. | `scheduler/scheduler.cpp` | *(Fixed)* | — |
| INF-7 | **[FIXED]** `LlamaCPUBackend::TokenCount()` exposes llama.cpp vocabulary-based token counting. `BatchExecutor` uses it when a backend is loaded; falls back to `SimpleTokenizer` otherwise. | `runtime/backends/cpu/llama_backend.cpp`, `runtime/execution/batch_executor.cpp` | *(Fixed)* | — |

### 3.3 Data Handling — MEDIUM

| ID | Issue | File(s) | Fix | Affects Docs |
|----|-------|---------|-----|-------------|
| DAT-1 | **[FIXED]** Adopted nlohmann/json v3.11.3 — replaced 4 hand-rolled parsers (~310 lines). CLI pending (CQ-1). | `external/nlohmann/json.hpp` | *(Fixed)* | — |
| DAT-2 | **[FIXED]** Hand-rolled YAML parser was fragile and has been replaced. | `server/main.cpp` | Replaced with `yaml-cpp` library. | — |
| DAT-3 | GGUF loader is a complete stub — returns dummy tensor with file size | `model/gguf/gguf_loader.cpp` | **Decision**: Delegate all GGUF parsing to the `llama.cpp` backend. This internal loader should not be implemented further and can be removed once the `LlamaBackend` is fully integrated. | `docs/Architecture.md` §Model |
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
| OBS-2 | **[FIXED]** Per-phase histograms (prefill/decode) added to MetricsRegistry; W3C traceparent parsed from incoming requests and emitted in responses; lightweight `Span` RAII abstraction in `server/tracing/span.h` (extension point for full OTel SDK). 78 unit tests (11 for tracing). | `server/metrics/metrics.cpp`, `runtime/execution/batch_executor.cpp`, `server/http/http_server.cpp`, `server/tracing/span.h` (new) | *(Fixed)* | `docs/Architecture.md` §Observability |
| OBS-3 | **[FIXED]** Added `/readyz` (model-aware), `/livez` (always 200), updated `/healthz` to show degraded status. | `server/http/http_server.cpp` | *(Fixed)* | — |
| OBS-4 | **[FIXED]** AuditLogger hashes prompts/responses by default (configurable via `debug_mode=true`) to prevent logging sensitive data and mitigate injection attacks. | `server/logging/audit_logger.cpp` | *(Fixed)* | `docs/Architecture.md` §Security |

### 3.6 Testing — MEDIUM

| ID | Issue | File(s) | Fix | Affects Docs |
|----|-------|---------|-----|-------------|
| TST-1 | **[FIXED]** 57 Catch2 tests across 9 files: ApiKeyAuth(6), RateLimiter(4), Guardrail(6), AuditLogger(6), Metrics(9), Scheduler(7), PolicyStore(5), OIDC(9), Tokenizer(2) + originals. | `tests/unit/` | *(Fixed)* | `docs/NFR.md` §Testing |
| TST-2 | **[FIXED]** Adopted Catch2 v3.7.1 amalgamated. All tests use TEST_CASE/REQUIRE. | `external/catch2/`, `CMakeLists.txt` | *(Fixed)* | — |
| TST-3 | **[FIXED]** Stub-mode integration tests added (`stub_integration_test.py`, 17 tests). Wired as `StubIntegration` ctest target unconditionally. | `tests/integration/stub_integration_test.py`, `CMakeLists.txt` | *(Fixed)* | — |
| TST-4 | Integration test `http_get` sends no auth header — may pass/fail depending on auth state | `tests/integration/sse_metrics_test.py` | Fix test helpers to include API key header | — |

### 3.7 Code Quality — LOW

| ID | Issue | File(s) | Fix | Affects Docs |
|----|-------|---------|-----|-------------|
| CQ-1 | **[FIXED]** CLI refactored from 648→310 lines using shared HttpClient and nlohmann/json. | `cli/main.cpp`, `net/http_client.cpp` | *(Fixed)* | — |
| CQ-2 | **[FIXED]** `CPUDeviceContext` now has a destructor that frees all tracked allocations. | `runtime/backends/cpu/cpu_backend.cpp/.h` | *(Fixed)* | — |
| CQ-3 | **[VERIFIED INCORRECT]** `Guardrail::Enabled()` returns false when only OPA endpoint is configured (empty blocklist). The implementation correctly checks both `blocklist_.empty()` and `opa_endpoint_.empty()`. | `server/policy/guardrail.cpp:74` | *(Not a bug)* | — |
| CQ-4 | **[FIXED]** `HttpClient::send()` now loops on partial sends and throws on failure. Also added 30s socket timeouts. | `net/http_client.cpp` | *(Fixed)* | — |
| CQ-5 | **[FIXED]** `Stop()` shuts down listening socket via `shutdown(SHUT_RDWR)` to unblock `accept()`. | `server/http/http_server.cpp` | *(Fixed)* | — |
| CQ-6 | **[FIXED]** Config path now resolves relative to the executable directory when the CWD-relative path does not exist, so `inferfluxd` can be invoked from any working directory. | `server/main.cpp` | *(Fixed)* | — |
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
| P0 | Fix OIDC signature verification | **Done** — JWKS fetch + 5-min TTL cache + RS256 verify + claim validation | SEC-1 | `server/auth/oidc_validator.cpp` |
| P0 | Hash API keys at rest and in memory | **Done** — ApiKeyAuth hashes with SHA-256; PolicyStore hashes on write; AddKeyHashed for load-from-disk path | SEC-2 | `server/auth/api_key_auth.cpp`, `policy/policy_store.cpp` |
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
| P0 | Queue-based scheduler + priority scheduling | **Done** — worker thread, fairness aging, BatchExecutor | INF-2, §2.9 | `scheduler/scheduler.cpp`, `runtime/execution/batch_executor.cpp` |
| P0 | **[UPGRADE]** Structured output / JSON mode | **Done** — HTTP parser validates `response_format`, `StructuredOutputAdapter` converts schemas to llama GBNF, `BatchExecutor`/`LlamaCPUBackend` enforce grammar sampling, unit tests cover adapter | §2.1 | `server/http/http_server.cpp`, `scheduler/request_batch.h`, `runtime/execution/batch_executor.cpp`, `runtime/structured_output/` |
| P0 | **[NEW]** Tool calling / function calling | **Done** — parse `tools[]`+`tool_choice`, inject schema as system preamble, detect `tool_call` JSON in output, emit `tool_calls` array with `finish_reason=tool_calls` | §2.3 | `server/http/http_server.cpp` |
| P0 | **[UPGRADE]** CUDA backend with FlashAttention | **Done (llama.cpp wiring)** — `ctx_params.flash_attn_type` wired in `LoadModel()`; `FlashAttentionEnabled()` accessor; `inferflux_flash_attention_enabled` gauge; 9 unit tests. FA3 CUDA kernels hardware-blocked (pending L40S/H100 build). | §2.7 | `runtime/backends/cpu/llama_backend.cpp`, `server/metrics/metrics.cpp` |
| P1 | **[NEW]** Prefix caching (radix tree KV reuse) | **Done** — `RadixPrefixCache` compressed trie; `Lookup` with `matched_tokens` out-param; LRU eviction; partial-hit Prometheus counters; 12 unit tests; scheduler + executor migrated | §2.4 | `runtime/prefix_cache/radix_prefix_cache.cpp` |
| P1 | Prompt/response hashing in audit logs | **Done** | OBS-4 | `server/logging/audit_logger.cpp` |
| P1 | **[DONE]** Multimodal / vision model support | **Done** — `ImagePreprocessor` (base64 decode, URL fetch, SHA-256 image IDs, `<__media__>` marker injection); `InferenceRequest.images` field; `LlamaCPUBackend::LoadMmproj()`/`GenerateWithImages()` (guarded by `INFERFLUX_HAS_MTMD`); multimodal Prometheus counters; 11 unit tests | §2.2 | `runtime/multimodal/`, `server/http/http_server.cpp`, `runtime/backends/cpu/llama_backend.cpp` |
| P1 | **[DONE]** Model pull from HuggingFace (`inferctl pull`) | Done | §2.8 | `cli/main.cpp` |
| P1 | TLS support (server + HttpClient) | **Done** — HttpServer TLS + HttpClient HTTPS both fully implemented with TLS handshake and cert verification | SEC-5 | `net/http_client.cpp` |
| P1 | Latency histograms and queue depth gauges in metrics | **Done** — 3 histograms (request/queue/batch), batch + prefix counters | OBS-1 | `server/metrics/` |
| P1 | OpenTelemetry trace integration | **Done** — W3C traceparent propagation, prefill/decode histograms, `server/tracing/span.h` RAII abstraction | OBS-2 | `server/tracing/span.h` (new) |
| P2 | Readiness vs liveness health probes | **Done** | OBS-3 | `server/http/http_server.cpp` |
| P2 | CORS headers | **Done** | DAT-4 | `server/http/http_server.cpp` |
| P2 | Refactor CLI to reuse HttpClient | **Done** | CQ-1 | `cli/main.cpp` |

**Existing Q3 items from `docs/Roadmap.md` remain:** CUDA/ROCm acceleration, NVMe-assisted KV
cache, speculative decoding, LoRA stacking, hot adapter reloads, autoscaler hints.

### Q4 — Enterprise, Scale & Differentiation

**Focus: Distributed operations, advanced scheduling, competitive differentiation.**

| Priority | Item | Status | Tracking IDs | Key Files |
|----------|------|--------|-------------|-----------|
| P0 | **[NEW]** Disaggregated prefill/decode | **Option A done** — in-process phased Prefill/Decode/FreeSequence, slot free-list, KVChannel gate; decode workers + cross-process transport pending | §2.5 | `runtime/disaggregated/`, `runtime/backends/cpu/llama_backend.cpp`, `scheduler/scheduler.cpp` |
| P0 | Expert parallelism for MoE models | Detection + stub done (IsMoE, EPDispatch, metrics); multi-GPU sharding pending | §2.6 | `runtime/backends/`, `scheduler/` |
| P1 | **[NEW]** Request priority & fairness scheduling | **Done** — `FairnessController` with timeslice, preemption, resume; `ApplyFairness()` in scheduler; fairness metrics (tokens/preemption/yield/resume); `Span` hooks for yield/resume; `UpdateFairnessConfig()` live API; 3 unit tests (`[fairness]`) | §2.9 | `scheduler/fairness_controller.h`, `scheduler/scheduler.h` |
| P1 | **[UPGRADE]** Tensor + pipeline parallelism | Not started | — | `runtime/backends/` |
| P2 | YAML parser replacement (yaml-cpp) | **Done** | DAT-2 | `server/main.cpp` |
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
- [X] §Functional Requirements: Add multimodal input (image content parts) — §2.2
- [X] §Functional Requirements: Add `inferctl pull` workflow — §2.8
- [X] §Functional Requirements: Add request priority parameter — §2.9
- [X] §Goals: Mention MoE model support explicitly — §2.6
- [X] §User Stories: Add agent workflow story (tool calling) — §2.3
- [X] §User Stories: Add vision model story — §2.2
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
- [X] §Observability: Document audit log redaction (hash by default) — OBS-4
- [X] §Modules: Add constrained decoding module — §2.1
- [X] §Modules: Add parallelism strategies section (TP/PP/EP) — §2.6, §2.7
- [X] §Data Flow: Add image preprocessing stage — §2.2
- [X] §Deployment View: Add disaggregated prefill/decode topology — §2.5
- [X] §Runtime: Add prefix cache subsystem — §2.4
- [x] §Runtime: Document attention kernel strategy (FA3) — §2.7 (Architecture.md §Flash Attention added)
- [X] §Scheduler: Add priority queue design — §2.9

### `docs/NFR.md`

- [X] §Performance: Add latency target for constrained vs unconstrained decoding — §2.1
- [X] §Performance: Add prefix cache hit rate KPIs — §2.4
- [X] §Scalability: Add KV transfer latency targets for disaggregation — §2.5
- [X] §Security: Require TLS support — SEC-5
- [X] §Security: Require JWT signature verification — SEC-1
- [X] §Operability: Add histogram and gauge metric requirements — OBS-1
- [X] §Testing: Require unit test coverage targets per module — TST-1

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

- [X] §Capabilities: Document Guardrail::Enabled() fix for OPA-only configs — CQ-3
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
| ARCH-4 | Wire `ModelRouter` into `Scheduler` | `scheduler/scheduler.h` | Replace direct `LlamaCPUBackend` dependency with `ModelRouter` abstraction. | **Done** |
| ARCH-5 | Wire `RequestBatch` into `Scheduler` | `scheduler/scheduler.h` | Replace `GenerateRequest`/`GenerateResponse` with `InferenceRequest` flow through `RequestBatch`. | In progress (HTTP + scheduler now emit `InferenceRequest`; batch builder/streaming pending) |
| ARCH-6 | Create `SingleModelRouter` | `scheduler/single_model_router.cpp` | Default `ModelRouter` implementation wrapping a single backend. Drop-in replacement for current behavior. | **Done** |

### 7.1 ModelRouter Activation Plan (Foundation)
- **Why now:** PRD §Functional Requirements #1/#10 and Roadmap Workstream A require multi-model routing before CUDA + prefix-cache milestones can land. Without it, features like `inferctl pull`, request priority, and fairness scheduling cannot target specific model pools.
- **Current status:** `ModelRouter` and `RequestBatch` interfaces exist but the scheduler, admin APIs, and CLI still assume a single backend.
- **Execution (see `docs/Architecture.md` for expanded design):**
  1. **[x] ARCH-6 — `SingleModelRouter` implementation:** add `scheduler/single_model_router.{h,cpp}` owning a registry map, mutex, and shared_ptr backends; cover load/unload/list tests in `tests/unit/test_scheduler.cpp`.
  2. **[x] ARCH-4 — Scheduler→Router wiring:** refactor `scheduler/scheduler.{h,cpp}` and request structs so every admission path resolves a model ID, handles `Resolve()` failures, and plumbs backend handles into batches/metrics.
  3. **[x] Registry + metrics:** introduce a `ModelRegistryEntry` struct (ID/path/backend/KV footprint/ready_ts) exposed via `/v1/admin/models` and Prometheus (`inferflux_model_routes_total`, `inferflux_model_load_seconds`).
  4. **[x] Control surface updates:** extend `server/http/http_server.cpp` and `cli/main.cpp` to list/load/unload models with RBAC scope checks; document new commands in `README.md` + `docs/Roadmap.md`.
  5. **[x] Config/bootstrap:** add `models` array to `config/server.yaml` plus env overrides, enabling declarative default model loading while allowing on-demand changes through the router.
  6. **[x] Rollout + observability:** update `/readyz` semantics (“ready when ≥1 model ready”), add structured logs for routing choices, and codify regression tests that exercise multi-model queues.
- **Exit criteria:** Scheduler never references concrete backends, admin/CLI/test suites can manipulate model inventory, and Workstream A KPIs (400 tok/s aggregate, >60% prefix cache hit rate) track per-model metrics.

### 7.2 RequestBatch Integration Plan (Continuous Batching Core)
- **Why now:** Workstream A (Throughput Foundation) and PRD Functional Requirement #10 (“per-request priority hints”) depend on phase-aware continuous batching with explicit `RequestBatch` plumbing. Without it, GPU prefill/decode overlap, fairness scheduling, and streaming cancellation remain bolted-on hacks.
- **Current status:** HTTP handlers and `Scheduler::Generate` now speak `InferenceRequest`, streaming/cancellation run through `on_token`, and batch metrics/streaming counters are live; remaining work is GPU overlap + fairness/preemption knobs.
- **Execution checklist (see `docs/Architecture.md` for detail):**
  1. **[x] ARCH-5 — Scheduler adopts `InferenceRequest`:** drop the legacy DTOs in `server/http/http_server.cpp`, `scheduler/scheduler.{h,cpp}`, and tests so requests flow through `InferenceRequest` end-to-end (priority, streaming callbacks, token buffers).
  2. **[x] BatchBuilder helper:** refactor `Scheduler::BuildBatchLocked` into a reusable component that emits `RequestBatch` objects with explicit token budgets, fairness aging, and queue metrics (batch-selection helper in `scheduler/scheduler.cpp`).
  3. **[x] Prefill/decode stages:** `BatchExecutor::ExecuteBatch()` and `ExecuteRequest()` now operate on `RequestBatch`, tag phases, and reserve/release KV pages per batch.
  4. **[x] Streaming/cancellation plumbing:** HTTP SSE flows install `InferenceRequest.on_token` callbacks + cancellation flags so streaming happens directly from the scheduler/executor.
  5. **[x] Metrics + observability:** batch-level prefill/decode durations recorded via `RecordPrefillDuration` / `RecordDecodeDuration`; histograms exposed in `server/metrics/metrics.cpp`.
  6. **[x] Testing/regression:** 3 new `[scheduler]` tests cover cancellation flag pre-set → `[cancelled]`, on_token callback via prefix cache hit, and max_tokens=0 clamped to 1. SSECancel integration test covers streaming + cancellation at HTTP level. 84 unit tests total.
- **Exit criteria:** Scheduler only deals with `RequestBatch`/`InferenceRequest`, metrics expose batch/phase telemetry, SSE streaming uses scheduler callbacks, and Workstream A GPU/prefix cache milestones can build on the shared abstraction.

---

## 8. Cross-Session Tracking

This document should be consulted at the start of each development session. Use the tracking
IDs (SEC-1, INF-2, etc.) in commit messages and PR descriptions for traceability.

**Quick reference — what to work on next, by role:**

| If you are... | Start with |
|---------------|-----------|
| Fixing security | SEC-5: wire HTTPS into HttpClient (ssl_ctx_ exists, ConnectAndSend needs TLS path for https:// URLs — needed for OIDC JWKS in production) |
| Adding Q3 features | §2.1: parse `response_format` in http_server (small — BatchExecutor already handles json_mode); §2.3: tool calling (medium); §2.2: multimodal (large) |
| Improving test coverage | TST-3 (stub integration tests without model), TST-4 (auth header in http_get), add PrefixCache unit tests |
| Improving observability | OBS-2 (OpenTelemetry spans for prefill/decode) |
| Improving core performance | True interleaved prefill/decode in BatchExecutor (INF-2 next phase) |
| Refactoring code quality | DAT-2 (YAML parser → yaml-cpp), CQ-6 (config path relative to binary) |
| Updating docs | See §5 Doc Update Tracker |
