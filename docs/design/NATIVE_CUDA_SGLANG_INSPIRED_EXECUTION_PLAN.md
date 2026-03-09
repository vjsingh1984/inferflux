# Native CUDA + Distributed Runtime Uplift Plan (SGLang-Informed)

Snapshot: March 8, 2026
Status: planned
Owner group: runtime + scheduler + distributed-runtime + QA

## 1) Objective

Raise production-grade throughput/concurrency while improving distributed runtime reliability:

- Throughput/continuous-batching: `B-/C+ -> B/B+`
- Distributed runtime: `C- -> B-`

Success is measured by contract gates, not ad-hoc benchmarks.

```mermaid
flowchart LR
  A[Current] --> B[Scheduler policy uplift]
  B --> C[Native async contract v2]
  C --> D[Quantized GGUF first-class path]
  D --> E[Distributed reliability contracts]
  E --> F[CI-enforced behavior gates]
```

## 2) Codebase Findings (What Exists vs What Is Missing)

| Pillar | Existing in code | Gap to close |
|---|---|---|
| Mixed prefill/decode execution | `Scheduler::ProcessBatch` + `ExecuteUnifiedBatchWithOverlap` are active | Batch admission is priority/age-only; no prefix-aware schedule policy |
| Prefix reuse | `RadixPrefixCache` + phased prefill/prefix-copy integration are active | Prefix reuse is used at execution time, not as first-class batch ranking signal |
| Native async unified batch | Async submit/collect methods exist | `NativeKernelExecutor::SupportsAsyncUnifiedBatch()` is hard-disabled (`false`) |
| Quantized runtime strategy layer | Strategy registry exists (`IWeightLayoutStrategy`, `IMatmulStrategy`, `IAttentionStrategy`) | Selection is mostly logged/validated; execution path is not fully strategy-driven |
| Quantized GGUF execution | Fused GEMV kernels exist and are called | Compatibility/dequant fallback still dominates on many paths; dequant lifecycle policy is coarse (`batch`/`model`) |
| KV precision policy | KV precision is load-scoped and configurable | `int8/fp8` paths are declared but downgraded to FP16 today |
| Distributed handoff | `IKVTransport` + channel/SHM transport + enqueue retry limit exist | No ticket ack/commit protocol; weak fault-domain visibility and recovery semantics |
| Session handles | TTL mapping layer exists | Disabled when decode workers are enabled; no distributed-session ownership model |
| Eviction/cleanup | Idle slot eviction loop exists | TODO remains for backend-sequence ownership cleanup in eviction path |

## 3) Foundations First (Implementation Order)

### Phase 0: Contract and observability foundation (low risk)

- [x] **P0-01 Scheduler policy enum + config plumbing**
  - Add `runtime.scheduler.policy: priority_age|lpm_priority|throughput_balanced`
  - Scope: `scheduler/scheduler.h`, `scheduler/scheduler.cpp`, `server/main.cpp`, `docs/CONFIG_REFERENCE.md`
  - Tests: `tests/unit/test_scheduler.cpp`

- [x] **P0-02 Prefix-affinity scoring in batch selection**
  - Add non-mutating prefix probe during `BuildBatchLocked()` to bias toward reusable prefixes when policy enables it
  - Scope: `scheduler/scheduler.cpp`, `runtime/prefix_cache/radix_prefix_cache.h/.cpp`
  - Tests: new scheduler unit tests for ranking determinism and starvation safety

- [x] **P0-03 Mixed-step tunables (SGLang-style knobs)**
  - Add `continuous_decode_steps`, `chunked_prefill_tokens`, `mixed_prefill_budget_ratio`
  - Scope: `scheduler/scheduler.h`, `runtime/execution/batch_executor.cpp`, `server/main.cpp`
  - Tests: unit tests for step/chunk bounds and fairness interaction

- [x] **P0-04 Metrics for policy efficacy**
  - Add counters/gauges: prefix-affinity hits, decode-step loops, prefill-chunk truncation, per-policy iteration mix
  - Scope: `server/metrics/metrics.h/.cpp`, `tests/unit/test_metrics.cpp`

### Phase 1: Native async contract v2 (throughput-critical)

- [x] **P1-01 Replace per-call `std::async` with persistent lane workers**
  - Add dedicated decode/prefill lane worker threads with bounded queues and CUDA event handoff
  - Scope: `runtime/backends/cuda/native_kernel_executor.h/.cpp`
  - Tests: dispatcher contract tests (`tests/unit/test_lane_dispatcher.cpp`)

- [x] **P1-02 Re-enable `SupportsAsyncUnifiedBatch()` behind readiness gate**
  - Return `true` only when lane workers/resources are healthy
  - Scope: `runtime/backends/cuda/native_kernel_executor.cpp`, `runtime/backends/cuda/native_cuda_backend.cpp`
  - Tests: native gate contract tests + async submit/collect negative contract tests (`tests/unit/test_native_forward.cpp`)

- [x] **P1-03 Async-lane backpressure + failure metrics**
  - Add queue depth, enqueue reject, timeout, lane-reset counters
  - Scope: `server/metrics/metrics.*`, `runtime/backends/cuda/native_kernel_executor.cpp`
  - Tests: metrics unit tests (`tests/unit/test_metrics.cpp`) + native metrics integration assertion (`tests/integration/native_metrics_test.py`)

### Phase 2: Quantized GGUF first-class runtime path

- [x] **P2-01 Strategy selection must drive execution mode**
  - Persist selected strategy IDs/modes in runtime state and use them for dispatch (not log-only)
  - Scope: `runtime/backends/cuda/native_kernel_executor.cpp`, `runtime/backends/cuda/native/strategy_registry.*`
  - Tests: `tests/unit/test_strategy_registry.cpp` + quantized policy gate tests (`tests/unit/test_quantized_weight_map.cpp`)

- [x] **P2-02 Add dequant cache policy `none` (memory-first default for quantized)**
  - Policy semantics: no persistent dequant cache for quantized projections; keep existing `batch`/`model` for compatibility
  - Scope: `runtime/backends/cuda/native/model_loader.*`, `runtime/backends/cuda/native/gguf_model_loader.cpp`, `runtime/backends/cuda/native_kernel_executor.cpp`, config docs
  - Tests: parser contract in `tests/unit/test_strategy_registry.cpp` + extended GGUF memory-contract integration test sections (`batch` + `none`)

- [ ] **P2-03 Coverage-complete fused path for `Q4_K`, `Q6_K`, `Q8_0`, `Q8_K`**
  - Ensure decode + short-prefill + batched decode path consistency with deterministic fallback
  - Scope: `runtime/backends/cuda/native/transformer_forward.cu`, `runtime/backends/cuda/native/fused_quant_gemm.*`
  - Tests: quantized correctness/regression matrix in `tests/integration/test_gguf_quantization.cpp`
  - Progress: strategy-selection coverage for all four target types + fused-threshold bounds and deterministic fallback assertions (policy helper + Gemv/RmsNorm runtime fallback checks for decode/short-prefill/batched decode envelopes) added in unit tests; `Q8_K` coverage added to quantized weight-map tests; CUDA runtime contract test scaffold added for fused launch/fallback behavior with synthetic tensors (skips cleanly when GPU unavailable); GGUF parser aliases now accept `_m` naming variants for `q6_k_m` and `q8_k_m` (`tests/unit/test_strategy_registry.cpp`, `tests/unit/test_native_forward.cpp`, `tests/unit/test_quantized_weight_map.cpp`, `tests/unit/test_gguf_parsing.cpp`)

### Phase 3: CUDA graph and overlap maturity

- [ ] **P3-01 CUDA graph bucket capture/reuse**
  - Capture per-lane buckets keyed by `(phase, batch_size, seq_block)`
  - Scope: `runtime/backends/cuda/native_kernel_executor.cpp`
  - Tests: kernel path selection tests + metric assertions

- [ ] **P3-02 Graph hit/miss + fallback observability**
  - Metrics for graph hit rate, recapture count, graph fallback reason
  - Scope: `server/metrics/metrics.*`, throughput gate parser
  - Tests: `tests/integration/throughput_gate_contract_test.py`

### Phase 4: Distributed runtime foundation (C- to B-)

- [ ] **P4-01 Ticketed KV transport protocol**
  - Introduce `ticket_id`, `ack`, `commit`, `timeout` states for KV handoff
  - Scope: `runtime/disaggregated/kv_channel.*`, `runtime/disaggregated/shm_kv_transport.*`, `scheduler/scheduler.cpp`
  - Tests: new distributed unit tests + SHM integration failure tests

- [ ] **P4-02 Decode worker health and readiness semantics**
  - `/readyz` degrade when worker pool unhealthy or transport saturation exceeds threshold
  - Scope: `scheduler/scheduler.cpp`, `server/http/http_server.cpp`, `server/metrics/metrics.*`
  - Tests: integration readiness/fault tests

- [ ] **P4-03 Sequence ownership registry for eviction cleanup**
  - Implement TODO in eviction path with backend-sequence ownership tracking
  - Scope: `scheduler/scheduler.cpp`, `scheduler/scheduler.h`, `runtime/scheduler/sequence_slot_manager.*`
  - Tests: scheduler eviction/cleanup unit tests

- [ ] **P4-04 Session handles with decode-worker mode**
  - Enable session lease path in decode-worker deployments via ownership-safe commit/release path
  - Scope: `scheduler/scheduler.cpp`, `scheduler/session_handle_manager.*`
  - Tests: scheduler + session manager integration tests

### Phase 5: CI gates and rollout safety

- [ ] **P5-01 Add explicit CI contract block for new scheduler policy tests**
  - Scope: `.github/workflows/ci.yml`
  - Tests: ensure pipeline logs list exact policy-contract test count

- [ ] **P5-02 Add distributed failure contract CI block**
  - Scope: `.github/workflows/ci.yml`, `tests/integration/shm_smoke_test.py`

- [ ] **P5-03 Throughput gate extensions**
  - Add checks: policy mode, continuous decode steps, prefix-affinity efficacy, graph hit-rate floor
  - Scope: `scripts/run_throughput_gate.py`, `tests/integration/throughput_gate_contract_test.py`

## 4) Definition of Done by Grade Target

### Throughput/continuous-batching to B/B+

- Native async unified-batch path is enabled by readiness gate and exercised in CI.
- Throughput gate passes with policy + overlap + native-forward + graph-hit contracts.
- Quantized GGUF path runs without persistent dequantized projection caches by default.

### Distributed runtime to B-

- Transport has deterministic ticket lifecycle (enqueue->ack->commit/timeout).
- Worker health is machine-visible and reflected in readiness.
- Eviction cleanup has no TODO/no-leak sequence ownership path.
- Failure-path integration suite is required in CI logs.

## 5) Risk Controls

- Keep policy default `priority_age` until P0 tests and canary metrics are stable.
- Feature-flag each major phase (`scheduler_policy`, `native_async_v2`, `kv_ticket_protocol`).
- Preserve deterministic fallback path to `cuda_llama_cpp` where policy allows.
- Add rollback docs for each flag in `docs/Troubleshooting.md` once implemented.

## 6) Immediate Next 3 Tasks (Recommended Start)

1. Implement `P0-01` + `P0-02` together (scheduler policy + prefix-affinity scoring).
2. Implement `P1-01` skeleton (persistent lane workers) with async still gated off.
3. Implement `P4-03` sequence ownership registry to close existing eviction TODO and de-risk distributed work.
