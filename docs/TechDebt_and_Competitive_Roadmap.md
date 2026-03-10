# InferFlux Tech Debt and Competitive Roadmap

**Snapshot date:** March 9, 2026  
**Current overall grade:** B-  
**Purpose:** Rank the debt that most directly blocks best-in-class runtime credibility.

```mermaid
flowchart TB
    A[Current B-] --> B[Strengths]
    A --> C[Debt hotspots]
    B --> B1[Control-plane rigor]
    B --> B2[Explicit backend identity]
    B --> B3[Memory-policy foundations]
    C --> C1[Quantized native throughput]
    C --> C2[Distributed ownership maturity]
    C --> C3[Required GPU release enforcement]
```

## 1) Dimension Grades

| Dimension | Grade | Strong today | Weak today |
|---|---|---|---|
| Vision/product coherence | B+ | Clear server-first product shape and explicit dual-CUDA strategy | Native-throughput story still runs ahead of proof on quantized GGUF |
| Capabilities | B+ | Strong API/admin/CLI contracts and machine-visible identity | Some native feature ownership still trails compatibility paths |
| Scalability/economy | C+ | Fairness, phased execution, prefix reuse, split roles, and transport-health semantics exist | Distributed ownership/cleanup semantics are still shallow |
| Resource efficiency | B | KV planner, load-scoped precision, memory-first dequant, 40+ fused GEMV kernels (Q8_1/dp4a/RmsNorm), batched decode ops | Native throughput still trails llama.cpp by 0.35-0.76x; batched decode path not yet default |
| Design/implementation | B | Clean provider split, bounded session-state model, and clearer transport policy | Transitional native/runtime complexity remains until hot paths and ownership close |
| TDD/CI maturity | B+ | Contract suites are explicit and visible | Required GPU/provider lane is still missing |
| OSS docs/operator clarity | A- | Canonical docs are compact and code-aligned | A few active design docs still need reality-first compression |

## 2) Revalidated Evidence

| Evidence | Result | Implication |
|---|---|---|
| Backend/provider contract | Explicit provider/fallback semantics in runtime, API, CLI, admin, and metrics | Strong automation and policy posture |
| Native memory-economy foundation | `dequant_cache_policy=none`, KV planner, and native KV metrics are wired | Good edge-device direction; Q8_1 path improved TinyLlama throughput 49% (161 -> 240 tok/s) |
| Native kernel maturity | 40+ fused GEMV kernels (Q8_1/dp4a/RmsNorm/packed variants), 3 batched decode kernels, adaptive dispatch geometry, 100+ TDD test cases | Kernel coverage is broad; throughput gap remains vs llama.cpp (weight bandwidth utilization ~40% vs ~60%) |
| Sync-first batching stance | Native keeps `SupportsAsyncUnifiedBatch()==false` and relies on sync batch execution. 2D grid GEMV handles all M values in single launch. | Batched decode opt-in via `INFERFLUX_ENABLE_BATCHED_DECODE=1`; needs compute-sanitizer validation before defaulting on |
| Session handle foundation | Optional TTL-based session leases exist in unified scheduler mode | Correct contract direction without changing default stateless behavior |
| Distributed transport foundation | Ticket lifecycle, timeout streak/debt, readiness impact, admin pools visibility, and optional fail-closed generation admission are implemented | Moves distributed runtime beyond scaffold-only claims, but not yet to robust ownership maturity |

## 3) Debt Register

| Priority | Debt item | Why it matters | Retirement gate | Tracking |
|---|---|---|---|---|
| P0 | Quantized GGUF native throughput parity | Native hot paths are fused and memory-first but still 0.35-0.76x of llama.cpp. Closing the weight bandwidth gap is the next throughput jump. | Native sequential decode reaches >= 0.8x llama.cpp on Qwen2.5-3B Q4_K_M | [P1-2](issues/P1-2-quantized-native-forward-productionization.md) |
| P0 | Batched decode default-on | BatchedRoPE/KvAppend/FlashDecodeMultiSeq and CUDA graph capture verified working with 8 concurrent Qwen2.5-3B requests. Currently opt-in via `INFERFLUX_ENABLE_BATCHED_DECODE=1`. | Promote batched decode to default-on after sustained load testing | [P1-1](issues/P1-1-native-flashattention-production-path.md) |
| P1 | Distributed sequence ownership cleanup | Transport signaling without deterministic cleanup still limits real distributed credibility | Eviction, timeout, and worker-loss cleanup are explicit and tested | [P1-6](issues/P1-6-distributed-failure-path-contract-tests.md) |
| P1 | Native-first parity independence | Delegate coupling hides real native feature gaps | Critical completion/chat/embeddings behavior is native-owned where practical | [P0-1](issues/P0-1-native-cuda-identity-contract.md), [P0-2](issues/P0-2-strict-native-request-policy.md) |
| P1 | GPU KV/page allocator maturity | Memory economy must hold under concurrency, not only at load time | Stable reuse metrics and predictable planner behavior under load | [P0-4](issues/P0-4-gpu-kv-page-allocator-prefix-reuse.md) |
| P1 | Mandatory GPU behavior lane | Native regressions should block merges, not be discovered later | Required CI block for native/provider/runtime gates | [P0-5](issues/P0-5-mandatory-gpu-behavioral-ci-gate.md) |

## 4) Outdated Patterns To Retire

| Pattern | Better practice |
|---|---|
| Treating async support as proof of throughput | Measure batch quality, graph residency, and native hot-path residency instead |
| Using compatibility fallback as invisible feature completion | Expose fallback and native ownership explicitly |
| Static VRAM reservations | Plan KV sizing against budget and publish the decision |
| Treating `/readyz` as passive-only observability | Allow transport-health to influence admission where operators choose fail-closed mode |
| Claiming distributed readiness from topology scaffolding alone | Require ticket lifecycle, ownership semantics, and failure-path tests |

See [MODERNIZATION_AUDIT](MODERNIZATION_AUDIT.md) for the migration table.

## 5) Two-CUDA-Backend Value Split

| Axis | `native_cuda` provider | `cuda_llama_cpp` provider |
|---|---|---|
| Why it exists | Native performance/control path | Stable compatibility and fallback path |
| What it does well now | Policy-visible identity, native loaders, memory-economy foundation, 40+ fused GEMV kernels with Q8_1/dp4a/RmsNorm fusion, batched decode kernels, adaptive dispatch geometry, 100+ TDD tests | Mature GGUF compatibility and higher throughput today |
| What still lags | Weight bandwidth utilization (~40% vs llama.cpp ~60%), batched decode stability under concurrent load | InferFlux-specific kernel/runtime headroom |
| Why both stay | They solve different operational risks today | They keep the control plane stable while native matures |

## 6) Competitive Direction

| Area | Keep | Close next |
|---|---|---|
| Control plane | API/admin/CLI rigor, routing policy, observability, admin pools | Keep current lead |
| Native runtime | Loader detection, memory policy, provider identity, Q8_1/dp4a/RmsNorm kernel family, 2D grid GEMV | Close weight bandwidth gap (40% -> 60%+), stabilize batched decode, enable CUDA graph by default |
| Distributed runtime | Honest bounded claims plus transport-health semantics | Close ownership cleanup, worker-loss handling, and failure CI before broadening claims |

## 7) Canonical References

- [Roadmap](Roadmap.md)
- [Architecture](Architecture.md)
- [GEMV_KERNEL_ARCHITECTURE](GEMV_KERNEL_ARCHITECTURE.md)
- [COMPETITIVE_POSITIONING](COMPETITIVE_POSITIONING.md)
- [MODERNIZATION_AUDIT](MODERNIZATION_AUDIT.md)
