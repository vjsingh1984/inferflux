# InferFlux Tech Debt and Competitive Roadmap

**Snapshot date:** March 5, 2026  
**Purpose:** Single-page debt heatmap tied to issue-backed retirement gates.

## 1) Grade Heatmap

```mermaid
flowchart TB
    A[Competitive Gradeboard] --> B[Strengths]
    A --> C[Debt Hotspots]
    B --> B1[Enterprise controls]
    B --> B2[API/admin contracts]
    B --> B3[Hardware breadth baseline]
    C --> C1[GPU batching + KV reuse]
    C --> C2[Native identity ambiguity]
    C --> C3[GPU CI enforcement gap]
    C --> C4[Distributed failure contracts]
```

| Dimension | Grade | What is strong | What is weak |
|---|---|---|---|
| Vision and product coherence | B | Clear OSS identity, OpenAI-compatible API, enterprise posture | Throughput narrative still ahead of full native implementation |
| Capabilities | C+ | Contract gates for models/admin/CLI are strong | Native CUDA provider path is still mixed in user perception |
| Scalability and economy | C | Fairness + phased execution + prefix cache foundation | No full GPU iteration scheduler or KV page allocator |
| Resource efficiency | C | Token-budget and batching instrumentation baseline exists | Economy SLO instrumentation is incomplete |
| Design and implementation quality | B- | Clear subsystem boundaries and capability structs | Transitional dual-path complexity in CUDA stack |
| TDD and CI maturity | B | Broad unit+integration coverage and focused contract suites | Mandatory GPU behavioral lane is not universal yet |

## 2) Debt Register (Actionable)

| Priority | Debt Item | Impact | Owner | Retirement Gate | Issue |
|---|---|---|---|---|---|
| P0 | Native CUDA identity not uniformly explicit | Operator confusion, policy risk, benchmark ambiguity | Runtime + CLI | Provider identity explicit in API/CLI/metrics and strict-fail path tested | [#1](https://github.com/vjsingh1984/inferflux/issues/1), [#2](https://github.com/vjsingh1984/inferflux/issues/2) |
| P1 | GPU continuous batching maturity gap | Throughput/cost lag vs vLLM/SGLang | Scheduler + Runtime | Iteration scheduler with benchmark and regression contracts | [#3](https://github.com/vjsingh1984/inferflux/issues/3) |
| P1 | GPU KV page allocator and prefix reuse gap | Recompute overhead, lower token economy | Runtime | Page allocator + correctness suite + reuse metrics | [#4](https://github.com/vjsingh1984/inferflux/issues/4) |
| P1 | Native attention/quantized kernel maturity gaps | Performance ceiling, fallback dependency | Runtime CUDA | Fused/production kernels with non-regression tests | [#6](https://github.com/vjsingh1984/inferflux/issues/6), [#7](https://github.com/vjsingh1984/inferflux/issues/7) |
| P1 | Scheduler lock contention | Queue latency under load | Scheduler | Lock partitioning and contention regression tests | [#8](https://github.com/vjsingh1984/inferflux/issues/8) |
| P1 | Economy metrics insufficient for autoscaling | Cost control and SLO blind spots | Observability | Efficiency metrics documented and consumed by policies | [#9](https://github.com/vjsingh1984/inferflux/issues/9) |
| P2 | GPU CI behavioral lane not fully mandatory | Regressions can slip by environment variance | QA + Runtime | Merge-blocking GPU behavior lane in CI | [#5](https://github.com/vjsingh1984/inferflux/issues/5), [#10](https://github.com/vjsingh1984/inferflux/issues/10) |
| P2 | Distributed failure-path contract coverage incomplete | Enterprise resilience risk | Distributed Runtime | Fault matrix tests for transport/prefill/decode failures | [#11](https://github.com/vjsingh1984/inferflux/issues/11) |

## 3) Competitive Position (Concise)

| Area | InferFlux current | Direction |
|---|---|---|
| Enterprise controls | Strong relative position | Keep lead via strict contracts + observability |
| Hardware and format breadth | Strong baseline | Maintain while hardening throughput core |
| Raw GPU throughput | Behind leaders | Close gap through #3/#4/#6/#7 |
| CI enforceability | Moderate | Raise with mandatory GPU behavior gates |
| Distributed resilience | Early | Mature via #11 and associated runbooks |

## 4) Execution Order (Minimal Critical Path)

1. Resolve identity and strict policy semantics first (`#1`, `#2`).
2. Land throughput foundation (`#3`, `#4`) before advanced optimizations.
3. Lock in mandatory GPU CI and coverage symmetry (`#5`, `#10`).
4. Complete kernel maturity and scheduler contention items (`#6`, `#7`, `#8`).
5. Finish economy metrics and distributed failure contracts (`#9`, `#11`).

## 5) Evidence Anchors (Code-Backed)

- Backend identity/policy selection: `runtime/backends/backend_factory.cpp`
- Native CUDA executor/delegation path: `runtime/backends/cuda/native_cuda_executor.cpp`
- Native kernel maturity surface: `runtime/backends/cuda/kernels/flash_attention.cpp`, `runtime/backends/cuda/native/quantized_forward.cpp`
- Scheduler contention surface: `scheduler/scheduler.cpp`, `scheduler/scheduler.h`
- CI enforcement surface: `.github/workflows/ci.yml`

## 6) Canonical References

- [Roadmap](Roadmap.md)
- [Architecture](Architecture.md)
- [API_SURFACE](API_SURFACE.md)
- [ARCHIVE_INDEX](ARCHIVE_INDEX.md)
