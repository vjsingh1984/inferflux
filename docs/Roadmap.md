# InferFlux Roadmap

**Snapshot date:** April 9, 2026
**Current overall grade:** B+
**Target overall grade:** A- after GPU CI lane and native structured output

```
Trajectory: B- (Mar 31) → B+ (Apr 9)

Completed since last snapshot:
  ✓ inferflux_cuda at parity with llama.cpp at c=8 (1.02x)
  ✓ 100% accuracy (chat template + repetition penalty)
  ✓ 1.87x faster than Ollama, 2.23x faster than LM Studio
  ✓ Design pattern audit (RAII, DIP, strategy, 0 bare catch)
  ✓ CPU-only builds: 43/43 tests passing
```

## 1) Grade Scorecard

| Dimension | Current | Evidence in code today | Blocker to next grade |
|---|---|---|---|
| Throughput | B- | Native CUDA exceeds llama.cpp on single-sequence (~1.1x); MMQ accumulate kernels landed for M=9-64 (0ccbad3), CUDA graphs re-enabled on primary forward. March 31 baseline: c=1 65.6, c=4 148.3, c=8 174.6 tok/s | Residual c=8 instability (~75% pass rate) prevents a clean concurrent win claim |
| Continuous batching | B- | Granular scheduler locks with fairness, prefix-affinity scoring, decode-worker pools, disaggregated KV channel; lane overlap race fixes (lane_overlap_mutex_) improved concurrent stability | Residual c=8 instability still not root-caused; sustained c>=8 wins require reliable pass rates |
| Capability identity | A- | Provider/fallback identity is explicit across API, admin, CLI, and metrics | Some advanced behavior still depends on compatibility fallback |
| Resource efficiency | B- | Memory-first GGUF direction, KV planner with multi-tier cache (GPU→host→disk), radix prefix cache, quantized execution are real | Native decode still spends too much work in its current down-proj kernels |
| CI and release enforcement | B- | 827 unit + 137 integration tests, docs contract gate, SBOM generation | Required GPU/provider lane is still not a release blocker |
| Distributed runtime | C+ | KV channel and SHM transport are production-tested, disaggregated health probes with timeout tracking, transport-aware readiness | Sequence ownership cleanup and worker-loss handling still need hardening |
| OSS release readiness | B | Canonical docs, release process, SBOM, CI contract gates, and conventional OSS metadata (LICENSE, CONTRIBUTING, SECURITY, CODE_OF_CONDUCT) are in place | Release surface still needs tighter benchmark/doc hygiene and stronger GPU validation |

## 2) Roadmap Priorities

| Priority | Workstream | Exit criteria |
|---|---|---|
| P0 | Residual c=8 concurrent instability | c=8 throughput gate passes reliably (>95%); clean runs already achieve 174.6 tok/s (32/32) but overall pass rate is ~75% |
| P0 | Required GPU/provider release lane | Native/provider/runtime checks become mandatory for release confidence |
| P1 | Structured-output native ownership | Grammar-constrained generation no longer relies on compatibility fallback for the CUDA path |
| P1 | Distributed ownership maturity | Cleanup and worker-loss behavior are deterministic and covered by tests |
| P2 | Benchmark and release hygiene | Release-facing benchmark narrative stays aligned with one maintained harness and current docs |

## 3) Quarter Targets

| Window | Target |
|---|---|
| Q2 2026 | Keep canonical docs, OSS metadata, and release process aligned with the actual codebase |
| Q3 2026 | Land a real native decode down-proj serving win and convert it into the default policy where appropriate |
| Q4 2026 | Make GPU/provider behavior part of required release gating and improve distributed ownership cleanup |

## 4) Grade Movement Rule

Grades move only when both are true:

1. A representative runtime path has evidence, not just a microbenchmark.
2. The supporting behavior is covered by tests, docs, or release gating as appropriate.

## 5) Immediate Engineering Plan

| Step | Why now |
|---|---|
| Diagnose and fix residual c=8 instability | MMQ accumulate and lane overlap mutex landed (0ccbad3) but ~75% pass rate at c=8 remains the top runtime risk |
| Investigate atomic decode_relay state (7561fc7) interaction with lane overlap | Concurrent state management is the likely area for remaining race conditions |
| Convert more GPU validation from ad hoc measurement into repeatable gates | Prevent regression churn during continued kernel work |

## 6) References

- [TechDebt_and_Competitive_Roadmap](TechDebt_and_Competitive_Roadmap.md)
- [benchmarks](benchmarks.md)
- [COMPETITIVE_POSITIONING](COMPETITIVE_POSITIONING.md)
