# Archive Index (Reference Evidence)

**Status:** Active (reference-only)

```mermaid
flowchart LR
    A[Canonical Docs] --> B[Release users]
    C[Archive Evidence] --> D[Deep-dive and historical context]
```

## Archive Policy

| Label | Meaning |
|---|---|
| `Reference-Evidence` | historical benchmark/debug/design context |
| `Canonical-Elsewhere` | source of truth moved to canonical docs |

## 1) Performance and Benchmark Evidence

| Document | Label | Notes |
|---|---|---|
| [NATIVE_CUDA_BENCHMARK_GUIDE](archive/evidence/NATIVE_CUDA_BENCHMARK_GUIDE.md) | Reference-Evidence | benchmark procedure |
| [NATIVE_vs_LLAMA_CPP_CUDA_BENCHMARK](archive/evidence/NATIVE_vs_LLAMA_CPP_CUDA_BENCHMARK.md) | Reference-Evidence | native vs universal snapshot |
| [COMPLETE_STRESS_TEST_REPORT_2026_03_04](archive/evidence/COMPLETE_STRESS_TEST_REPORT_2026_03_04.md) | Reference-Evidence | stress/debug summary |
| [GGUF_COMPLETE_IMPLEMENTATION_2026_03_04](archive/evidence/GGUF_COMPLETE_IMPLEMENTATION_2026_03_04.md) | Reference-Evidence | GGUF milestone snapshot |
| [NSIGHT_QWEN3_VS_TINYLLAMA_COMPARISON](archive/evidence/NSIGHT_QWEN3_VS_TINYLLAMA_COMPARISON.md) | Reference-Evidence | Nsight comparison |
| [GPU_OPTIMIZATION_FINDINGS_2026_03_03](archive/evidence/GPU_OPTIMIZATION_FINDINGS_2026_03_03.md) | Reference-Evidence | optimization findings |
| [ROCM_BACKEND_IMPLEMENTATION_PLAN](archive/evidence/ROCM_BACKEND_IMPLEMENTATION_PLAN.md) | Reference-Evidence | ROCm planning context |

## 2) Consolidated Product Narrative Snapshots

| Document | Label | Canonical replacement |
|---|---|---|
| [VISION_2026_03_05](archive/evidence/VISION_2026_03_05.md) | Canonical-Elsewhere | [PRD](PRD.md), [Roadmap](Roadmap.md), [TechDebt](TechDebt_and_Competitive_Roadmap.md) |
| [COMPETITIVE_POSITIONING_2026_03_05](archive/evidence/COMPETITIVE_POSITIONING_2026_03_05.md) | Canonical-Elsewhere | [PRD](PRD.md), [TechDebt](TechDebt_and_Competitive_Roadmap.md) |
| [NFR_2026_03_05](archive/evidence/NFR_2026_03_05.md) | Canonical-Elsewhere | [PRD](PRD.md), [Roadmap](Roadmap.md) |

## 3) Consolidated Operations and Tuning Deep-Dives

| Document | Label | Canonical replacement |
|---|---|---|
| [PERFORMANCE_TUNING_2026_03_05](archive/evidence/PERFORMANCE_TUNING_2026_03_05.md) | Canonical-Elsewhere | [MONITORING](MONITORING.md), [CONFIG_REFERENCE](CONFIG_REFERENCE.md) |
| [PROFILING_OPERATIONS_GUIDE_2026_03_05](archive/evidence/PROFILING_OPERATIONS_GUIDE_2026_03_05.md) | Canonical-Elsewhere | [MONITORING](MONITORING.md), [Developer Guide](DeveloperGuide.md) |
| [INFERCTL_SERVER_MANAGEMENT_2026_03_05](archive/evidence/INFERCTL_SERVER_MANAGEMENT_2026_03_05.md) | Canonical-Elsewhere | [AdminGuide](AdminGuide.md) |

## 4) Consolidated GGUF Deep-Dives

| Document | Label | Canonical replacement |
|---|---|---|
| [GGUF_NATIVE_KERNEL_IMPLEMENTATION_2026_03_05](archive/evidence/GGUF_NATIVE_KERNEL_IMPLEMENTATION_2026_03_05.md) | Canonical-Elsewhere | [GGUF_NATIVE_KERNEL_IMPLEMENTATION](GGUF_NATIVE_KERNEL_IMPLEMENTATION.md) |
| [GGUF_QUANTIZATION_REFERENCE_2026_03_05](archive/evidence/GGUF_QUANTIZATION_REFERENCE_2026_03_05.md) | Canonical-Elsewhere | [GGUF_NATIVE_KERNEL_IMPLEMENTATION](GGUF_NATIVE_KERNEL_IMPLEMENTATION.md), [GGUF_SMOKE_TEST_GUIDE](GGUF_SMOKE_TEST_GUIDE.md) |
| [GGUF_SMOKE_TEST_GUIDE_2026_03_05](archive/evidence/GGUF_SMOKE_TEST_GUIDE_2026_03_05.md) | Canonical-Elsewhere | [GGUF_SMOKE_TEST_GUIDE](GGUF_SMOKE_TEST_GUIDE.md) |

## 5) Consolidated Startup Advisor Deep-Dives

| Document | Label | Canonical replacement |
|---|---|---|
| [DYNAMIC_SLOT_ALLOCATION_STARTUP_ADVISOR_2026_03_05](archive/evidence/DYNAMIC_SLOT_ALLOCATION_STARTUP_ADVISOR_2026_03_05.md) | Canonical-Elsewhere | [STARTUP_ADVISOR](STARTUP_ADVISOR.md), [CONFIG_REFERENCE](CONFIG_REFERENCE.md) |
| [STARTUP_ADVISOR_DYNAMIC_SLOTS_SUMMARY_2026_03_05](archive/evidence/STARTUP_ADVISOR_DYNAMIC_SLOTS_SUMMARY_2026_03_05.md) | Canonical-Elsewhere | [STARTUP_ADVISOR](STARTUP_ADVISOR.md), [CONFIG_REFERENCE](CONFIG_REFERENCE.md) |
| [STARTUP_ADVISOR_CONFIGURABLE_CONSTANTS_2026_03_05](archive/evidence/STARTUP_ADVISOR_CONFIGURABLE_CONSTANTS_2026_03_05.md) | Canonical-Elsewhere | [STARTUP_ADVISOR](STARTUP_ADVISOR.md), [CONFIG_REFERENCE](CONFIG_REFERENCE.md) |
| [LARGE_CONTEXT_CONFIGURATION_GUIDE_2026_03_05](archive/evidence/LARGE_CONTEXT_CONFIGURATION_GUIDE_2026_03_05.md) | Canonical-Elsewhere | [CONFIG_REFERENCE](CONFIG_REFERENCE.md), [STARTUP_ADVISOR](STARTUP_ADVISOR.md) |

## Canonical Sources of Truth

- [README](../README.md)
- [INDEX](INDEX.md)
- [Quickstart](Quickstart.md)
- [API Surface](API_SURFACE.md)
- [Architecture](Architecture.md)
- [PRD](PRD.md)
- [Roadmap](Roadmap.md)
- [TechDebt and Competitive Roadmap](TechDebt_and_Competitive_Roadmap.md)
