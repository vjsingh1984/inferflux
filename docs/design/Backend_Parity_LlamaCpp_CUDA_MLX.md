# Backend Parity: llama.cpp, CUDA, and MLX

## Goal
Keep backend behavior consistent across hardware targets while preserving
target-specific optimization paths and minimizing duplicated control logic.

## Current Parity Snapshot (March 8, 2026)

| Area | llama.cpp provider path (`cuda_llama_cpp`) | native provider path (`native_cuda`) | MLX path |
| --- | --- | --- | --- |
| Model load lifecycle | Shared in `LlamaCPUBackend` | Native runtime via `NativeCudaRuntime` + strict-native policy hooks | Separate MLX-native loader/engine |
| Phased prefill/decode APIs | Supported | Supported (native unified batch + decode/prefill overlap path) | Supported (native overrides) |
| Unified batch async lanes | Supported | Contract intentionally disabled today (`SupportsAsyncUnifiedBatch()==false`) | Supported (native overrides) |
| GGUF quantized overlap safety | N/A (llama.cpp path) | Active lane-local quantized map/adapter ownership (no hard GGUF overlap disablement) | N/A |
| Capability surface | Broad compatibility baseline | Endpoint parity contracts are closed for completion/chat/embeddings through native parity delegate paths; speculative decoding contract is native-owned | Hardware-specific capabilities |
| Backend selection | Factory + router | Factory + router (native-preferred with explicit strict/fallback policy) | Factory + router |
| Primary role | Stability + model-format compatibility | Performance/control core with active kernel maturity roadmap | Apple Silicon performance path |

## Design Principles

1. Single control-plane path:
   - Scheduler, executor, and router should not branch on backend internals.
2. Sharded backend policy modules:
   - Backend selection and tuning live outside concrete backend implementations.
3. Hardware optimization stays local:
   - CUDA/MLX optimization hooks remain inside their backend classes.
4. Capability-first evolution:
   - New backend features are surfaced via traits/capabilities, not hard-coded
     checks in scheduler/executor.

## Current Value Split (Why Keep Two CUDA Backends)

| Backend | What it gives today | Why it stays |
| --- | --- | --- |
| `native_cuda` | InferFlux-owned runtime control, strict provider identity, active mixed decode/prefill overlap path, explicit endpoint contracts (completion/chat/embeddings) | Throughput and architecture headroom |
| `cuda_llama_cpp` | Stable compatibility baseline with mature feature surface and broad GGUF behavior | Production safety net and fallback path |

## Current Gaps to Close

1. Native async unified-batch contract parity:
   - `SupportsAsyncUnifiedBatch()` is still disabled in native runtime.
2. Native-first parity independence:
   - completion/chat/embeddings parity currently depends on delegate availability
     for some model-format layouts.
3. Native quantized heavy-batch maturity:
   - GGUF overlap safety is now in place, but fused-kernel + large-model perf
     proof is still pending.

## Next Steps

1. Re-enable native async unified-batch contract with latency/correctness gates.
2. Move parity surfaces from delegate-backed implementation to native-first
   implementation where practical.
3. Validate quantized GGUF throughput on larger batches/models and wire
   non-regression checks into GPU CI.
