# Backend Parity: llama.cpp, CUDA, and MLX

## Goal
Keep backend behavior consistent across hardware targets while preserving
target-specific optimization paths and minimizing duplicated control logic.

## Current Parity Snapshot (March 2026)

| Area | llama.cpp CPU/MPS path | CUDA path | MLX path |
| --- | --- | --- | --- |
| Model load lifecycle | Shared in `LlamaCPUBackend` | Uses same lifecycle with CUDA-tuned config | Separate MLX-native loader/engine |
| Phased prefill/decode APIs | Supported | Supported (inherits llama.cpp path) | Supported (native overrides) |
| Unified batch step API | Supported | Supported (inherits llama.cpp path) | Supported (native overrides) |
| Backend selection | Factory + router | Factory + router | Factory + router |
| Config tuning policy | Target traits shard | Target traits shard | Selected via factory (recorded follow-up for trait adapter) |

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

## Implemented This Session (llama.cpp + CUDA)

1. Added `runtime/backends/llama/llama_backend_traits.{h,cpp}`:
   - canonical backend target parsing,
   - per-target traits (`gpu_accelerated`, flash-attn capability),
   - centralized config tuning (`TuneLlamaBackendConfig`).
2. Refactored `BackendFactory` to use the traits shard:
   - selection result now carries target + traits metadata,
   - merged config tuning is centralized and consistent,
   - backend exposure policy is explicit: native-preferred with universal
     llama fallback (`INFERFLUX_BACKEND_PREFER_NATIVE`,
     `INFERFLUX_BACKEND_ALLOW_LLAMA_FALLBACK`).
3. Kept CUDA backend as a thin specialization over llama.cpp runtime:
   - CUDA-specific load tuning remains backend-local,
   - scheduler/executor code remains backend-agnostic.
4. Added backend exposure provenance + deterministic priority chains:
   - `BackendFactory::NormalizeHintList(...)` centralizes backend priority
     normalization.
   - `SingleModelRouter` now resolves backend candidates from a priority chain
     (`runtime.backend_priority` / `INFERFLUX_BACKEND_PRIORITY`) and records
     per-model provenance (`requested_backend`, `backend_provider`,
     `backend_fallback`, `backend_fallback_reason`).
   - `/v1/models`, `/v1/models/{id}`, and `/v1/admin/models` now expose
     `backend_exposure`, and Prometheus exports
     `inferflux_backend_exposures_total`.
5. Added request-time capability fallback routing (default route only):
   - When the default model cannot satisfy request features (e.g., logprobs),
     scheduler can reroute to another loaded backend with the same model path.
   - Explicit `model` requests do not auto-switch (guardrail against surprise
     model changes).
   - Selection logic is centralized in `scheduler/model_selection.*` and reused
     by both scheduler and HTTP preflight checks to avoid control-plane drift.
   - Prometheus exports `inferflux_capability_route_fallbacks_total`.

## Recorded Follow-ups (MLX, no code change in this session)

1. Add a lightweight MLX trait adapter so MLX capabilities are reported through
   the same trait model used by llama.cpp/CUDA selection.
2. Add MLX throughput baselines using the same per-model token-rate metrics
   (`rate(inferflux_model_*_tokens_total[...])`) used for CUDA/llama.cpp.
3. Align MLX and llama.cpp sampler option coverage in a shared conformance
   checklist (temperature/top-p/top-k/min-p/penalties + stop behavior).

## Next Implementation Steps (CUDA-focused)

1. Stream-aware prefill/decode overlap in executor/scheduler.
2. GPU KV residency and reuse policy to remove host-side bottlenecks.
3. CI throughput guardrail target (tokens/sec floor) on CUDA runner hardware.
