# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

InferFlux is a C++17 inference server delivering OpenAI-compatible REST/gRPC/WebSocket APIs across CUDA, ROCm, Metal (MPS), and CPU backends via a unified device abstraction. Models in GGUF and safetensors formats are loaded through an integrated llama.cpp backend. The project also ships `inferctl`, a CLI client with interactive chat, streaming, and admin commands.

## Build & Run

```bash
# First-time setup: init llama.cpp submodule
git submodule update --init --recursive

# Full build (Release, auto-detects cores)
./scripts/build.sh

# Fast incremental build
cmake -S . -B build && cmake --build build -j

# CPU-only build (no GPU SDK required, matches CI)
cmake -S . -B build -DENABLE_CUDA=OFF -DENABLE_ROCM=OFF -DENABLE_MPS=OFF -DENABLE_VULKAN=OFF && cmake --build build -j

# Start dev server (auto-builds if needed, logs to logs/server.log)
./scripts/run_dev.sh --config config/server.yaml

# Format before committing (CI checks these directories)
find server runtime scheduler model cli net io policy \
  \( -name '*.cpp' -o -name '*.h' \) ! -path '*/external/*' \
  | xargs clang-format -i
```

**Key CMake options:** `-DENABLE_CUDA=ON|OFF`, `-DENABLE_ROCM=ON|OFF`, `-DENABLE_MPS=ON|OFF`, `-DENABLE_VULKAN=ON|OFF`, `-DENABLE_MLX=OFF`, `-DENABLE_MTMD=OFF`, `-DENABLE_COVERAGE=ON` (Debug+gcov, adds `coverage` target)

**Build outputs:** `build/inferfluxd` (server), `build/inferctl` (CLI), `build/inferflux_tests` (test binary)

**Dependencies:** llama.cpp (git submodule at `external/llama.cpp` — **READONLY, never modify**), yaml-cpp (auto-fetched via CMake FetchContent), OpenSSL, nlohmann/json v3.11.3 (single-header at `external/nlohmann/json.hpp`), Catch2 v3.7.1 (amalgamated at `external/catch2/`)

**IMPORTANT:** `external/llama.cpp` is a readonly git submodule used for reference and build only. **Never edit, patch, or write files inside `external/llama.cpp/`**. If llama.cpp behavior needs to change, wrap or override it in InferFlux code instead. To update the submodule version, use `git submodule update` — do not commit changes inside the submodule directory.

## Testing

```bash
# Run all unit tests
ctest --test-dir build --output-on-failure

# Run a single test case by name
./build/inferflux_tests "test case name here"

# Run tests by tag (e.g., only auth tests)
./build/inferflux_tests "[auth]"

# Run tests by ctest label
ctest --test-dir build -R "(paged_kv|unified_batch)"

# List all available test cases
./build/inferflux_tests --list-tests

# Stub integration tests (no model required, always available)
ctest --test-dir build -R StubIntegration --output-on-failure

# Integration tests requiring a real model
# Requires: INFERFLUX_MODEL_PATH, INFERCTL_API_KEY, optional INFERFLUX_PORT_OVERRIDE
ctest --test-dir build -R IntegrationSSE --output-on-failure

# Coverage report (requires -DENABLE_COVERAGE=ON build)
cmake --build build-cov --target coverage
# Output: build-cov/coverage/html/index.html, build-cov/coverage/lcov.info

# Throughput gate (performance regression testing)
./scripts/run_throughput_gate.py --server-bin ./build/inferfluxd --config config/server.cuda.yaml \
  --model tinyllama --backend cuda --min-completion-tok-per-sec 120
```

Test files are in `tests/unit/` (one per module). Framework: Catch2 v3.7.1. Available ctest labels: `paged_kv`, `unified_batch`, `parallel`, `ep`, `backend_factory`, `backend_capabilities`, `moe`, `flash_attn`, `shm_transport`, `chat_template`, `sampling`, `logger`, `structured`, `model_registry`, `model_format`, `model_paths`, `model_identity`, `embeddings_routing`, `stop_sequences`, `fairness`.

Integration test suites (Python, require built `inferfluxd`): `StubIntegration`, `IntegrationCLIModelListContract`, `IntegrationEmbeddingsRoutingContract`, `IntegrationModelIdentityContract`, `SSECancel`, `ShmSmoke`, `SSEMetrics`, `BackendIdentityContract`, `BenchmarkHarnessDefaults`, `BenchmarkResponseClassifier`, `InferctlAdminPools`, `NativePhaseTiming`, `ThroughputGateContract`, `ThroughputGateFailureContract`, `IntegrationSSE` (needs model).

**First-token parity probe** (`tests/tools/first_token_probe.cpp`): builds `inferflux_first_token_probe` binary that runs a single forward pass and emits top-N logit distributions as JSON. Used by `tests/integration/first_token_parity_probe_test.py` to validate numeric parity across backends. Set `INFERFLUX_FIRST_TOKEN_PROBE_BIN` to override the binary path (defaults to `build/inferflux_first_token_probe`).

## Architecture

The CMake target `inferflux_core` links all modules into a single library consumed by both `inferfluxd` (server) and `inferctl` (CLI).

**Request flow:** Client → `HttpServer` (multi-threaded, server/http/) → auth middleware (API-key SHA-256/OIDC RS256/rate-limiting in server/auth/) → guardrail enforcement (server/policy/) → `Scheduler` (scheduler/) → `BackendManager` (runtime/backends/) → llama.cpp backend. Responses stream back as SSE when `stream: true`.

**Plugin interfaces** (pure-virtual C++ classes at key boundaries):
- `PolicyBackend` (`policy/policy_backend.h`) — policy storage/enforcement. Implemented by `PolicyStore` (encrypted INI). HttpServer depends on the interface, not the concrete store.
- `ModelRouter` (`scheduler/model_router.h`) — multi-model serving (list, load, unload, resolve). Implemented by `SingleModelRouter` with backend provider tracking (native vs universal), format routing (gguf/safetensors/hf), and capability-based fallback.
- `DeviceContext` (`runtime/device_context.h`) — hardware abstraction. Implemented by `CPUDeviceContext`.
- `RequestBatch` (`scheduler/request_batch.h`) — per-request state and batch grouping for continuous batching. Interface only.

**Key modules:**
- `runtime/` — Device abstraction (`DeviceContext`), paged KV cache with LRU/Clock eviction, speculative decoding (draft + validator), NVMe offload via async file writer (io/)
- `runtime/backends/` — Backend factory with native/universal provider paths, CUDA phase overlap and flash attention tuning, backend exposure policy with capability-based routing
- `model/` — GGUF loader (via llama.cpp submodule in external/), tokenizer, model format auto-detection (`model_format.cpp` supports gguf/safetensors/hf with HuggingFace URI resolution)
- `scheduler/` — Scheduler with global mutex (to be replaced by continuous batching via `RequestBatch`), `ModelRouter` with multi-model serving and backend provider tracking
- `server/` — Multi-threaded HTTP server (thread pool), auth (API-key, OIDC, rate limiter), metrics (Prometheus /metrics), audit logging, guardrails, health probes (/healthz, /readyz, /livez)
- `policy/` — `PolicyBackend` interface, `PolicyStore` (encrypted INI with AES-GCM via OpenSSL), OPA client
- `cli/` — `inferctl` client (chat, completion, admin commands) using shared `HttpClient` and nlohmann/json
- `net/` — Shared `HttpClient` (Get/Post/Put/Delete/SendRaw)
- `config/` — `server.yaml` (primary config), `policy_store.conf` (encrypted policy persistence)

**Tech debt tracker:** `docs/TechDebt_and_Competitive_Roadmap.md` — consult at session start for priorities.

**Canonical docs (keep in sync with code changes):**
- `docs/GEMV_KERNEL_ARCHITECTURE.md` — kernel geometry, dispatch priority, TDD coverage
- `docs/GGUF_NATIVE_KERNEL_IMPLEMENTATION.md` — native GGUF runtime guide, operator status
- `docs/MONITORING.md` — observability signals, tuning levers, profiling workflow
- `docs/design/NATIVE_GGUF_QUANTIZED_RUNTIME_ARCHITECTURE.md` — design rules and next gates
- `docs/API_SURFACE.md` — all HTTP endpoints and CLI contracts (source-aligned)
- `docs/CONFIG_REFERENCE.md` — full config map including all env vars and YAML knobs
- `docs/BACKEND_DEVELOPMENT.md` — guide to adding or extending backends

## Coding Conventions

- **C++17**, all symbols in `inferflux` namespace, helpers in anonymous namespaces
- snake_case files, PascalCase public types (`ApiKeyAuth`, `PagedKvCache`), member fields end with `_`
- RAII resource management — no naked `new`/`delete`; use `std::unique_ptr`/`std::shared_ptr`
- Headers live beside their `.cpp` files; sorted includes, local before system
- 2-space indent (clang-format enforced)

## Backend Selection & Model Format Routing

**Backend types:** `cpu`, `cuda`, `inferflux_cuda` (InferFlux CUDA implementation), `llama_cpp_cuda` (llama.cpp-backed), `mps`, `rocm`

**Model formats:** `auto` (default), `gguf`, `safetensors`, `hf` (HuggingFace URI-style `hf://org/repo`)

**Backend resolution logic:**
1. Explicit backend hints (`inferflux_cuda`, `llama_cpp_cuda`) are honored when available
2. Backend priority (`runtime.backend_priority`, `INFERFLUX_BACKEND_PRIORITY`) determines fallback order
3. Capability routing (`runtime.capability_routing.*`) enables graceful degradation when requested capabilities aren't available
4. Backend exposure policy controls which backends are exposed via `/v1/models`

**Model format resolution:**
1. HuggingFace URIs (`hf://org/repo`) resolve to `${INFERFLUX_HOME:-$HOME/.inferflux}/models/org/repo`
2. Auto-detection from file extension (`.gguf`, `.safetensors`)
3. GGUF sidecar fallback for non-GGUF formats when llama.cpp backends are used
4. Format-specific load path resolution for MLX vs llama.cpp backends

## Configuration

All config knobs live in `config/server.yaml` and can be overridden with `INFERFLUX_*` environment variables. Key env vars for development:
- `INFERFLUX_MODEL_PATH` — path to GGUF model file
- `INFERFLUX_MODELS` — multi-model configuration string (`id=model1,path=/path/to/model.gguf,format=gguf,backend=cuda,default=true`)
- `INFERCTL_API_KEY` — API key matching server config (default dev key: `dev-key-123`)
- `INFERFLUX_POLICY_PASSPHRASE` — enables AES-GCM encryption on the policy store
- `INFERFLUX_MPS_LAYERS` — number of layers to offload to Metal
- `INFERFLUX_PORT_OVERRIDE` / `INFERFLUX_HOST_OVERRIDE` — network overrides
- `INFERFLUX_BACKEND_PREFER_INFERFLUX` — prefer InferFlux implementations over llama.cpp
- `INFERFLUX_BACKEND_ALLOW_LLAMA_FALLBACK` — allow fallback to llama.cpp backends
- `INFERFLUX_CUDA_STRICT` — fail load if native CUDA runtime reports fallback
- `INFERFLUX_MODEL_FORMAT` — override model format detection

## CUDA Development

**Two-backend architecture:** The CUDA path has two providers that both accept GGUF models:
- `inferflux_cuda` (`runtime/backends/cuda/inferflux_cuda_backend.cpp`, `inferflux_cuda_executor.cpp`) — first-party CUDA kernels, no llama.cpp dependency at inference time. Owns logprobs, embeddings, batched decode, 50+ fused GEMV kernels (v1 column-major + v2 cooperative-warp). Still trails llama.cpp on single-sequence throughput (~0.35-0.76x).
- `llama_cpp_cuda` — delegates to llama.cpp for inference. Higher throughput today, lower ceiling for InferFlux-specific innovation.

Only structured output (grammar-constrained generation) still delegates to the llama.cpp parity backend. Logprobs and embeddings are native.

**Key CUDA env vars:** (centralized in `NativeExecutionPolicy::FromEnv()`)
- `INFERFLUX_DISABLE_BATCHED_DECODE=1` — opt out of batched decode (default-on)
- `INFERFLUX_DISABLE_Q8_1_ACTIVATIONS=1` — disable pre-quantized Q8_1 activation path
- `INFERFLUX_CUDA_TIMING_SAMPLE_RATE=N` — record CUDA event timing every Nth batch (0=off)
- `INFERFLUX_CUDA_PHASE_OVERLAP` — enable prefill/decode lane overlap
- `INFERFLUX_CUDA_ATTENTION_KERNEL` — force attention kernel (`auto`, `fa2`, `standard`)
- `INFERFLUX_CUDA_KV_MAX_BATCH` / `INFERFLUX_CUDA_KV_MAX_SEQ` — KV cache sizing

**InferFlux CUDA kernel files:**
- `runtime/backends/cuda/native/kernels/fused_dequant_gemv.cuh` — V1 GEMV kernels (column-major, 8 warps/block, used for pair/triple M>8 fallback)
- `runtime/backends/cuda/native/kernels/mmvq.cuh` — MMVQ weight-read-first kernels (batch 1-8, primary dispatch path)
- `runtime/backends/cuda/native/kernels/mmq.cuh` — MMQ tiled quantized GEMM kernels (batch 9-64)
- `runtime/backends/cuda/native/kernels/quant_common.cuh` — shared quantization primitives (Dp4aS8, Vsubss4, LoadPacked*)
- `runtime/backends/cuda/native/fused_quant_gemm.cu` — dispatch tables, MMVQ/MMQ selection, threshold logic
- `runtime/backends/cuda/native/transformer_forward.cu` — forward pass wiring
- `runtime/backends/cuda/native/cuda_kernels.cu` — batched RoPE/KvAppend, MeanPool, utility kernels
- `runtime/backends/cuda/kernels/flash_attention.cu` — FlashAttention-2 and FlashDecodeMultiSeq
- `runtime/backends/cuda/kernels/flash_attention_mma.cuh` — MMA tensor-core prefill kernel (m16n8k16, Br=16, auto-selected for query_len≥16)

**InferFlux CUDA policy and execution files:**
- `runtime/backends/cuda/native/native_execution_policy.h` — `NativeExecutionPolicy` struct, env var parsing
- `runtime/backends/cuda/native/native_dispatch_policy.{h,cpp}` — operator selection, dispatch decisions
- `runtime/backends/cuda/native/native_dispatch_registry.{h,cpp}` — per-phase/batch-bucket dispatch winner registry
- `runtime/backends/cuda/native/native_bootstrap_config.{h,cpp}` — KV cache sizing, startup config
- `runtime/backends/cuda/native/native_linear_executor.h` — projection stage execution with fallback chains
- `runtime/backends/cuda/native/cuda_sync_trace.h` — CUDA sync latency tracing at named pipeline sites

**InferFlux CUDA metrics:** Prometheus at `/metrics`: `inferflux_cuda_forward_passes_total{phase}`, `inferflux_cuda_forward_batch_tokens_total`, `inferflux_cuda_forward_duration_ms`, `inferflux_cuda_sampling_duration_ms`, `inferflux_cuda_kv_active_sequences`, `inferflux_cuda_kv_max_sequences`, FFN/down-proj operator counters. NVTX annotations for Nsight Systems profiling.

**Throughput validation:**
```bash
# Performance regression gate (canonical entry point)
bash scripts/benchmark.sh throughput-gate

# Native vs llama.cpp comparison benchmark
bash scripts/benchmark.sh gguf-compare

# Multi-backend benchmark (inferflux_cuda, llama_cpp_cuda, ollama, vllm, sglang, lmstudio)
bash scripts/benchmark.sh multi-backend
# AUTOSTART_VLLM=true / AUTOSTART_SGLANG=true to launch external engines automatically
# INFERFLUX_BENCH_SINGLE_BACKEND=<id> to run one backend through the full harness

# Profiling entry points
bash scripts/profile.sh backend           # nsys backend profile
bash scripts/profile.sh backend-ncu       # ncu kernel profile
bash scripts/profile.sh phase-timing      # per-phase timing breakdown

# Smoke tests (no model required)
bash scripts/smoke.sh gguf-native
bash scripts/smoke.sh backend-identity
```

**Script archive policy:** One-off probes and superseded wrappers live in `scripts/archive/`. New scripts should extend the entry points above, not add new top-level files.

## Disaggregated Runtime

`runtime/disaggregated/` implements split prefill/decode with KV transfer:
- `kv_channel.h` — ticket-based KV handoff with lifecycle tracking (create/transfer/consume/timeout)
- `shm_kv_transport.h` — shared-memory transport for process-local KV transfer
- Health signals: timeout streak/debt metrics influence `/readyz` and optional fail-closed admission

## Commits & PRs

Short imperative subjects under ~72 chars mentioning scope (e.g., `Wire speculative validation and async NVMe writes`). PR bodies should link the tracking issue, enumerate config/env changes, and paste ctest output. Update README.md, docs/, and Helm/Docker assets alongside code changes.
