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

**Dependencies:** llama.cpp (git submodule at `external/llama.cpp` — treat as vendor code), yaml-cpp (auto-fetched via CMake FetchContent), OpenSSL, nlohmann/json v3.11.3 (single-header at `external/nlohmann/json.hpp`), Catch2 v3.7.1 (amalgamated at `external/catch2/`)

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

Integration test suites (Python, require built `inferfluxd`): `StubIntegration`, `IntegrationCLIModelListContract`, `IntegrationEmbeddingsRoutingContract`, `IntegrationModelIdentityContract`, `SSECancel`, `ShmSmoke`, `IntegrationSSE` (needs model).

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

## Coding Conventions

- **C++17**, all symbols in `inferflux` namespace, helpers in anonymous namespaces
- snake_case files, PascalCase public types (`ApiKeyAuth`, `PagedKvCache`), member fields end with `_`
- RAII resource management — no naked `new`/`delete`; use `std::unique_ptr`/`std::shared_ptr`
- Headers live beside their `.cpp` files; sorted includes, local before system
- 2-space indent (clang-format enforced)

## Backend Selection & Model Format Routing

**Backend types:** `cpu`, `cuda`, `cuda_native` (native CUDA implementation), `cuda_llama_cpp`/`cuda_llama` (llama.cpp-backed), `mps`, `rocm`

**Model formats:** `auto` (default), `gguf`, `safetensors`, `hf` (HuggingFace URI-style `hf://org/repo`)

**Backend resolution logic:**
1. Explicit backend hints (`cuda_native`, `cuda_llama_cpp`) are honored when available
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
- `INFERFLUX_BACKEND_PREFER_NATIVE` — prefer native implementations over universal
- `INFERFLUX_BACKEND_ALLOW_LLAMA_FALLBACK` — allow fallback to llama.cpp backends
- `INFERFLUX_NATIVE_CUDA_STRICT` — fail load if native CUDA runtime reports fallback
- `INFERFLUX_MODEL_FORMAT` — override model format detection

## CUDA Development

**Native CUDA backend:** `runtime/backends/cuda/native_cuda_backend.cpp` and `runtime/backends/cuda/native_cuda_runtime.cpp` provide the canonical native CUDA path used by `cuda_native`.

**Phase overlap:** Mixed-batch decode prioritization with configurable prefill/decode overlap. Enable via `runtime.cuda.phase_overlap.enabled` or `INFERFLUX_CUDA_PHASE_OVERLAP`. Monitor via Prometheus metrics: `inferflux_cuda_lane_submissions_total`, `inferflux_cuda_lane_completions_total`, `inferflux_cuda_lane_overlap_events_total`, `inferflux_cuda_lane_overlap_duration_ms_total`.

**Attention kernels:** Multiple CUDA attention implementations with automatic selection. Configure via `runtime.cuda.attention.kernel` or `INFERFLUX_CUDA_ATTENTION_KERNEL` (`auto`, `fa2`, `standard`). Monitor fallbacks via `inferflux_cuda_attention_kernel_fallbacks_total`.

**Native CUDA metrics:** The native kernel executor (`runtime/backends/cuda/native_kernel_executor.cpp`) reports per-forward-pass timing and batching metrics via Prometheus: `inferflux_native_forward_passes_total{phase}`, `inferflux_native_forward_batch_tokens_total`, `inferflux_native_forward_duration_ms` (histogram), `inferflux_native_sampling_duration_ms` (histogram), `inferflux_native_kv_active_sequences`, `inferflux_native_kv_max_sequences`. NVTX annotations are added for Nsight Systems profiling (ranges: Forward, Embedding, Layer, QKV_Projection, RoPE, KV_Append, FlashAttention2, O_Projection, FFN, LM_Head, Sampling).

**Throughput validation:** Use `scripts/run_throughput_gate.py` for performance regression testing. The script validates tok/s thresholds, CUDA lane submissions, overlap metrics, native forward pass counters (`--require-native-forward-passes`), and backend provider exposure.

## Commits & PRs

Short imperative subjects under ~72 chars mentioning scope (e.g., `Wire speculative validation and async NVMe writes`). PR bodies should link the tracking issue, enumerate config/env changes, and paste ctest output. Update README.md, docs/, and Helm/Docker assets alongside code changes.
