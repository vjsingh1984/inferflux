# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

InferFlux is a C++17 inference server (vLLM-inspired) providing OpenAI-compatible REST/gRPC/WebSocket APIs. It supports CUDA, ROCm, Metal (MPS), and CPU backends via a unified device abstraction. Models in GGUF and safetensors formats are loaded through an integrated llama.cpp backend. The project also ships `inferctl`, a CLI client with interactive chat, streaming, and admin commands.

## Build & Run

```bash
# Full build (Release, auto-detects cores)
./scripts/build.sh

# Fast incremental build
cmake -S . -B build && cmake --build build -j

# Start dev server (auto-builds if needed, logs to logs/server.log)
./scripts/run_dev.sh --config config/server.yaml

# Format before committing
clang-format -i server/**/*.cpp runtime/**/*.h
```

## Testing

```bash
# Run all unit tests (44 Catch2 test cases)
ctest --test-dir build --output-on-failure

# Run tests by tag (e.g., only auth tests)
./build/inferflux_tests "[auth]"

# List all available test cases
./build/inferflux_tests --list-tests

# Integration tests (SSE streaming, guardrails, metrics)
# Requires: INFERFLUX_MODEL_PATH, INFERCTL_API_KEY, optional INFERFLUX_PORT_OVERRIDE
ctest -R IntegrationSSE --output-on-failure
```

Test files are in `tests/unit/` (one per module). Framework: Catch2 v3.7.1 amalgamated at `external/catch2/`.

## Architecture

The CMake target `inferflux_core` links all modules into a single library consumed by both `inferfluxd` (server) and `inferctl` (CLI).

**Request flow:** Client → `HttpServer` (multi-threaded, server/http/) → auth middleware (API-key SHA-256/OIDC RS256/rate-limiting in server/auth/) → guardrail enforcement (server/policy/) → `Scheduler` (scheduler/) → `BackendManager` (runtime/backends/) → llama.cpp backend. Responses stream back as SSE when `stream: true`.

**Plugin interfaces** (pure-virtual C++ classes at key boundaries):
- `PolicyBackend` (`policy/policy_backend.h`) — policy storage/enforcement. Implemented by `PolicyStore` (encrypted INI). HttpServer depends on the interface, not the concrete store.
- `ModelRouter` (`scheduler/model_router.h`) — multi-model serving (list, load, unload, resolve). Interface only — wire-up pending.
- `DeviceContext` (`runtime/device_context.h`) — hardware abstraction. Implemented by `CPUDeviceContext`.
- `RequestBatch` (`scheduler/request_batch.h`) — per-request state and batch grouping for continuous batching. Interface only.

**Key modules:**
- `runtime/` — Device abstraction (`DeviceContext`), paged KV cache with LRU/Clock eviction, speculative decoding (draft + validator), NVMe offload via async file writer (io/)
- `model/` — GGUF loader (via llama.cpp submodule in external/), tokenizer
- `scheduler/` — Scheduler with global mutex (to be replaced by continuous batching via `RequestBatch`), `ModelRouter` interface
- `server/` — Multi-threaded HTTP server (thread pool), auth (API-key, OIDC, rate limiter), metrics (Prometheus /metrics), audit logging, guardrails, health probes (/healthz, /readyz, /livez)
- `policy/` — `PolicyBackend` interface, `PolicyStore` (encrypted INI with AES-GCM via OpenSSL), OPA client
- `cli/` — `inferctl` client (chat, completion, admin commands) using shared `HttpClient` and nlohmann/json
- `net/` — Shared `HttpClient` (Get/Post/Put/Delete/SendRaw)
- `config/` — `server.yaml` (primary config), `policy_store.conf` (encrypted policy persistence)

**External dependencies:** llama.cpp (git submodule at external/llama.cpp, pinned — treat as vendor code), OpenSSL (AES-GCM for policy encryption, SHA-256 for API key hashing, RS256 for JWT verification), nlohmann/json v3.11.3 (single-header at external/nlohmann/json.hpp), Catch2 v3.7.1 (amalgamated at external/catch2/).

**Tech debt tracker:** `docs/TechDebt_and_Competitive_Roadmap.md` — consult at session start for priorities.

## Coding Conventions

- **C++17**, all symbols in `inferflux` namespace, helpers in anonymous namespaces
- snake_case files, PascalCase public types (`ApiKeyAuth`, `PagedKvCache`), member fields end with `_`
- RAII resource management — no naked `new`/`delete`; use `std::unique_ptr`/`std::shared_ptr`
- Headers live beside their `.cpp` files; sorted includes, local before system
- 2-space indent (clang-format enforced)

## Configuration

All config knobs live in `config/server.yaml` and can be overridden with `INFERFLUX_*` environment variables. Key env vars for development:
- `INFERFLUX_MODEL_PATH` — path to GGUF model file
- `INFERCTL_API_KEY` — API key matching server config (default dev key: `dev-key-123`)
- `INFERFLUX_POLICY_PASSPHRASE` — enables AES-GCM encryption on the policy store
- `INFERFLUX_MPS_LAYERS` — number of layers to offload to Metal
- `INFERFLUX_PORT_OVERRIDE` / `INFERFLUX_HOST_OVERRIDE` — network overrides

## Commits & PRs

Short imperative subjects under ~72 chars mentioning scope (e.g., `Wire speculative validation and async NVMe writes`). PR bodies should link the tracking issue, enumerate config/env changes, and paste ctest output. Update README.md, docs/, and Helm/Docker assets alongside code changes.
