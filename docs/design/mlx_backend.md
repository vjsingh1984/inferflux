# MLX Backend Design

## Goal
Introduce an MLX-backed inference path that plugs into the existing OpenAI-compatible HTTP surface and scheduling architecture. The new backend should align with how CUDA/ROCm/MPS are structured so future accelerators can reuse the same patterns.

## Guiding principles
- **Adapter pattern:** `BackendManager` remains the single entry point for device-specific backends. Each backend implements a shared interface (`Prefill`, `Decode`, sampler lifecycle, KV access).
- **Declarative configs:** `runtime.backends[]` (or future `models[*].backend`) selects the backend; no backend-specific code leaks into the HTTP/API layer.
- **Incremental CI:** add compile-check jobs similar to CUDA so we validate new backends without blocking existing pipelines.
- **Extensibility:** the design should make CUDA/ROCm parity trivial (i.e., `ENABLE_{CUDA,ROCM,MLX}` follow the same plumbing, configuration, and error-reporting patterns).

## Capabilities & API contract
### MLX expectations
- **Model format:** GGUF via llama.cpp conversion in the short term; direct MLX checkpoint consumption is optional once the backend matures.
- **Execution model:** MLX runs on top of Metal, so batching rules mirror MPS. We require:
  - batched matmul + attention kernels
  - tensor creation from raw buffers (for KV hydration/serialization)
  - sampler hooks (logits exposure per token)
- **KV support:** If MLX lacks llama.cpp-style KV copy APIs, InferFlux must expose a capability flag so the scheduler can disable warm prefix donation/rehydration for MLX models.
- **Feature flag contract:** Each backend advertises:
  - `supports_structured_output`
  - `supports_speculative_decoding`
  - `supports_kv_prefix_copy`
  - `supports_multimodal`

### Integration choice
- **Phase 1 (stub adapter):** Reuse llama.cpp’s CPU path while wiring `backend: mlx` through config/CLI/CI. This keeps the API stable, lets us verify packaging/CI plumbing, and provides a place to plug the real MLX device later.
- **Phase 2 (full MLX):** When the MLX SDK exposes the required primitives, either:
  - Upstream MLX support into llama.cpp (preferred, keeps one execution engine), or
  - Implement `MlxBackend` in InferFlux that mirrors the llama.cpp API (more control but more code).

## Task breakdown

### 1. Requirements & API contract
- Document MLX capabilities we rely on: supported GGUF (or conversion path), tensor dtypes, KV caching APIs, batching constraints.
- Decide whether to upstream MLX support into llama.cpp or ship a parallel backend in InferFlux (matrix of pros/cons).
- Write a spec for the backend interface extensions (e.g., new capability flags for features that MLX lacks).

### 2. Build system plumbing
1. Add `option(ENABLE_MLX "Enable MLX backend" OFF)` in `CMakeLists.txt`.
2. Vendor the upstream `mlx-c` repository under `external/mlx-c` (submodule) so developers have a default implementation.
3. CMake behavior:
   - If `MLX_C_ROOT`/`$ENV{MLX_C_ROOT}` is supplied, use `find_path`/`find_library` to detect the headers (`mlx/c/mlx.h`) and library (`libmlxc.dylib`/`libmlx.so`).
   - Otherwise, when the submodule exists and `INFERFLUX_EMBED_MLXC=ON`, call `add_subdirectory(external/mlx-c EXCLUDE_FROM_ALL)` and link the resulting `mlxc` target directly (note: the upstream `mlx-c` build will fetch `mlx` from GitHub unless `MLX_C_USE_SYSTEM_MLX` is toggled).
4. Only set `INFERFLUX_HAS_MLX` when the include dir + library have both been found/built; otherwise log a warning and keep the backend disabled.
5. Update CI/scripts to set `MLX_C_ROOT` for the MLX compile-check job once the dependency is available on the runner.

#### Tensor mapping (GGUF → MLX)
- GGUF floats map to `mlx_dtype_float32`, `mlx_dtype_float16`, `mlx_dtype_bfloat16` (see `mlx/c/array.h`).
- Integer embeddings / vocab IDs map to `mlx_dtype_int32` (GGUF uses 32-bit token IDs).
- KV cache tensors use `[n_layers, n_heads, seq, head_dim]` layout; they will map to MLX arrays with consistent strides so `mlx_memory_view` can expose them to Metal.
- Rotary embeddings and normalization weights remain f32/f16 tensors; we’ll convert GGUF buffers to MLX arrays via `mlx_array_from_buffer`.

### 3. Implementation plan (multi-stage)

#### Stage 0 — Baseline scaffolding (landing now)
1. **Backend plumbing** — `MlxBackend`, `MlxWeightLoader`, and `MlxExecutionEngine` exist (stubbed) so the router/config can target `backend: mlx`.
2. **Build detection** — `ENABLE_MLX`, `INFERFLUX_EMBED_MLXC`, and `MLX_C_ROOT` decide whether to link `mlx-c`. When headers/libs aren’t found we log a warning and keep `INFERFLUX_HAS_MLX=0`.
3. **Docs/CLI** — Quickstart/Troubleshooting/Installer docs enumerate the new backend; CLI accepts `--backend mlx`.

> ✅ This stage is already landed (placeholders + build detection).

#### Stage 1 — Model metadata ingestion
1. **Checkpoints** — Decide on the MLX-native format (likely the same `.npz`/`npy`/MLX serialization the Python API emits). We will not attempt GGUF conversion initially.
2. **Loader** — Extend `MlxWeightLoader` to parse the MLX checkpoint metadata and expose a list of tensors (name, dtype, shape, file offset). Use `mlx_array_from_buffer` (from `mlx/c/array.h`) to materialize MLX arrays directly from the file.
3. **Tokenizer** — Integrate the same tokenization we use today (SentencePiece via llama.cpp, or a native MLX tokenizer if available) and store vocabulary → token id tables for MLX models.

#### Stage 2 — Execution engine foundations
4. **Device/stream** — Wrap `mlx_device_t`, `mlx_stream_t`, and memory allocators so each backend instance owns its Metal context and command queues. Mirror the patterns shown in MLX’s “Using MLX in C++” docs.
5. **Graph builder** — Add helpers for composing MLX ops (RMSNorm, attention, MLP). Start with eager evaluation before moving to compiled graphs (`mlx_compile`).
6. **KV cache** — Define an MLX-native KV representation (likely as MLX arrays with `[layers, heads, seq, head_dim]` layout). Provide serialization/hydration stubs so the scheduler can manage sequence slots.

#### Stage 3 — Prefill & Decode kernels
7. **Prefill path** — Implement the transformer forward pass using MLX ops for the initial prompt: embedding lookup → attention → MLP. Update the KV cache after each layer.
8. **Decode path** — Implement the autoregressive loop: feed the previous token, run attention/MLP with the KV cache, compute logits, and return the sampled token/logprobs.
9. **Performance hooks** — Populate structured-output callbacks (grammar enforcement), speculative decoding flags, and logprob collection to match existing API contracts.

#### Stage 4 — Scheduler & API integration
10. **Capability flags** — Set `supports_structured_output`, `supports_speculative_decoding`, `supports_kv_prefix_copy` based on what Stage 3 implements. Update scheduler logic to honor them (e.g., skip prefix reuse if MLX can’t copy KV ranges).
11. **/v1/models metadata** — Report backend-specific details (device label, ready status).
12. **CI/testing** — Add integration tests that launch InferFlux with an MLX-native checkpoint, run `inferctl completion/chat`, and verify admin APIs.

### 4. Configuration, CLI, and docs
1. Extend `config/server.yaml` schema with `backend: mlx` and MLX-specific parameters (e.g., memory pool size).
2. Update CLI (`inferctl quickstart`) to accept `--backend mlx` when scaffolding configs.
3. Document installation steps (MLX prerequisites, environment variables) in `docs/Installer.md` and `docs/Quickstart.md`.
4. Update `docs/Architecture.md` to include the new backend in the device matrix.

### 5. Testing & CI
1. Add unit tests that instantiate `MlxBackend` (guarded by `INFERFLUX_HAS_MLX`).
2. Add a CI job (similar to `build-check-cuda`) that downloads MLX headers and runs a compile check.
3. Define manual validation steps (e.g., run TinyLlama with MLX backend on Apple Silicon) and document them in `Testing/README.md`.

## Tracking
- All MLX tasks are tracked in this document and referenced from `docs/TechDebt_and_Competitive_Roadmap.md` under the Ease-of-setup and Hardware breadth rows.
- Future backends (ROCm/CUDA) will follow the same structure: `ENABLE_<backend>` flag, backend class implementing the shared interface, and CI compile-check job.
