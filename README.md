# InferFlux

> **Deliver high-performance LLM serving with predictable operations, consistent APIs, and a single codebase spanning CPU, CUDA/ROCm, Metal, and MLX backends.**

| Guide | Audience | Focus | Location |
| --- | --- | --- | --- |
| 🧑‍💻 Developer Guide | contributors | build, debug, test | `docs/DeveloperGuide.md` |
| 🙋 User Guide | practitioners | run, query, configure | `docs/UserGuide.md` |
| 🛠️ Admin Guide | operators | deploy, secure, observe | `docs/AdminGuide.md` |
| 📐 Design / PRD | everyone | intent & roadmap | `docs/Architecture.md`, `docs/PRD.md` |

### System-at-a-glance
```
+----------+   HTTP/SSE   +------------------+   Device APIs   +------------+
| Clients  |─────────────▶|  InferFlux Core  |───────────────▶| Backends   |
| (SDK/CLI)|◀──stream─────|  Scheduler+HTTP  |◀──metrics──────| (CPU/MPS/…)|
+----------+              +------------------+                +------------+
         ▲                 │      │      │
         │                 │      │      └─ Guardrails / Auth / Registry
         └────── CLI (inferctl)   └─ Paged KV / Speculative / Prefix Cache
```

### Platform pillars
- **Any Runtime, One Binary** – CPU, CUDA/ROCm, Metal (MPS), and MLX share a unified backend interface with paged KV caches, speculative decoding, and streaming SSE.
- **Operational Guardrails** – OIDC/API-key auth, rate limiting, RBAC scopes, encrypted policy store, external OPA integrations, and audit logs.
- **Cloud-native Observability** – Prometheus metrics, tracing hooks, autoscaler hints, streaming counters, and admin APIs mirroring OpenAI semantics.
- **Developer Ergonomics** – `inferctl` for streaming/chat/admin, hot reloads, policy tooling, and clear contribution workflows.

---

## Quick Start Matrix

| Task | Command / File | Notes |
| --- | --- | --- |
| Build | `./scripts/build.sh` | Generates `build/` binaries |
| Run dev server | `./scripts/run_dev.sh --config config/server.yaml` | Uses stub model if none provided |
| Invoke CLI | `./build/inferctl completion --prompt "Hello"` | Set `INFERCTL_API_KEY` beforehand |
| Run tests | `ctest --test-dir build --output-on-failure` | Full suite; see Dev Guide for filters |

> 🔗 Dive deeper via the [Developer Guide](docs/DeveloperGuide.md) and [User Guide](docs/UserGuide.md).

---

## Feature Capsules

### Speculative decoding & metrics
- Toggle via `runtime.speculative_decoding.enabled` or `INFERFLUX_SPECULATIVE_ENABLED`.
- Control draft cadence with `INFERFLUX_SPEC_DRAFT_MODEL`, `INFERFLUX_SPEC_MAX_PREFILL`, `INFERFLUX_SPEC_CHUNK_SIZE`.
- Observe adoption with Prometheus counters: `inferflux_spec_chunks_total`, `inferflux_spec_chunks_accepted_total`, `inferflux_spec_tokens_reused_total`.

### NVMe paged KV offload
- Point `runtime.nvme_offload.path` (or `INFERFLUX_NVME_OFFLOAD_PATH`) to fast storage.
- Tune writers via `runtime.nvme_offload.workers` and `.queue_depth`.
- Host cache sizing: `runtime.paged_kv.cpu_pages`, `runtime.paged_kv.eviction` (override with `INFERFLUX_KV_CPU_PAGES`, `INFERFLUX_KV_EVICTION`).

### Guardrails + policy flow
- `guardrails.blocklist` for deny lists, `guardrails.opa_endpoint` (file/http) for contextual decisions.
- Policy storage: `config/policy_store.conf` (override `INFERFLUX_POLICY_STORE`), encrypt via `INFERFLUX_POLICY_PASSPHRASE`.
- Runtime routing policy updates from `/v1/admin/routing` are persisted in the policy store and restored on restart.
- Admin policy mutations fail closed with `{"error":"policy_persist_failed"}` when disk persistence fails.
- Admin CLI snippets:
  - `inferctl admin guardrails --set secret,pii --api-key $ADMIN_KEY`
  - `inferctl admin rate-limit --set 200 --api-key $ADMIN_KEY`
  - `inferctl admin routing --set --fallback-scope any_compatible --api-key $ADMIN_KEY`

### Scheduler throughput tuning
- Batch admission budgets are configurable via `runtime.scheduler.max_batch_size` and `runtime.scheduler.max_batch_tokens`.
- Env overrides: `INFERFLUX_SCHED_MAX_BATCH_SIZE`, `INFERFLUX_SCHED_MAX_BATCH_TOKENS`.
- Observe pressure with `inferflux_scheduler_batch_token_budget_skips_total` and configured limits via `inferflux_scheduler_batch_limit_*`.
- CUDA mixed-batch decode prioritization scaffold: `runtime.cuda.phase_overlap.enabled` and
  `runtime.cuda.phase_overlap.min_prefill_tokens` (env:
  `INFERFLUX_CUDA_PHASE_OVERLAP`, `INFERFLUX_CUDA_PHASE_OVERLAP_MIN_PREFILL_TOKENS`).
- Optional dual-context overlap mode: `runtime.cuda.phase_overlap.prefill_replica` (env:
  `INFERFLUX_CUDA_PHASE_OVERLAP_PREFILL_REPLICA`) to run prefill on a replica context and hand off KV to decode.
- CUDA async lane telemetry:
  `inferflux_cuda_lane_submissions_total`,
  `inferflux_cuda_lane_completions_total`,
  `inferflux_cuda_lane_queue_depth`,
  `inferflux_cuda_lane_overlap_events_total`,
  `inferflux_cuda_lane_overlap_duration_ms_total`,
  `inferflux_cuda_lane_inflight`.
- CUDA attention telemetry:
  `inferflux_cuda_attention_kernel_selected`,
  `inferflux_cuda_attention_kernel_fallbacks_total`,
  `inferflux_cuda_attention_kernel_switches_total`.

### Throughput guardrail (local + CI)
- Use `scripts/run_throughput_gate.py` to run a reproducible load test and fail when tok/s drops below your baseline.
- Local example:
  `./scripts/run_throughput_gate.py --server-bin ./build/inferfluxd --config config/server.cuda.yaml --model tinyllama --backend cuda --server-env "INFERFLUX_MODELS=id=tinyllama,path=/abs/path/model.gguf,format=gguf,backend=cuda,default=true" --min-completion-tok-per-sec 120 --require-metrics --require-cuda-lanes --require-cuda-overlap`
- Ada RTX 4000 shortcut profile:
  `./scripts/run_throughput_gate.py --gpu-profile ada_rtx_4000 --server-bin ./build/inferfluxd --config config/server.cuda.yaml --model tinyllama --server-env "INFERFLUX_MODELS=id=tinyllama,path=/abs/path/model.gguf,format=gguf,backend=cuda,default=true" --min-completion-tok-per-sec 120 --require-metrics`
- GPU profiles do not pin a single CUDA attention kernel by default (`auto` may resolve to `fa2` or `standard` depending on runner build/hardware). Add `--expect-cuda-attention-kernel ...` only on pinned fleets.
- Mixed-phase pressure flag: `--mixed-prompt-workload` sends alternating long/short prompts to exercise prefill+decode overlap in one run.
- Mixed-phase pressure budget: `--target-total-prefill-tokens` controls how much aggregate prefill prompt pressure mixed workload requests can generate (higher values increase overlap pressure at the cost of heavier batches).
- `--require-cuda-lanes` always requires prefill-lane submissions; decode-lane submissions are required when the run generated completion tokens (zero-token EOS runs are treated as valid decode short-circuits).
- Optional guardrail: add `--max-cuda-attention-fallbacks 0` when you expect no kernel fallback under a pinned attention mode.
- Optional overlap floor: add `--min-cuda-overlap-duration-ms <ms>` to require meaningful overlap windows (useful on Ada-class developer GPUs).
- Native-path guardrail: add `--require-backend-provider native` to fail fast when `/v1/models` reports universal llama fallback instead of native runtime.
- Fallback guardrail: add `--require-no-backend-fallback` to fail when backend exposure reports scaffold/delegate fallback.
- Native scaffold executor selection: `INFERFLUX_NATIVE_CUDA_EXECUTOR=delegate|direct_llama` (used when `backend=cuda_native`).
- CI wiring is available in `.github/workflows/ci.yml` (`cuda-throughput-gate` job), gated by:
  `INFERFLUX_ENABLE_CUDA_THROUGHPUT_GATE=true` and `INFERFLUX_CUDA_MODEL_PATH`.
- Useful tuning variables in CI:
  `INFERFLUX_TP_CONCURRENCY`, `INFERFLUX_TP_REQUESTS`, `INFERFLUX_TP_WARMUP`,
  `INFERFLUX_TP_MAX_TOKENS`, `INFERFLUX_TP_MIN_TOK_S`, `INFERFLUX_TP_MIN_SUCCESS_RATE`,
  `INFERFLUX_TP_MIN_OVERLAP_MS`, `INFERFLUX_TP_MAX_ATTN_FALLBACKS`,
  `INFERFLUX_TP_TARGET_PREFILL_TOKENS`.

### Streaming CLI cheatsheet

| Mode | Command | Purpose |
| --- | --- | --- |
| Status | `inferctl status --host 127.0.0.1 --port 8080` | Health probe |
| Models list | `inferctl models --json` | Raw OpenAI-compatible `/v1/models` output for scripts (omit `--json` for table view) |
| Model detail | `inferctl models --id llama3 --json` | Direct `/v1/models/{id}` lookup (non-zero exit on non-2xx) |
| Completion | `inferctl completion --prompt "Hi" --model llama3 --stream` | SSE streaming |
| Chat (scripted) | `inferctl chat --message "user:Hello" --stream` | One-shot chat |
| Chat (interactive) | `inferctl chat --interactive --model tinyllama` | Maintains dialogue |
| Admin ops | `inferctl admin models --load foo.gguf --id llama3 --default` | Model lifecycle (`/v1/admin/routing` controls runtime capability fallback policy) |

---

## Configuration Field Guide

| Category | Keys / Env | Purpose |
| --- | --- | --- |
| Backend selection | `models[].backend` (`cpu`, `cuda`, `cuda_native`, `cuda_universal`/`cuda_llama`, `mps`, `rocm`), `runtime.backend_priority`, `runtime.backend_exposure.*`, `runtime.capability_routing.*`, `INFERFLUX_BACKEND_PRIORITY`, `INFERFLUX_BACKEND_PREFER_NATIVE`, `INFERFLUX_BACKEND_ALLOW_LLAMA_FALLBACK`, `INFERFLUX_ROUTING_ALLOW_DEFAULT_CAPABILITY_FALLBACK`, `INFERFLUX_ROUTING_REQUIRE_READY_BACKEND`, `INFERFLUX_ROUTING_FALLBACK_SCOPE`, `INFERFLUX_NATIVE_CUDA_STRICT`, `INFERFLUX_NATIVE_CUDA_EXECUTOR` | Choose backend chain + explicit provider path + capability fallback behavior |
| Model format routing | `models[].format`, `model.format`, `INFERFLUX_MODEL_FORMAT`, `INFERFLUX_MODELS` (`format=...`) with `auto`, `gguf`, `safetensors`, `hf` | Auto-detect model format from path or force loader selection per model (`hf://org/repo` resolves to local cache under `${INFERFLUX_HOME:-$HOME/.inferflux}/models/org/repo`; non-MLX backends can consume cached `.gguf` sidecars when present; model APIs expose both `source_path` and `effective_load_path`) |
| CUDA runtime tuning | `runtime.cuda.attention.kernel`, `runtime.cuda.flash_attention.*`, `runtime.cuda.phase_overlap.*`, `INFERFLUX_CUDA_ATTENTION_KERNEL`, `INFERFLUX_CUDA_FLASH_ATTENTION`, `INFERFLUX_CUDA_FLASH_TILE`, `INFERFLUX_CUDA_PHASE_OVERLAP`, `INFERFLUX_CUDA_PHASE_OVERLAP_MIN_PREFILL_TOKENS`, `INFERFLUX_CUDA_PHASE_OVERLAP_PREFILL_REPLICA` | Control CUDA attention policy, mixed-batch decode/prefill arbitration, and optional dual-context prefill overlap |
| Scheduler batching | `runtime.scheduler.max_batch_size`, `runtime.scheduler.max_batch_tokens`, `INFERFLUX_SCHED_MAX_BATCH_SIZE`, `INFERFLUX_SCHED_MAX_BATCH_TOKENS` | Tune continuous-batching throughput/latency tradeoff |
| Fairness | `runtime.fairness.enable_preemption`, `.high_priority_threshold`, `.max_timeslice_tokens` | Continuous batching fairness |
| Registry (CQ-8) | `registry.path`, `registry.poll_interval_ms` | Hot-reload model manifests |
| Authentication | `auth.api_keys[].scopes`, `INFERCTL_API_KEY` | RBAC for generate/read/admin |
| Guardrails | `guardrails.blocklist`, `guardrails.opa_endpoint` | Content filters & policy |
| Observability | `/metrics`, `/readyz`, `/livez` | Prometheus endpoints |

### API-key example
```yaml
auth:
  api_keys:
    - key: dev-key-123
      scopes: [generate, read, admin]
```

---

## Audience Guides

- 🧑‍💻 **Developers** – build/test/debug instructions, coding standards, and profiling tips live in [docs/DeveloperGuide.md](docs/DeveloperGuide.md).
- 🙋 **Users** – CLI walkthroughs, configuration snippets, and sample workflows live in [docs/UserGuide.md](docs/UserGuide.md).
- 🛠️ **Admins** – deployment matrices, registry automation, observability dashboards, and security knobs live in [docs/AdminGuide.md](docs/AdminGuide.md).

---

## Tests & Validation

| Scope | Command | Notes |
| --- | --- | --- |
| Unit | `ctest --test-dir build --output-on-failure` | Full suite |
| Integration (SSE) | `ctest -R IntegrationSSE --output-on-failure` | Requires `INFERFLUX_MODEL_PATH`, `INFERCTL_API_KEY` |
| CLI contract gate | `ctest --test-dir build -R IntegrationCLIModelListContract --output-on-failure -V` | Verifies `inferctl models` / `inferctl admin models --list` JSON+table semantics and auth-failure behavior |
| Embeddings routing contract | `ctest --test-dir build -R "EmbeddingsRoutingTests|IntegrationEmbeddingsRoutingContract" --output-on-failure -V` | Verifies `/v1/embeddings` capability-routing invariants (fallback, explicit pinning, backend-unavailable) |
| Model identity contract | `ctest --test-dir build -R "ModelIdentityTests|IntegrationModelIdentityContract" --output-on-failure -V` | Verifies strict explicit-ID behavior for `/v1/models/{id}` and admin model lifecycle operations (`load` validation, `unload`, `set-default`) |
| Admin argument contract | `ctest --test-dir build -R IntegrationCLIAdminArgContract --output-on-failure -V` | Verifies fail-fast `inferctl` argument semantics for `models --id`, `admin models`, `admin cache`, `admin api-keys`, `admin guardrails`, `admin rate-limit`, and `admin routing` (required values, operation exclusivity, non-2xx exit codes) |
| Targeted | `ctest -R "<pattern>"` | Curated labels: `[paged_kv]`, `[unified_batch]`, `[parallel]`, etc. |

---

For design deep-dives, non-functional requirements, and roadmap context, explore `docs/`. Contributions are always welcome—open a PR referencing the relevant guide and follow the coding checklists in the Developer Guide. Happy hacking! 🚀
