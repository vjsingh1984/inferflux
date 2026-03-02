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
- Admin CLI snippets:
  - `inferctl admin guardrails --set secret,pii --api-key $ADMIN_KEY`
  - `inferctl admin rate-limit --set 200 --api-key $ADMIN_KEY`

### Streaming CLI cheatsheet

| Mode | Command | Purpose |
| --- | --- | --- |
| Status | `inferctl status --host 127.0.0.1 --port 8080` | Health probe |
| Completion | `inferctl completion --prompt "Hi" --model llama3 --stream` | SSE streaming |
| Chat (scripted) | `inferctl chat --message "user:Hello" --stream` | One-shot chat |
| Chat (interactive) | `inferctl chat --interactive --model tinyllama` | Maintains dialogue |
| Admin ops | `inferctl admin models --load foo.gguf --id llama3 --default` | Model lifecycle |

---

## Configuration Field Guide

| Category | Keys / Env | Purpose |
| --- | --- | --- |
| Backend selection | `runtime.backend_priority`, `runtime.backend_exposure.*`, `INFERFLUX_BACKEND_PRIORITY`, `INFERFLUX_BACKEND_PREFER_NATIVE`, `INFERFLUX_BACKEND_ALLOW_LLAMA_FALLBACK` | Choose backend chain + fallback policy |
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
| Targeted | `ctest -R "<pattern>"` | Curated labels: `[paged_kv]`, `[unified_batch]`, `[parallel]`, etc. |

---

For design deep-dives, non-functional requirements, and roadmap context, explore `docs/`. Contributions are always welcome—open a PR referencing the relevant guide and follow the coding checklists in the Developer Guide. Happy hacking! 🚀
