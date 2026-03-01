# InferFlux

InferFlux is a high-performance, C++-based inference server inspired by vLLM. It targets drop-in compatibility with LM Studio and Ollama style APIs while supporting CUDA, ROCm, Metal (MPS), and CPU runtimes. Models in GGUF and tensor-based (safetensors) formats can be loaded directly from local storage or Hugging Face mirrors.

## Highlights
- Continuous batching scheduler with paged KV cache, speculative decoding hooks, and streaming SSE.
- Modular device backends so CUDA/ROCm/MPS/CPU share a common interface, with automatic Metal offload knobs.
- OpenAI-compatible REST/gRPC/WebSocket APIs plus a lightweight CLI (`inferctl`) featuring interactive transcripts and SSE playback.
- Config-driven deployments (YAML/Terraform/Helm) and enterprise guardrails (OIDC, RBAC, rate limiting, audit logs).
- Built-in metrics, tracing hooks, authentication, adapter hot-reload flows, and policy plugins for guardrails/PII scrubbing.

### Unique Selling Propositions
1. **Any-Backend Runtime** – one binary spans CPU laptops, MPS-capable Macs, CUDA/ROCm servers, with auto-tuned paged KV caches.
2. **Enterprise-Grade Security** – OIDC/API-key auth, per-tenant rate limiting, RBAC scopes, encrypted adapter storage, audit logs.
3. **Cloud-Native Observability** – Prometheus + OpenTelemetry, `/metrics` endpoint, autoscaler hints, structured logs, debug UI hooks.
4. **Model Portability** – GGUF + safetensors, manifest signing, LoRA stacking, adapter hot reloads, CLI conversions.
5. **Developer Ergonomics** – `inferctl` streaming mode, interactive chat, SSE viewer, and admin APIs mirroring OpenAI.

## Getting Started
```bash
./scripts/build.sh
./scripts/run_dev.sh --config config/server.yaml
```

## CLI
Set an API key (matching `config/server.yaml`) and query the server:
```bash
export INFERCTL_API_KEY=dev-key-123
export INFERFLUX_MODEL_PATH=$HOME/models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
export INFERFLUX_RATE_LIMIT_PER_MINUTE=60
export INFERFLUX_GUARDRAIL_BLOCKLIST=classified,pii
export INFERFLUX_AUDIT_LOG=logs/audit.log
export INFERFLUX_OIDC_ISSUER=https://issuer.example.com
export INFERFLUX_OIDC_AUDIENCE=inferflux
export INFERFLUX_POLICY_STORE=config/policy_store.conf
export INFERFLUX_POLICY_PASSPHRASE="super-secret-passphrase"
./build/inferctl status --host 127.0.0.1 --port 8080
./build/inferctl completion --prompt "Hello from InferFlux" --api-key "$INFERCTL_API_KEY" --model llama3
./build/inferctl chat --message "user:Hello" --stream --api-key "$INFERCTL_API_KEY"
./build/inferctl chat --interactive --model tinyllama --stream
./build/inferctl admin guardrails --list --api-key "$INFERCTL_API_KEY"
./build/inferctl admin guardrails --set secret,pii --api-key "$INFERCTL_API_KEY"
./build/inferctl admin rate-limit --set 200 --api-key "$INFERCTL_API_KEY"
./build/inferctl admin api-keys --list --api-key "$INFERCTL_API_KEY"
./build/inferctl admin api-keys --add new-key --scopes generate,read --api-key "$INFERCTL_API_KEY"
./build/inferctl admin models --list --api-key "$INFERCTL_API_KEY"
./build/inferctl admin models --load "$HOME/models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf" --id llama3 --default --api-key "$INFERCTL_API_KEY"
```

## Speculative Decoding & Metrics
Enable the draft/diffuser path via `runtime.speculative_decoding.enabled` or `INFERFLUX_SPECULATIVE_ENABLED=true`. `INFERFLUX_SPEC_DRAFT_MODEL`, `INFERFLUX_SPEC_MAX_PREFILL`, and `INFERFLUX_SPEC_CHUNK_SIZE` shape how many prompt tokens seed the draft model and how large each validation chunk is. When speculation is active the `/metrics` endpoint publishes `inferflux_spec_chunks_total`, `inferflux_spec_chunks_accepted_total`, and `inferflux_spec_tokens_reused_total`, so you can plot how often draft tokens survive the target validation pass.

## NVMe Paged KV Offload
Point `runtime.nvme_offload.path` (or `INFERFLUX_NVME_OFFLOAD_PATH`) at a fast local disk to persist KV pages. Tune the async writer with `runtime.nvme_offload.workers` / `.queue_depth` or `INFERFLUX_NVME_WORKERS` / `INFERFLUX_NVME_QUEUE_DEPTH`; InferFlux logs the active values on startup. Host cache sizing is controlled by `runtime.paged_kv.cpu_pages` + `runtime.paged_kv.eviction` and can also be overridden with `INFERFLUX_KV_CPU_PAGES` / `INFERFLUX_KV_EVICTION` for quick experiments.

## Guardrails & OPA Policies
`guardrails.blocklist` remains the simplest way to stop known-bad strings, but you can now supply `guardrails.opa_endpoint` (file:// or http://) to evaluate every prompt through an external policy even when no blocklist terms are defined. Local JSON fixtures (e.g., `file://config/policy.json`) are handy for smoke tests, while HTTP endpoints let you connect to live OPA/Cedar deployments. Denials bubble up to clients as 400 responses and are also captured in the audit log.

When `INFERFLUX_MODEL_PATH` (or `model.path` in `config/server.yaml`) points to a GGUF file, the server loads it through the integrated `llama.cpp` backend and returns human-readable completions. Without a model, a friendly stub response is returned for smoke tests.

- Pass `--stream` to `inferctl completion|chat` to receive Server-Sent Events live, chunk by chunk, just like the OpenAI API.
- Interactive mode (`inferctl chat --interactive`) reads multi-turn prompts from stdin and maintains the running transcript.

The HTTP service now exposes:

- `/metrics` with Prometheus counters for requests, errors, and token counts (tagged by backend type).
- Fairness knobs (`runtime.fairness.*` in `config/server.yaml`) let you enable preemption, set high-priority thresholds, and clamp timeslice tokens on CPU/MPS before CUDA validation lands.
- Streaming counters (`inferflux_stream_tokens_total`, `inferflux_stream_cache_hits_total`) surface SSE throughput vs cached completions for ops visibility.
- Streaming responses when `{ "stream": true }` is supplied to `/v1/completions` or `/v1/chat/completions`.
- Admin APIs to inspect/update guardrail blocklists, rate limits, API keys, and loaded models, protected by RBAC scopes (`generate`, `read`, `admin`).

API keys are defined in `config/server.yaml` with explicit scopes:

```yaml
auth:
  api_keys:
    - key: dev-key-123
      scopes: [generate, read, admin]
```

Use admin-scoped keys with `inferctl admin ...` to modify guardrails, rate limits, API keys, and model inventory without restarts.

The persistent policy store lives at `config/policy_store.conf` (override via `INFERFLUX_POLICY_STORE`) and can be AES-GCM encrypted transparently by setting `INFERFLUX_POLICY_PASSPHRASE`. Admin updates mutate the encrypted store so multiple replicas stay consistent even across restarts.

To offload layers to Metal (MPS), set either `runtime.mps_layers` in `config/server.yaml` or export `INFERFLUX_MPS_LAYERS=<layer-count>` before launching `inferfluxd`. The metrics endpoint reports the active backend label (`cpu`, `mps`, or `stub`).

## Configuration Highlights
- `runtime.speculative_decoding` toggles draft-model speculation (enable flag, draft model path, max prefill tokens, and `chunk_size` to control validation granularity).
- `runtime.nvme_offload.path` configures an NVMe-backed paged KV cache directory.
- `runtime.nvme_offload.workers` and `runtime.nvme_offload.queue_depth` tune the async writer (override via `INFERFLUX_NVME_WORKERS` / `INFERFLUX_NVME_QUEUE_DEPTH`).
- `runtime.paged_kv.cpu_pages` and `runtime.paged_kv.eviction` (lru or clock, also adjustable with `INFERFLUX_KV_CPU_PAGES` / `INFERFLUX_KV_EVICTION`) size and prioritize the host cache.
- `guardrails.opa_endpoint` reserves a future OPA/Cedar decision endpoint for contextual policies.
- `models` allows you to declare multiple GGUF entries (`id`, `path`, optional `backend`, and `default`), and `INFERFLUX_MODELS` can override the list at runtime (`id=llama3,path=/models/llama3.gguf,backend=mps,default=true;...`).
- `auth.api_keys[].scopes` map to RBAC scopes (`generate`, `read`, `admin`), and `INFERFLUX_POLICY_PASSPHRASE` encrypts the policy store.
- `INFERFLUX_POLICY_STORE` overrides the location of the encrypted policy file (`config/policy_store.conf` by default).

See `docs/` for the PRD, design, and non-functional requirements, and browse `docs/Roadmap.md` for milestone details.

## Tests
- Unit tests: `ctest --test-dir build --output-on-failure`
- Integration (requires `INFERFLUX_MODEL_PATH` pointing to a GGUF and `INFERCTL_API_KEY`):  
  `ctest -R IntegrationSSE --output-on-failure`
