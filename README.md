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
./build/inferctl status --host 127.0.0.1 --port 8080
./build/inferctl completion --prompt "Hello from InferFlux" --api-key "$INFERCTL_API_KEY" --model llama3
./build/inferctl chat --message "user:Hello" --stream --api-key "$INFERCTL_API_KEY"
./build/inferctl chat --interactive --model tinyllama --stream
```

When `INFERFLUX_MODEL_PATH` (or `model.path` in `config/server.yaml`) points to a GGUF file, the server loads it through the integrated `llama.cpp` backend and returns human-readable completions. Without a model, a friendly stub response is returned for smoke tests.

- Pass `--stream` to `inferctl completion|chat` to receive Server-Sent Events live, chunk by chunk, just like the OpenAI API.
- Interactive mode (`inferctl chat --interactive`) reads multi-turn prompts from stdin and maintains the running transcript.

The HTTP service now exposes:

- `/metrics` with Prometheus counters for requests, errors, and token counts (tagged by backend type).
- Streaming responses when `{ "stream": true }` is supplied to `/v1/completions` or `/v1/chat/completions`.

To offload layers to Metal (MPS), set either `runtime.mps_layers` in `config/server.yaml` or export `INFERFLUX_MPS_LAYERS=<layer-count>` before launching `inferfluxd`. The metrics endpoint reports the active backend label (`cpu`, `mps`, or `stub`).

See `docs/` for the PRD, design, and non-functional requirements, and `config/server.yaml` for a reference deployment configuration.
