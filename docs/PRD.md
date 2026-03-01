# InferFlux Product Requirements

## Vision

> **InferFlux: The enterprise-native inference server that runs anywhere.**
>
> While vLLM and SGLang optimize for raw throughput on GPU clusters, and Ollama
> optimizes for local simplicity, InferFlux is the only inference server built from
> the ground up with integrated enterprise controls (RBAC, encrypted policy, audit,
> guardrails) that runs identically on a developer's MacBook and a Kubernetes cluster.

## Overview
InferFlux is a C++17 inference server that combines the enterprise features missing
from open-source inference servers (RBAC, encrypted policy store, audit logging,
guardrails) with the model compatibility of llama.cpp. It exposes OpenAI-compatible
APIs and ships a CLI (`inferctl`) for local and remote management.

The strategic approach is to deeply integrate llama.cpp as the inference engine
(which already supports CUDA, Metal, Vulkan, speculative decoding, grammar sampling,
multimodal) and focus InferFlux's value-add on the serving, security, and operations
layer above it.

## Personas
1. **Indie Builder** — wants a local LM Studio replacement with GPU acceleration and adapter hot-reload.
2. **Platform Engineer** — needs an autoscaling inference tier inside a Kubernetes cluster that integrates with existing auth/logging/metrics.
3. **Research Scientist** — swaps between custom LoRA adapters and quantizations for experiments, requires scriptable control through CLI.
4. **Agent Developer** — builds multi-step AI workflows requiring structured output, tool calling, and prefix caching for efficient multi-turn conversations.

## Persona KPI Scorecard
| Persona | KPI | Target | Measurement Cadence |
| --- | --- | --- | --- |
| Indie Builder | Time-to-first chat over SSE after install | <5 minutes on M3/M1 laptops | Weekly smoke test via `inferctl chat` |
| Platform Engineer | Tail latency (TTFT) with audit + guardrails enabled | <250 ms P95 on CPU, <120 ms on CUDA | Perf suite per release |
| Research Scientist | Adapter swap latency + failure-free reloads | <5 s with 0 dropped connections | Nightly adapter regression |
| Agent Developer | JSON-schema compliant responses | 99.5% success with constrained decoding enabled | Contract tests each minor release |
| Compliance Lead | Policy/audit replication lag | <30 s propagation across replicas | Observability dashboard, daily |

## Goals & Non-Goals
- **Goals**: OpenAI-compatible API serving via llama.cpp, drop-in compatibility with existing SDKs, enterprise-grade auth/policy/audit, multi-backend runtime (CPU/CUDA/ROCm/MPS), observability (Prometheus metrics), and safe hot-reloads.
- **Non-Goals**: Training, fine-tuning pipelines, bespoke frontend UX, or reimplementing inference kernels that llama.cpp already provides.

### Current Status (May 2025)
- **Streaming & Cancellation**: `InferenceRequest.on_token` now drives SSE responses directly from the scheduler, with shared cancellation flags so dropped clients stop generation in-flight.
- **Batch Observability**: Prefill/decode timing and streamed token counters are exposed via `/metrics`; `inferflux_stream_tokens_total` vs `inferflux_stream_cache_hits_total` highlight live SSE health.
- **Tool Calling Stubs**: When no backend is loaded the server synthesizes JSON tool-call envelopes so agent frameworks still receive valid scaffolding (logged via `INFERFLUX_LOG_TOOL_CALLS`).
- **Next Focus**: CPU/MPS fairness + preemption (priority queue aging, cancellation regression tests), llama.cpp tokenizer metrics, and GPU work once compatible hardware is available.
- **Upcoming knobs (Q3)**: `scheduler.fairness.*` config path will control priority thresholds, timeslice tokens, and per-priority budgets so operators can tune fairness on CPU/MPS hardware before CUDA validation lands.

### Strategic Modules In Flight
- **Constrained Decoder**: Grammar/JSON-aware decoding path inserted between scheduler and runtime to deliver the 99.5% schema KPI.
- **Prefix Cache**: Radix-tree cache of validated KV prefixes feeding both single-tenant (agent) and multi-tenant workloads.
- **Multimodal Adapter**: Image/audio preprocessing pipeline leveraging llama.cpp `libmtmd` so chat completions accept base64/URL media blobs.

## Unique Selling Propositions
1. **Integrated Policy Engine**: Built-in OIDC/API-key auth, per-tenant rate limiting, RBAC scopes, audit logging, encrypted policy store (AES-GCM), and pluggable guardrails. No competitor has this built-in.
2. **Single Binary, Any Hardware**: Auto-detect CPU, CUDA, ROCm, or Apple MPS via llama.cpp. One binary spans laptops to GPU clusters.
3. **Cloud-Native from Day One**: Prometheus metrics, Kubernetes health probes (`/readyz`, `/livez`), CORS support, and Docker/Helm assets built into the architecture.
4. **Agent-Ready**: Structured output (JSON mode, grammar constraints), tool/function calling, and prefix caching for efficient multi-turn and agentic workloads.
   - OpenAI-style `response_format` requests are the canonical contract; internal adapters translate them into backend-native constraints (llama.cpp GBNF today, future engines tomorrow) so agents never have to reason about hardware-specific formats.
5. **Developer Ergonomics**: Rich CLI (`inferctl`) with interactive chat, streaming, admin commands, and model management.

## User Stories
- As an indie builder, I can run `inferctl chat --model llama3:8b --interactive` and stream chat completions from my laptop GPU.
- As a platform engineer, I can deploy InferFlux via Helm, configure API keys and RBAC scopes, and scrape Prometheus metrics for autoscaling decisions.
- As a researcher, I can load GGUF weights plus multiple LoRA adapters without restarting the server.
- As an agent developer, I can use tool calling and structured JSON output with prefix caching for efficient multi-turn agentic workflows.

## Functional Requirements
1. Serve `/v1/chat/completions`, `/v1/completions`, `/healthz`, `/readyz`, `/livez`, `/metrics`, and admin endpoints (`/v1/admin/guardrails`, `/v1/admin/rate_limit`, `/v1/admin/api_keys`).
2. Support GGUF model loading via llama.cpp (git submodule).
3. Provide device backends for CPU and MPS (via llama.cpp) while keeping the runtime ready for CUDA and vendor accelerators (Intel GPU, AMD ROCm) once compatible hardware is on hand.
4. Offer CLI (`inferctl`) to manage models, run chat/completion requests, stream SSE responses, and administer guardrails/rate-limits/API keys.
5. Export Prometheus metrics (success/error counters, backend label, speculative stats).
6. Support API-key (SHA-256 hashed) and OIDC (RS256 JWT) authentication, per-key rate limiting, RBAC scopes, audit logs, and encrypted policy persistence.
7. Provide configuration via YAML + environment variable overrides.
8. Support `response_format` parameter for structured output (JSON mode, grammar constraints via llama.cpp GBNF). The server must accept OpenAI-style `json_schema` payloads and raw grammar strings, validate size/complexity, and document the relationship with tool outputs so agent builders can rely on 99.5% schema fidelity.
9. Support `tools` and `tool_choice` parameters for function/tool calling.
10. Support per-request priority hints so fairness scheduling can honor latency SLOs.
11. Support CUDA/FlashAttention execution path with disaggregated prefill/decode to hit GPU throughput targets.
12. Support model pull from HuggingFace Hub (`inferctl pull`).

## Security Caveats & Dependencies
- Built-in TLS termination can be enabled via `server.tls.*` (or `INFERFLUX_TLS_ENABLED` + cert/key env vars); clusters that terminate at an external ingress can keep it disabled.
- API keys are SHA-256 hashed before persisting to the encrypted PolicyStore; rotations for legacy plaintext files require the `inferctl admin api-keys` tooling to rewrite entries.
- Guardrail verdicts hash prompts/responses by default; enabling raw logging requires the explicit `debug_mode` flag in `AuditLogger`.

## Acceptance Criteria
- Compatibility: Ollama and LM Studio clients can point to InferFlux without code changes.
- Latency: <500ms P99 prompt handling for 2k token prompts on CPU backend.
- Reliability: 99.9% uptime target with graceful shutdown <5 seconds.
- Security: API keys never stored in plaintext at rest; OIDC tokens cryptographically verified.
- Return 503 Service Unavailable when no model backend is loaded.

## Success Metrics
- Tokens-per-second per backend, API error rate <0.5%, policy update latency <250ms.
- Adoption: 3 pilot deployments (desktop, single-node GPU, Kubernetes cluster) for MVP.

## Milestones & SLAs
| Phase | Scope | KPI/Target |
| --- | --- | --- |
| **MVP (Q2)** | CPU/MPS backends, SSE streaming, policy store, auth, CLI | >=30 tok/s per request on CPU; policy update latency <250 ms; admin CLI task success 95% |
| **Performance (Q3)** | CUDA/ROCm offload, continuous batching, structured output, tool calling | >=400 tok/s aggregate on L40S 7B Q4K; structured output overhead <5% |
| **Enterprise (Q4)** | Disaggregated prefill/decode, expert parallelism, model registry | Guardrail verdict latency <500 ms; >99.95% policy replication consistency |
