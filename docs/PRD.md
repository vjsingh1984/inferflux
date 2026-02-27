# InferFlux Product Requirements

## Overview
InferFlux is a C++17 inference server designed to replace desktop and cloud LLM providers such as LM Studio and Ollama while offering the performance profile of vLLM. The product must support GGUF and safetensors models (including quantized variants) and expose OpenAI-compatible APIs, a CLI, and admin controls for adapters and multi-tenant use. The initial release targets single-node deployments with CPU and CUDA backends, followed by ROCm/MPS scale-out.

## Personas
1. **Indie Builder** – wants a local LM Studio replacement with GPU acceleration and adapter hot-reload.
2. **Platform Engineer** – needs an autoscaling inference tier inside a Kubernetes cluster that integrates with existing auth/logging/metrics.
3. **Research Scientist** – swaps between custom LoRA adapters and quantizations for experiments, requires scriptable control through CLI.

## Goals & Non-Goals
- **Goals**: Continuous batching throughput comparable to vLLM, drop-in API compatibility, multi-backend runtime, observability, enterprise-grade guardrails, and safe hot-reloads.
- **Non-Goals**: Training, fine-tuning pipelines, or bespoke frontend UX.

## Unique Selling Propositions
1. **Any-Backend Runtime**: Auto-detect CPU, CUDA, ROCm, or Apple MPS, with per-device tuning (paged KV cache, speculative decode) so one binary spans laptops to GPU clusters.
2. **Enterprise Controls**: Built-in OIDC/API-key auth, per-tenant rate limiting, RBAC scopes, audit logging, encrypted adapter storage, and policy hooks for guardrails or PII scrubbing.
3. **Cloud-Native Ops**: Prometheus/OTel telemetry, streaming SSE/WebSocket APIs, Helm/Terraform assets, autoscaling hints (queued tokens, KV page pressure), and hot-swappable LoRA/adapters.
4. **Model Portability**: GGUF + safetensors ingestion, on-the-fly conversion (`inferctl pull`), adapter stacking, and manifest signing so teams reuse the same artifacts across InferFlux and Ollama.
5. **Developer Ergonomics**: Rich CLI (`inferctl`) with interactive chat, transcript mode, streaming playback, and local debug UI endpoints to inspect KV caches and token traces.

## User Stories
- As an indie builder, I can run `inferctl run --model llama3:8b` and stream chat completions from my laptop GPU.
- As a platform engineer, I can deploy InferFlux via Helm, configure API keys, and scrape Prometheus metrics for autoscaling decisions.
- As a researcher, I can load GGUF weights plus multiple LoRA adapters without restarting the server.

## Functional Requirements
1. Serve `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/healthz`, `/metrics`, `/admin/reload` endpoints.
2. Support GGUF (via llama.cpp reader) and safetensors (via safetensors-cpp) weight ingestion.
3. Provide device backends for CUDA, ROCm, MPS, and CPU with a shared scheduler + paged KV cache, including Metal offload controls (`runtime.mps_layers`).
4. Offer CLI (`inferctl`) to manage models, run local servers, stream SSE responses, and inspect status (health, metrics, transcript mode).
5. Export Prometheus metrics and OpenTelemetry traces, including tokens/sec, backend mode, success/error counters, and adapter usage.
6. Support API-key and OIDC authentication, per-tenant rate limiting, RBAC scopes, audit logs, and encrypted at-rest secrets.
7. Provide configuration-driven deployment (YAML + CLI overrides) and Docker/Helm assets, plus Terraform examples for managed GPU clusters.
8. Enable policy/guardrail hooks (PII scrubbing, content filters), signed model manifests, LoRA stacking, and adapter hot-reload without downtime.

## Acceptance Criteria
- Latency: <250 ms P99 prompt handling for 2k token prompts on A100 with CUDA backend.
- Throughput: >= 400 tokens/s on single L40S for 7B models quantized to Q4K.
- Reliability: 99.9% uptime target with rolling restarts < 5 seconds per replica.
- Compatibility: LM Studio and Ollama clients can point to InferFlux without code changes.

## Success Metrics
- Tokens-per-second per GPU, average queue depth, KV cache hit rate, API error rate <0.5%.
- Adoption: 3 pilot deployments (desktop, single-node GPU, Kubernetes cluster) for MVP.

## Milestones & SLAs
| Phase | Scope | KPI/Target |
| --- | --- | --- |
| **MVP (Q2)** | CPU/MPS backends, SSE streaming, policy store | ≥30 tok/s per request on CPU; policy update latency <250 ms; admin CLI task success 95% |
| **Performance (Q3)** | CUDA/ROCm offload, speculative decoding, NVMe cache | ≥400 tok/s aggregate on L40S 7B Q4K; speculative draft reduces latency 30%; NVMe cache miss <5% |
| **Enterprise (Q4)** | OPA integration, LoRA stacking, distributed scheduler | Guardrail verdict latency <500 ms; >99.95% policy replication consistency; admin UX SUS score ≥80 |

SLA highlights:
- **Tokens/sec under load**: maintain ≥400 tok/s on L40S for 7B Q4K with 80% GPU utilization.
- **Policy propagation**: guardrail/rate-limit updates visible on all replicas within 2 seconds (P99).
- **Admin UX**: CLI/web tasks (list/update guardrails, rate limits, API keys) complete within 3 steps and <5 seconds end-to-end.
