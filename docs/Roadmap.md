# InferFlux Roadmap

## Q2 – MVP
- CPU & MPS backends with SSE streaming.
- Policy store with RBAC scopes, guardrail/rate-limit admin APIs.
- CLI interactive mode, Prometheus metrics, audit logging.
- **KPIs**: 30 tok/s/request on CPU; policy updates <250 ms; admin CLI task success ≥95%.

## Q3 – Performance & Scale
- CUDA/ROCm acceleration with NVMe-assisted paged KV cache.
- Speculative decoding (draft model + validator) with automatic fallback.
- LoRA stacking and hot adapter reloads.
- Autoscaler hints (queue depth, KV fragmentation) and NVMe telemetry.
- **KPIs**: 400 tok/s aggregate on L40S; speculative path reduces P99 latency by 30%; NVMe miss <5%.

## Q4 – Enterprise & Distributed Ops
- Distributed scheduler (multi-node KV metadata, MIG awareness).
- OPA/Cedar policy engine integration for contextual guardrails.
- Model registry + signed manifests; adapter sandboxing and per-tenant quotas.
- Web admin console with policy editor, live traces, and GPU health.
- **KPIs**: guardrail verdict latency <500 ms; policy replication consistency 99.95%; admin UX SUS ≥80.

## Stretch Goals
- Multi-region active/active with workload-aware routing.
- Budget-aware autoscaling (cost per token) and GPU sharing.
- Frontend SDKs for major frameworks (LangChain, LlamaIndex).
