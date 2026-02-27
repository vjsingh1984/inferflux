# InferFlux Architecture

## System Overview
```
[Clients / SDKs]
      |
[HTTP/gRPC Frontend] -- websockets --> streaming clients
      |
[Scheduler + Admission Control]
      |
      +--> [Model Manager] -- loaders --> GGUF / safetensors weights
      +--> [Runtime Core]
              |- DeviceContext (CUDA/ROCm/MPS/CPU)
              |- Tensor / kernel registry
              |- Paged KV cache (GPU/CPU pages)
              |- Speculative decoding helper
      |
[Telemetry] -> Prometheus / OpenTelemetry
[Admin API] -> reload adapters/models
[Security] -> OIDC provider / RBAC / Audit log sinks
```

## Modules
- **Runtime**: Provides `DeviceContext` abstraction, memory allocators, paged KV cache, speculative decoding (draft models), NVMe-backed offload, and fused attention/MoE kernels. Supports CUDA, ROCm, MPS (Metal), and CPU backends with adaptive sharding.
- **Model**: Handles tokenizer plugins, GGUF and safetensors loaders, quantization metadata, and adapter composition.
- **Scheduler**: Implements continuous batching, prefill/decoding queues, request prioritization, and speculative decoding hooks.
- **Server**: Hosts HTTP/gRPC/WebSocket frontends, SSE streaming, auth (OIDC/API-key), rate limiting, RBAC enforcement, and structured/audit logging.
- **Policy Store**: Persists API keys, scopes, guardrails, and rate limits, exposing admin APIs and CLI hooks for dynamic updates.
- **CLI**: Offers `inferctl` commands to pull models, run local servers, and inspect runtime health.
- **Security Services**: Integrations for secrets management, KMS-backed encryption of adapters, and policy plugins for guardrails/PII scrubbing.
- **Observability**: Metrics registry, Prometheus exporter, OpenTelemetry spans, live debug UI, and hooks for autoscaler hints (queue depth, KV pages, backend mix).

## Data Flow
1. Client sends REST/gRPC request with prompt + parameters.
2. Auth middleware validates API key and injects tenant metadata.
3. Scheduler registers request, runs tokenizer, and batches prefill tokens.
4. Runtime executes prefill kernels, stores KV cache pages, and flips request to decode queue.
5. Device contexts iterate decode steps until termination; results stream back to clients.
6. Metrics and traces are emitted for each stage.

## Deployment View
- **Standalone**: single binary, config YAML, runs on developer desktops (CPU/MPS).
- **GPU Node**: Docker image with CUDA runtime, NVMe-backed cache, Prometheus sidecar.
- **Kubernetes**: Helm chart provisions ConfigMaps/Secrets, Horizontal Pod Autoscaler based on queue depth, optional Redis for distributed scheduling state.
- **Multi-region cloud**: Terraform/Helm combo for GKE/EKS/AKS, with managed Redis/etcd, service mesh policies, and workload identity (OIDC) for per-tenant RBAC.

## Security & Operations
- API keys stored as hashed values in config or secret store.
- Request + response payloads hashed before logging; raw text only available when debug flag enabled.
- Health checks expose readiness (model loaded, cache warmed) vs liveness (event loop running).
- Audit logs include tenant, model, adapter, tokens, and policy decisions; can stream to SIEM.
- Guardrail hooks for content classification, PII redaction, and function-calling approval flows.
