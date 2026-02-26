# Non-Functional Requirements

## Performance
- P99 prompt latency < 250 ms for 2k token prompts on A100 with CUDA backend.
- Streaming throughput >= 30 tok/s per request; aggregate >= 400 tok/s per GPU.
- CPU fallback must sustain 8 tok/s for demo workloads.

## Scalability
- Horizontal scaling to 64 GPUs per cluster with shared KV cache metadata.
- Support for MIG partitioning and multi-node deployments using Redis/etcd for scheduler coordination.
- Multi-region active/active design with weighted routing, autoscaler hints (queue depth, cache pressure), and BYO object-storage for weights/adapters.

## Reliability
- Target 99.9% availability with graceful degradation when GPUs fail (automatic CPU fallback for small prompts).
- Hot-reload adapters/models in <5 s with no dropped connections.
- Persistent KV cache metadata to recover within 30 s after restart.

## Security
- API-key + OIDC auth (PKCE + workload identity), per-tenant rate limiting, TLS termination via ingress.
- Logs redact prompts by default; enable debug logs via config.
- Signed model packages (checksum verification) before loading weights.
- RBAC scopes (admin, read, generate, adapter-manager) enforced at gateway and CLI.
- Audit logging (structured JSON) shipped to SIEM/KMS-encrypted storage; adapter secrets encrypted with KMS key.
- Guardrail API to plug in policy engines (PII scrubbing, jailbreak detection, classifier enforcement).

## Operability
- Prometheus metrics for latency, queue depth, KV cache usage, GPU memory fragmentation, adapter hits, backend mode.
- OpenTelemetry traces for tokenizer→scheduler→backend pipeline.
- Configurable structured logging (JSON) with log rotation and log-sampling per tenant, plus `/metrics` and `/debug` endpoints (token traces, adapter status).
- CLI + admin UI for live streaming, SSE playback, interactive transcripts, and health dashboards.

## Compliance & Testing
- Deterministic CPU integration tests for reproducibility.
- Performance regression suite replaying trace logs.
- Dependency license audit (MIT/BSD/Apache 2.0 only) and SBOM generation per build, with attestation/publishing via `gh release`.
