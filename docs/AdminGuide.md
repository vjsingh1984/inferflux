# Admin Guide

| Theme | Description |
| --- | --- |
| Deployment matrix | how to run in dev/prod |
| Security checklist | auth + guardrails |
| Monitoring & registry | keep clusters healthy |

## Deployment matrix

| Environment | Command / Tool | Notes |
| --- | --- | --- |
| Local dev | `./scripts/run_dev.sh --config config/server.yaml` | auto-stub when model missing |
| Docker | `docker build -t inferflux . && docker run -p 8080:8080 inferflux` | mount `/models` |
| Helm | `helm upgrade --install inferflux charts/inferflux -f deploy/values.yaml` | integrates with K8s secrets |
| Terraform | `terraform apply -chdir=deploy/terraform` | optional module for cloud VMs |

## Distributed toggles

| Variable | Meaning |
| --- | --- |
| `INFERFLUX_DIST_RANK`, `INFERFLUX_DIST_WORLD_SIZE` | enable ParallelContext |
| `INFERFLUX_DIST_BACKEND` | choose stub/real NCCL backend (future) |

## Security checklist

- [ ] Configure API keys + scopes in `config/server.yaml`.
- [ ] Enable OIDC (`auth.oidc_issuer`, `auth.oidc_audience`) where SSO is required.
- [ ] Set `guardrails.blocklist` **and** `guardrails.opa_endpoint` for layered defenses.
- [ ] Store policies in encrypted form (`INFERFLUX_POLICY_PASSPHRASE`).
- [ ] Rotate audit logs (`logging.audit_log`) via logrotate or cloud logging sinks.

## Model registry flow
```
registry.yaml ──poll──▶ ModelRegistry ──update──▶ Backend Manager ──▶ Scheduler slots
        ▲                      │                         │
        │                      └── emits events to logs & metrics
        └── Git / Ops edits (id,path,format,backend,default)
```

`GET /v1/admin/models` now reports both `source_path` and
`effective_load_path` so operators can verify when `hf://` or safetensors
entries resolved to a concrete local GGUF artifact.
Identity contract: unknown IDs consistently return `model_not_found` for
`GET /v1/models/{id}`, `DELETE /v1/admin/models`, and
`PUT /v1/admin/models/default`; missing admin IDs return `id is required`.

## Monitoring table

| Signal | Endpoint / Metric | Action |
| --- | --- | --- |
| Queue depth | `/metrics` → `inferflux_queue_depth` | scale replicas or adjust fairness |
| KV cache health | `inferflux_kv_pages_in_use` | increase `runtime.paged_kv.cpu_pages` or NVMe |
| SSE throughput | `inferflux_stream_tokens_total` | watch for drops (network / client issues) |
| Guardrail hits | `inferflux_guardrail_blocks_total` | audit policy store |

## Incident playbook

| Scenario | First steps |
| --- | --- |
| High latency | check queue depth, fairness config, backend utilization |
| Memory pressure | inspect paged KV metrics, adjust NVMe offload, shrink batch size |
| Auth errors | confirm API-key scopes and OIDC tokens |
| Model churn | use registry (CQ-8) to add/remove models without restart |
| Admin policy write fails (`policy_persist_failed`) | verify `INFERFLUX_POLICY_STORE` path is writable, check filesystem permissions/space, and confirm backup file (`.bak`) integrity |

Keep this guide handy for ops runbooks. For developer-centric workflows see `docs/DeveloperGuide.md`; for CLI usage see `docs/UserGuide.md`. ✅
