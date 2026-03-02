# User Guide

| Section | Focus |
| --- | --- |
| Quick-start checklist | env vars + CLI |
| Request flows | ASCII diagram |
| Configuration cheat sheet | user-facing knobs |
| Troubleshooting table | fast fixes |

## Quick-start checklist

| Step | Command |
| --- | --- |
| 1. Set API key | `export INFERCTL_API_KEY=dev-key-123` |
| 2. Point to model | `export INFERFLUX_MODEL_PATH=$HOME/models/llama3.gguf` |
| 3. Launch server | `./scripts/run_dev.sh --config config/server.yaml` |
| 4. Make a request | `./build/inferctl completion --prompt "Hello" --stream --model llama3` |

## Request flow diagram
```
Client CLI ──HTTP JSON/SSE──▶ HTTP Server ──▶ Scheduler ──▶ Backend (CPU/MPS/CUDA)
     ▲                            │                 │              │
     │                            │                 ├─ Prefix Cache│
     └── stream events ◀──────────┘                 └─ Metrics/Logs │
```

## Configuration cheatsheet

| Feature | YAML / Env | Description |
| --- | --- | --- |
| Streaming | `request.stream=true` | SSE token streaming |
| Model registry | `registry.path=config/registry.yaml` | Hot reload models without restart |
| Guardrails | `guardrails.blocklist`, `guardrails.opa_endpoint` | Basic + contextual filtering |
| Rate limit | `auth.rate_limit_per_minute` | Per-key throttling |

## CLI recipes

| Goal | Command |
| --- | --- |
| Chat multi-turn | `inferctl chat --interactive --model llama3` |
| Admin list models | `inferctl admin models --list` |
| Load model & mark default | `inferctl admin models --load path.gguf --id llama3 --default` |
| Update guardrails | `inferctl admin guardrails --set pii,classified` |

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `no_backend` response | model not loaded | set `INFERFLUX_MODEL_PATH` or use registry entry |
| SSE stops mid-stream | client dropped | restart CLI or check `inferflux_stream_tokens_total` |
| Guardrail block | term matched or OPA deny | review blocklist / OPA response |
| Rate limit | exceeded per-key quota | increase `auth.rate_limit_per_minute` |

Need deeper assistance? Visit the [Admin Guide](AdminGuide.md) for deployment and observability tips or file an issue. 👍
