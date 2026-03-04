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
| Model format routing | `models[].format`, `model.format`, `INFERFLUX_MODEL_FORMAT`, `INFERFLUX_MODELS` (`format=...`) | Detect or force model loader format (`auto`, `gguf`, `safetensors`, `hf`); `hf://org/repo` maps to `${INFERFLUX_HOME:-$HOME/.inferflux}/models/org/repo` |
| Throughput tuning | `runtime.scheduler.max_batch_size`, `runtime.scheduler.max_batch_tokens`, `INFERFLUX_SCHED_MAX_BATCH_SIZE`, `INFERFLUX_SCHED_MAX_BATCH_TOKENS` | Control scheduler batch packing |
| CUDA attention policy | `runtime.cuda.attention.kernel`, `INFERFLUX_CUDA_ATTENTION_KERNEL` | Select `auto`, `fa3`, `fa2`, or `standard` with safe fallback |
| CUDA mixed-phase tuning | `runtime.cuda.phase_overlap.enabled`, `runtime.cuda.phase_overlap.min_prefill_tokens`, `runtime.cuda.phase_overlap.prefill_replica`, `INFERFLUX_CUDA_PHASE_OVERLAP`, `INFERFLUX_CUDA_PHASE_OVERLAP_MIN_PREFILL_TOKENS`, `INFERFLUX_CUDA_PHASE_OVERLAP_PREFILL_REPLICA` | Decode-first arbitration plus optional dual-context prefill overlap for mixed prefill/decode unified batches |
| Guardrails | `guardrails.blocklist`, `guardrails.opa_endpoint` | Basic + contextual filtering |
| Rate limit | `auth.rate_limit_per_minute` | Per-key throttling |

For model provenance debugging, `/v1/models` and `/v1/admin/models` include
`path`, `source_path`, and `effective_load_path`.

## CLI recipes

| Goal | Command |
| --- | --- |
| Chat multi-turn | `inferctl chat --interactive --model llama3` |
| List public models | `inferctl models` (table default; add `--json` for raw `/v1/models` output) |
| Describe one model | `inferctl models --id llama3 --json` (calls `/v1/models/{id}`) |
| Admin list models | `inferctl admin models --list` (table includes source/effective loader paths; add `--json` for script-friendly raw output) |
| Load model & mark default | `inferctl admin models --load path.gguf --id llama3 --default` |
| Update guardrails | `inferctl admin guardrails --set pii,classified` |
| Inspect routing policy | `inferctl admin routing --get` |

For automation, `inferctl models --json` and
`inferctl admin models --list --json` return non-zero on auth or other non-2xx HTTP responses.
OpenAI-style requests treat `model: "default"` as an alias for default-model routing;
other unknown explicit model IDs return `model_not_found`.
`GET /v1/models/{id}` returns the same `model_not_found` contract for unknown IDs, and
admin model lifecycle APIs require explicit IDs for `unload`/`set-default` (`400 id is required`, `404 model_not_found`).
`inferctl admin models --load/--unload/--set-default` now also returns non-zero
on non-2xx responses so shell automation can rely on exit codes.
`inferctl` enforces required values for model identity flags (`--id`, `--load`,
`--unload`, `--set-default`) and fails fast on malformed invocations.
`inferctl admin models` enforces exactly one operation flag
(`--list`, `--load`, `--unload`, `--set-default`) to prevent ambiguous scripts.
`inferctl admin cache`, `inferctl admin api-keys`, `inferctl admin guardrails`,
`inferctl admin rate-limit`, and `inferctl admin routing` follow the same
fail-fast contract for required values, operation exclusivity, and non-2xx
exit codes.

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `no_backend` response | model not loaded | set `INFERFLUX_MODEL_PATH` or use registry entry |
| `model_not_found` response | explicit `model` id is unknown | verify ID via `inferctl models --json` or `/v1/models` |
| SSE stops mid-stream | client dropped | restart CLI or check `inferflux_stream_tokens_total` |
| Guardrail block | term matched or OPA deny | review blocklist / OPA response |
| Rate limit | exceeded per-key quota | increase `auth.rate_limit_per_minute` |

Need deeper assistance? Visit the [Admin Guide](AdminGuide.md) for deployment and observability tips or file an issue. 👍
