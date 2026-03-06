# Session Handle Layer (Phase 1)

## Contract Snapshot

| Item | Contract |
|---|---|
| Default API mode | Stateless (OpenAI-compatible) |
| Optional stateful mode | `session_id` on request + `runtime.scheduler.session_handles.enabled=true` |
| Session mapping | `session_id -> {sequence_slot, sequence_id, block_table, prompt_tokens}` |
| TTL | Configurable (`ttl_ms`), with background expiry cleanup |
| KV precision | Server/model-load scoped (`runtime.cuda.kv_cache_dtype`), never per-session |
| Scope | Unified scheduler mode (decode-worker pool path is out-of-scope for Phase 1) |

## Flow

```mermaid
flowchart LR
    A[HTTP request] --> B{session_id provided?}
    B -->|No| C[Stateless scheduling path]
    B -->|Yes + feature enabled| D[Acquire session lease]
    D --> E{existing compatible state?}
    E -->|Yes| F[Reuse sequence slot + suffix prefill]
    E -->|No| G[Fresh prefill + new slot]
    F --> H[Decode]
    G --> H
    H --> I{request success?}
    I -->|Yes| J[Commit lease state + refresh TTL]
    I -->|No| K[Release lease + free sequence resources]
```

## Lease State Machine

```mermaid
stateDiagram-v2
    [*] --> Missing
    Missing --> Leased: AcquireLease(session_id)
    Leased --> Stored: CommitLease(state)
    Leased --> Missing: ReleaseLease(no state)
    Stored --> Leased: AcquireLease(session_id)
    Stored --> Expired: ttl elapsed
    Expired --> Missing: Cleanup/free sequence
```

## Request Contract

- `session_id` may be passed in JSON body (`session_id`) or header (`x-inferflux-session-id`).
- If omitted, server behavior is unchanged and fully stateless.
- Phase 1 expects prompt continuity semantics (new prompt extends previous prompt token prefix for reuse).

## Config

```yaml
runtime:
  scheduler:
    session_handles:
      enabled: false
      ttl_ms: 300000
      max_sessions: 1024
```

Environment overrides:

- `INFERFLUX_SESSION_HANDLES_ENABLED`
- `INFERFLUX_SESSION_TTL_MS`
- `INFERFLUX_SESSION_MAX`
