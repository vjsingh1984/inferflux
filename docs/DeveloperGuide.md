# Developer Guide

| Topic | Why it matters | TL;DR |
| --- | --- | --- |
| Build & toolchain | determinism across devices | `cmake .. && ninja` or `./scripts/build.sh` |
| Testing strategy | protect batching/kv/cache logic | `ctest -R <label>` |
| Workflow checklist | clean PRs + fast reviews | follow the diagram below |
| Debug/playbooks | shorten MTTR | `lldb`, `perf`, `clangd`, `sanitizers` |

## Workflow Blueprint
```
┌──────────────┐    format/lint    ┌────────────┐   ctest labels   ┌────────────┐
│ Edit code    │ ───────────────▶ │ ./scripts/ │ ───────────────▶ │ Review PR  │
│ (feature)    │                   │ format.sh  │                  │ (docs incl)│
└──────┬───────┘                   └─────┬──────┘                  └────┬───────┘
       │                                 │                            │
       ▼                                 ▼                            ▼
  Run unit tests                 ctest -R "[paged_kv]"         update diagrams
```

## Build + Tooling Matrix

| Scenario | Command | Notes |
| --- | --- | --- |
| Full build | `./scripts/build.sh` | uses `build/` out dir |
| Focused target | `cmake --build build --target inferflux_tests -j` | incremental |
| Sanitizers | `ENABLE_ASAN=ON ./scripts/build.sh` | ASan + UBSan |
| Formatting | `./scripts/format.sh` | clang-format + cmake-format |
| Include-what-you-use | `IWYU=ON ./scripts/build.sh` | optional |

## Test Labels

| Label | Coverage | Command |
| --- | --- | --- |
| `[paged_kv]` | paged cache, offload, ref-counts | `ctest -R paged_kv` |
| `[unified_batch]` | chunked prefill + mixed decode | `ctest -R unified_batch` |
| `[parallel]` | parallel context / comm stubs | `ctest -R parallel` |
| `[stop_sequences]` | stop handling edge cases | `ctest -R stop_sequences` |

> Tip: combine labels via regex OR, e.g. `ctest -R "(paged_kv|unified_batch)"`.

## Coding Checklist
- [ ] Update/consult relevant document (Dev/User/Admin guide).
- [ ] Add/refresh diagrams or tables describing behavior.
- [ ] Document env vars when introducing new config.
- [ ] Cover success + failure paths with tests (unit or integration).
- [ ] Ensure log messages contain context (`request_id`, backend, etc.).

## Debug Recipes

| Goal | Command / Tool | Notes |
| --- | --- | --- |
| Inspect scheduler queues | `./build/inferctl admin metrics` | look at queue depth + fairness metrics |
| Sample CPU hotspots | `perf record -g ./build/inferfluxd ...` | view via `perf report` |
| LLM backend traces | set `LLAMA_LOG_LEVEL=debug` | forwarded through InferFlux logger |
| SSE replay | `./build/inferctl completion --stream --dump-sse` | saves raw SSE payloads |

## Document & Diagram Expectations
- Use tables for parameter mappings.
- Add ASCII diagrams or Mermaid blocks when modifying flows (place them near the change).
- Reference diagrams from `docs/Diagrams.md` when applicable.

Happy building! Reach out via GitHub discussions if a workflow is unclear. 🚀
