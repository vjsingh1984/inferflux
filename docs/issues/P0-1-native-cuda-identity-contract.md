# [P0-1] Native CUDA Identity Contract Across API/CLI/Metrics

Priority: P0
Owner: Runtime + CLI
Effort: 1.5 eng-weeks
Risk: Medium
Dependencies: None
Labels: runtime, cli, api-contract, observability

## Problem
`cuda_universal` and `cuda_native` behavior is not surfaced consistently enough for operators and automation. This creates benchmarking ambiguity and policy-routing confusion.

## Scope Files
- `runtime/backends/backend_factory.cpp`
- `runtime/backends/backend_factory.h`
- `scheduler/model_registry.cpp`
- `server/http/http_server.cpp`
- `cli/main.cpp`
- `tests/unit/test_backend_factory.cpp`
- `tests/integration/stub_integration_test.py`
- `docs/API_SURFACE.md`
- `docs/UserGuide.md`
- `docs/AdminGuide.md`
- `docs/CONFIG_REFERENCE.md`

## Test Plan
1. Add unit tests for backend hint normalization and identity mapping (`cuda`, `cuda_universal`, `cuda_native`).
2. Add integration tests for `/v1/models` and `/v1/models/{id}` to assert explicit backend identity fields.
3. Add CLI integration tests for `inferctl models --json` and `inferctl models --id ... --json` identity parity with HTTP.
4. Run contract tests and docs gate.

## Acceptance Checklist
- [ ] API model payload includes explicit backend identity contract field(s) with stable enum values.
- [ ] CLI JSON output mirrors API identity exactly.
- [ ] Human-readable CLI table output clearly distinguishes universal vs native.
- [ ] Existing clients are backward-compatible (no breaking removal of current fields).
- [ ] Unit + integration tests for identity contract are green.
- [ ] Docs updated to describe backend identity semantics and examples.
