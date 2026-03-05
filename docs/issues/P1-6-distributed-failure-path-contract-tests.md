# [P1-6] Distributed Failure-Path Contract Tests (Prefill/Decode Transport)

Priority: P1
Owner: Distributed Runtime + QA
Effort: 2 eng-weeks
Risk: High
Dependencies: P0-4
Labels: distributed-runtime, qa, resilience, integration-test

## Problem
Distributed and disaggregated path lacks strong failure-path contracts (restart, backpressure, ticket recovery) needed for enterprise reliability claims.

## Scope Files
- `runtime/disaggregated/kv_channel.h`
- `runtime/disaggregated/kv_channel.cpp`
- `runtime/disaggregated/shm_kv_transport.cpp`
- `scheduler/scheduler.cpp`
- `tests/integration/shm_smoke_test.py`
- `tests/integration/stub_integration_test.py` (if admin and readiness checks are extended)
- `docs/Architecture.md`
- `docs/AdminGuide.md`

## Test Plan
1. Add integration tests for decode worker restart during active tickets.
2. Add integration tests for KV transport saturation and backpressure.
3. Add failure-injection test for transport interruption and recovery.
4. Validate readiness and health semantics under partial pool failure.
5. Run SHM smoke and stub integration suites.

## Acceptance Checklist
- [ ] No ticket loss under worker restart and transport interruption scenarios.
- [ ] Recovery behavior is deterministic and observable via metrics and logs.
- [ ] `/readyz` and admin pool views reflect degraded and recovered states correctly.
- [ ] Failure-path tests are wired into CI and visible in pipeline logs.
- [ ] Admin and architecture docs include rollback and incident guidance.
