# [P0-2] Strict Native-Request Policy (No Silent Fallback)

Priority: P0
Owner: Runtime
Effort: 1 eng-week
Risk: Medium
Dependencies: P0-1
Labels: runtime, backend-policy, reliability

## Problem
Explicit native requests can still degrade into fallback behavior without a strict fail-fast contract suitable for production policy enforcement.

## Scope Files
- `runtime/backends/backend_factory.cpp`
- `runtime/backends/backend_factory.h`
- `server/main.cpp`
- `docs/CONFIG_REFERENCE.md`
- `docs/Troubleshooting.md`
- `tests/unit/test_backend_factory.cpp`
- `tests/integration/stub_integration_test.py`

## Test Plan
1. Add unit tests for strict mode behavior when native is unavailable or not ready.
2. Add integration test that explicit native load request fails with clear error unless fallback opt-in is set.
3. Validate fallback opt-in path still functions and emits explicit diagnostics.
4. Run backend factory and stub integration suites.

## Acceptance Checklist
- [ ] New strict policy switch is documented and configurable.
- [ ] Explicit `cuda_native` request fails fast if native path is not eligible.
- [ ] Fallback only occurs when explicitly allowed by policy.
- [ ] Error messages are actionable and machine-parseable.
- [ ] Backend factory tests cover all strict and fallback permutations.
