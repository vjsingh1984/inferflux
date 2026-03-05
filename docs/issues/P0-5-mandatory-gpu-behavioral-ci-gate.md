# [P0-5] Mandatory GPU Behavioral CI Gate

Priority: P0
Owner: QA + Runtime
Effort: 1 eng-week (+ runner infra)
Risk: Medium
Dependencies: P0-3
Labels: ci, qa, throughput, release-gate

## Problem
GPU validation is partially advisory and environment-dependent, reducing confidence in throughput and native-path regressions.

## Scope Files
- `.github/workflows/ci.yml`
- `scripts/run_throughput_gate.py`
- `docs/ReleaseProcess.md`
- `docs/DeveloperGuide.md`

## Test Plan
1. Make one GPU lane required (not advisory) for protected branch workflows.
2. Ensure the lane executes throughput gate plus focused capability and contract assertions.
3. Verify CI artifact publication for perf diagnostics on pass or fail.
4. Validate branch protection docs reference the required check name.

## Acceptance Checklist
- [ ] At least one CUDA behavioral job is required for merge.
- [ ] Required job runs throughput gate and fails on threshold breach.
- [ ] GPU job logs and artifacts are consistently uploaded.
- [ ] Release docs define GPU gate expectations and fallback process.
- [ ] CI names are stable so repository protection rules remain valid.
