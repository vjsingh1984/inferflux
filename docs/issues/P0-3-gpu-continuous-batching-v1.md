# [P0-3] GPU Continuous Batching v1 (Iteration-Level Scheduler)

Priority: P0
Owner: Scheduler + Runtime
Effort: 3 eng-weeks
Risk: High
Dependencies: P0-1
Labels: scheduler, runtime, throughput, cuda

## Problem
Current batching foundations exist, but missing iteration-level GPU scheduling blocks major throughput and cost-per-token gains.

## Scope Files
- `scheduler/scheduler.cpp`
- `scheduler/scheduler.h`
- `runtime/execution/batch_executor.cpp`
- `runtime/backends/cuda/native_cuda_executor.cpp`
- `runtime/backends/backend_capabilities.cpp`
- `server/metrics/metrics.cpp`
- `tests/unit/test_scheduler.cpp`
- `tests/unit/test_unified_batching.cpp`
- `tests/unit/test_native_batching.cpp`
- `tests/integration/throughput_gate_contract_test.py`
- `scripts/run_throughput_gate.py`

## Test Plan
1. Add unit tests for iteration scheduling decisions and fairness constraints under mixed prefill/decode loads.
2. Add unit tests for cancellation and preemption behavior during iteration scheduling.
3. Extend throughput contract test expectations for batching-health metrics.
4. Run fairness, unified-batch, and throughput contract suites plus GPU throughput gate.

## Acceptance Checklist
- [ ] Iteration-level scheduling is active on CUDA path for mixed workloads.
- [ ] Fairness and cancellation semantics remain correct.
- [ ] Batching utilization metrics improve and are exported.
- [ ] Throughput gate passes with tightened batch health thresholds.
- [ ] No regression in model and capability routing behavior.
