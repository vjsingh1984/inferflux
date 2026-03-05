# [P1-3] Scheduler Lock Partitioning and Contention Reduction

Priority: P1
Owner: Scheduler
Effort: 2 eng-weeks
Risk: Medium
Dependencies: P0-3
Labels: scheduler, concurrency, scalability

## Problem
Single lock-domain contention (`queue_mutex_`) can become a throughput and scalability bottleneck under high concurrency.

## Scope Files
- `scheduler/scheduler.cpp`
- `scheduler/scheduler.h`
- `scheduler/fairness_controller.cpp`
- `scheduler/fairness_controller.h`
- `tests/unit/test_scheduler.cpp`
- `tests/unit/test_fairness.cpp` (or existing fairness-tagged tests)
- `docs/Architecture.md`

## Test Plan
1. Add concurrency-focused unit tests covering enqueue/dequeue/preemption interactions.
2. Add stress test with high parallel request admission to validate correctness.
3. Add optional microbenchmark for lock contention reporting.
4. Run fairness and scheduler-related test targets.

## Acceptance Checklist
- [ ] Queue synchronization is partitioned or otherwise optimized with no behavior regressions.
- [ ] Fairness and cancellation semantics remain intact.
- [ ] Contention metrics or benchmark show measurable reduction.
- [ ] Architecture docs reflect the new locking model.
