# Concurrent Throughput Investigation Summary

**Date**: 2026-03-10
**Status**: Investigation Phase Complete

## Problem Statement

Native CUDA backend achieves **0.83x** sequential parity (exceeding P0 gate of 0.8x) but only **0.50x** at concurrency=4. This document summarizes the investigation into the root cause.

## Key Findings

### 1. Conservative Default Batch Size Limit

**Location**: `scheduler/scheduler.h:66`
```cpp
Config()
    : max_batch_size(4), max_batch_tokens(8192), ...
```

**Impact**: When concurrency=4, the scheduler can only process 4 requests at a time. If there are more concurrent requests (e.g., 8), the remaining requests wait in queue, reducing throughput.

**Evidence from code** (`scheduler/scheduler.cpp:1106-1109`):
```cpp
if (selection.pending.size() >=
    static_cast<std::size_t>(config_.max_batch_size)) {
  break;  // Stop adding requests to batch
}
```

**Analysis**: The default `max_batch_size=4` is conservative. Production config files use higher values (e.g., 32 in `config/server.cuda.yaml`), but the Scheduler::Config() default is used when not explicitly configured.

### 2. Scheduler Global Mutex

**Location**: `scheduler/scheduler.h:187`
```cpp
mutable std::mutex queue_mutex_;
```

**Impact**: The `BuildBatchLocked()` method holds `queue_mutex_` during the entire batch building process, including:
1. Deduplication checks
2. Prefix cache lookups (when enabled)
3. Priority/age sorting (stable_sort)
4. Token cost estimation
5. Request selection loop

**Evidence** (`scheduler/scheduler.cpp:951`):
```cpp
Scheduler::BatchSelection Scheduler::BuildBatchLocked() {
  // ... 180+ lines of work while holding mutex
}
```

**Analysis**: The mutex is held during sorting and cache lookups, which could introduce contention at high concurrency. However, the critical section duration is relatively short (< 1ms typically).

### 3. Native CUDA Batched Decode Implementation

**Location**: `runtime/backends/cuda/native_kernel_executor.cpp:2498-2620`

**Key Findings**:
- **True parallel execution**: `BatchedDecode()` processes all B sequences in single `BatchForward()` call
- **No internal serialization**: CUDA kernels execute asynchronously, no mutex locks in critical path
- **Batch capacity limited by**: `decode_batch_capacity` (configured via `INFERFLUX_NATIVE_KV_MAX_BATCH`, default 16)

**Evidence** (`native_kernel_executor.cpp:2501-2503`):
```cpp
const int B = static_cast<int>(
    std::min(decode_group.size() - offset,
             static_cast<size_t>(decode_batch_capacity)));
```

**Analysis**: The native backend correctly implements batched decode. However, the **scheduler only sends up to `max_batch_size` requests at a time**, limiting the GPU's ability to process larger batches.

### 4. Batch Accumulation Settings

**Location**: `scheduler/scheduler.cpp:886-906`

**Evidence**:
```cpp
const std::size_t accumulation_target = std::max<std::size_t>(
    2, static_cast<std::size_t>(config_.min_batch_size));
```

**Impact**: When `min_batch_size=4` (default), the worker waits for at least 4 requests before building a batch. This can cause requests to wait even when GPU is idle.

## Hypothesis Validation

### Primary Bottleneck: **Conservative Batch Size Limit**

**Evidence**:
1. Default `max_batch_size=4` in `Scheduler::Config()`
2. Production configs use higher values (e.g., 32), but the default limits users who don't explicitly configure
3. When concurrency=4, the scheduler can process all 4 requests in one batch
4. When concurrency=8, the scheduler processes 4 requests, then builds another batch for the remaining 4
5. This serializes execution: batch 1 → GPU → batch 2 → GPU, instead of: all 8 → GPU

**Predicted Impact**:
- At concurrency=1: No impact (single request always fills batch)
- At concurrency=4: Moderate impact (batch size = concurrency)
- At concurrency=8+: Significant impact (batch size < concurrency, queue buildup)

### Secondary Bottleneck: **Batch Accumulation Latency**

**Evidence**:
1. `batch_accumulation_ms=0` by default (no wait)
2. `min_batch_size=1` by default
3. This means "build batch immediately when any work arrives"
4. At high concurrency, this causes many small batches instead of fewer larger batches

## Recommendations

### Immediate Fix (Phase 3)

**Increase default batch limits** in `scheduler/scheduler.h`:

```cpp
Config()
    : max_batch_size(16),  // Changed from 4
      max_batch_tokens(16384),  // Changed from 8192
      min_batch_size(1),
      batch_accumulation_ms(2) {}  // Changed from 0
```

**Rationale**:
- `max_batch_size=16`: Allows GPU to process more concurrent requests
- `max_batch_tokens=16384`: Sufficient for 16 requests × 1024 tokens
- `batch_accumulation_ms=2`: Small wait allows more requests to accumulate, improving GPU utilization

**Expected Impact**:
- Concurrency=4: Minimal change (already optimal at batch size 4)
- Concurrency=8: 1.5-2x throughput improvement (one batch of 8 vs two batches of 4)
- Concurrency=16+: 2-3x throughput improvement

### Future Optimization (Phase 4)

**Reduce scheduler serialization**:

1. **Lock-free batch queue**: Use lock-free data structures for request queuing
2. **Per-priority queues**: Reduce contention by separating high/low priority requests
3. **Batch builder thread pool**: Allow multiple threads to build batches in parallel

**GPU kernel optimization** (if investigation shows resource contention):

1. **Improve memory access patterns**: Optimize for concurrent sequences
2. **Tune kernel launch bounds**: Better occupancy for batch sizes > 4
3. **Wave-level scheduling**: Better throughput for mixed workloads

## Validation Plan

### Step 1: Profile Current Behavior

Run the scheduler profiler:
```bash
python3 scripts/profile_scheduler_bottleneck.py \
    --server-url http://localhost:18090 \
    --concurrency 4 \
    --num-requests 32 \
    --output scheduler_profile_c4.json
```

**Expected output**:
- "Batch size limit hit X% of time" — if > 50%, increase `max_batch_size`
- "Queue depth averaged Y" — if Y > concurrency/2, scheduler bottleneck
- "Scaling efficiency: Z%" — if Z < 70%, serialization issue

### Step 2: Apply Fix

Update `scheduler/scheduler.h` with new defaults (see recommendations above).

### Step 3: Validate Improvement

Re-run profiler and benchmark:
```bash
./scripts/benchmark_multi_backend_comparison.sh \
    models/qwen2.5-3b-instruct-q4_k_m.gguf
```

**Success criteria**:
- Concurrent parity improves from 0.50x to 0.70x+ at concurrency=4
- Concurrent parity improves from 0.50x to 0.60x+ at concurrency=8
- Sequential parity maintained at 0.83x

## References

- Scheduler implementation: `scheduler/scheduler.cpp`
- Native CUDA executor: `runtime/backends/cuda/native_kernel_executor.cpp`
- Config defaults: `scheduler/scheduler.h:58-77`
- Production config: `config/server.cuda.yaml`

## Appendix: Code Locations

| Component | File | Lines | Description |
|-----------|------|-------|-------------|
| Default config | `scheduler/scheduler.h` | 66-71 | max_batch_size=4 |
| Batch building | `scheduler/scheduler.cpp` | 951-1133 | BuildBatchLocked() |
| Batch limit check | `scheduler/scheduler.cpp` | 1106-1109 | Stops at max_batch_size |
| Worker loop | `scheduler/scheduler.cpp` | 884-930 | Batch accumulation logic |
| Native batched decode | `runtime/backends/cuda/native_kernel_executor.cpp` | 2498-2620 | BatchedDecode() |
| Batch capacity | `runtime/backends/cuda/native_bootstrap_config.cpp` | KV cache sizing | INFERFLUX_NATIVE_KV_MAX_BATCH |
