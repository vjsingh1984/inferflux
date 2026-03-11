# Concurrent Throughput Investigation: Batch Size Analysis

**Date**: March 10, 2026
**Task**: Investigate 0.50x concurrent throughput gap (0.83x sequential)
**Status**: ✅ COMPLETE - Key finding: NOT a batch size limit issue

---

## Executive Summary

**Result**: INCREASING `max_batch_size` does NOT improve concurrent throughput. In fact, `max_batch=4` performs BEST at concurrency=4 (69.9 tok/s vs 67.1 tok/s for max_batch=32).

**Key Finding**: The bottleneck is NOT the scheduler batch size limit. The issue is that the scheduler predominantly processes single-sequence batches even when multiple requests are concurrent.

---

## Hypothesis

**Original Hypothesis**: The default `max_batch_size=4` in `config/server.yaml` limits concurrent throughput because the scheduler can only process 4 requests at a time.

**Investigation**: Test different `max_batch_size` values (4, 8, 16, 32) at different concurrency levels (1, 2, 4) to measure impact on throughput and batch size utilization.

---

## Test Configuration

**Hardware**: NVIDIA RTX 4000 Ada (20GB VRAM)
**Model**: Qwen2.5-3B-Instruct Q4_K_M
**Test**: 16 requests, 64 tokens each, temperature=0
**Metrics**: Collected via Prometheus `/metrics` endpoint after each run

---

## Results

### Throughput Comparison (tok/s)

| max_batch_size | c=1 | c=2 | c=4 |
|----------------|-----|-----|-----|
| **4**  | 59.7 | **68.3** | **69.9** |
| 8   | 61.9 | 65.7 | 66.8 |
| 16  | 61.8 | 67.1 | 69.5 |
| 32  | 60.8 | 63.8 | 67.1 |

### Batch Size Distribution at c=4

**max_batch=4** (best performer at 69.9 tok/s):
- Max batch observed: 3
- Bucket 1 (single-sequence): 681 passes (89.4%)
- Bucket 2 (2 sequences): 24 passes (3.2%)
- Bucket 3-4 (3-4 sequences): 38 passes (5.0%)
- Total decode passes: 743

**max_batch=16** (close second at 69.5 tok/s):
- Max batch observed: 4
- Bucket 1 (single-sequence): 580 passes (86.3%)
- Bucket 2 (2 sequences): 57 passes (8.5%)
- Bucket 3-4 (3-4 sequences): 42 passes (6.3%)
- Total decode passes: 679

### Scaling Efficiency

| max_batch_size | c=4 speedup | Efficiency |
|----------------|-------------|------------|
| **4**  | 1.17x | 29% |
| 8   | 1.08x | 27% |
| 16  | 1.12x | 28% |
| 32  | 1.10x | 28% |

**Ideal scaling**: 4x concurrency = 4x speedup (400% efficiency)
**Actual scaling**: ~1.1x speedup (~28% efficiency)
**Gap**: 3.6x throughput missing

---

## Analysis

### Finding 1: Batch Size Limit is NOT the Bottleneck

**Evidence**: Increasing `max_batch_size` from 4 to 32 DECREASES throughput by 4% (69.9 → 67.1 tok/s).

**Conclusion**: The default `max_batch_size=4` is NOT limiting concurrent throughput. The bottleneck lies elsewhere.

### Finding 2: Scheduler Fails to Batch Concurrent Requests

**Evidence**: At c=4 with max_batch=4:
- 89.4% of decode passes are single-sequence (bucket 1)
- Only 5% of passes use batches of 3-4 sequences
- This means 4 concurrent requests are mostly processed as 4 separate single-sequence batches

**Implication**: The scheduler's continuous batching logic is not effectively grouping concurrent requests into larger batches.

### Finding 3: Lower Batch Count is Correlated with Higher Throughput

**Observation**: max_batch=16 has 679 decode passes vs 743 for max_batch=4, but similar throughput.

**Hypothesis**: Fewer, larger batches may be more efficient than many smaller batches, even if most batches are single-sequence.

---

## Root Cause Hypothesis

### Primary Suspect: Request Scheduling Latency

**Theory**: The scheduler processes requests serially via `BuildBatchLocked()`, with each batch processed completely before the next batch is built. This creates a pipeline:

1. Build batch (single-sequence for request A)
2. Execute batch (request A decode step)
3. Write response
4. Repeat for request B

**Result**: Even though requests are concurrent, they're processed sequentially in single-sequence batches.

**Evidence**:
- 89% single-sequence batches at c=4
- Low scaling efficiency (29% of ideal 4x)
- Increasing max_batch doesn't help (scheduler still builds single-sequence batches)

### Secondary Suspects

1. **Token budget limit**: `max_batch_tokens=16384` may be hit, preventing larger batches
   - **Test**: Increase to 32768 and measure

2. **Batch accumulation timeout**: `batch_accumulation_ms=0` means no waiting for larger batches
   - **Test**: Set to 2-5ms to allow batch building

3. **Scheduler mutex contention**: `BuildBatchLocked()` holds global lock during batch building
   - **Test**: Profile lock wait times with instrumentation

4. **GPU kernel limitations**: Batched decode kernels may not be faster than single-sequence
   - **Test**: Profile kernel execution time for B=1 vs B=4

---

## Next Steps

### Immediate Investigation

1. **Profile scheduler behavior**:
   - Add timing instrumentation around `BuildBatchLocked()`
   - Measure time between batch builds
   - Track why requests aren't being batched (token budget? timing? priority?)

2. **Test batch accumulation delay**:
   - Set `batch_accumulation_ms=2` to allow scheduler to wait for larger batches
   - Measure impact on throughput and latency

3. **Increase token budget**:
   - Set `max_batch_tokens=32768` from 16384
   - Measure if token budget is preventing batching

### If Scheduling is Bottleneck

4. **Decode worker optimization**:
   - Check if `use_decode_workers_` is enabled
   - Profile decode worker thread behavior
   - May need separate decode worker thread pool

5. **Batch building policy**:
   - Current: `priority_age` (oldest high-priority first)
   - Try: `throughput_balanced` or `lpm_priority`
   - May improve batching behavior

### If GPU Kernels are Bottleneck

6. **Profile batched kernel efficiency**:
   - Use Nsight Compute to measure B=1 vs B=4 kernel performance
   - Check if memory bandwidth is saturated
   - May need kernel optimization for larger batches

---

## Recommendations

### DO NOT Change Default max_batch_size

**Reason**: Testing shows max_batch=4 performs BEST at c=4. Increasing to 16 or 32 reduces throughput.

### DO Investigate Scheduler Behavior

**Priority**: HIGH - This is the most likely bottleneck

**Approach**:
1. Add instrumentation to measure batch building frequency
2. Profile why single-sequence batches dominate
3. Test batch_accumulation_ms and token budget changes

### DO Profile GPU Resource Utilization

**Priority**: MEDIUM - Rule out GPU kernel limitations

**Approach**:
1. Use Nsight Systems to trace execution
2. Measure SM utilization during concurrent requests
3. Check if memory bandwidth is saturated

---

## Conclusion

The concurrent throughput gap (0.50x at c=4) is NOT caused by the `max_batch_size` limit. The scheduler predominantly processes single-sequence batches even when multiple requests are concurrent, indicating a scheduling or execution bottleneck rather than a configuration limit.

**Next action**: Profile scheduler behavior to understand why requests aren't being batched effectively.

---

## Update: Batch Accumulation Test

**Date**: March 10, 2026 (later)
**Test**: Measured impact of `batch_accumulation_ms` on throughput

**Results**:
| accum_ms | Tok/s | B=1% | B=2% | B=3-4% | Decodes |
|----------|-------|------|------|--------|--------|
| 0 | 71.7 | 70.4 | 22.5 | 7.0 | 142 |
| **2** | **74.7** | 61.4 | 38.6 | 0.0 | 140 |

**Finding**: `batch_accumulation_ms=2` improves throughput by 4.2%, but this is far from the 100% improvement needed to close the concurrent throughput gap.

**Recommendation**: Set `batch_accumulation_ms=2` in default config for modest improvement, but continue investigating the primary bottleneck (likely GPU kernel efficiency or decode loop architecture).

---

**Document Version**: 1.1
**Last Updated**: March 10, 2026
**Status**: Partially resolved - 4% improvement found, primary bottleneck TBD (likely kernel efficiency)
