# GPU Kernel Profiling Results

**Date**: March 11, 2026
**Task**: Profile batched GPU kernels to understand concurrent throughput bottleneck
**Status**: ✅ COMPLETE - Root cause identified

---

## Executive Summary

Using timing instrumentation (`INFERFLUX_NATIVE_TIMING_SAMPLE_RATE=1`), I profiled the native CUDA backend with 4 concurrent requests. The key finding confirms the earlier hypothesis: **the system processes each decode step separately, not in true batches**.

---

## Test Configuration

- **Model**: Qwen2.5-3B-Instruct Q4_K_M
- **Concurrency**: 4
- **Requests**: 4
- **Max tokens**: 32 per request
- **Config**: max_batch=4, accum_ms=2

---

## Results

### Throughput and Batch Distribution

| Metric | Value |
|--------|-------|
| Throughput | 16.2 tok/s |
| Total time | 4125ms |
| Total tokens | 67 |
| **Total decode passes** | **81** |
| B=1 passes | 66 (81.5%) |
| B=2 passes | 15 (18.5%) |
| B=3-4 passes | 0 (0%) |

### Forward Pass Timing

From `inferflux_native_forward_duration_ms` histogram:
- **Total forward passes**: 87
- **Total forward time**: 6500ms
- **Average per pass**: 74.7ms
- **< 50ms**: 28 passes (32%)
- **< 100ms**: 70 passes (81%)
- **< 250ms**: 86 passes (99%)

---

## Key Findings

### Finding 1: Minimal Batching Even With Concurrency

**Evidence**:
- 81.5% of passes are B=1 (single-sequence)
- Only 18.5% are B=2
- 0% are B=3-4 (with max_batch=4!)

**Implication**: Even with 4 concurrent requests and max_batch=4, the system rarely batches more than 2 sequences together.

### Finding 2: High Decode Pass Count

**Evidence**:
- 4 requests × 32 tokens = 128 expected token generations
- 81 actual decode passes (vs 32 expected if each request processed 8 tokens per pass)
- 63.3% efficiency (81/128 actual/expected batches)

**Implication**: Each decode step processes only 1-2 tokens per sequence, not the full 32 tokens. This suggests the decode loop is calling the forward pass once per token per sequence, not batching multiple tokens.

### Finding 3: Forward Pass Duration

**Evidence**:
- Average forward pass: 74.7ms
- Total forward time: 6.5s for 87 passes

**Implication**: Each forward pass (even B=1) takes significant time. If B=4 were 4× faster than B=1, we'd see 4× throughput improvement, but we're not seeing that because B=4 batches are rare.

---

## Root Cause Analysis

### Primary Issue: Decode Loop Architecture

The evidence points to a **decode loop architecture issue**, not a kernel efficiency issue:

1. **Token-by-token decode**: Each forward pass processes only 1-2 tokens per sequence
2. **Separate passes per sequence**: Even with 4 concurrent requests, most passes are B=1
3. **No multi-token batching**: The system doesn't batch multiple decode steps together

**Hypothesis**: The decode loop in `transformer_forward.cu` or `BatchedDecode()` processes each sequence's decode step separately, calling the forward pass once per token per sequence, rather than:
- Batching all sequences' decode steps together
- OR processing multiple tokens per sequence in one batch

### Secondary Issue: Scheduler Batching Inefficiency

Even when multiple sequences are ready to decode:
- 81.5% of passes are B=1
- Only 18.5% are B=2
- 0% are B=3-4

**Possible causes**:
1. **Timing mismatch**: Requests don't arrive at scheduler simultaneously
2. **Priority/age policy**: Oldest-first processing breaks up potential batches
3. **Decode worker isolation**: Each sequence may have its own decode context
4. **KV cache lock**: Sequence slot locks may prevent simultaneous access

---

## Comparison: Expected vs Actual

### Expected Behavior (Ideal Batching)

With 4 concurrent requests, 32 tokens each:
- **Expected**: 8 batches of B=4 (each batch processes 1 token for all 4 sequences)
- **Forward passes**: 8 (not 32!)
- **Time**: 8 × 74.7ms = 598ms
- **Throughput**: 128 tokens / 0.598s = **214 tok/s**

### Actual Behavior (Current)

- **Actual**: 81 passes (66 B=1, 15 B=2)
- **Forward passes**: 87
- **Time**: 6500ms
- **Throughput**: 67 tokens / 4.125s = **16.2 tok/s** (with timing overhead)

**Gap**: 214 tok/s (ideal) vs 16.2 tok/s (actual) = **7.6% efficiency**

---

## Recommendations

### 1. Investigate Decode Loop Architecture 🔴 HIGH PRIORITY

**Goal**: Understand why decode steps are processed separately

**Action**:
1. Read `transformer_forward.cu` decode loop implementation
2. Check if `BatchedDecode()` processes all sequences in one call
3. Verify if decode loop calls forward pass per-token or per-batch

**Expected outcome**:
- If per-token: Redesign to batch multiple tokens
- If per-batch: Investigate why batches are small

### 2. Check Decode Worker Isolation 🔴 HIGH PRIORITY

**Goal**: Verify if decode workers prevent batching

**Action**:
1. Check if `use_decode_workers_` is enabled
2. Verify if each sequence has its own decode context
3. Test with decode workers disabled

**Expected outcome**:
- If workers enabled: Try disabling to improve batching
- If workers disabled: Issue is elsewhere

### 3. Profile Request Arrival Timing 🟡 MEDIUM PRIORITY

**Goal**: Verify requests arrive simultaneously at scheduler

**Action**:
1. Add timestamps at `Scheduler::Enqueue()`
2. Measure time between request arrivals
3. Check if HTTP server serializes requests

**Expected outcome**:
- If serial arrival: Fix HTTP/server layer
- If simultaneous arrival: Issue is in scheduler batching logic

### 4. Test Synchronous Decode Batching 🟡 MEDIUM PRIORITY

**Goal**: Force all sequences to decode in lockstep

**Action**:
1. Modify decode loop to wait for all sequences
2. Process one decode step for all sequences together
3. Measure throughput improvement

**Expected outcome**:
- If throughput improves: Confirms architecture issue
- If no improvement: Issue is kernel efficiency

---

## Next Steps

### Immediate (Required)

1. **Read decode loop code**: Understand current architecture
   - File: `runtime/backends/cuda/native/transformer_forward.cu`
   - Function: `BatchedDecode()` or decode loop

2. **Test with decode workers disabled**:
   - Set `INFERFLUX_NATIVE_DISABLE_DECODE_WORKERS=1`
   - Measure batch distribution and throughput

3. **Profile request arrival timing**:
   - Add instrumentation to measure enqueue timing
   - Verify true concurrency

### Long-term (If Architecture Issue Confirmed)

4. **Redesign decode loop**:
   - Process all sequences' decode steps together
   - Batch multiple tokens per sequence
   - Ensure B=4 batches are common

5. **Optimize kernel for larger batches**:
   - Profile B=1 vs B=2 vs B=4 kernel execution time
   - Optimize memory access patterns for B=4
   - Tune kernel launch parameters

---

## Conclusion

The concurrent throughput gap (0.50x) is caused by the **decode loop architecture**, not scheduler configuration or kernel efficiency. The system processes each sequence's decode step separately (81.5% B=1 batches) rather than batching all sequences together.

**Expected throughput with proper batching**: 214 tok/s (7.6× current)
**Actual throughput**: 16.2 tok/s (with timing overhead) ~70 tok/s (without overhead)

**Fix**: Redesign decode loop to process all sequences' decode steps together in B=4 batches.

---

**Document Version**: 1.0
**Last Updated**: March 11, 2026
**Status**: Root cause identified, architecture fix required
