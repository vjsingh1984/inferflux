# Sprint 2 Summary: Template-Based Batch Kernel Design

**Date**: March 10, 2026
**Status**: Design Complete, Complexity Discovered
**Outcome**: Architectural differences identified, implementation plan created

---

## What Was Accomplished

### 1. Template-Based Batch Kernel Prototype ✅

**File**: `runtime/backends/cuda/native/kernels/fused_dequant_gemv.cuh`

Created `fused_dequant_gemv_q4k_batched<BatchSize>()` kernel with:
- Template parameter for batch size (compile-time specialization)
- Batch-aware accumulator: `float batch_acc[BatchSize]`
- Loop unrolling for batch dimension: `#pragma unroll`
- Processes multiple sequences in single kernel invocation

**Key features**:
```cpp
template <int BatchSize>
__global__ void fused_dequant_gemv_q4k_batched(
    const block_q4_k *__restrict__ weight,
    const half *__restrict__ x,  // [BatchSize][K]
    half *__restrict__ output,   // [BatchSize][N]
    int N, int K) {

  float batch_acc[BatchSize];
  #pragma unroll
  for (int b = 0; b < BatchSize; ++b) {
    batch_acc[b] = 0.0f;
  }

  // Process each sequence, sharing weight loads
  #pragma unroll
  for (int b = 0; b < BatchSize; ++b) {
    // Load activation for sequence b
    // Compute GEMV (weights loaded once above)
    for (int blk = 0; blk < num_blocks; ++blk) {
      batch_acc[b] += ...;
    }
  }

  // Warp reduction and write all outputs
}
```

### 2. Architectural Analysis ✅

**Finding**: cuda_native and llama.cpp use FUNDAMENTALLY DIFFERENT approaches to batching:

| Aspect | cuda_native | llama.cpp |
|--------|-------------|-----------|
| **Batch dimension** | M (rows) via blockIdx.y | N (columns) via template param |
| **Output per kernel** | 1 element | Multiple elements (BatchSize) |
| **Weight sharing** | None (per sequence) | Shared across batch |
| **Batch size** | Runtime (loop over M) | Compile-time (template) |
| **Memory layout** | `x[M][K] × weight[N][K] → output[M][N]` | `weight[K] × y[BatchSize][K] → output[BatchSize]` |

### 3. Implementation Plan Created ✅

**File**: `docs/SPRINT2_IMPLEMENTATION_PLAN.md`

Comprehensive plan with:
- 4-phase implementation approach
- Complexity assessment (6-9 weeks)
- Risk analysis and mitigation
- Alternative incremental improvements

---

## Key Discovery: Architectural Mismatch

### Why This Is Complex

**cuda_native current design**:
- Uses `blockIdx.y` for the batch dimension (M)
- Each kernel computes ONE output element per sequence
- Batch processing happens at HOST level (multiple kernel launches)

**llama.cpp design**:
- Uses template parameter `ncols_dst` for batch size (N dimension)
- Each kernel computes MULTIPLE output elements per sequence
- Batch processing happens at KERNEL level (single launch)

**Critical insight**: To match llama.cpp's efficiency, cuda_native needs to:
1. Change kernel to process multiple OUTPUT elements (not just input rows)
2. Restructure memory layout for cooperative processing
3. Refactor dispatch logic for template specialization
4. Optimize register allocation per batch size

This is a **substantial architectural refactoring**, not a simple kernel addition.

---

## Implementation Complexity

### Effort Estimation

| Phase | Description | Effort | Risk |
|-------|-------------|--------|------|
| 1 | Proof of concept (test prototype) | 3-5 days | LOW |
| 2A | Redesign output computation | 2-3 weeks | HIGH |
| 2B | Batch-optimized activations | 1-2 weeks | MEDIUM |
| 3 | Dispatch implementation | 1 week | MEDIUM |
| 4 | Integration & optimization | 1-2 weeks | MEDIUM |

**Total**: 6-9 weeks for full implementation

### Technical Challenges

1. **Memory layout restructuring**
   - Current: `weight[N][K]` where N=output dimension
   - Required: Access pattern that enables weight sharing across batch
   - Impact: Major data layout changes

2. **Output computation redesign**
   - Current: One output element per kernel per sequence
   - Required: Multiple outputs per kernel per sequence
   - Impact: Core kernel logic changes

3. **Dispatch complexity**
   - Current: Generic kernel for all batch sizes
   - Required: Template specializations for batch sizes 1, 2, 4, 8, 16
   - Impact: New dispatch table, runtime selection logic

4. **Backward compatibility**
   - Must maintain existing API contracts
   - Need fallback logic for edge cases
   - Impact: Careful migration path required

---

## Alternative Approaches

Given the complexity, consider these **lower-risk, shorter-term alternatives**:

### Option A: Kernel Fusion (2-3 weeks, 20-30% benefit)

**Instead of** template-based batching, fuse multiple operations:
- RoPE + KV append + attention → single kernel
- Gate + up + down projections → single kernel
- Reduce kernel launch count (5,667 → ~4,000 target)

**Pros**:
- Lower risk (doesn't require architectural changes)
- Shorter implementation time
- Clear benefit (fewer launches)

**Cons**:
- Won't achieve full llama.cpp parity
- Still need separate optimization

### Option B: Memory Coalescing (1-2 weeks, 10-20% benefit)

**Optimize memory access patterns** in current design:
- Restructure weight loading for better cache utilization
- Optimize shared memory usage patterns
- Reduce memory bank conflicts

**Pros**:
- Incremental improvement
- Low risk
- Can combine with other optimizations

**Cons**:
- Limited benefit (memory bandwidth is only one bottleneck)
- Doesn't address kernel count issue

### Option C: CUDA Graph Optimization (1 week, 5-10% benefit)

**Better use of CUDA graphs**:
- Capture graph for common batch patterns
- Replay graph for reduced launch overhead
- Profile and optimize

**Pros**:
- Quick to implement
- Low risk
- Complementary to other approaches

**Cons**:
- Limited benefit (launch overhead is small % of total time)
- Doesn't address fundamental efficiency issues

---

## Recommendation

### Proceed with Phase 1: Proof of Concept

**Next steps**:
1. Create test harness for `fused_dequant_gemv_q4k_batched` kernel
2. Run correctness tests with batch sizes 1, 2, 4, 8
3. Benchmark against current implementation
4. Profile with Nsight Systems to validate memory bandwidth improvement

**Decision criteria**:
- **Go**: If proof of concept shows >20% improvement with reasonable complexity
- **Pivot**: If improvement is <20% or complexity is too high
  - Implement Option A (kernel fusion) instead
  - Or Option B+C (memory coalescing + CUDA graphs)

### If Proof of Concept Succeeds

**Proceed with**:
1. Phase 2: Architectural refactoring (2-3 weeks)
2. Phase 3: Dispatch implementation (1 week)
3. Phase 4: Integration and optimization (1-2 weeks)

**Expected outcome**: 1.5-2.0x scaling improvement (from 1.11x baseline)

### If Proof of Concept Fails

**Pivot to**:
- Option A: Kernel fusion for immediate 20-30% benefit
- Option B+C: Memory coalescing + CUDA graphs for 15-30% combined benefit
- Revisit template-based batching after incremental improvements

**Expected outcome**: 1.3-1.4x scaling improvement (more modest but still valuable)

---

## Files Created/Modified

### Created
- `docs/SPRINT2_IMPLEMENTATION_PLAN.md` - Detailed implementation plan
- `docs/SPRINT2_SUMMARY.md` - This summary document

### Modified
- `runtime/backends/cuda/native/kernels/fused_dequant_gemv.cuh`
  - Added `fused_dequant_gemv_q4k_batched<BatchSize>()` kernel (lines 260-345)

---

## Status Summary

### Completed ✅
- Template-based batch kernel prototype created
- Architectural analysis completed
- Implementation plan documented
- Alternative approaches identified

### In Progress 🔄
- Proof of concept testing (test harness needed)
- Performance validation
- Go/no-go decision

### Pending ⏳
- Phase 2: Architectural refactoring (awaiting PoC results)
- Phase 3: Dispatch implementation
- Phase 4: Integration and optimization

---

## Key Takeaways

1. **Template-based batch processing is valid** but requires architectural changes
2. **Current cuda_native design** uses different batch dimension than llama.cpp
3. **Full implementation is 6-9 weeks** of high-complexity work
4. **Proof of concept is essential** before committing to full refactoring
5. **Alternative approaches exist** with better ROI/risk ratios

**Next step**: Create test harness and validate prototype → Make go/no-go decision
