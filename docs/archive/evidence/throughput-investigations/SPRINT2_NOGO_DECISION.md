# Sprint 2 NO-GO Decision: Pivot to Incremental Improvements

**Date**: March 10, 2026
**Status**: ❌ NO-GO - Template-based batch processing abandoned
**Pivot**: Implement incremental improvements (30-40% expected benefit)

---

## Executive Summary

The proof-of-concept testing revealed that the template-based batch kernel approach **does NOT provide performance benefits**. In fact, it makes things significantly worse, with performance degrading from -2% at batch size 1 to -606% at batch size 8.

**Decision**: **ABANDON** template-based batch processing. **PIVOT** to incremental improvements with better ROI.

---

## Benchmark Results

### Performance Comparison

| Batch Size | Baseline Time | Batch Time | Speedup | Change | Verdict |
|-----------|--------------|-----------|--------|--------|---------|
| **1** | 0.006 ms | 0.006 ms | 0.98x | -2% | ❌ NO |
| **2** | 0.007 ms | 0.011 ms | 0.63x | -59% | ❌ NO |
| **4** | 0.011 ms | 0.036 ms | 0.30x | -236% | ❌ NO |
| **8** | 0.017 ms | 0.122 ms | 0.14x | -606% | ❌ NO |

**Clear pattern**: Performance gets WORSE with larger batch sizes.

---

## Root Cause Analysis

### The Fundamental Flaw

The batch kernel implementation has a critical architectural error:

```cpp
// WRONG: Processes sequences sequentially
template <int BatchSize>
__global__ void fused_dequant_gemv_q4k_batched(...) {
    #pragma unroll
    for (int b = 0; b < BatchSize; ++b) {
        // Load activation for sequence b
        LoadHalfToSmem(x + b*K*blockDim.y + row*K, sx, K, tid);
        __syncthreads();

        // Compute GEMV for this sequence
        for (int blk = 0; blk < num_blocks; ++blk) {
            batch_acc[b] += ...;  // Weights REFETCHED for each sequence
        }

        __syncthreads();  // EXTRA synchronization
    }
}
```

**Problems**:
1. **Sequential processing**: Each sequence processed one after another
2. **No weight sharing**: Weights fetched from global memory for each sequence
3. **Extra overhead**: Additional `__syncthreads()` calls
4. **M× work**: Doing M full passes over weight matrix instead of 1

**Result**: Batch kernel does M× the work for zero benefit.

---

### What I Got Wrong

**My understanding**:
- Batch processing means processing multiple sequences (M dimension) cooperatively
- Template parameter controls number of sequences to process
- Weight loading can be shared across sequences

**Reality** (llama.cpp architecture):
- Batch processing means processing multiple OUTPUT ELEMENTS (N dimension) cooperatively
- Template parameter controls number of output columns to compute
- Weight loading is shared because they compute the SAME position (K) for multiple outputs
- Different sequences are handled via separate kernel invocations or different N positions

**Key insight**: llama.cpp doesn't process multiple sequences in one kernel. It processes multiple output dimensions (hidden size) for a SINGLE sequence position, then moves to the next position.

### Architectural Mismatch

| Dimension | cuda_native | llama.cpp (MMVQ) |
|-----------|-------------|-----------------|
| **N (output)** | Output dimension (4096) | `ncols_dst` (batch size, 1-8) |
| **M (batch)** | Batch size (1-16) | Different batches or positions |
| **K (input)** | Hidden size (2048) | Input dimension |
| **Per kernel** | ONE element of N per sequence | ALL of ncols_dst outputs |

**Critical difference**: llama.cpp's "batch" is the output dimension N, not the sequence batch M!

---

## Why This Changes Everything

### Original Plan (Based on Wrong Understanding)

Implement template-based batch processing:
- 6-9 weeks of development
- Architectural refactoring
- Expected benefit: 1.5-2.0x scaling improvement

**Reality**: This would require:
- Completely different kernel design (process N dimension, not M)
- Major memory layout restructuring
- Different batching paradigm (output-level, not sequence-level)
- Essentially rewriting the entire kernel system

**Risk**: HIGH complexity for uncertain benefit

---

## New Plan: Incremental Improvements

### Approach: Focus on What Works

Since the batch processing approach is fundamentally incompatible with cuda_native's architecture without a complete rewrite, we'll pursue **incremental improvements** that:

1. Build on existing strengths
2. Have clear ROI
3. Lower risk
4. Shorter time to value

### Option A: Kernel Fusion (2-3 weeks, 20-30% benefit)

**Goal**: Reduce kernel launch count by fusing operations

**Implementation**:
- Fuse RoPE + KV append + attention into single kernel
- Fuse gate + up + down projections into single kernel
- Reduce launch overhead from 5,667 → ~4,000 kernels

**Expected benefit**: 20-30% reduction in kernel launch overhead

### Option B: Memory Coalescing (1-2 weeks, 10-20% benefit)

**Goal**: Optimize memory access patterns

**Implementation**:
- Restructure weight loading for better cache utilization
- Optimize shared memory access patterns
- Reduce memory bank conflicts

**Expected benefit**: 10-20% memory bandwidth improvement

### Option C: CUDA Graph Optimization (1 week, 5-10% benefit)

**Goal**: Reduce launch overhead for common patterns

**Implementation**:
- Capture CUDA graphs for common operation sequences
- Replay graphs to reduce launch overhead
- Profile and optimize graph capture/replay

**Expected benefit**: 5-10% reduction in launch overhead

### Combined Approach (4-5 weeks, 30-40% benefit)

Implement Options A + B + C:
- Week 2-3: Kernel fusion
- Week 3-4: Memory coalescing
- Week 4: CUDA graph optimization
- Week 5: Integration and testing

**Expected outcome**: 1.3-1.4x scaling improvement (from 1.11x baseline)

---

## Updated Timeline

### Abandoned ❌
- Sprint 2 Phase 2: Architectural refactoring (6-9 weeks)
- Sprint 2 Phase 3: Dispatch implementation (1 week)
- Sprint 2 Phase 4: Integration (1-2 weeks)

### New Plan ✅
- **Week 2-3**: Kernel fusion (Option A)
- **Week 3-4**: Memory coalescing (Option B)
- **Week 4**: CUDA graph optimization (Option C)
- **Week 5**: Integration and testing

**Total**: 4-5 weeks to production (vs 6-9 weeks for abandoned approach)

---

## Key Learnings

### 1. Architecture Matters More Than Code

**Lesson**: Template-based batch processing isn't a magic bullet. It requires:
- Compatible memory layout
- Cooperative computation patterns
- Shared work across threads

**Implication**: Copying techniques without understanding the architecture doesn't work.

### 2. Batch Dimension Confusion

**Lesson**: llama.cpp's "batch size" refers to output columns (N), not sequence batch (M).

**Implication**: Need to carefully study architectural assumptions before implementing.

### 3. Proof of Concept is Essential

**Lesson**: Testing validates (or invalidates) theoretical approaches.

**Implication**: Always validate assumptions with real benchmarks before committing to major work.

### 4. Incremental Improvements Have Better ROI

**Lesson**: Three 20% improvements (60% total) beat one 100% improvement that might not work.

**Implication**: Focus on achievable wins with clear timelines.

---

## Success Criteria (Updated)

### Previous (Abandoned)

- ~~1.5x scaling with batch processing~~
- ~~Match llama.cpp kernel count (57% reduction)~~
- ~~-20% memcpy time at c=16~~

### New (Achievable)

**Combined incremental improvements**:
- Reduce kernel launches by 20-30% (fusion)
- Improve memory bandwidth by 10-20% (coalescing)
- Reduce launch overhead by 5-10% (CUDA graphs)
- **Total: 30-40% improvement in throughput**

**Expected scaling**:
- Current: 1.11x (c=1 → c=16)
- Target: 1.4x (c=1 → c=16)
- Still better than cuda_native's current 1.11x
- More achievable with lower risk

---

## Decision Rationale

### Why Abandon Template-Based Batching?

1. **Fundamental incompatibility**: Requires processing N dimension, not M
2. **No benefit**: 2-6× SLOWER, not faster
3. **High complexity**: Would require complete rewrite
4. **Better alternatives exist**: Incremental improvements with clear ROI

### Why Pivot to Incremental?

1. **Lower risk**: Each option is independent and can be validated
2. **Faster value**: 4-5 weeks vs 6-9 weeks
3. **Clear benefit**: 30-40% improvement vs. uncertain benefit
4. **Builds on strengths**: Works within existing architecture

---

## Next Steps

### Immediate: Start Kernel Fusion (Option A)

1. Identify kernel fusion opportunities
2. Implement RoPE + KV append + attention fusion
3. Implement gate + up + down projection fusion
4. Benchmark and validate

### Then: Memory Coalescing (Option B)

1. Profile current memory access patterns
2. Restructure weight loading
3. Optimize shared memory usage
4. Validate with Nsight Systems

### Finally: CUDA Graphs (Option C)

1. Identify common operation sequences
2. Capture CUDA graphs
3. Profile graph replay overhead
4. Integrate into dispatch logic

---

## Files Updated

### Task Status
- Task #6 marked complete with NO-GO decision

### Documentation

- Created: This document
- To update: `docs/cuda_native_scaling_roadmap.md`
- To update: `memory/MEMORY.md`

---

## Conclusion

**The template-based batch processing approach is fundamentally incompatible with cuda_native's current architecture**. Pursuing it would require a complete rewrite with uncertain benefits.

**The new incremental approach** provides:
- Clear 30-40% improvement target
- Lower risk (4-5 weeks vs 6-9 weeks)
- Builds on existing strengths
- Proven techniques (fusion, coalescing, CUDA graphs)

**Recommendation**: Proceed with incremental improvements (Options A+B+C) for achievable 1.3-1.4x scaling improvement.

---

## Victory in Learning

While the batch processing approach failed, we learned:
- ✅ Validation early saves months of work
- ✅ Architectural understanding is critical before implementation
- ✅ Incremental improvements often beat "magic bullets"
- ✅ Proof-of-concept testing is essential

**The investigation is still valuable** - it prevented us from pursuing a 6-9 week path that wouldn't work.
