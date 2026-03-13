# Sprint 2: Template-Based Batch Kernel Implementation Plan

**Date**: March 10, 2026
**Status**: Design Complete, Prototype In Progress
**Complexity**: HIGH - Requires significant architectural refactoring

---

## Executive Summary

After deeper analysis of both cuda_native and llama.cpp architectures, implementing template-based batch processing requires substantial refactoring of cuda_native's kernel design. The current cuda_native architecture uses `blockIdx.y` for the batch dimension (M), while llama.cpp uses template specialization for the output column dimension (N/sequences).

**Recommendation**: Implement in phases with proof-of-concept validation at each step.

---

## Architectural Differences

### cuda_native Current Design

```cpp
// Current: Each kernel processes ONE output element per sequence
__global__ void fused_dequant_gemv_q4k(
    const block_q4_k *weight,  // [N][K] quantized
    const half *x,              // [M][K] activations (M=batch size)
    half *output,               // [M][N] outputs
    int N, int K) {             // N=output_dim, K=input_dim

    const int row = blockIdx.y;  // M dimension (sequence in batch)
    const int out_idx = blockIdx.x * 8 + warp_id;  // N dimension

    // Process ONE output element
    float acc = 0.0f;
    for (int blk = 0; blk < num_blocks; ++blk) {
        acc += ... * sx[...];
    }

    output[row * N + out_idx] = __float2half(acc);  // ONE output
}
```

**Characteristics**:
- `blockIdx.y` = sequence index in batch (M dimension)
- Each kernel computes ONE output element per sequence
- Batch processing via separate kernel launches per sequence
- No template specialization for batch size

### llama.cpp Design

```cpp
// llama.cpp: Template-based batch processing
template <int ncols_dst>  // Compile-time batch size!
__global__ void mul_mat_vec_q(
    const void *vx,        // [K] weights (loaded ONCE)
    const void *vy,        // [ncols_dst][K] activations
    float *dst,            // [ncols_dst] outputs
    ...) {

    // Batch-sized accumulator
    float tmp[ncols_dst][rows_per_block] = {{0.0f}};

    // Process ALL output columns (sequences) cooperatively
    for (int kbx = ...; kbx < blocks_per_row_x; kbx += ...) {
        #pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {  // ALL sequences
            tmp[j][i] += vec_dot_q_cuda(vx, &y[j*stride + kby], ...);
        }
    }
}
```

**Characteristics**:
- Template parameter `ncols_dst` = number of sequences (output columns)
- Batch-sized accumulator computes ALL sequences in ONE kernel
- Weights loaded ONCE, shared across ALL sequences
- Compile-time optimization per batch size

---

## Implementation Challenges

### Challenge 1: Different Batch Dimension Usage

| Aspect | cuda_native | llama.cpp |
|--------|-------------|-----------|
| **Batch dimension** | M (rows via blockIdx.y) | N (columns via template) |
| **Output per kernel** | 1 element per sequence | Multiple elements per sequence |
| **Batch specialization** | Runtime (loop over M) | Compile-time (template) |
| **Memory layout** | [M][K] x [N] → [M][N] | [K] x [M][K] → [M] |

**Impact**: Cannot directly template the current cuda_native design without changing the output computation pattern.

### Challenge 2: Memory Layout Differences

**cuda_native**:
- Activations: `x[M][K]` where M=batch size
- Weights: `weight[N][K]` where N=output dimension
- Output: `output[M][N]`
- Kernel processes: One element of N per invocation per M

**llama.cpp**:
- Activations: `y[M][K]` (but indexed as `[ncols_dst][K]`)
- Weights: `x[K]` (per output element, loaded once)
- Output: `dst[M]` (all outputs for batch)
- Kernel processes: ALL M outputs per invocation

**Impact**: Memory coalescing patterns are fundamentally different.

### Challenge 3: Dispatch Logic Complexity

Current cuda_native dispatch:
- Runtime batch size detection
- Generic kernel for all batch sizes
- Single kernel implementation

Required llama.cpp-style dispatch:
- Template specialization for each batch size (1, 2, 4, 8, 16)
- Compile-time optimization per variant
- Multiple kernel instantiations

---

## Implementation Strategy: Phased Approach

### Phase 1: Proof of Concept (Week 2)

**Goal**: Demonstrate template-based batch processing works

**Tasks**:
1. ✅ Create template-based batch kernel variant (COMPLETED)
   - File: `runtime/backends/cuda/native/kernels/fused_dequant_gemv.cuh`
   - Function: `fused_dequant_gemv_q4k_batched<BatchSize>()`
   - Status: Kernel code written, needs testing

2. [ ] Create simple test/benchmark
   ```cpp
   // Test with batch sizes 1, 2, 4, 8
   // Compare against current implementation
   // Validate correctness and performance
   ```

3. [ ] Profile with Nsight Systems
   - Compare kernel counts
   - Measure memory bandwidth
   - Validate weight sharing

**Expected outcome**: Proof that template-based approach improves efficiency

### Phase 2: Architectural Refactoring (Weeks 3-4)

**Goal**: Restructure kernel design for template-based batching

**Approach A: Redesign output computation** (HIGH effort, HIGH reward)

1. Change kernel to process multiple output elements per sequence
2. Template parameter for batch size (number of output elements)
3. Modify memory layout for cooperative processing

```cpp
template <int BatchSize>
__global__ void fused_dequant_gemv_q4k_redesigned(
    const block_q4_k *weight,  // [N/BatchSize][K] restructured
    const half *x,              // [M][K]
    half *output,               // [M][N]
    int N, int K) {

    // Each block processes BatchSize output elements
    const int base_out_idx = blockIdx.x * kGemvWarpsPerBlock * BatchSize;
    const int row = blockIdx.y;

    float batch_acc[BatchSize] = {0.0f};

    // Load weights ONCE for all BatchSize outputs
    for (int blk = 0; blk < num_blocks; ++blk) {
        // Same weights used for all BatchSize accumulators
        const block_q4_k &b = wrow[blk];

        #pragma unroll
        for (int b = 0; b < BatchSize; ++b) {
            // Compute for each output element
            batch_acc[b] += ...;
        }
    }

    // Write all outputs
    #pragma unroll
    for (int b = 0; b < BatchSize; ++b) {
        if (base_out_idx + b < N) {
            output[row * N + base_out_idx + b] = ...;
        }
    }
}
```

**Approach B: Batch-optimized activations** (MEDIUM effort, MEDIUM reward)

1. Restructure activation loading for better coalescing
2. Process multiple rows (sequences) more efficiently
3. Template-based shared memory optimization

```cpp
template <int BatchRows>
__global__ void fused_dequant_gemv_q4k_batch_rows(
    const block_q4_k *weight,
    const half *x,              // [BatchRows][K]
    half *output,               // [BatchRows][N]
    int N, int K) {

    // Process BatchRows sequences in one kernel
    const int base_row = blockIdx.y * BatchRows;

    // Load activations for all BatchRows cooperatively
    // Better memory coalescing
    // ...
}
```

### Phase 3: Dispatch Implementation (Week 5)

**Goal**: Add runtime dispatch for template variants

1. Create dispatch table for batch sizes
   ```cpp
   template <int BatchSize>
   void DispatchBatchedGEMV_Q4_K(...) {
       fused_dequant_gemv_q4k_batched<BatchSize>
           <<<grid, block, shmem_size, stream>>>(...);
   }

   using BatchDispatchFn = void(*)(...);
   BatchDispatchFn batch_dispatch_table[] = {
       DispatchBatchedGEMV_Q4_K<1>,
       DispatchBatchedGEMV_Q4_K<2>,
       DispatchBatchedGEMV_Q4_K<4>,
       DispatchBatchedGEMV_Q4_K<8>,
       nullptr,  // Fall back to generic for >8
   };
   ```

2. Runtime batch size detection
3. Fallback to generic kernel for unsupported sizes

### Phase 4: Integration and Optimization (Weeks 6-7)

1. Integrate into `FusedQuantGemm` dispatch
2. Optimize register allocation per batch size
3. Adaptive warp count based on batch size
4. Comprehensive testing and benchmarking

---

## Updated Implementation Plan

### Immediate Next Steps

1. **Test current batch kernel prototype**
   - Create simple benchmark
   - Verify correctness
   - Profile with Nsight Systems

2. **Validate architectural assumptions**
   - Can we actually share weights across outputs?
   - What's the optimal batch size for memory coalescing?
   - How does this interact with existing KV cache?

3. **Design memory layout reorganization**
   - Current: `weight[N][K]` × `x[M][K]` → `output[M][N]`
   - Proposed: Restructure for better batch processing
   - Trade-offs: Memory overhead vs. bandwidth savings

4. **Create detailed refactoring plan**
   - Step-by-step migration path
   - Backward compatibility strategy
   - Testing approach

---

## Complexity Assessment

### Estimated Effort

| Phase | Task | Effort | Risk | Dependency |
|-------|------|--------|------|-------------|
| 1 | Proof of concept | 3-5 days | LOW | None |
| 2A | Redesign output computation | 2-3 weeks | HIGH | Phase 1 complete |
| 2B | Batch-optimized activations | 1-2 weeks | MEDIUM | Phase 1 complete |
| 3 | Dispatch implementation | 1 week | MEDIUM | Phase 2 complete |
| 4 | Integration & optimization | 1-2 weeks | MEDIUM | Phase 3 complete |

**Total**: 6-9 weeks for full implementation

---

## Alternative: Incremental Improvements

Given the complexity, consider incremental improvements that DON'T require full redesign:

### Option A: Kernel Fusion (2-3 weeks)

Instead of template-based batching, focus on reducing kernel launch count:

1. Fuse RoPE + KV append + attention into single kernel
2. Fuse multiple projection layers (gate + up + down)
3. Reduce launch overhead without template complexity

**Expected benefit**: 20-30% reduction in kernel launches

### Option B: Memory Coalescing (1-2 weeks)

Optimize memory access patterns in current design:

1. Restructure weight loading for better cache utilization
2. Optimize shared memory usage patterns
3. Reduce memory bank conflicts

**Expected benefit**: 10-20% memory bandwidth improvement

### Option C: CUDA Graph Optimization (1 week)

Better use of CUDA graphs for repetitive patterns:

1. Capture graph for common batch sizes
2. Replay graph for reduced launch overhead
3. Profile and optimize graph capture/replay

**Expected benefit**: 5-10% launch overhead reduction

---

## Recommendation

**Proceed with Phase 1 (Proof of Concept)** to validate the approach:

1. Test the `fused_dequant_gemv_q4k_batched` kernel
2. Benchmark against current implementation
3. Profile with Nsight Systems
4. Make go/no-go decision based on results

If proof of concept shows >20% improvement, proceed with Phase 2 (Architectural Refactoring).

Otherwise, pivot to incremental improvements (Option A/B/C above) which may provide better ROI with lower risk.

---

## Next Steps

1. ✅ Kernel prototype created
2. [ ] Create test harness
3. [ ] Run correctness tests
4. [ ] Benchmark performance
5. [ ] Profile with Nsight Systems
6. [ ] Analyze results and decide on Phase 2

**Current Status**: Design complete, prototype ready for testing
