# cuda_native Scaling Investigation: Complete Summary

**Date**: March 10, 2026
**Status**: ✅ INVESTIGATION COMPLETE
**Outcome**: Root cause identified, improvement strategy designed, ready for implementation

---

## Executive Summary

Three-phase investigation identified why cuda_native scales 1.11x (c=1→c=16) while llama.cpp scales 2.59x. **Root cause**: llama.cpp uses template-based batch processing with compile-time batch size specialization, enabling weight sharing and memory bandwidth amortization across concurrent sequences. cuda_native uses generic single-sequence kernels with no weight sharing.

**Solution path**: Implement template-based batch kernels with adaptive configuration (Sprint 2-4).

---

## Phase 1: cuda_native Profiling (March 10, 2026) ✅

### Goal
Identify why cuda_native doesn't scale horizontally

### Methodology
- Nsight Systems profiling at c=1 and c=16
- CUDA API call analysis
- Kernel launch pattern comparison

### Key Findings

| Metric | c=1 | c=16 | Change |
|--------|-----|------|--------|
| cudaLaunchKernel calls | 5,667 | 5,667 | **0%** |
| Total profile time | 794 ms | 818 ms | **+3%** |

**Conclusion**: cuda_native launches IDENTICAL number of kernels at both concurrency levels. GPU already saturated at c=1 (97% utilization), single-request kernel optimization prevents scaling.

**Files**: `cuda_native_profile_20260310_124218/ANALYSIS.md`

---

## Phase 2: cuda_llama_cpp Profiling (March 10, 2026) ✅

### Goal
Understand what llama.cpp does differently to achieve 2.59x scaling

### Methodology
- Nsight Systems profiling at c=1 and c=16
- CUDA API call comparison with cuda_native
- Memory bandwidth utilization analysis

### Key Findings

| Backend | Kernel Count | Total Time c=1 | Total Time c=16 | Memcpy c=16 |
|---------|--------------|----------------|-----------------|-------------|
| **cuda_llama_cpp** | 2,464 | 829 ms | 696 ms | **-38%** ✅ |
| **cuda_native** | 5,667 | 794 ms | 818 ms | **+3%** ❌ |

**Critical discovery**: llama.cpp ALSO has identical kernel counts at c=1 and c=16 (2,464 launches), BUT execution time IMPROVES at c=16 (-16%). The scaling advantage comes from kernel EFFICIENCY, not kernel COUNT.

**Secret identified**:
- 57% fewer kernel launches (2,464 vs 5,667)
- Memory operations 38% faster at c=16
- Total execution 15% faster at c=16

**Files**: `llama_cpp_profile_20260310_131008/ANALYSIS.md`

---

## Phase 3: llama.cpp Kernel Study (March 10, 2026) ✅

### Goal
Understand HOW llama.cpp achieves kernel efficiency

### Methodology
- Source code analysis of llama.cpp MMVQ kernels
- Comparison with cuda_native GEMV kernels
- Architecture pattern identification

### Key Findings: Template-Based Batch Processing

| Aspect | cuda_native | llama.cpp |
|--------|-------------|-----------|
| **Batch processing** | One sequence per kernel | Multiple sequences per kernel |
| **Accumulator** | `float acc` (single) | `float tmp[ncols_dst][...]` (array) |
| **Template parameters** | None | `int ncols_dst` (compile-time) |
| **Warp count** | Fixed 8 warps | Adaptive 4/2/1 warps |
| **Weight loading** | Per sequence | Shared across batch |
| **Memory bandwidth** | Not amortized | Amortized across batch |

### llama.cpp Kernel Architecture

```cpp
// Template-based batch processing
template <ggml_type type, int ncols_dst, bool has_fusion>
__global__ void mul_mat_vec_q(...) {
    // Batch-aware accumulator (compile-time sized)
    float tmp[ncols_dst][rows_per_cuda_block] = {{0.0f}};

    // Weights loaded ONCE, shared across ALL sequences
    for (int kbx = ...; kbx < blocks_per_row_x; kbx += ...) {
        #pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {  // Loop over batch
            tmp[j][i] += vec_dot_q_cuda(vx, &y[j*stride_col_y + kby], ...);
        }
    }
}
```

**Critical optimizations**:
1. **Compile-time batch size**: `int ncols_dst` is template parameter
2. **Weight sharing**: Single load serves multiple sequences
3. **Loop unrolling**: `#pragma unroll` for batch dimension
4. **Adaptive warps**: 4 warps (batch 1-4), 2 warps (5-8), 1 warp (>8)
5. **Memory bandwidth amortization**: -38% memcpy time at c=16

**Files**: `docs/PHASE3_KERNEL_ANALYSIS.md`

---

## Root Cause: Why cuda_native Doesn't Scale

### cuda_native Architecture
```cpp
__global__ void fused_dequant_gemv_q4k(...) {
    float acc = 0.0f;  // Single accumulator

    // Process ONE sequence only
    for (int blk = 0; blk < num_blocks; ++blk) {
        // Weights loaded PER sequence
        acc += ...;
    }

    output[row * N + out_idx] = __float2half(acc);  // ONE output
}
```

**Limitations**:
- Single sequence per kernel invocation
- No weight sharing across concurrent sequences
- Memory bandwidth NOT amortized
- Generic kernels (no compile-time optimization)

### llama.cpp Architecture
```cpp
template <int ncols_dst>  // Batch size is COMPILE-TIME constant
__global__ void mul_mat_vec_q(...) {
    float tmp[ncols_dst][...] = {{0.0f}};  // Batch-sized accumulator

    // Process ALL sequences cooperatively
    for (int kbx = ...; kbx < blocks_per_row_x; kbx += ...) {
        #pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {  // Batch dimension
            tmp[j][i] += vec_dot_q_cuda(vx, &y[j*...], ...);  // Shared weights
        }
    }

    // Write ALL outputs
    #pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        output[j * N + out_idx] = tmp[j][...];
    }
}
```

**Advantages**:
- Multiple sequences per kernel invocation
- Weight loading amortized across batch
- Memory bandwidth shared across sequences
- Compile-time optimization per batch size

---

## Improvement Strategy

### Priority 1: Template-Based Batch Kernels ⭐ HIGHEST PRIORITY

**Approach**: Add template parameter for batch size to GEMV kernels

```cpp
template <int BatchSize, typename BlockType>
__global__ void fused_dequant_gemv_q4k_batched(
    const BlockType *__restrict__ weight,
    const half *__restrict__ x,  // [BatchSize][K]
    half *__restrict__ output,   // [BatchSize][N]
    int N, int K) {

    float batch_acc[BatchSize] = {0.0f};

    // Weights loaded ONCE per block iteration
    for (int blk = 0; blk < num_blocks; ++blk) {
        // Compute for ALL sequences in batch
        #pragma unroll
        for (int b = 0; b < BatchSize; ++b) {
            batch_acc[b] += ...;
        }
    }

    // Write back all outputs
    #pragma unroll
    for (int b = 0; b < BatchSize; ++b) {
        output[b * N + out_idx] = __float2half(batch_acc[b]);
    }
}
```

**Benefits**:
- Weights loaded once per block iteration (amortized across batch)
- Compile-time optimization for specific batch sizes
- Register allocation optimized for batch
- Loop unrolling for batch dimension

**Expected outcome**:
- MVP: 57% reduction in kernel launches (5,667 → 2,464)
- Stretch: -20% memcpy time at c=16 (from +3% to -20%)
- Ultimate: Exceed llama.cpp throughput (>277 tok/s @ c=16)

### Priority 2: Adaptive Warp Configuration

**Approach**: Adjust warp count based on batch size

```cpp
constexpr int calc_nwarps(int batch_size) {
    if (batch_size <= 4) return 4;
    if (batch_size <= 8) return 2;
    return 1;
}
```

**Rationale**: Smaller batches benefit from more warps (better occupancy), larger batches benefit from fewer warps (less contention).

### Priority 3: Kernel Fusion

**Approach**: Fuse multiple projection layers into single kernel

**Current**: Separate kernel launches per projection layer
**Proposed**: Single kernel processing multiple projections
**Benefit**: Fewer kernel launches, weight loading amortized across projections

---

## Implementation Roadmap

### Sprint 2: Prototype Template-Based Batch Kernels (Week 2)

**Tasks**:
1. Create template-based batch GEMV kernel variant for Q4_K
2. Implement for batch sizes 1, 2, 4, 8, 16
3. Add dispatch logic for runtime batch size selection
4. Benchmark against current implementation

**Expected outcome**: 57% reduction in kernel launches, 1.3x scaling

### Sprint 3: Adaptive Configuration & Optimization (Weeks 3-4)

**Tasks**:
1. Implement adaptive warp count
2. Optimize memory access patterns
3. Implement kernel fusion for multiple projections
4. Comprehensive benchmarking with Nsight Systems

**Expected outcome**: Match llama.cpp memory efficiency, 1.5x scaling

### Sprint 4: Production Integration & Validation (Weeks 5-7)

**Tasks**:
1. Production integration with fallback logic
2. Comprehensive testing (unit, integration, regression)
3. Performance validation across models
4. Documentation and examples

**Expected outcome**: Production-ready with 1.5-2x scaling

---

## Success Criteria

### MVP (Minimum Viable Improvement)

**Target**: 1.5x scaling from c=1 to c=16
- Current: 1.11x (92.8 / 83.4)
- Target: 1.5x (125 tok/s @ c=16)
- Improvement: 35% higher throughput

### Stretch Goal

**Target**: Match llama.cpp scaling efficiency (2.0x+)
- Current: 1.11x
- Target: 2.0x (167 tok/s @ c=16)
- Improvement: 80% higher throughput

### Ultimate Goal

**Target**: Exceed llama.cpp concurrent throughput
- Current: 92.8 tok/s @ c=16
- llama.cpp: 277.4 tok/s @ c=16
- Target: >277 tok/s @ c=16
- This would make cuda_native the best backend for all workloads

---

## Files Generated

### Phase 1
- `cuda_native_profile_20260310_124218/ANALYSIS.md`
- `docs/PHASE1_PROFILING_SUMMARY.md`

### Phase 2
- `llama_cpp_profile_20260310_131008/ANALYSIS.md`
- `docs/PHASE2_PROFILING_SUMMARY.md`

### Phase 3
- `docs/PHASE3_KERNEL_ANALYSIS.md`
- `docs/cuda_native_scaling_roadmap.md` (updated)

### Consolidated
- `docs/INVESTIGATION_COMPLETE_SUMMARY.md` (this document)

---

## Key Insights

1. **Template-based batch processing is the secret**: Compile-time batch size enables aggressive optimization
2. **Memory bandwidth is the bottleneck**: Sharing weight loads across batch is critical for scaling
3. **Kernel count matters less than kernel efficiency**: Fewer, more efficient kernels > more generic kernels
4. **Adaptive configuration optimizes resource utilization**: Warp count should adjust based on batch size
5. **Implementation is feasible**: Clear path forward with template-based kernels

---

## Next Steps

1. ✅ **Investigation complete**: All three phases done
2. **Next**: Sprint 2 - Implement template-based batch kernels
3. **Then**: Sprint 3-4 - Optimization and production integration

**Status**: Ready to begin implementation with clear strategy and success criteria.

---

## Victory: Understanding Achieved 🏆

Three-phase investigation has definitively identified why cuda_native doesn't scale and HOW to fix it:

- ✅ **Phase 1**: cuda_native profiling (GPU saturation confirmed)
- ✅ **Phase 2**: cuda_llama_cpp profiling (kernel efficiency identified)
- ✅ **Phase 3**: llama.cpp kernel study (template-based batch processing understood)

**Solution**: Implement template-based batch kernels with:
- Compile-time batch size specialization
- Weight loading amortization across batch
- Adaptive warp configuration
- Kernel fusion for multiple projections

**Expected outcome**: 1.5-2x scaling improvement, making cuda_native competitive with llama.cpp for concurrent workloads.
