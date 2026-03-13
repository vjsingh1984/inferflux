# Phase 3: llama.cpp vs cuda_native Kernel Architecture Analysis

**Date**: March 10, 2026
**Status**: ✅ COMPLETE
**Outcome**: Identified fundamental architectural differences enabling llama.cpp's 2.59x scaling

---

## Executive Summary

**Root cause identified**: llama.cpp uses template-based batch processing where multiple sequences are processed cooperatively within a SINGLE kernel invocation, sharing weight loads. cuda_native processes ONE sequence per kernel invocation with NO weight sharing across concurrent sequences.

---

## Kernel Architecture Comparison

### llama.cpp MMVQ (Matrix-Matrix-Vector Quantized)

**File**: `external/llama.cpp/ggml/src/ggml-cuda/mmvq.cu`

#### Template-Based Batch Processing

```cpp
// Template parameters include batch size (ncols_dst) as COMPILE-TIME constant
template <ggml_type type, int ncols_dst, bool has_fusion, bool is_multi_token_id = false>
__launch_bounds__(calc_nwarps(ncols_dst, get_device_table_id())*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q(
        const void * __restrict__ vx, const void * __restrict__ vy,
        const int32_t * __restrict__ ids,
        const ggml_cuda_mm_fusion_args_device fusion,
        float * __restrict__ dst, ...)
```

**Key characteristics**:
- **Batch size is template parameter**: `int ncols_dst` - compile-time constant
- **Adaptive warp count**: `calc_nwarps(ncols_dst, ...)` - adjusts warps based on batch size
  - Batch 1-4: 4 warps
  - Batch 5-8: 2 warps
  - Batch >8: 1 warp
- **Multiple rows per block**: `calc_rows_per_block(ncols_dst, ...)` - processes 2 rows/block for batch 2-8

#### Batch-Aware Accumulator

```cpp
// Accumulator sized for batch at compile time
float tmp[ncols_dst][rows_per_cuda_block] = {{0.0f}};

// Weight loading ONCE per kbx iteration
for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
    const int kby = kbx * (qk/QK8_1);

    // Compute ALL sequences in batch using SAME loaded weights
    #pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {  // Loop over batch
        #pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
            tmp[j][i] += vec_dot_q_cuda(
                vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
        }
    }
}
```

**Critical optimization**:
- **Weights loaded once**: `vx` pointer is same for ALL `j` (batch) iterations
- **Loop unrolling**: `#pragma unroll` allows compiler to optimize for specific batch sizes
- **Memory bandwidth amortization**: Single weight load serves multiple sequences

#### Shared Memory Aggregation

```cpp
// Aggregate results from all warps in shared memory
__shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];

if (threadIdx.y > 0) {
    #pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        #pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
            tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
        }
    }
}
__syncthreads();

// Warp-level reduction for each sequence
#pragma unroll
for (int j = 0; j < ncols_dst; ++j) {
    #pragma unroll
    for (int i = 0; i < rows_per_cuda_block; ++i) {
        #pragma unroll
        for (int l = 0; l < nwarps-1; ++l) {
            tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
        }
        tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
    }
}
```

---

### cuda_native Fused GEMV

**File**: `runtime/backends/cuda/native/kernels/fused_dequant_gemv.cuh`

#### Single-Sequence Per Kernel

```cpp
__global__ void fused_dequant_gemv_q4k(const block_q4_k *__restrict__ weight,
                                       const half *__restrict__ x,
                                       half *__restrict__ output,
                                       int N, int K) {
    extern __shared__ float sx[];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
    const int row = blockIdx.y;  // Input row index (0..M-1)

    // Load x[row] into shared memory (ONE row only)
    const half *x_row = x + row * K;
    LoadHalfToSmem(x_row, sx, K, tid);
    __syncthreads();

    if (out_idx >= N)
        return;

    // Process ONE output element (ONE sequence)
    float acc = 0.0f;

    for (int blk = 0; blk < num_blocks; ++blk) {
        const block_q4_k &b = wrow[blk];

        // Compute single output
        #pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            // ... dequantization and accumulation
            acc += (d_sc_lo * q_lo - dm_m_lo) * sx[base + lane];
            acc += (d_sc_hi * q_hi - dm_m_hi) * sx[base + 32 + lane];
        }
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (lane == 0) {
        output[row * N + out_idx] = __float2half(acc);  // ONE output
    }
}
```

**Key characteristics**:
- **Single accumulator**: `float acc = 0.0f` - ONE sequence only
- **Single output per kernel**: `output[row * N + out_idx]` - writes ONE element
- **Fixed warp count**: `kGemvWarpsPerBlock = 8` - no adaptation to batch size
- **No batch processing**: Each kernel invocation processes ONE sequence

---

## Critical Differences Summary

| Aspect | cuda_native | llama.cpp |
|--------|-------------|-----------|
| **Batch processing** | One sequence per kernel | Multiple sequences per kernel |
| **Accumulator** | Single `float acc` | Array `float tmp[ncols_dst][...]` |
| **Template parameters** | None | Batch size (`ncols_dst`) is template param |
| **Warp count** | Fixed (8 warps) | Adaptive (4/2/1 warps based on batch) |
| **Weight loading** | Per sequence | Shared across sequences (amortized) |
| **Loop structure** | Single sequence | Nested loops with batch unrolling |
| **Memory bandwidth** | Not amortized | Amortized across batch |
| **Compiler optimization** | Generic | Specialized per batch size |

---

## Why llama.cpp Scales 2.59x vs cuda_native 1.11x

### Memory Bandwidth Amortization

**llama.cpp at c=16**:
- Weights loaded ONCE per kernel iteration
- Serve 16 sequences simultaneously
- Memory bandwidth divided by 16
- Result: **-38% memcpy time** (209ms vs 339ms at c=1)

**cuda_native at c=16**:
- Weights loaded PER sequence (same kernel launched 16x)
- No memory bandwidth sharing
- Memory contention between concurrent kernels
- Result: **+3% memcpy time** (387ms vs 375ms at c=1)

### Kernel Launch Efficiency

**llama.cpp**:
- 2,464 kernel launches total
- Aggressive kernel fusion
- Batch operations in FEWER launches

**cuda_native**:
- 5,667 kernel launches total (130% more)
- More granular kernels
- Launch overhead higher

### Compile-Time Optimization

**llama.cpp**:
- Batch size is template parameter (`int ncols_dst`)
- Compiler generates specialized code for each batch size
- Loop unrolling (`#pragma unroll`) optimal for specific batch
- Register allocation optimized for batch size

**cuda_native**:
- Generic kernels (no template specialization)
- Runtime batch handling via batching wrapper
- No compile-time batch optimization

---

## Implementation Recommendations for cuda_native

### Priority 1: Template-Based Batch Kernels ⭐ HIGHEST PRIORITY

**Approach**: Add template parameter for batch size to GEMV kernels

```cpp
// Proposed: Template-based batch processing
template <int BatchSize, typename BlockType>
__global__ void fused_dequant_gemv_q4k_batched(
    const BlockType *__restrict__ weight,
    const half *__restrict__ x,  // [BatchSize][K]
    half *__restrict__ output,   // [BatchSize][N]
    int N, int K) {

    extern __shared__ float sx[];

    // Load activation for each sequence in batch
    float batch_acc[BatchSize] = {0.0f};

    // Process BatchSize sequences cooperatively
    for (int blk = 0; blk < num_blocks; ++blk) {
        // Load weights ONCE
        const BlockType &b = wrow[blk];

        // Compute for ALL sequences in batch
        #pragma unroll
        for (int b = 0; b < BatchSize; ++b) {
            const half *x_b = x + b * K;
            // ... dequantization and accumulation
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

**Implementation steps**:
1. Create template variants for batch sizes 1, 2, 4, 8, 16
2. Add dispatch logic to select specialized kernel
3. Profile to identify optimal batch sizes
4. Adaptive warp count based on batch size (optional)

### Priority 2: Adaptive Warp Configuration

**Approach**: Adjust warp count based on batch size

```cpp
constexpr int calc_nwarps(int batch_size) {
    if (batch_size <= 4) return 4;
    if (batch_size <= 8) return 2;
    return 1;
}

template <int BatchSize>
__launch_bounds__(calc_nwarps(BatchSize) * 32, 1)
__global__ void fused_dequant_gemv_q4k_batched(...) {
    constexpr int nwarps = calc_nwarps(BatchSize);
    // ... kernel implementation
}
```

**Rationale**:
- Smaller batches: More warps for better occupancy
- Larger batches: Fewer warps to reduce resource contention

### Priority 3: Kernel Fusion for Multiple Projections

**Approach**: Fuse gate/up/down projections into single kernel

**Current** (cuda_native):
- Separate kernel launches per projection layer
- Multiple weight loads per layer

**Proposed**:
- Single kernel processing multiple projections
- Weight loading amortized across projections
- Fewer kernel launches

---

## Success Criteria

### MVP: Match llama.cpp Kernel Count

- **Target**: Reduce from 5,667 to 2,464 kernel launches (57% reduction)
- **Approach**: Template-based batch kernels + projection fusion
- **Expected scaling**: 1.5x (125 tok/s @ c=16)

### Stretch: Match llama.cpp Memory Efficiency

- **Target**: Achieve -20% memcpy time at c=16 (from +3% to -20%)
- **Approach**: Weight sharing across batch in single kernel
- **Expected scaling**: 2.0x (167 tok/s @ c=16)

### Ultimate: Exceed llama.cpp Throughput

- **Target**: >277 tok/s @ c=16
- **Approach**: Custom optimizations beyond llama.cpp
- **Expected scaling**: 3.0x (250+ tok/s @ c=16)

---

## Implementation Roadmap

### Sprint 3A: Prototype Batch Kernels (1 week)

1. Create template-based batch GEMV kernel variant
2. Implement for Q4_K format (most common)
3. Test with batch sizes 1, 2, 4, 8
4. Benchmark against current implementation

### Sprint 3B: Adaptive Configuration (1 week)

1. Implement adaptive warp count
2. Add batch size dispatch logic
3. Optimize register allocation
4. Profile and tune

### Sprint 3C: Production Integration (1 week)

1. Integrate into `FusedQuantGemm` dispatch
2. Add fallback logic for unsupported batch sizes
3. Comprehensive testing
4. Documentation

---

## Files Analyzed

### llama.cpp
- `external/llama.cpp/ggml/src/ggml-cuda/mmvq.cuh` - MMVQ interface
- `external/llama.cpp/ggml/src/ggml-cuda/mmvq.cu` - MMVQ implementation
- Key functions: `mul_mat_vec_q`, `calc_nwarps`, `calc_rows_per_block`

### cuda_native
- `runtime/backends/cuda/native/kernels/fused_dequant_gemv.cuh` - GEMV kernels
- `runtime/backends/cuda/native/fused_quant_gemm.cu` - Dispatch logic
- Key functions: `fused_dequant_gemv_q4k`, `fused_dequant_gemv_q6k`

---

## Next Steps

1. ✅ Phase 1 complete: cuda_native profiling (GPU saturation confirmed)
2. ✅ Phase 2 complete: cuda_llama_cpp profiling (kernel efficiency identified)
3. ✅ Phase 3 complete: llama.cpp kernel study (template-based batch processing identified)
4. **Next**: Sprint 3 - Implement template-based batch kernels
5. **Then**: Sprint 4 - Performance validation and refinement

---

## Key Insights for Implementation

1. **Template-based batch processing is the secret**: Compile-time batch size enables optimization
2. **Weight loading amortization**: Single load serves multiple sequences
3. **Adaptive warp count**: Adjust resources based on batch size
4. **Loop unrolling**: Compiler optimizes for specific batch sizes
5. **Memory bandwidth is the bottleneck**: Sharing weight loads across batch is critical

**Implementation priority**: Template-based batch kernels > Adaptive warp config > Kernel fusion
