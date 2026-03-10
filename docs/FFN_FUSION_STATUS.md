# FFN Kernel Fusion Implementation Status

**Date**: March 10, 2026
**Task**: #12 - Implement FFN kernel fusion (gate+up+SiLU+down)
**Status**: ⏳ IN PROGRESS - Proof of Concept Complete

---

## Summary

FFN kernel fusion implementation is in progress. Phase 1 (analysis and profiling) is complete. Phase 2 (implementation) has begun with initial kernel design, but requires further development and testing.

---

## Completed Work

### Phase 1: Analysis and Profiling ✅

**Profiling Results** (from `INFERFLUX_NATIVE_PHASE_TIMING=1`):

**Prefill** (tokens=16):
- `ffn_proj` (gate+up): 50.02 ms (42% of FFN time)
- `ffn_silu`: 0.00 ms ✅ **Already fused into down_proj!**
- `ffn_down`: 68.95 ms (58% of FFN time)
- **FFN total**: 118.97 ms (40% of total 299.26 ms)

**Decode** (tokens=1):
- `ffn_proj` (gate+up): ~2.0 ms (48% of FFN time)
- `ffn_silu`: 0.00 ms ✅ **Already fused!**
- `ffn_down`: ~2.1 ms (52% of FFN time)
- **FFN total**: ~4.1 ms (38% of total ~10.7 ms)

**Key Finding**: SiLU activation is already fused into the down_proj kernel via `TryQ8_1SiluMulGemv`. The remaining fusion opportunity is to combine gate+up+down into a single kernel.

### Expected Improvement (Revised)

- **Current**: gate+up (separate kernel) → down_proj (with SiLU fused)
- **Proposed**: gate+up+SiLU+down (single kernel)
- **Eliminate**: Intermediate gate/up writes to global memory
- **Expected savings**: 10-20% of FFN time = 3.8-8% overall improvement ✅

### Documentation Created

1. **`docs/FFN_FUSION_ANALYSIS.md`** - Comprehensive technical analysis
   - Current architecture breakdown
   - Memory traffic patterns
   - Implementation complexity analysis
   - Risk assessment
   - Success criteria

2. **`scripts/profile_ffn_breakdown.sh`** - Profiling script
   - Automated FFN performance profiling
   - Nsight Systems integration
   - Phase timing extraction

3. **Kernel Design Document** - This document

---

## Implementation Progress

### Files Created

1. **`runtime/backends/cuda/native/kernels/fused_ffn_gemm.cuh`** ⏳ PARTIAL
   - ✅ SwiGLU activation function
   - ✅ Kernel signature and interface
   - ✅ Basic kernel structure
   - ⏳ **Needs**: Bug fixes and optimization
   - ⏳ **Needs**: Correct down_proj weight indexing

2. **`runtime/backends/cuda/native/native_execution_policy.h`** ✅ UPDATED
   - ✅ Added `enable_fused_ffn` flag
   - ✅ Added `INFERFLUX_ENABLE_FUSED_FFN` env var parsing

### Files To Be Created

3. **Dispatch wrapper** ⏳ TODO
   - `fused_quant_gemm.cu`: Add `DispatchFusedFFN` function
   - Runtime selection based on `enable_fused_ffn` flag
   - Fallback to current implementation

4. **Integration** ⏳ TODO
   - `transformer_forward.cu`: Replace 3-stage FFN with fused kernel call
   - Handle geometry constraints (N_inter size limits)

5. **Correctness Test** ⏳ TODO
   - `tests/unit/test_fused_ffn.cu`
   - Bit-exact comparison with baseline
   - Test various geometries

6. **Performance Benchmark** ⏳ TODO
   - Isolated kernel benchmark
   - Full-model throughput measurement
   - Comparison to baseline

---

## Kernel Design Challenges

### Challenge 1: Intermediate Dimension Size ❌

**Problem**: The intermediate dimension (N_inter) is large (e.g., 5632 for Qwen2.5-3B with hidden=2048, expansion=2.75).

**Initial Approach**: Store all intermediate gate+up outputs in shared memory
- **Issue**: 5632 × 4 bytes = 22KB per intermediate value
- **Result**: Doesn't fit in shared memory (typically 48KB max)

**Revised Approach**: Stream intermediate dimensions
- Compute gate+up for one intermediate dimension at a time
- Apply SwiGLU
- Immediately multiply by down_proj weight and accumulate
- **Status**: ⏳ Implemented but needs debugging

### Challenge 2: Weight Layout Complexity ⏳

**Problem**: Three weight matrices with different layouts:
- `gate_weight`: [K × N_inter] stored as [N_inter, num_blocks]
- `up_weight`: [K × N_inter] stored as [N_inter, num_blocks]
- `down_weight`: [N_inter × N_hidden] stored as [N_hidden, num_blocks]

**Issue**: Correct indexing for down_weight when iterating through intermediate dimensions

**Status**: ⏳ Needs correction in kernel code

### Challenge 3: Thread Block Organization ⏳

**Current Design**:
- Each thread block computes one output dimension (N_hidden)
- 8 warps per block (256 threads)
- Warps stride through intermediate dimensions

**Potential Issues**:
- Load imbalance if N_inter not evenly divisible by 8
- Memory coalescing for down_weight access
- Occupancy and register pressure

**Status**: ⏳ Needs profiling and optimization

---

## Current Kernel Implementation

### File: `fused_ffn_gemm.cuh`

**Components**:
1. ✅ `SwiGLU()` - Activation function
2. ✅ `FusedFFNGemmQ4K()` - Main kernel
3. ✅ Shared memory for activation
4. ✅ Warp-level reductions
5. ⏳ Down projection indexing (needs fix)

**Kernel Signature**:
```cpp
template <typename BlockType>
__global__ void FusedFFNGemmQ4K(
    const BlockType *__restrict__ gate_weight,
    const BlockType *__restrict__ up_weight,
    const BlockType *__restrict__ down_weight,
    const half *__restrict__ activation,
    half *__restrict__ output,
    int N_inter, int N_hidden, int M, int K
);
```

**Algorithm**:
1. Load activation row into shared memory
2. For each intermediate dimension (strided across warps):
   a. Compute gate_proj(activation)
   b. Compute up_proj(activation)
   c. Warp reduction
   d. Apply SwiGLU
   e. Multiply by down_proj weight
   f. Accumulate
3. Final warp reduction
4. Write output

---

## Known Issues and Fixes Needed

### Issue 1: Down Projection Weight Indexing

**Current Code** (incorrect):
```cpp
const BlockType *down_wrow = down_weight + out_idx * num_blocks;
```

**Problem**: This assumes down_weight is indexed by output dimension first, but we need to iterate through intermediate dimensions.

**Correct Approach**:
```cpp
// For each intermediate dimension 'inter_idx':
//   We need down_weight[inter_idx][out_idx]
//   In row-major storage: down_weight[inter_idx * num_blocks_per_row + out_idx/threads_per_block]
```

**Fix Needed**: ⏳ Correct the down_weight indexing logic

### Issue 2: Quantization in Down Projection

**Current Code** (incorrect):
```cpp
const float q_val = (pair == 0) ? q_lo : q_hi;
const float d_sc = (pair == 0) ? d_sc_lo : d_sc_hi;
down_acc += (d_sc * q_val - dm_m) * activated;
```

**Problem**: Down_proj should dequantize the weight properly, not use raw quantized values.

**Fix Needed**: ⏳ Use proper dequantization for down_proj weights

### Issue 3: Activation Scaling

**Current Code** (missing):
```cpp
down_acc += (d_sc * q_val - dm_m) * activated;
```

**Problem**: The down_proj multiplies its dequantized weights by the activated intermediate value. But the activated value is a scalar for the intermediate dimension, while the quantized weights are for the K dimension.

**Fix Needed**: ⏳ Re-evaluate the down_proj computation logic

---

## Revised Kernel Design (Recommendation)

Given the complexity, I recommend a **simplified two-pass approach**:

### Pass 1: Compute Activated Intermediate (gate+up+SiLU)
- Each thread block computes all intermediate dimensions
- Writes activated intermediate to global memory
- Still saves 1 kernel launch vs current (3 → 2)

### Pass 2: Down Projection
- Reads activated intermediate from global memory
- Computes down_proj
- This is the current approach with SiLU fused

**Benefit**: Simpler, less bug-prone
**Drawback**: Still writes intermediate to global memory
**Estimated Improvement**: 2-4% (vs 3-8% for full fusion)

---

## Next Steps

### Immediate (Required for Completion)

1. **Fix Kernel Bugs** ⏳ CRITICAL
   - Correct down_proj weight indexing
   - Fix quantization logic in down projection
   - Verify activation scaling

2. **Create Correctness Test** ⏳ CRITICAL
   - Test against baseline for bit-exact match
   - Start with small geometry (K=128, N_inter=256, N_hidden=128)
   - Scale up to real model sizes

3. **Add Dispatch Logic** ⏳ REQUIRED
   - Create wrapper function in `fused_quant_gemm.cu`
   - Add runtime selection based on `enable_fused_ffn`
   - Handle fallback for unsupported geometries

4. **Integrate into Transformer Forward** ⏳ REQUIRED
   - Modify `transformer_forward.cu`
   - Replace 3-stage FFN with single kernel call
   - Ensure compatibility with existing code

### Short-term (Week 1)

5. **Debug and Optimize** ⏳
   - Use cuda-memcheck to detect memory errors
   - Profile with Nsight Compute
   - Optimize memory access patterns
   - Tune thread block configuration

6. **Create Benchmark** ⏳
   - Isolated kernel performance test
   - Full-model throughput comparison
   - Measure actual improvement

### Medium-term (Week 2)

7. **Evaluate Results** ⏳
   - If ≥ 3% improvement: Default-enable
   - If < 3%: Keep as opt-in, document findings
   - Consider alternative optimizations

---

## Timeline Estimate

| Task | Estimate | Status |
|------|----------|--------|
| Analysis and profiling | 4 hours | ✅ Complete |
| Initial kernel implementation | 6 hours | ⏳ In progress (bugs remain) |
| Debug and fix kernel | 4-8 hours | ⏳ TODO |
| Dispatch integration | 2 hours | ⏳ TODO |
| Correctness test | 2 hours | ⏳ TODO |
| Performance benchmark | 2 hours | ⏳ TODO |
| Optimization and tuning | 4-8 hours | ⏳ TODO |
| **Total remaining** | **18-32 hours** | - |

**Estimated completion**: 1-2 weeks of focused work

---

## Risk Assessment

### High Risks

1. **Kernel Correctness**: Complex quantization logic, easy to introduce bugs
   - **Mitigation**: Extensive testing, start with small geometries

2. **Performance May Not Meet Target**: Actual improvement could be < 3%
   - **Mitigation**: Profile early, be ready to pivot to alternative optimizations

3. **Limited Geometry Support**: May not work for all model sizes
   - **Mitigation**: Fallback to current implementation, document constraints

### Medium Risks

1. **Code Complexity**: Harder to maintain and debug
   - **Mitigation**: Clean documentation, opt-in initially

2. **Compatibility**: May not work with all quantization types
   - **Mitigation**: Start with Q4_K only, expand later if successful

### Low Risks

1. **Regressions**: Opt-in flag minimizes impact
   - **Mitigation**: Already implemented

2. **Build Integration**: Straightforward addition
   - **Mitigation**: Following existing patterns

---

## Alternative Approaches

If full fusion proves too complex or doesn't meet targets:

### Option A: Two-Pass Fusion (Simpler)
- Compute gate+up+SiLU in one kernel
- Write intermediate to global memory
- Compute down_proj in separate kernel (current approach)
- **Benefit**: Still reduces 1 kernel launch
- **Expected**: 2-4% improvement
- **Complexity**: Low

### Option B: Kernel Fusion for Other Operations
- Fuse Q+K+V projections in attention
- Or fuse RmsNorm with other projections
- **Expected**: 3-5% improvement
- **Complexity**: Medium
- **Status**: Not investigated

### Option C: Concurrent Throughput Focus
- Fix 0.50x gap at concurrency=4
- May have higher business impact
- **Expected**: 0.50x → 0.70x (40% improvement)
- **Complexity**: Medium-High
- **Status**: Separate investigation needed

---

## Recommendations

### Immediate Actions

1. **Complete current implementation** ⏳
   - Fix kernel bugs (down_proj indexing, quantization)
   - Create correctness test
   - Validate with small geometries first

2. **If bugs prove difficult** ⏳
   - Implement two-pass fusion (simpler)
   - Or pivot to concurrent throughput optimization

3. **Decision gate** ⏳
   - After 1 week: Evaluate progress
   - If < 50% complete: Consider pivot
   - If ≥ 50% complete: Continue to completion

### Long-term Strategy

- **FFN fusion** is one of several optimization paths
- **Vectorized loads** failed (0.55% vs 5% target)
- **Kernel fusion** has higher complexity but proven benefits in other frameworks
- **Concurrent throughput** may have higher ROI

**Recommendation**: Complete FFN fusion proof-of-concept, then evaluate whether to:
- Continue with full FFN fusion (1-2 more weeks)
- Pivot to simpler fusion approaches
- Focus on concurrent throughput gap (0.50x → 0.70x)

---

## Success Metrics

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Analysis complete | ✅ | ✅ | Done |
| Kernel implemented | ✅ | ⏳ | Has bugs |
| Correctness test | Bit-exact | ⏳ | TODO |
| Micro-kernel speedup | ≥ 5% | ⏳ | TODO |
| Full-model improvement | ≥ 3% | ⏳ | TODO |
| Code quality | Clean, documented | ⏳ | Partial |

---

## Conclusion

FFN kernel fusion implementation is **in progress** with Phase 1 complete and Phase 2 partially complete. The kernel has been designed and implemented but has bugs that need fixing. The expected benefit (3-8% overall improvement) justifies continued work, but the complexity suggests we should evaluate progress after 1 week and consider alternative approaches if needed.

**Current Status**: ⏳ 30% complete (analysis done, kernel skeleton created, bugs remain)

**Estimated Time to Completion**: 1-2 weeks

**Go/No-Go Decision**: Evaluate after fixing bugs and running initial correctness tests

---

**Document Version**: 1.0
**Last Updated**: March 10, 2026
**Status**: Work in progress
