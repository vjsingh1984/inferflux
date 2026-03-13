# Incremental Improvements Analysis

**Date**: March 10, 2026
**Status**: Analysis phase for Options A, B, C from NO-GO decision
**Goal**: 30-40% throughput improvement via incremental optimizations

---

## Executive Summary

Following the NO-GO decision on template-based batch processing, this document analyzes three incremental improvement options:

- **Option A**: Kernel fusion (2-3 weeks, 20-30% benefit)
- **Option B**: Memory coalescing (1-2 weeks, 10-20% benefit)
- **Option C**: CUDA graph optimization (1 week, 5-10% benefit)

This analysis focuses on **Option A: Kernel Fusion** as the highest-ROI first step.

---

## Current Kernel Launch Analysis

### Kernel Count Breakdown

Per transformer layer (single decode step):
| Operation | Kernels | Notes |
|-----------|---------|-------|
| Q projection | 1 | Fused RmsNorm+GEMV (when applicable) |
| K projection | 1 | Fused RmsNorm+GEMV (when applicable) |
| V projection | 1 | Fused RmsNorm+GEMV (when applicable) |
| RoPE | 1 | Position encoding |
| KV append | 1 | Decode only, append to KV cache |
| FlashAttention | 1 | FlashAttention2 or FlashDecodeMultiSeq |
| O projection | 1 | GEMV |
| Gate projection | 1 | Fused RmsNorm+GEMV (when applicable) |
| Up projection | 1 | Fused RmsNorm+GEMV (when applicable) |
| SiluMul | 1 | SwiGLU activation |
| Down projection | 1 | GEMV |
| **Total per layer** | **11** | |

For Qwen2.5-3B (26 layers, 127 decode steps):
- 26 layers × 11 kernels = 286 kernels per step
- 286 kernels × 1 step (seq) = 286 kernel launches
- 286 kernels × 127 steps (full prefill+decode) = 36,322 kernel launches

Wait, this doesn't match the 5,667 kernel count from profiling. Let me recalculate...

Actually, looking at the profiling data from Phase 2:
- c=1: 5,667 kernels
- c=16: 5,667 kernels (cuda_native)

This is for a single request (one prompt completion), not 127 decode steps.

Let me estimate more accurately. For a typical completion (e.g., 512 tokens):

**Qwen2.5-3B architecture**:
- 26 transformer layers
- Hidden size: 2048
- Intermediate size: 5632 (gate/up projections)

**Per-token kernels**:
- 26 layers × 10 operations (Q,K,V,RoPE,attn,O,gate,up,SiluMul,down) = 260 kernels
- Plus: embedding (1), RMSNorm (1), LM head (1) = 263 kernels/token
- For 512 tokens: 263 × 512 = 134,656 kernels

This still doesn't match. Let me check the profiling output more carefully...

**Revised estimate** (based on actual profiling):
- 5,667 kernels for a typical benchmark run
- Likely includes prefill (multiple tokens in parallel) + decode (one token at a time)
- Average of ~100-200 tokens generated

**Key insight**: The exact kernel count matters less than the relative reduction from fusion.

---

## Option A: Kernel Fusion Opportunities

### 1. RoPE + FlashAttention Fusion

**Current implementation** (3 kernel launches):
```cpp
// Step 1: RoPE in-place
cuda_kernel::RoPE(d_q_, d_k_new_, seq_len, num_heads_, num_kv_heads_,
                  head_dim_, n_past, freq_base, stream_, rope_type);

// Step 2: Append K/V to cache
cuda_kernel::BatchedKvAppend(d_k_new_, d_v_new_, k_ap, v_ap,
                             B, kv_dim, stream_);

// Step 3: Attention computation
cuda_kernel::FlashDecodeMultiSeq(d_q_, d_k_ptrs, d_v_ptrs, d_o_,
                                 B, num_heads_, num_kv_heads_,
                                 head_dim, scale, true, stream_);
```

**Fused implementation** (2 kernel launches):
```cpp
// Step 1: RoPE + KV append (or separate)
// Step 2: Attention with RoPE applied inline
```

**Benefits**:
- Eliminate one kernel launch per layer per step
- Reduce global memory writes (RoPE output stays in registers/shared memory)
- Better cache utilization

**Challenges**:
- RoPE needs to happen before attention, but K/V need to be cached
- Can't fully fuse all three operations (KV append is needed for future steps)
- Limited benefit (~9% launch reduction)

**Expected improvement**: 5-10%

### 2. Gate + Up + Down Projection Fusion

**Current implementation** (3 kernel launches):
```cpp
// Gate projection (with fused RmsNorm+GEMV)
FusedQuantGemm(..., gate_weight, activation, gate_output, ...);

// Up projection (with fused RmsNorm+GEMV)
FusedQuantGemm(..., up_weight, activation, up_output, ...);

// SwiGLU activation
cuda_kernel::SiluMul(gate_output, up_output, activation, count, stream_);

// Down projection
FusedQuantGemm(..., down_weight, activation, output, ...);
```

**Issue**: These are already partially optimized via Q8_1 grouped kernels.

**Current optimizations**:
- Q8_1 grouped kernels process gate+up together (for M=1, seq_len=1)
- SiluMul is already fused with quantization for down projection

**Limited fusion opportunity**: The current implementation already has most of these fusions.

**Expected improvement**: 3-5% (marginal gains)

### 3. Q + K + V Projection Fusion

**Current**: 3 separate kernel launches

**Fused**: Single kernel processing all three projections

**Benefits**:
- Better memory coalescing (read activation once)
- Reduced kernel launch overhead

**Challenges**:
- Different output dimensions (Q, K, V may have different sizes)
- Complex to generalize across model architectures

**Expected improvement**: 5-10%

---

## Realistic Fusion Targets

Based on analysis, the achievable fusions are:

1. **RoPE + attention partial fusion** (5-10% improvement)
   - Apply RoPE to Q in registers before attention
   - Keep K/V append separate (needed for KV cache)
   - Moderate complexity (2-3 days)

2. **Q + K + V grouped projection** (5-10% improvement)
   - Similar to existing gate+up grouping
   - Read activation once, compute all three projections
   - Higher complexity (1 week)

**Combined Option A estimate**: 10-15% improvement (not 20-30%)

---

## Option B: Memory Coalescing (Higher ROI)

### Current Memory Access Patterns

**GEMV kernel** (v1 column-major):
```cpp
// Each warp reads 8 consecutive rows of quantized weights
// Access pattern: [row0, row1, row2, row3, row4, row5, row6, row7]
// For K=2048: 256 blocks of 8 elements each
```

**Potential optimizations**:
1. **Vectorized loads**: Use `float4` / `uint4` for weight loading
2. **Shared memory tiling**: Reduce global memory accesses
3. **Cache alignment**: Ensure 128-byte aligned accesses

**Expected improvement**: 10-20% memory bandwidth improvement

**Complexity**: Low-Medium (1-2 weeks)

**ROI**: Higher than fusion (memory bandwidth is usually the bottleneck)

---

## Option C: CUDA Graph Optimization

### Current CUDA Graph Usage

**Existing graphs**: Captured for batch sizes 1-4 in `BatchedDecode`

**Gap**: Not all operation sequences are graph-captured

**Opportunities**:
1. Capture prefill stage graphs (seq_len > 1)
2. Capture projection+activation sequences
3. Capture multi-layer graphs (aggressive)

**Expected improvement**: 5-10% launch overhead reduction

**Complexity**: Low (1 week)

---

## Recommended Implementation Order

### Phase 1: Quick Wins (Week 1)
1. **CUDA graph expansion** (Option C)
   - Capture prefill graphs
   - Graph multi-layer sequences
   - Expected: 5-10% improvement

### Phase 2: Memory Optimization (Week 2-3)
2. **Memory coalescing** (Option B)
   - Vectorized weight loads
   - Shared memory tiling
   - Expected: 10-20% improvement

### Phase 3: Advanced Fusion (Week 4-5, if needed)
3. **Selective kernel fusion** (Option A)
   - Q+K+V grouped projections
   - RoPE+attention partial fusion
   - Expected: 5-10% improvement

**Total**: 4-5 weeks, 20-40% combined improvement

---

## Next Steps

### Immediate (This Session)

1. **Profile memory bandwidth**
   - Run Nsight Compute on current kernels
   - Identify memory bottlenecks
   - Establish baseline metrics

2. **Test CUDA graph expansion**
   - Identify uncaptured operation sequences
   - Implement prefill graph capture
   - Benchmark improvement

### Week 2-3

3. **Implement memory coalescing**
   - Vectorize weight loads in GEMV kernels
   - Add shared memory tiling
   - Validate with Nsight Compute

4. **Benchmark and iterate**
   - Measure memory bandwidth improvement
   - Compare throughput gains
   - Decide if fusion is needed

---

## Success Criteria

### Minimum Viable
- 15% throughput improvement (from current 72.8 tok/s → 83.7 tok/s)
- Maintained correctness (all tests pass)
- No regression in sequential performance

### Target
- 30% throughput improvement (72.8 tok/s → 94.6 tok/s)
- 1.3-1.4x scaling at c=16 (from 1.11x)
- Memory overhead reduced by 10-20%

### Stretch Goal
- 40% throughput improvement (72.8 tok/s → 101.9 tok/s)
- Match or exceed llama.cpp at c=4 (0.50x → 0.70x+)

---

## Risk Assessment

### Low Risk
- CUDA graph expansion (well-understood technique)
- Memory coalescing (standard optimization)

### Medium Risk
- Q+K+V fusion (complexity, may not generalize)
- RoPE+attention fusion (limited benefit)

### High Risk
- Multi-layer graph capture (complex, may exceed GPU memory)
- Aggressive fusion (may hurt readability/maintainability)

---

## Conclusion

**Key finding**: Kernel fusion (Option A) has lower ROI than initially estimated.

**Recommendation**: Start with Options C and B:
1. Week 1: CUDA graph expansion (5-10% improvement)
2. Week 2-3: Memory coalescing (10-20% improvement)
3. Week 4-5: Selective fusion (5-10% improvement, if needed)

**Expected outcome**: 20-30% improvement in 2-3 weeks, 30-40% in 4-5 weeks.

**Parallel work**: Can implement Options B and C concurrently, then evaluate if Option A is needed.

---

## Related Documents

- NO-GO decision: `docs/SPRINT2_NOGO_DECISION.md`
- Kernel architecture: `docs/GEMV_KERNEL_ARCHITECTURE.md`
- Profiling data: `llama_cpp_profile_*/ANALYSIS.md`
- Implementation plan: `docs/SPRINT2_IMPLEMENTATION_PLAN.md`
