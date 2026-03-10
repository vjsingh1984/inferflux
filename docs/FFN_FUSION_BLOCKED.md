# FFN Kernel Fusion Implementation Status: BLOCKED on Down Proj Weight Layout

**Date**: March 10, 2026
**Task**: #12 - FFN kernel fusion
**Status**: ❌ BLOCKED - Cannot proceed without understanding weight layout

---

## Current Situation

I've attempted to fix the FFN fusion kernel but encountered a **fundamental blocker**: I don't understand how the down_proj weights are actually stored and accessed in Q4_K format.

### What I Know ✅

1. **Current FFN flow** (from phase timing):
   - gate+up: Fused (50ms prefill) → writes to `d_ffn_gate_`, `d_ffn_up_`
   - SiLU: Fused into down_proj (0ms)
   - down_proj: Separate kernel (69ms prefill)

2. **Matrix dimensions** (for Qwen2.5-3B):
   - gate_proj: [K=2048, N_inter=5632]
   - up_proj: [K=2048, N_inter=5632]
   - down_proj: [N_inter=5632, N_hidden=2048]

3. **Q4_K Quantization**:
   - `block_q4_k` structure: 140 bytes total
   - Stores QK_K=256 elements per block
   - Each block: scales[8], qs[128]
   - Designed for matrices where one dimension is a multiple of 256

### What I Don't Understand ❌

**Question**: How are down_proj weights stored in Q4_K format?

**Issue**: The down_proj weight matrix is [N_inter=5632, N_hidden=2048]
- 5632 is not a multiple of 256
- Q4_K expects dimension to be multiple of 256 for quantization
- How is this actually stored?

**Hypotheses** (need to verify):
1. Maybe down_proj weights are stored differently (not Q4_K?)
2. Maybe N_inter is padded to be a multiple of 256?
3. Maybe the quantization is applied differently for this layer?
4. Maybe I'm looking at the wrong weight matrix entirely?

### Current Kernel State

**File**: `kernels/fused_ffn_gemm.cuh`

**Status**:
- ✅ SwiGLU activation function - correct
- ✅ Kernel signature - correct
- ✅ Gate/Up projection logic - correct (copied from existing kernels)
- ❌ Down projection logic - **INCORRECT** (placeholder code)

**The Bug**: I'm trying to access down_weight without understanding the layout

---

## Root Cause Analysis

### Why Am I Stuck?

1. **Didn't study existing code carefully enough**: Should have traced how `TryQ8_1SiluMulGemv` actually accesses down_proj weights

2. **Made assumptions about layout**: Assumed down_weight follows same pattern as gate/up weights

3. **Didn't validate with real model**: Should have printed actual weight dimensions and storage

### What Should I Have Done Differently?

1. **Step 1**: Study existing down_proj implementation
   - Read `TryQ8_1SiluMulGemv` function in detail
   - Understand how it accesses down_weight
   - Trace the actual memory addresses and indices

2. **Step 2**: Verify weight dimensions
   - Print actual weight tensor shapes from model loading
   - Check if N_inter is padded
   - Verify block_q4_k usage for down_proj

3. **Step 3**: Start with simpler fusion
   - Maybe don't fuse all three at once
   - Start with gate+up fusion (already done)
   - Then add down_proj separately

---

## Options Forward

### Option A: Investigate Existing Code ✅ RECOMMENDED

**Action**: Spend 2-4 hours studying how down_proj actually works

**Steps**:
1. Read `TryQ8_1SiluMulGemv` implementation line-by-line
2. Trace down_proj weight loading in quantized_weight_map
3. Add debug prints to understand memory layout
4. Create a test to verify understanding

**Expected outcome**: Clear understanding of weight layout, can fix kernel

**Time estimate**: 2-4 hours

### Option B: Use Different Fusion Strategy

**Action**: Simplify to two-pass fusion

**Approach**:
- Keep gate+up as is (already fused)
- Write activated intermediate to global memory
- Read back for down_proj (current approach)
- Only save 1 kernel launch instead of 2-3

**Expected improvement**: 2-4% (less than full fusion but simpler)

**Time estimate**: 4-8 hours

### Option C: Pivot to Different Optimization

**Action**: Abandon FFN fusion, focus on concurrent throughput

**Rationale**:
- FFN fusion is proving very complex
- Concurrent throughput gap (0.50x → 0.70x) may have higher business impact
- Different problem space might be easier

**Time estimate**: Unknown (needs investigation)

### Option D: Use Existing Fused Kernel ✅ ALREADY EXISTS

**Discovery**: The current implementation already has significant fusion!
- Gate+Up: Fused via `TryQ8_1ProjectionGroup`
- SiLU+Down: Fused via `TryQ8_1SiluMulGemv`
- Only 2 kernel launches for entire FFN

**Remaining optimization**: Combine into 1 kernel launch
- Benefit: Save 1 kernel launch (~5-10 μs)
- Expected improvement: ~1-2% (minimal)

**Time estimate**: 8-16 hours (low ROI)

---

## Recommendation

### Immediate: Take a Break from FFN Fusion ⏸️

**Reasoning**:
- I've hit a fundamental blocker (weight layout unknown)
- Continuing to guess will likely produce more bugs
- Better to step back and investigate properly

### Next Steps

1. **Investigate existing code** (Option A) - 2-4 hours
   - Study `TryQ8_1SiluMulGemv` implementation
   - Understand down_proj weight access pattern
   - Document findings
   - Resume kernel implementation with correct knowledge

2. **Or pivot to concurrent throughput** (Option C)
   - Different problem, fresh start
   - Potentially higher impact (0.50x → 0.70x)
   - Avoids current blocker entirely

3. **Or document current state as "partial completion"**
   - Accept that FFN fusion is blocked for now
   - Move on to other optimizations
   - Return to it later with fresh perspective

---

## Technical Debt

### Files Created
- `kernels/fused_ffn_gemm.cuh` - Has bugs, needs complete rewrite or careful fix
- `docs/FFN_FUSION_ANALYSIS.md` - Good analysis document
- `docs/FFN_FUSION_STATUS.md` - Status document (needs update)

### Files Modified
- `native_execution_policy.h` - Added `enable_fused_ffn` flag

### Learnings
1. **Study existing code before modifying**: Should have traced through down_proj implementation first
2. **Validate assumptions early**: Should have printed weight dimensions and layout
3. **Break down complex problems**: Should have started with simpler fusion approach

---

## Decision Matrix

| Option | Time | Complexity | Success Probability | ROI |
|--------|------|------------|---------------------|-----|
| Investigate code | 2-4h | Low | High | High |
| Two-pass fusion | 4-8h | Medium | High | Medium |
| Concurrent throughput | ? | Medium-High | ? | High |
| Continue guessing | 4-8h | Very High | Low | Negative |

**Recommendation**: Stop guessing, investigate existing code (Option A)

---

## Conclusion

The FFN kernel fusion implementation is **blocked** due to lack of understanding about how down_proj weights are stored and accessed in Q4_K format.

**Current kernel**: Has placeholder code, will not work correctly

**Path forward**:
1. Investigate existing code to understand weight layout (recommended)
2. Or pivot to different optimization (concurrent throughput)
3. Return later with fresh perspective after studying code more carefully

**Status**: Waiting on decision - cannot proceed without more information

---

**Document Version**: 1.0
**Last Updated**: March 10, 2026
**Status**: BLOCKED
