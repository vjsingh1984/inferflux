# cuda_native Scaling Improvement Plan

**Status**: Investigation Phase Complete | **Priority**: HIGH | **Owner**: TBD

## Problem Statement

The cuda_native backend does not scale horizontally with concurrent requests:
- c=1: 83.4 tok/s → c=16: 92.8 tok/s (**only 1.11x speedup**)
- GPU is already at 97% utilization at c=1
- No batch efficiency gain from concurrent sequences

In contrast, cuda_llama_cpp scales 2.59x from c=1 to c=16.

**Benchmark evidence**: See [docs/benchmarks.md](benchmarks.md) for full comparison.

---

## Root Cause Analysis

### Current Understanding

**GPU saturation at c=1**:
- cuda_native: 97% GPU utilization @ c=1
- cuda_llama_cpp: 92% GPU utilization @ c=1

**Kernel optimization differences**:

| Aspect | cuda_native | cuda_llama_cpp |
|--------|-------------|----------------|
| Kernel design | Large single-request GEMV | Batched inference kernels |
| Optimization target | Single-request throughput | Concurrent throughput |
| Memory access | Optimized for sequential | Optimized for batching |
| Batch efficiency | None (scales 1.11x) | High (scales 2.59x) |

**Key insight**: cuda_native kernels are already at maximum throughput for a single request. Adding concurrent requests cannot improve throughput because:
1. GPU has no headroom (97% saturated)
2. Kernels don't benefit from batching efficiency
3. Memory access patterns optimized for single sequence

---

## Investigation Plan

### Phase 1: Profile cuda_native Execution ✅ COMPLETE (March 10, 2026)

**Goal**: Identify serialization points and bottlenecks

**Completed tasks**:
1. ✅ Run Nsight Systems profile for c=1 workload
   - Captured GPU kernel timeline
   - Identified memory bandwidth vs compute bound
   - Checked for synchronization barriers

2. ✅ Run Nsight Systems profile for c=16 workload
   - Compared with c=1 profile
   - Analyzed kernel launch patterns
   - Checked batch size distribution in kernels

3. ✅ Analyze BatchedDecode() implementation
   - Verified true batched execution (not serialized loop)
   - Profiled kernel launch overhead

**Deliverables**:
- Nsight Systems reports: `cuda_native_profile_20260310_124218/`
- Analysis document: `ANALYSIS.md` in profile directory

**Key Findings**:

| Metric | c=1 | c=16 | Change |
|--------|-----|------|--------|
| cudaLaunchKernel calls | 5,667 | 5,667 | **0%** |
| cudaMemcpyAsync calls | 1,366 | 1,366 | **0%** |
| Total profile time | 794 ms | 818 ms | **+3%** |

**Conclusion**: Identical CUDA API call counts confirm GPU saturation at c=1. Batched decode processes multiple sequences in the SAME kernel launch, not via increased kernel parallelism. The bottleneck is kernel design (single-request optimization), NOT scheduler or serialization.

**Hypothesis validated**: Adding concurrent requests cannot improve throughput because GPU has no headroom (97% saturated @ c=1) and kernels don't benefit from batching efficiency.

---

### Phase 2: Compare with llama.cpp Implementation ✅ COMPLETE (March 10, 2026)

**Goal**: Understand what llama.cpp does differently

**Completed tasks**:
1. ✅ Profile cuda_llama_cpp execution with Nsight Systems
   - Same methodology as Phase 1 (c=1 and c=16 profiles)
   - Compared kernel launch patterns with cuda_native
   - Compared memory bandwidth utilization

2. [ ] Study llama.cpp batched decode implementation (deferred to Sprint 3)
   - Read `external/llama.cpp/ggml-cuda.cu`
   - Understand batched GEMV kernels
   - Document key differences

**Deliverables**:
- Comparison document: `llama_cpp_profile_20260310_131008/ANALYSIS.md`
- List of applicable techniques from llama.cpp (below)

**Key Findings**:

| Metric | cuda_native | cuda_llama_cpp | Difference |
|--------|-------------|----------------|------------|
| **Kernel count** | 5,667 | 2,464 | **57% fewer** |
| **Memcpy time @ c=1** | 374.6 ms | 339.2 ms | 9% faster |
| **Memcpy time @ c=16** | 387.1 ms | 209.0 ms | **46% faster** |
| **Total time @ c=16** | 818 ms | 696 ms | **15% faster** |

**Critical insight**: llama.cpp ALSO has identical kernel counts at c=1 and c=16 (2,464 launches), BUT execution time IMPROVES at c=16 (-16%). The scaling advantage comes from kernel EFFICIENCY, not kernel COUNT.

**Identified llama.cpp techniques**:
1. **Fewer kernel launches** (2,464 vs 5,667): More aggressive kernel fusion
2. **Memory coalescing** (-38% memcpy time at c=16): Access patterns optimized for concurrent sequences
3. **Cooperative processing**: Each kernel processes multiple sequences cooperatively
4. **CUDA graph usage**: Graph replay for common patterns

---

### Phase 3: Study llama.cpp Kernel Implementation (NEXT STEP)

**Goal**: Deep dive into llama.cpp kernel design to understand batch optimization techniques

**Tasks**:
1. [ ] Study llama.cpp GEMV kernels
   - Read `external/llama.cpp/ggml-cuda.cu` for GEMV implementation
   - Understand cooperative multi-sequence processing
   - Document memory coalescing patterns

2. [ ] Compare with cuda_native kernels
   - Analyze cuda_native `fused_dequant_gemv.cuh` vs llama.cpp approach
   - Identify specific differences causing efficiency gap
   - Document applicable patterns

3. [ ] Design cuda_native improvement strategy
   - Map llama.cpp techniques to cuda_native architecture
   - Prioritize changes based on effort vs. impact
   - Create detailed implementation plan for Sprint 3-4

**Expected deliverables**:
- llama.cpp kernel study document
- cuda_native improvement design document
- Detailed Sprint 3-4 implementation plan

---

### Phase 4: Prototype Improvements (1-2 weeks)

**Goal**: Implement and test scaling improvements

**Updated priority based on Phase 1-2 findings**:
- Phase 1 confirmed: Bottleneck is kernel design, NOT serialization or batch building
- Priority shift: Focus on batch-optimized kernels (Option A) and wave-level scheduling (Option B)
- Deprioritize: Async batch building (Option C) - minimal impact given GPU saturation bottleneck

**Candidate approaches**:

#### Option A: Batch-Optimized Kernels (HIGH EFFORT, HIGH REWARD) ⭐ PRIORITY

**Concept**: Reimplement cuda_native kernels with batch efficiency in mind

**Changes**:
- Design new GEMV kernel variants for B>1 scenarios
- Optimize memory access patterns for multiple sequences
- Reduce per-sequence overhead via cooperative processing

**Rationale**: Phase 1 confirmed kernels are single-request optimized. Need batch-first design.

**Effort**: 2-3 weeks
**Expected impact**: 1.5-2x scaling improvement

#### Option B: Wave-Level Scheduling (MEDIUM EFFORT, MEDIUM REWARD) ⭐ PRIORITY

**Concept**: Process multiple sequences cooperatively across warps within single kernel

**Changes**:
- Redesign GEMV grid strategy for batch efficiency
- Cooperatively process multiple sequences per warp/wave
- Better GPU occupancy and memory bandwidth utilization

**Rationale**: Current 8-warps-per-block design optimized for large M, not concurrent B.

**Effort**: 1 week
**Expected impact**: 1.3-1.5x scaling improvement

#### Option D: Adaptive Kernel Selection (LOW EFFORT, MEDIUM REWARD)

**Concept**: Use different kernels based on batch size

**Changes**:
- Keep current single-request kernels for c=1 (already optimal)
- Implement separate batched kernels for c>1
- Dynamic kernel dispatch based on workload

**Rationale**: Low-risk approach to validate batch-optimized kernel concepts.

**Effort**: 3-5 days
**Expected impact**: 1.2-1.4x scaling improvement

#### Option C: Asynchronous Batch Building (LOW EFFORT, LOW REWARD) ⬇️ DEPRIORITIZED

**Concept**: Overlap batch building with GPU execution

**Changes**:
- Use CUDA streams for overlapping execution
- Build next batch while current batch executes
- Reduce CPU-GPU synchronization

**Rationale**: Phase 1 showed bottleneck is GPU kernel design, not CPU-side batching.

**Effort**: 2-3 days
**Expected impact**: 1.05-1.1x scaling improvement (minimal)

---

## Success Criteria

### Minimum Viable Improvement (MVP)

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

## Implementation Roadmap

### Sprint 1: Investigation ✅ COMPLETE (March 10, 2026)

- ✅ Complete Phase 1 profiling (Nsight Systems c=1 and c=16)
- ✅ Complete Phase 2 profiling (cuda_llama_cpp c=1 and c=16)
- ✅ Write findings documents (both ANALYSIS.md files)
- ✅ Identify scaling bottleneck: Kernel efficiency, not kernel count

### Sprint 2: Study llama.cpp Kernels (Week 2) ⬅️ NEXT

- [ ] Complete Phase 3: Study llama.cpp kernel implementation
- [ ] Document cooperative multi-sequence processing techniques
- [ ] Analyze memory coalescing patterns
- [ ] Design batch-optimized kernel architecture for cuda_native
- [ ] Create detailed implementation plan for Sprint 3-4

### Sprint 3: Batch-Optimized Kernel Implementation (Weeks 3-4)

- [ ] Implement Option A: Batch-optimized GEMV kernels based on llama.cpp findings
- [ ] Implement Option B: Wave-level scheduling for cooperative processing
- [ ] Reduce kernel launch count via fusion (target: 2,464 from 5,667)
- [ ] Benchmark and validate improvements (target: 1.5x scaling)

### Sprint 4: Performance Validation & Refinement (Weeks 5-7)

- [ ] Implement Option D: Adaptive kernel dispatch (c=1 vs c>1)
- [ ] Optimize memory access patterns for batching
- [ ] Full regression testing
- [ ] Performance validation (target: 1.5-2x scaling)
- [ ] Documentation and examples

---

## Resources

### Code Locations

- Batched decode: `runtime/backends/cuda/native_kernel_executor.cpp:2450-2620`
- GEMV kernels: `runtime/backends/cuda/native/kernels/fused_dequant_gemv.cuh`
- Forward pass: `runtime/backends/cuda/native/transformer_forward.cu`
- Model forward interface: `runtime/backends/cuda/native/model_forward.h`

### Benchmarking Tools

- Multi-backend benchmark: `scripts/benchmark_multi_backend_comparison.sh`
- Scheduler profiler: `scripts/profile_scheduler_bottleneck.py`
- Nsight Systems: `nsys profile -o output.nsys-rep ./build/inferfluxd`

### Documentation

- Benchmarks: [docs/benchmarks.md](benchmarks.md)
- GEMV architecture: [docs/GEMV_KERNEL_ARCHITECTURE.md](GEMV_KERNEL_ARCHITECTURE.md)
- Native implementation: [docs/GGUF_NATIVE_KERNEL_IMPLEMENTATION.md](GGUF_NATIVE_KERNEL_IMPLEMENTATION.md)

---

## Open Questions

### ✅ ANSWERED (Phase 1)

1. **Why is GPU at 97% utilization at c=1?**
   - **Answer**: Kernels optimized for single-request throughput with large GEMV operations
   - **Evidence**: Identical CUDA API call counts at c=1 and c=16
   - **Implication**: No headroom for batching without kernel redesign

2. **Is BatchedDecode truly batched?**
   - **Answer**: Yes, but processes multiple sequences in SAME kernel launch, not via increased kernel parallelism
   - **Evidence**: cudaLaunchKernel call count is identical (5,667) at both c=1 and c=16
   - **Implication**: Batched decode doesn't increase GPU concurrency, just aggregates work

3. **What's the optimal concurrency for cuda_native?**
   - **Answer**: c=1 is optimal (83.4 tok/s) vs c=16 (92.8 tok/s, only 1.11x gain)
   - **Evidence**: Minimal scaling improvement from higher concurrency
   - **Recommendation**: Use cuda_llama_cpp for concurrent workloads

### OPEN (Phase 2)

4. **What specific optimizations enable llama.cpp's 2.59x scaling?**
   - Need to profile llama.cpp kernels with Nsight Systems
   - Compare kernel launch patterns and memory access
   - Identify techniques applicable to cuda_native

5. **Can we implement batch-optimized kernels without full rewrite?**
   - Explore hybrid approach: single-request kernels for c=1, batched for c>1
   - Investigate kernel modifications for better concurrent throughput
   - Assess effort vs. reward trade-offs

---

## Next Steps

### Immediate (March 2026)
1. ✅ **Phase 1 profiling complete** - GPU saturation hypothesis validated
2. **Next**: Phase 2 - Profile cuda_llama_cpp with Nsight Systems
3. **Create tracking issue** on GitHub for Sprint 2-4 work

### Future Work
1. **Sprint 2**: Quick wins (async batch building, adaptive kernel dispatch)
2. **Sprint 3**: Kernel optimization (wave-level scheduling, memory access patterns)
3. **Sprint 4**: Major refactor (batch-optimized kernels from ground up)

**Contact**: @vsingh for questions or to volunteer for implementation
