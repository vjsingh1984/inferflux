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

### Phase 3: Study llama.cpp Kernel Implementation ✅ COMPLETE (March 10, 2026)

**Goal**: Deep dive into llama.cpp kernel design to understand batch optimization techniques

**Completed tasks**:
1. ✅ Study llama.cpp GEMV kernels
   - Read `external/llama.cpp/ggml/src/ggml-cuda/mmvq.cu` for GEMV implementation
   - Understood cooperative multi-sequence processing
   - Documented memory coalescing patterns

2. ✅ Compare with cuda_native kernels
   - Analyzed cuda_native `fused_dequant_gemv.cuh` vs llama.cpp approach
   - Identified specific differences causing efficiency gap
   - Documented applicable patterns

3. ✅ Design cuda_native improvement strategy
   - Mapped llama.cpp techniques to cuda_native architecture
   - Prioritized changes based on effort vs. impact
   - Created detailed implementation plan for Sprint 3-4

**Deliverables**:
- llama.cpp kernel study document: `docs/PHASE3_KERNEL_ANALYSIS.md`
- cuda_native improvement design document (included in PHASE3_KERNEL_ANALYSIS.md)
- Detailed Sprint 3-4 implementation plan

**Key Findings - Template-Based Batch Processing**:

| Aspect | cuda_native | llama.cpp |
|--------|-------------|-----------|
| **Batch processing** | One sequence per kernel | Multiple sequences per kernel |
| **Accumulator** | Single `float acc` | Array `float tmp[ncols_dst][...]` |
| **Template parameters** | None | Batch size (`int ncols_dst`) is template param |
| **Warp count** | Fixed (8 warps) | Adaptive (4/2/1 warps based on batch) |
| **Weight loading** | Per sequence | Shared across sequences (amortized) |
| **Loop structure** | Single sequence | Nested loops with batch unrolling |
| **Memory bandwidth** | Not amortized | Amortized across batch |

**Root cause identified**: llama.cpp uses template-based batch processing where:
- Batch size is a compile-time template parameter (`int ncols_dst`)
- Accumulator sized for batch: `float tmp[ncols_dst][rows_per_cuda_block]`
- Weights loaded ONCE per iteration, shared across ALL sequences
- Loop unrolling: `#pragma unroll for (int j = 0; j < ncols_dst; ++j)`
- Result: Memory bandwidth amortized across batch (-38% memcpy time at c=16)

**Implementation priorities**:
1. ⭐ **Template-based batch kernels** (HIGH effort, HIGH reward)
2. **Adaptive warp configuration** (MEDIUM effort, MEDIUM reward)
3. **Kernel fusion for multiple projections** (HIGH effort, MEDIUM reward)

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
- ✅ Complete Phase 3: Study llama.cpp kernel implementation
- ✅ Identify scaling bottleneck: Template-based batch processing
- ✅ Design improvement strategy with clear priorities

### Sprint 2: Prototype Template-Based Batch Kernels (Week 2) ⬅️ NEXT

**Goal**: Implement llama.cpp-style template-based batch processing

**Tasks**:
1. Create template-based batch GEMV kernel variant for Q4_K
   - Add template parameter for batch size: `template <int BatchSize>`
   - Implement batch-aware accumulator: `float batch_acc[BatchSize]`
   - Weight loading ONCE per block iteration
   - Loop unrolling for batch dimension

2. Implement for batch sizes 1, 2, 4, 8, 16
   - Specialized kernels for each batch size
   - Compile-time optimization via template instantiation
   - Register allocation optimized per batch size

3. Add dispatch logic
   - Runtime batch size detection
   - Select appropriate specialized kernel
   - Fallback to current implementation for unsupported sizes

4. Benchmark and validate
   - Compare with current implementation
   - Profile memory bandwidth utilization
   - Measure kernel launch count reduction

**Expected outcome**: 57% reduction in kernel launches (5,667 → 2,464), 1.3x scaling improvement

### Sprint 3: Adaptive Configuration & Optimization (Weeks 3-4)

**Goal**: Implement adaptive warp count and optimize for throughput

**Tasks**:
1. Implement adaptive warp count
   - 4 warps for batch 1-4
   - 2 warps for batch 5-8
   - 1 warp for batch >8

2. Optimize memory access patterns
   - Shared memory allocation per batch
   - Coalesced weight loading across batch
   - Reduce memory bandwidth contention

3. Implement kernel fusion for multiple projections
   - Fuse gate/up/down projections
   - Reduce kernel launch count
   - Amortize weight loading

4. Comprehensive benchmarking
   - Profile with Nsight Systems
   - Compare with llama.cpp baseline
   - Identify remaining bottlenecks

**Expected outcome**: Match llama.cpp memory efficiency (-20% memcpy time at c=16), 1.5x scaling

### Sprint 4: Production Integration & Validation (Weeks 5-7)

**Goal**: Production-ready implementation with comprehensive testing

**Tasks**:
1. Production integration
   - Integrate into `FusedQuantGemm` dispatch
   - Add fallback logic for edge cases
   - Update documentation

2. Comprehensive testing
   - Unit tests for new kernel variants
   - Integration tests with real models
   - Regression tests for existing functionality

3. Performance validation
   - Benchmark across models (TinyLlama, Qwen2.5-3B, etc.)
   - Profile at concurrency levels 1, 2, 4, 8, 16
   - Compare with llama.cpp and cuda_native baseline

4. Documentation and examples
   - Kernel architecture documentation
   - Performance tuning guide
   - Backend selection recommendations

**Expected outcome**: Production-ready implementation with 1.5-2x scaling, comprehensive testing, documentation

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
