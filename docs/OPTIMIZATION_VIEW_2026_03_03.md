# Optimization View & Architecture Analysis

**Date**: 2026-03-03
**Status**: Post-Batch Accumulation Implementation

---

## Executive Summary

Batch accumulation implementation completed successfully (+23.2% throughput, -16.3% latency). However, significant optimization opportunities remain, and backend architecture needs refactoring to support multi-architecture FlashAttention (NVIDIA Ada/Ampere/Hopper, AMD ROCm CDNA2/3).

---

## 1. Current Performance State

### Achieved Optimizations
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Throughput** | 311.3 tok/s | 383.7 tok/s | +23.2% |
| **p50 Latency** | 619.1 ms | 518.4 ms | -16.3% |
| **Batch Config** | 0ms, batch=1 | 5ms, batch=4 | Optimal |

### Remaining Bottlenecks
- **GPU Utilization**: ~15% (target: 60-80%)
- **Primary issue**: Batch sizes still too small for maximum throughput
- **Memory transfers**: Synchronous (not overlapping compute)
- **Kernel efficiency**: Using llama.cpp delegation (not native)

---

## 2. Backend Architecture Status

### Isolation Analysis Results

```
Backend Files: 33 total
  - common/: 3 files ✅
  - cuda/: 14 files
  - llama/: 2 files
  - rocm/: 2 files
  - mlx/: 8 files
  - cpu/: 4 files
```

### Cross-Dependency Assessment

**Current State**: ⚠️ PARTIAL ISOLATION

| Coupling Type | Count | Status |
|---------------|-------|--------|
| CUDA → llama.cpp | 5 | ⚠️ Needs refactoring |
| llama.cpp → CUDA | 0 | ✅ Good |
| Common module usage | 4 | ✅ Working |

**Specific Issues**:
1. `cuda_backend.h` includes `cpu/llama_backend.h` (concrete type dependency)
2. `native_cuda_executor.h` includes `cpu/llama_backend.h` (scaffolding)
3. `native_cuda_backend.h` includes `cpu/llama_backend.h` (type sharing)
4. `cuda_backend.cpp` includes `llama/llama_backend_traits.h` (trait sharing)

**Impact**: Blocks clean ROCm integration and multi-architecture FlashAttention

### ROCm Readiness: 35/100

| Component | Status |
|-----------|--------|
| ✅ Common module | EXISTS |
| ⚠️ Isolation | PARTIAL (5 cross-links) |
| ❌ Device abstraction | INCOMPLETE |
| ⚠️ Factory flexibility | PARTIAL |
| ✅ FlashAttention | EXISTS |

**Blockers**:
1. Device context abstraction incomplete
2. 5 cross-dependencies between CUDA and llama.cpp

---

## 3. Circular Dependency Analysis

**Status**: ✅ NO CIRCULAR DEPENDENCIES DETECTED

This is good news - the codebase has clean dependency chains. However, the high coupling between CUDA and llama.cpp needs to be addressed.

---

## 4. FlashAttention Multi-Architecture Readiness

| Feature | Status |
|---------|--------|
| Interface abstraction | ✅ Implemented |
| CUDA FA2 support | ❌ Not implemented |
| CUDA FA3 support | ❌ Not implemented |
| ROCm FlashAttention | ✅ Basic implementation |
| Architecture detection | ✅ Implemented |
| Kernel registry | ❌ Not implemented |

**Readiness Score**: 3/6 (50%)

---

## 5. Optimization Opportunities (Priority by ROI)

### 🚀 QUICK WINS (1-2 weeks)

#### 1. GPU Utilization Optimization
- **ROI**: 150% gain per week
- **Potential**: +200-400% throughput
- **Effort**: 2-3 days
- **Actions**:
  - Increase batch size to 32-64
  - Implement request prioritization (short requests first)
  - Add dynamic batch sizing based on queue depth

#### 2. Memory Transfer Optimization
- **ROI**: 7.5% gain per week
- **Potential**: +10-20% throughput
- **Effort**: 3-5 days
- **Actions**:
  - Use CUDA streams for async transfers
  - Allocate pinned memory for host buffers
  - Overlap compute and transfer

### ⚡ MEDIUM TERM (1-2 months)

#### 3. Native CUDA Kernels
- **ROI**: 38% gain per week
- **Potential**: +57-96% throughput
- **Effort**: 6-8 weeks
- **Actions**:
  - Implement native attention kernel
  - Optimize memory layout (SoA)
  - Implement FlashAttention-2 for Ada
  - Tune for architecture specifics

#### 4. Speculative Decoding
- **ROI**: 25% gain per week
- **Potential**: +2-3x throughput
- **Effort**: 2-3 weeks
- **Actions**:
  - Integrate draft model
  - Implement verification
  - Add token-level parallelism

#### 5. CUDA Graphs
- **ROI**: 10% gain per week
- **Potential**: +15-25% throughput
- **Effort**: 2-3 weeks
- **Actions**:
  - Capture execution graph
  - Instantiate graph for replay
  - Handle variable sequence lengths

### 🔮 LONG TERM (3-6 months)

#### 6. Multi-GPU Tensor Parallelism
- **ROI**: 25% gain per week
- **Potential**: +2.5-3x per GPU
- **Effort**: 4-6 weeks
- **Actions**:
  - Implement tensor parallel sharding
  - Add all-reduce for attention heads
  - Distribute KV cache across GPUs

---

## 6. Refactoring Roadmap

### Phase 1: Decouple CUDA from llama.cpp (HIGH PRIORITY)
**Effort**: 2-3 days

**Tasks**:
1. Extract `LlamaCPUBackend` interface to `common/`
2. Move shared types to `backend_types.h`
3. Create `ILlamaBackend` interface
4. Use dependency injection instead of direct includes

**Files to modify**:
- `runtime/backends/cuda/cuda_backend.h`
- `runtime/backends/cuda/native_cuda_executor.h`
- `runtime/backends/cuda/native_cuda_backend.h`
- `runtime/backends/cuda/cuda_backend.cpp`

**Expected outcome**: 0 cross-dependencies, clean separation

### Phase 2: Create FlashAttention Architecture Registry (HIGH PRIORITY)
**Effort**: 3-5 days

**Tasks**:
1. Create `IFlashAttention` interface in `common/`
2. Create kernel registry by compute capability
3. Add architecture detection (Ada, Ampere, Hopper, CDNA2, CDNA3)
4. Implement per-architecture kernel selection

**Files to create**:
- `runtime/backends/common/flash_attention_interface.h`
- `runtime/backends/common/kernel_registry.h`
- `runtime/backends/common/architecture.h`

**Expected outcome**: FlashAttention readiness 5/6 (83%)

### Phase 3: Implement ROCm FlashAttention Backend (MEDIUM PRIORITY)
**Effort**: 5-7 days

**Tasks**:
1. Port FA2/FA3 to HIP/ROCm
2. Implement ROCm-specific optimizations
3. Add CDNA architecture support
4. Test on AMD hardware

**Files to create**:
- `runtime/backends/rocm/rocm_flash_attention.cpp`
- `runtime/backends/rocm/rocm_flash_attention.h`

**Expected outcome**: Full AMD GPU support

### Phase 4: Add Per-Architecture Kernel Selection (MEDIUM PRIORITY)
**Effort**: 2-3 days

**Tasks**:
1. Detect GPU architecture at runtime
2. Select optimal kernel for architecture
3. Add fallback chains
4. Add metrics for kernel selection

**Expected outcome**: Automatic optimization for all GPU generations

**Total Refactoring Effort**: 12-18 days (3-4 weeks)

---

## 7. Immediate Next Steps

### Week 1: Quick Wins + Profiling
1. **Day 1-2**: Run Nsight Systems profiling
   - Capture 10s trace during workload
   - Identify top 5 kernels by time
   - Analyze memory transfer patterns

2. **Day 3-4**: Implement GPU utilization optimization
   - Increase batch size to 32
   - Add request prioritization
   - Benchmark improvements

3. **Day 5**: Profile memory transfers
   - Identify sync points
   - Plan async transfer strategy

### Week 2: Backend Decoupling
1. **Day 1-2**: Phase 1 refactoring
   - Extract LlamaCPUBackend interface
   - Remove cross-dependencies
   - Test with existing backends

2. **Day 3-5**: Phase 2 implementation
   - Create FlashAttention registry
   - Add architecture detection
   - Implement kernel selection

### Week 3-4: Medium Term Optimizations
1. Choose based on profiling results:
   - Native kernels (if kernel time dominates)
   - Speculative decoding (if decode dominates)
   - CUDA graphs (if launch overhead dominates)

---

## 8. Success Metrics

### Short-Term (2 weeks)
- [ ] GPU utilization > 50%
- [ ] Throughput > 600 tok/s (+100%)
- [ ] 0 CUDA-llama cross-dependencies
- [ ] FlashAttention registry implemented

### Medium-Term (2 months)
- [ ] GPU utilization > 70%
- [ ] Throughput > 1000 tok/s (+233%)
- [ ] Native kernels or speculative decoding
- [ ] ROCm support functional

### Long-Term (6 months)
- [ ] GPU utilization > 85%
- [ ] Throughput > 1500 tok/s (+367%)
- [ ] Multi-GPU support
- [ ] All GPU generations supported

---

## 9. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Regression from refactoring | Medium | High | Comprehensive test suite |
| ROCm compatibility issues | Medium | Medium | Early testing on AMD hardware |
| Kernel optimization takes longer | High | Medium | Focus on highest-ROI items first |
| Multi-GPU complexity | High | High | Incremental implementation |

---

## 10. Resource Requirements

### Development
- **Engineers**: 1-2 senior C++/CUDA engineers
- **Timeline**: 3-6 months for full optimization
- **Hardware**: NVIDIA GPU (for CUDA), AMD GPU (for ROCm testing)

### Tools
- **Profiling**: Nsight Systems, Nsight Compute (✅ Available)
- **Testing**: CTest, pytest (✅ Available)
- **CI/CD**: GitHub Actions (✅ Available)

---

## Conclusion

**Current State**: Solid foundation with batch accumulation implemented

**Immediate Priority**:
1. Complete backend decoupling (enables ROCm)
2. Implement FlashAttention registry (enables multi-architecture)
3. Optimize GPU utilization (highest ROI quick win)

**Long-term Vision**: Multi-architecture inference server supporting:
- NVIDIA: Ada, Ampere, Hopper (FA2/FA3)
- AMD: CDNA2, CDNA3 (ROCm)
- Apple: Metal (MPS)
- CPU: x86, ARM (optimized backends)

**Path Forward**: Execute Phase 1-2 refactoring, then proceed with high-ROI optimizations.

---

**Status**: ✅ Analysis complete, roadmap defined
**Next Action**: Begin Phase 1 refactoring (backend decoupling)
