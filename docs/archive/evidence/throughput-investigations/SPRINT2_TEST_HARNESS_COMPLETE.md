# Sprint 2 Phase 1: Test Harness Complete ✅

**Date**: March 10, 2026
**Status**: Test harness ready for validation
**Next**: Run benchmarks to make go/no-go decision

---

## What Was Created

### 1. Standalone Benchmark ✅

**File**: `tests/unit/benchmark_batch_gemv.cu`

**Features**:
- Self-contained CUDA file (no external dependencies)
- Tests batch sizes: 1, 2, 4, 8
- Compiles with nvcc directly
- Measures execution time for baseline vs batch kernels
- Calculates speedup and percentage improvement

**Usage**:
```bash
nvcc -O3 -arch=sm_89 --std=c++17 \
    -Iruntime/backends/cuda/native/kernels \
    -o benchmark_batch_gemv \
    tests/unit/benchmark_batch_gemv.cu

./benchmark_batch_gemv
```

### 2. Google Test Suite ✅

**File**: `tests/unit/test_batch_gemv_kernel.cpp`

**Features**:
- Full correctness tests for batch sizes 1, 2, 4, 8
- Performance comparison benchmarks
- Random test data generation
- Output validation with tolerance checking
- Google Test framework integration

**Test cases**:
- `BatchSize1_Correctness` - Validates BatchSize=1 matches baseline
- `BatchSize2_Correctness` - Validates BatchSize=2 matches baseline
- `BatchSize4_Correctness` - Validates BatchSize=4 matches baseline
- `BatchSize8_Correctness` - Validates BatchSize=8 matches baseline
- `Performance_BaselineVsBatch` - Measures performance improvement

### 3. Build Script ✅

**File**: `scripts/build_and_test_batch_kernel.sh`

**Features**:
- Automatic CUDA architecture detection (Ada, Ampere, Turing)
- Compiles standalone benchmark
- Runs benchmark automatically
- Displays performance comparison

**Usage**:
```bash
./scripts/build_and_test_batch_kernel.sh
```

### 4. Profiling Script ✅

**File**: `scripts/profile_batch_kernel.sh`

**Features**:
- Profiles baseline kernel with Nsight Systems
- Profiles batch kernels (BatchSize=1,2,4,8)
- Captures GPU metrics (execution time, memory bandwidth, occupancy)
- Generates analysis document
- Exports to CSV for further analysis

**Usage**:
```bash
./scripts/profile_batch_kernel.sh
nsys-ui ./batch_kernel_profile_*/profile_baseline.nsys-rep
nsys-ui ./batch_kernel_profile_*/profile_batch2.nsys-rep
```

### 5. Documentation ✅

**File**: `tests/unit/README_BATCH_KERNEL_TESTS.md`

**Contents**:
- Quick start guide
- Usage instructions for all test variants
- Troubleshooting guide
- Performance analysis guidelines
- Go/no-go decision criteria
- Related documentation references

---

## Next Steps: Validation Phase

### Step 1: Build and Run Benchmark (5-10 minutes)

```bash
# Clone/build inferflux if needed
cd /home/vsingh/code/inferflux

# Run the test harness
./scripts/build_and_test_batch_kernel.sh
```

**Expected output**:
```
========================================
Batch GEMV Kernel Benchmark
========================================

Batch size: 1
  Output dim (N): 512
  Input dim (K): 2048
  ---------------------------
  Baseline kernel: X.XXX ms
  Batch kernel:    Y.YYY ms
  Speedup:         Z.ZZX
  Improvement:     +P.P%

[... repeats for batch sizes 2, 4, 8 ...]
```

### Step 2: Analyze Results

**Look for**:
- **Speedup > 1.0**: Batch kernel is faster
- **Improvement increases with batch size**: Should see better speedup at M=4, M=8
- **No errors or CUDA failures**: Correctness validation

### Step 3: (Optional) Profile with Nsight Systems

```bash
./scripts/profile_batch_kernel.sh
```

**Key metrics to check**:
- GPU time per kernel
- Memory bandwidth utilization
- GPU occupancy percentage
- Kernel launch overhead

---

## Go/No-Go Decision Criteria

### ✅ GO: Proceed with Full Implementation

**Trigger**: **ANY** of the following:
- BatchSize=2 shows >10% speedup
- BatchSize=4 shows >20% speedup
- BatchSize=8 shows >30% speedup
- Nsight Systems shows better memory bandwidth utilization

**Action**: Proceed to Phase 2 (Architectural Refactoring)
- 6-9 weeks of development
- Implement template-based batch processing throughout
- Expected outcome: 1.5-2.0x scaling improvement

### ❌ NO-GO: Pivot to Incremental Improvements

**Trigger**: ALL of the following:
- All batch sizes show <10% improvement (or regression)
- Performance degrades with larger batch sizes
- Nsight Systems shows memory bandwidth contention

**Action**: Pivot to incremental improvements
- Option A: Kernel fusion (2-3 weeks, 20-30% benefit)
- Option B: Memory coalescing (1-2 weeks, 10-20% benefit)
- Option C: CUDA graph optimization (1 week, 5-10% benefit)
- Combined B+C: 2-3 weeks, 15-30% benefit

---

## Decision Matrix

| Batch Size | Expected Improvement | Go if | No-Go if |
|-----------|---------------------|--------|----------|
| M=2 | >10% speedup | ✅ YES | ❌ NO |
| M=4 | >20% speedup | ✅ YES | ❌ NO |
| M=8 | >30% speedup | ✅ YES | ❌ NO |

**Note**: Even if only M=2 shows improvement but M=4/8 don't, still consider GO if M=2 shows >15-20%.

---

## Files Summary

### Created Files
1. `tests/unit/test_batch_gemv_kernel.cpp` - Google Test suite
2. `tests/unit/benchmark_batch_gemv.cu` - Standalone benchmark
3. `scripts/build_and_test_batch_kernel.sh` - Build script
4. `scripts/profile_batch_kernel.sh` - Profiling script
5. `tests/unit/README_BATCH_KERNEL_TESTS.md` - Documentation

### Modified Files
- `runtime/backends/cuda/native/kernels/fused_dequant_gemv.cuh`
  - Added `fused_dequant_gemv_q4k_batched<BatchSize>()` kernel

---

## Timeline

### Completed ✅
- Kernel prototype implementation
- Test harness creation
- Documentation

### In Progress 🔄
- Benchmark execution and validation
- Performance analysis
- Go/no-go decision

### Pending ⏳
- Phase 2: Architectural refactoring (awaiting decision)
- Phase 3: Dispatch implementation
- Phase 4: Integration and optimization

---

## Testing Checklist

Before making go/no-go decision, ensure:

- [ ] Benchmark compiles and runs successfully
- [ ] All batch sizes (1, 2, 4, 8) tested
- [ ] Correctness validated: output matches baseline
- [ ] Performance measured: timing data collected
- [ ] Nsight Systems profile analyzed (optional but recommended)
- [ ] Results documented
- [ ] Go/no-go decision made

---

## Expected Timeline

**If GO decision**:
- Week 3-4: Phase 2 (Architectural refactoring)
- Week 5: Phase 3 (Dispatch implementation)
- Week 6-7: Phase 4 (Integration and optimization)
- **Total**: 5-7 additional weeks to production

**If NO-GO decision**:
- Week 3-4: Option A (Kernel fusion)
- Week 4-5: Option B+C (Memory coalescing + CUDA graphs)
- Week 6: Integration and testing
- **Total**: 4-5 weeks to production

---

## Key Takeaways

1. **Test harness is complete and ready to validate the approach**
2. **Go/no-go decision depends on empirical results, not theory**
3. **Success criteria are clear**: 10-30% improvement depending on batch size
4. **Both paths forward have clear timelines and expected benefits**
5. **Even NO-GO provides value**: Incremental improvements still yield 15-30% scaling**

**Next action**: Run `./scripts/build_and_test_batch_kernel.sh` to validate the prototype!

---

## Documentation References

- **Implementation plan**: `docs/SPRINT2_IMPLEMENTATION_PLAN.md`
- **Sprint summary**: `docs/SPRINT2_SUMMARY.md`
- **Investigation summary**: `docs/INVESTIGATION_COMPLETE_SUMMARY.md`
- **Kernel analysis**: `docs/PHASE3_KERNEL_ANALYSIS.md`
- **Scaling roadmap**: `docs/cuda_native_scaling_roadmap.md`
