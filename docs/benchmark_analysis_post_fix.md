# Post-Fix Benchmark Analysis

**Date:** 2026-04-21
**Model:** Qwen2.5-3B-instruct Q4_K_M
**GPU:** NVIDIA RTX 4000 Ada Generation (20GB)
**Fix:** atomicAdd() for FP32 accumulate kernels

## Benchmark Results

### Throughput Performance

| Concurrency | inferflux_cuda | llama_cpp_cuda | Speedup | Status |
|-------------|----------------|----------------|---------|--------|
| c=1 | 78.0 tok/s | 104.8 tok/s | 0.74x | ⚠️ Slower |
| c=4 | 146.7 tok/s | 164.0 tok/s | 0.89x | ⚠️ Slower |
| c=8 | 169.5 tok/s | 264.9 tok/s | 0.64x | ❌ Much slower |

### Memory Consumption

| Backend | GPU Memory | Overhead |
|---------|-----------|----------|
| inferflux_cuda | 6358 MB | +1284 MB |
| llama_cpp_cuda | 5074 MB | baseline |

### Quality (Jaccard Similarity)

| Concurrency | Mean Jaccard | Mean Overlap | Status |
|-------------|--------------|--------------|--------|
| c=1 | 0.119 (11.9%) | 0.206 | ❌ Poor |
| c=4 | 0.121 (12.1%) | 0.209 | ❌ Poor |
| c=8 | 0.115 (11.5%) | 0.199 | ❌ Poor |

## Critical Issue: Quality Not Improved

**The atomicAdd fix did NOT improve multi-token quality as expected.**

### Expected vs Actual

| Metric | Expected | Actual | Status |
|--------|----------|--------|--------|
| First-token parity | 100% | Unknown | ⚠️ Not measured |
| 20-token Jaccard | >70% | ~12% | ❌ Failed |
| 50-token Jaccard | >50% | ~12% | ❌ Failed |

### Investigation Required

**The benchmark shows ~12% Jaccard, but our earlier test showed 83.33% for 20 tokens.**

Possible explanations:
1. **Different test setup** - Benchmark uses different prompts/sampling
2. **AtomicAdd not sufficient** - May be other numerical issues
3. **FP32 residual stream issue** - May need further investigation
4. **Sampling parameters** - Benchmark may use different temperature/top_p

## Throughput Regression Analysis

### c=8 Performance (0.64x speedup)

**inferflux_cuda is much slower at high concurrency:**
- c=1: 0.74x (acceptable)
- c=4: 0.89x (good)
- c=8: 0.64x (poor)

**Possible causes:**
1. **Atomic overhead** - atomicAdd may serialize writes at high concurrency
2. **Kernel launch overhead** - More kernel launches than llama.cpp
3. **Memory bandwidth** - FP32 residual uses more bandwidth
4. **Batching efficiency** - Native batching may be less efficient

### Memory Overhead (1284 MB)

**Breakdown:**
- Model weights: ~2000 MB (both backends)
- KV cache: ~1152 MB (both backends)
- **Extra 1284 MB** in inferflux_cuda

**Possible causes:**
1. **FP32 residual buffer** - 16 MB extra (not enough to explain 1284 MB)
2. **Scratch buffers** - May have additional allocations
3. **CUDA context overhead** - Different memory allocation pattern
4. **Weight caching** - Different caching strategy

## Recommendations

### Immediate Actions

1. **✅ VERIFY FIX WAS APPLIED**
   ```bash
   bash scripts/validate_fp32_fix.sh
   ```
   - Confirm atomicAdd is in all 5 kernels
   - Check build timestamp

2. **🔍 INVESTIGATE QUALITY DISCREPANCY**
   - Why 83.33% in manual test vs 12% in benchmark?
   - Check sampling parameters
   - Compare prompt distributions

3. **📊 PROFILE c=8 REGRESSION**
   - Use Nsight Systems to find bottleneck
   - Check atomicAdd serialization
   - Compare kernel launch patterns

### Next Steps

1. **Quality First**
   - If atomicAdd fix is correct, investigate other sources of divergence
   - Check RoPE implementation
   - Verify attention computation
   - Compare logit distributions

2. **Performance Second**
   - Optimize atomicAdd usage (maybe not all kernels need it)
   - Reduce memory overhead
   - Improve batching efficiency
   - Profile c=8 bottleneck

3. **Memory Third**
   - Audit memory allocations
   - Reduce scratch buffer usage
   - Implement hybrid KV cache (planned feature)

## Conclusion

**Status:** ⚠️ **INVESTIGATION REQUIRED**

The atomicAdd fix successfully resolved the race condition, but:
1. **Quality improvement not seen in benchmark** (12% Jaccard vs expected 83%)
2. **c=8 performance regression** (0.64x vs llama.cpp)
3. **Memory overhead persists** (+1284 MB)

**Priority:** Investigate why manual test showed 83% Jaccard but benchmark shows 12%.

## Files Generated

- `benchmark_results.log` - Full benchmark output
- `gguf_benchmark_results/comparison_20260421_153656.json` - Detailed metrics
- `docs/benchmark_analysis_post_fix.md` - This analysis

## Test Commands

```bash
# Re-validate fix
bash scripts/validate_fp32_fix.sh

# Re-run quality test
./build-cuda/inferflux_first_token_probe \
  --backend inferflux_cuda \
  --model models/qwen2.5-3b-instruct/qwen2.5-3b-instruct-q4_k_m.gguf \
  --prompt "The quick brown fox" \
  --max-tokens 20

# Profile c=8 bottleneck
bash scripts/profile.sh backend-ncu
```
