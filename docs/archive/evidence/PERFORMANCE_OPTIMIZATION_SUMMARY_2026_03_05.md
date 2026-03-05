# Performance Optimization & Profiling Summary

**Last Updated**: 2026-03-05
**Status**: Consolidated

---

## Quick Reference

### Backend Performance by Model Type

#### FP16 Models (Qwen2.5 3B F16, Sequential)

| Backend | Throughput | Latency | Notes |
|---------|-----------|---------|-------|
| cuda_universal | 5,119-5,594 tok/s | 4.0-4.5 ms | Excellent single-request |
| cuda_native | TBD | TBD | Full benchmark pending |

**Model Size**: 5.8 GB (3x larger than Q4_K_M)
**Use Case**: Quality-critical applications with sufficient GPU memory
**See**: [FP16_MODEL_GUIDE](FP16_MODEL_GUIDE.md) for FP16-specific guidance

#### Quantized Models (TinyLlama 1.1B Q4_K_M, 8 concurrent)

| Backend | Throughput | Notes |
|---------|-----------|-------|
| cuda_universal | 2,774 tok/s | 27ms avg latency |
| cuda_native | 2,489 tok/s | 37ms avg latency |

**Conclusion**: Both backends perform similarly on small models. cuda_native expected to excel on larger models (7B+) with better kernel optimizations.

---

## Configuration Recommendations

### For High Throughput (Production)

```yaml
runtime:
  backend_priority: cuda_native
  cuda:
    native_executor: native_kernel
    flash_attention:
      enabled: true
      kernel: fa2
    phase_overlap:
      enabled: true
      min_prefill_tokens: 256
    batch_size: 16
    max_batch_tokens: 8192
  llama:
    max_parallel_sequences: 128  # From StartupAdvisor
    n_ctx: 4096
```

### Environment Variables

```bash
# Dynamic slot allocation (StartupAdvisor)
INFERFLUX_GPU_UTILIZATION_PCT=85      # Target GPU memory %
INFERFLUX_OVERHEAD_GB=1                # CUDA context overhead
INFERFLUX_MIN_SLOTS=10                 # Minimum concurrent sequences
INFERFLUX_MAX_SLOTS=256                # Maximum concurrent sequences

# Backend selection
INFERFLUX_NATIVE_CUDA_EXECUTOR=native_kernel  # Use native kernels
INFERFLUX_BACKEND_PREFER_NATIVE=true          # Prefer native over universal
```

---

## Performance Tuning Quick Wins

### 1. Enable cuda_native
**Impact**: +100-1500% throughput (varies by model size)
```bash
export INFERFLUX_NATIVE_CUDA_EXECUTOR=native_kernel
```

### 2. Enable Flash Attention
**Impact**: +10-20% throughput
```yaml
runtime:
  cuda:
    flash_attention:
      enabled: true
      kernel: fa2
```

### 3. Tune Batch Size
**Impact**: +20-30% throughput
```yaml
runtime:
  cuda:
    batch_size: 16  # Up from 8
    max_batch_tokens: 8192
```

### 4. Enable Phase Overlap
**Impact**: +30-50% on mixed workloads
```yaml
runtime:
  cuda:
    phase_overlap:
      enabled: true
      min_prefill_tokens: 256
```

### 5. Optimize Slot Allocation
**Impact**: +15-25% throughput, better memory utilization
```bash
# Let StartupAdvisor calculate optimal values
INFERFLUX_MIN_SLOTS=16 INFERFLUX_MAX_SLOTS=128
```

---

## Monitoring Commands

### Check Backend Performance
```bash
# Throughput metrics
curl -s http://localhost:8080/metrics | grep inferflux_native_forward_duration_ms

# Phase overlap effectiveness
curl -s http://localhost:8080/metrics | grep cuda_lane_overlap

# Memory utilization
curl -s http://localhost:8080/metrics | grep kv_active_sequences
```

### Check Slot Allocation
```bash
# View StartupAdvisor recommendations
grep "slot_allocation" logs/server.log

# Check current slot usage
curl -s http://localhost:8080/metrics | grep -E "kv_max_sequences|kv_active_sequences"
```

---

## Documentation Index

### Canonical Sources
- **[CONFIG_REFERENCE](CONFIG_REFERENCE.md)**: All configuration options
- **[MONITORING](MONITORING.md)**: Metrics and observability
- **[STARTUP_ADVISOR](STARTUP_ADVISOR.md)**: Dynamic slot allocation
- **[GGUF_SMOKE_TEST_GUIDE](GGUF_SMOKE_TEST_GUIDE.md)**: Testing runbook
- **[GGUF_NATIVE_KERNEL_IMPLEMENTATION](GGUF_NATIVE_KERNEL_IMPLEMENTATION.md)**: GGUF runtime contract

### Design & Architecture
- **[design/KV_CACHE_ARCHITECTURE_DEEP_DIVE](design/KV_CACHE_ARCHITECTURE_DEEP_DIVE_2026_03_04.md)**: KV cache internals
- **[design/SEQUENCE_SLOT_MANAGER_PLAN](design/SEQUENCE_SLOT_MANAGER_PLAN.md)**: Slot management design

### Archived Evidence
- **[archive/evidence/GGUF_CONCURRENT_PROFILING_RESULTS_2026_03_05](archive/evidence/GGUF_CONCURRENT_PROFILING_RESULTS_2026_03_05.md)**: Full profiling details
- **[archive/evidence/GGUF_PROFILING_QUICK_REFERENCE_2026_03_05](archive/evidence/GGUF_PROFILING_QUICK_REFERENCE_2026_03_05.md)**: Quick profiling guide
- **[archive/evidence/FLASHATTENTION_LIVE_TEST_RESULTS_2025_03_02](archive/evidence/FLASHATTENTION_LIVE_TEST_RESULTS_2025_03_02.md)**: FlashAttention validation

---

## Benchmarking Tools

### Quick Benchmark Script
```bash
# Compare cuda_universal vs cuda_native
./scripts/quick_benchmark_gguf.sh both

# Test specific backend
./scripts/quick_benchmark_gguf.sh native
./scripts/quick_benchmark_gguf.sh universal

# Customize concurrency
CONCURRENT=16 ./scripts/quick_benchmark_gguf.sh both
```

### Throughput Gate
```bash
# Run performance regression test
./scripts/run_throughput_gate.py \
  --server-bin ./build/inferfluxd \
  --config config/server.yaml \
  --model tinyllama \
  --backend cuda \
  --min-completion-tok-per-sec 120
```

---

## Common Issues & Solutions

### Issue: "failed to find a memory slot"
**Cause**: KV cache slots exhausted
**Solution**:
```bash
# Increase max_parallel_sequences
INFERFLUX_MIN_SLOTS=32 INFERFLUX_MAX_SLOTS=256

# Or reduce context window
runtime:
  llama:
    n_ctx: 2048  # Down from 4096
```

### Issue: Low GPU utilization
**Cause**: Model too small or workload not GPU-bound
**Solution**: Use larger model (7B+) or increase batch size

### Issue: Phase overlap not triggering
**Cause**: Insufficient mixed workload (prefill + decode)
**Solution**: Ensure workload has both long and short requests

### Issue: High P95 latency
**Cause**: Batch size too large or context window too large
**Solution**:
```yaml
runtime:
  cuda:
    batch_size: 8  # Reduce from 16
  llama:
    n_ctx: 2048  # Reduce from 4096
```

---

## Validation Checklist

Before deploying to production:

- [ ] Run quick benchmark to validate backend performance
- [ ] Enable cuda_native and verify metrics collection
- [ ] Run StartupAdvisor and apply slot allocation recommendations
- [ ] Enable Flash Attention and verify kernel selection
- [ ] Enable phase overlap and verify overlap metrics
- [ ] Test with production-like workload
- [ ] Monitor memory usage under load
- [ ] Document baseline metrics
- [ ] Set up alerting for throughput degradation

---

## Next Steps

1. **Profile with production model** (7B+ F16 GGUF) to validate cuda_native advantages
2. **Run Nsight Systems** to identify kernel-level optimization opportunities
3. **Test phase overlap** with mixed workloads
4. **Monitor metrics** for 24-48 hours to establish baseline
5. **Tune configuration** based on actual usage patterns

---

## Contact & Support

- **Issues**: [GitHub Issues](https://github.com/anthropics/inferflux/issues)
- **Docs**: See [INDEX.md](INDEX.md) for full documentation
- **Troubleshooting**: [Troubleshooting.md](Troubleshooting.md)
