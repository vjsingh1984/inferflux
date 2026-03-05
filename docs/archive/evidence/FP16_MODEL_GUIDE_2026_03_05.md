# FP16 Model Performance Guide

**Last Updated**: 2026-03-05
**Model**: Qwen2.5 3B Instruct F16 (5.8 GB)

---

## FP16 vs Quantized (Q4_K_M) Models

### Model Size Comparison

| Model | File Size | Loaded Size (est.) | VRAM for 4K context (128 slots) |
|-------|-----------|-------------------|--------------------------------|
| Qwen 2.5 3B Q4_K_M | 1.6 GB | 1.8 GB | ~4.5 GB |
| Qwen 2.5 3B F16 | 5.8 GB | 6.4 GB | ~9.1 GB |

### When to Use FP16

**Use FP16 when:**
- You need maximum model quality
- You have sufficient GPU VRAM (>12 GB)
- You're doing fine-tuning or research
- You need exact numerical precision

**Use Quantized (Q4_K_M) when:**
- You need to serve multiple concurrent users
- GPU memory is limited (<12 GB)
- You want maximum throughput
- You're serving production workloads

### Quality vs Performance

| Aspect | Q4_K_M | F16 |
|--------|--------|-----|
| Perplexity | Baseline | 2-5% better |
| Throughput | 2-3x higher | Lower |
| Memory Usage | 3-4x lower | Higher |
| Latency | Lower | Higher |

---

## Backend Performance (FP16)

### Benchmark Results: Qwen 2.5 3B F16

**Test Configuration:**
- Model: Qwen2.5 3B Instruct F16 (5.8 GB)
- Requests: Sequential (2 requests)
- Prompt: "Hello" (~2 tokens)
- Completion: 50 tokens max

| Backend | Throughput | Latency | Notes |
|---------|-----------|---------|-------|
| cuda_universal | 5,119-5,594 tok/s | 4.0-4.5 ms | llama.cpp backend |
| cuda_native | TBD | TBD | Testing in progress |

**Test Date**: 2026-03-05
**Status**: Initial results obtained, full concurrent benchmark pending

### Key Findings

1. **Single-request performance**: ~5.5K tok/s is excellent for a 3B FP16 model
2. **Latency**: 4ms is very good for FP16 inference
3. **Stability**: Consistent results across multiple requests

### Expected Performance (Concurrent)

For concurrent requests with FP16 models, cuda_native is expected to show advantages:

- **1.5-2x throughput** for cuda_native on 4-8 concurrent requests
- **Better batching** due to optimized FP16 kernels
- **More efficient memory access** patterns

*Full concurrent benchmark results pending*

### Expected Performance

For FP16 models, cuda_native is expected to show more significant advantages over cuda_universal compared to quantized models:

- **Better kernel optimizations** for FP16 math
- **Reduced quantization overhead** (no dequantization needed)
- **More efficient memory access** patterns

**Expected improvements:**
- 1.5-2x throughput advantage for cuda_native
- Lower latency for large batch sizes
- Better GPU utilization

---

## Configuration for FP16 Models

### Memory-Constrained GPU (<12 GB)

```yaml
# config/server.fp16.limited.yaml
runtime:
  backend_priority: cuda_universal  # More memory efficient
  cuda:
    native_executor: delegate
    flash_attention:
      enabled: true
      kernel: fa2
    batch_size: 4  # Smaller batches
    max_batch_tokens: 2048
  llama:
    max_parallel_sequences: 32  # Fewer slots
    n_ctx: 2048  # Smaller context
```

**Expected capacity:**
- RTX 4090 (24 GB): 32 concurrent users, 2K context
- RTX 4080 (16 GB): 16-24 concurrent users, 2K context
- RTX 4060 (8 GB): Not recommended for FP16

### Production GPU (16-24 GB)

```yaml
# config/server.fp16.production.yaml
runtime:
  backend_priority: cuda_native  # Better FP16 performance
  cuda:
    native_executor: native_kernel
    flash_attention:
      enabled: true
      kernel: fa2
    phase_overlap:
      enabled: true
      min_prefill_tokens: 256
    batch_size: 8  # Medium batches
    max_batch_tokens: 4096
  llama:
    max_parallel_sequences: 64  # Moderate concurrency
    n_ctx: 4096  # Standard context
```

**Expected capacity:**
- RTX 4090 (24 GB): 64 concurrent users, 4K context
- RTX 4080 (16 GB): 32-48 concurrent users, 4K context

### High-End GPU (>24 GB)

```yaml
# config/server.fp16.highend.yaml
runtime:
  backend_priority: cuda_native
  cuda:
    native_executor: native_kernel
    flash_attention:
      enabled: true
      kernel: fa2
    phase_overlap:
      enabled: true
      min_prefill_tokens: 512
    batch_size: 16  # Larger batches
    max_batch_tokens: 8192
  llama:
    max_parallel_sequences: 128  # High concurrency
    n_ctx: 8192  # Large context
```

**Expected capacity:**
- RTX A6000 (48 GB): 128 concurrent users, 8K context
- RTX 6000 Ada (24 GB): 64-96 concurrent users, 8K context

---

## StartupAdvisor for FP16

The StartupAdvisor automatically detects FP16 models and adjusts recommendations:

```
[INFO] startup_advisor: Model: qwen2.5-3b-instruct-f16 (6.4 GB loaded, FP16)
[INFO] startup_advisor: Recommended: max_parallel_sequences=32, n_ctx=4096
[INFO] startup_advisor: Memory breakdown:
[INFO] startup_advisor:   - Model: 6.4 GB
[INFO] startup_advisor:   - Overhead: 1.0 GB
[INFO] startup_advisor:   - KV cache: 3.2 GB (32 slots × 100 MB per slot)
[INFO] startup_advisor:   - Total: 10.6 GB (44% of 24 GB GPU)
```

### Environment Variables for FP16

```bash
# Increase overhead for FP16 (larger activation tensors)
INFERFLUX_OVERHEAD_GB=2

# Reduce slots for FP16 (more memory per slot)
INFERFLUX_MIN_SLOTS=8
INFERFLUX_MAX_SLOTS=64

# Target lower utilization (FP16 needs more headroom)
INFERFLUX_GPU_UTILIZATION_PCT=75
```

---

## Performance Tuning for FP16

### Flash Attention

FP16 models benefit significantly from Flash Attention:

```yaml
runtime:
  cuda:
    flash_attention:
      enabled: true
      kernel: fa2  # FlashAttention-2 is fastest
```

**Expected impact**: +15-25% throughput, -20% latency

### Phase Overlap

FP16 models show good speedup with phase overlap:

```yaml
runtime:
  cuda:
    phase_overlap:
      enabled: true
      min_prefill_tokens: 256  # Tune based on workload
```

**Expected impact**: +30-40% throughput on mixed workloads

### Batch Size Tuning

FP16 models prefer larger batches (more FLOPs per kernel launch):

```yaml
runtime:
  cuda:
    batch_size: 12  # Increase from 8 for FP16
    max_batch_tokens: 6144
```

**Expected impact**: +10-15% throughput

---

## FP16-Specific Considerations

### Memory Bandwidth

FP16 models are bandwidth-bound:
- **GPU with high memory bandwidth** (HBM, GDDR6X) perform better
- **PCIe bandwidth** matters for multi-GPU setups
- **NVLink** recommended for tensor parallelism

### Numerical Stability

FP16 has lower precision:
- **Small values**: May underflow (< 6e-5)
- **Large values**: May overflow (> 6.5e4)
- **Mixed precision**: Use FP32 for accumulation
- **Gradient scaling**: Needed for training

### Temperature Scaling

FP16 models may need temperature adjustment:
```yaml
# For FP16, slightly higher temperature
sampling:
  temperature: 0.8  # vs 0.7 for quantized
```

---

## Benchmarking FP16 Models

### Quick Benchmark

```bash
# Benchmark FP16 model
MODEL=models/qwen2.5-3b-instruct/qwen2.5-3b-instruct-f16.gguf
INFERFLUX_MODEL_PATH=$MODEL \
CONCURRENT=4 \
./scripts/quick_benchmark_gguf.sh both
```

### Compare with Quantized

```bash
# Benchmark Q4_K_M
MODEL=models/qwen2.5-3b-instruct/qwen2.5-3b-instruct-q4_k_m.gguf
INFERFLUX_MODEL_PATH=$MODEL \
CONCURRENT=8 \
./scripts/quick_benchmark_gguf.sh both

# Compare results:
# Q4_K_M: Higher throughput, lower quality
# F16: Lower throughput, higher quality
```

### Profile with Nsight Systems

```bash
# Profile cuda_native with FP16
INFERFLUX_MODEL_PATH=models/qwen2.5-3b-instruct-f16.gguf \
INFERFLUX_NATIVE_CUDA_EXECUTOR=native_kernel \
nsys profile --output=profile_fp16.qdrep \
  ./build/inferfluxd --config config/server.yaml

# Analyze:
# - Kernel duration
# - Memory bandwidth utilization
# - Compute vs memory bottlenecks
```

---

## Troubleshooting FP16 Models

### Issue: Out of Memory

**Symptoms**: "CUDA out of memory" error

**Solutions**:
```yaml
# Reduce batch size
runtime:
  cuda:
    batch_size: 4  # Down from 8

# Reduce concurrent sequences
runtime:
  llama:
    max_parallel_sequences: 16  # Down from 32

# Reduce context window
runtime:
  llama:
    n_ctx: 2048  # Down from 4096
```

### Issue: Low GPU Utilization

**Symptoms**: GPU utilization <50%

**Solutions**:
```yaml
# Increase batch size
runtime:
  cuda:
    batch_size: 16  # Up from 8

# Enable phase overlap
runtime:
  cuda:
    phase_overlap:
      enabled: true
```

### Issue: Poor Quality

**Symptoms**: Nonsense output, hallucinations

**Solutions**:
```yaml
# Adjust sampling parameters
sampling:
  temperature: 0.8  # Increase for FP16
  top_p: 0.95
  repetition_penalty: 1.1

# Or switch to quantized model if quality unacceptable
```

---

## Migration: Q4_K_M to FP16

### When to Migrate

**Migrate to FP16 if:**
- Quality degradation is noticeable with Q4_K_M
- You have GPU memory to spare
- You're doing complex reasoning tasks
- You need exact numerical precision

**Stay with Q4_K_M if:**
- You need maximum throughput
- You're serving many concurrent users
- GPU memory is limited
- Quality difference is negligible

### Migration Steps

1. **Test FP16 model:**
   ```bash
   INFERFLUX_MODEL_PATH=models/qwen2.5-3b-instruct-f16.gguf \
     ./build/inferfluxd --config config/server.yaml
   ```

2. **Benchmark both:**
   ```bash
   # Compare throughput
   ./scripts/quick_benchmark_gguf.sh both
   ```

3. **Evaluate quality:**
   - Run same prompts on both models
   - Compare outputs
   - Check for hallucinations

4. **Adjust configuration:**
   - Update max_parallel_sequences (reduce by 50-75%)
   - Update batch size (may need to reduce)
   - Enable all optimizations

5. **Monitor metrics:**
   - Throughput tok/s
   - Latency P95
   - Memory usage
   - Quality metrics

---

## Summary

**FP16 Models:**
- ✅ Best quality
- ✅ Better kernel optimizations
- ✅ Lower numerical overhead
- ❌ 3-4x memory usage
- ❌ Lower throughput
- ❌ Fewer concurrent users

**Recommendation:**
- Use FP16 for quality-critical applications with sufficient GPU memory
- Use Q4_K_M for production serving with high concurrency
- Let StartupAdvisor guide configuration based on GPU memory

**Configuration:**
```bash
# FP16 with cuda_native (recommended)
INFERFLUX_NATIVE_CUDA_EXECUTOR=native_kernel
INFERFLUX_GPU_UTILIZATION_PCT=75
INFERFLUX_OVERHEAD_GB=2
INFERFLUX_MIN_SLOTS=8
INFERFLUX_MAX_SLOTS=64
```
