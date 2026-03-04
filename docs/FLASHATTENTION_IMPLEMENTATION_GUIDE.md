# FlashAttention Integration Guide for Ada RTX 4000

## Executive Summary

**Good news:** llama.cpp already has FlashAttention-2 implementations optimized for Ada (your RTX 4000)!

**Your task:** Integrate it properly into InferFlux's architecture with monitoring, metrics, and adaptive selection.

## Learning Path

### Phase 1: Understanding (Week 1)

**Goal:** Learn how llama.cpp's FlashAttention works

**Tasks:**
1. Read `external/llama.cpp/ggml/src/ggml-cuda/fattn.cu`
2. Understand the kernel selection logic
3. Learn about GQA (Grouped-Query Attention)
4. Study memory tiling strategies

**Key Files:**
- `fattn.cu` - Main FlashAttention implementation
- `fattn.cuh` - Interface
- `fattn-tile.cu` - Tile-based implementation (Ada/Ampere)
- `fatnn-mma-f16.cu` - Matrix multiply accumulate with FP16
- `fattn-vec.cu` - Vectorized implementation
- `fattn-common.cuh` - Shared utilities

**Concepts to Understand:**
- Memory tiling: Q, K, V matrices are split into tiles that fit in shared memory
- Online softmax: Computes softmax in a single pass to reduce memory accesses
- GQA optimization: Reuses K/V across multiple attention heads
- Kernel variants: Different implementations for different GPU architectures

### Phase 2: Integration (Week 2-3)

**Goal:** Wire llama.cpp's FlashAttention into InferFlux's backend

**Current Situation:**
- llama.cpp has `ggml_cuda_flash_attn_ext()` function
- Your `CudaBackend` wraps llama.cpp
- Need to expose FlashAttention metrics and control

**Implementation Steps:**

1. **Add FlashAttention status to backend capabilities:**
```cpp
// In runtime/backends/backend_capabilities.h
struct BackendCapabilities {
  bool supports_structured_output;
  bool supports_flash_attention;  // Add this
  std::string flash_attention_version;  // "fa2" or "standard"
  // ...
};
```

2. **Query FlashAttention availability in CudaBackend:**
```cpp
// In runtime/backends/cuda/cuda_backend.cpp
bool CudaBackend::LoadModel(...) {
  // ... existing code ...

#ifdef INFERFLUX_HAS_CUDA
  // Query FlashAttention support
  int device = 0;  // Or your device ID
  bool fa_supported = ggml_cuda_flash_attn_ext_supported(device, nullptr);

  capabilities_.supports_flash_attention = fa_supported;
  capabilities_.flash_attention_version = fa_supported ? "fa2" : "standard";

  log::Info("cuda_backend",
    "FlashAttention-2: " + std::string(fa_supported ? "YES" : "NO"));
#endif
}
```

3. **Add FlashAttention metrics:**
```cpp
// In server/metrics/metrics.h
// Add these metrics:

// FlashAttention kernel selection
void RecordFlashAttentionKernelSelected(const std::string& kernel_type);

// FlashAttention performance
void RecordFlashAttentionExecutionTime(double duration_ms);
void RecordFlashAttentionTokensProcessed(int num_tokens);

// FlashAttention fallback
void RecordFlashAttentionFallback(const std::string& reason);
```

4. **Wire metrics in backend:**
```cpp
// In runtime/backends/cuda/cuda_backend.cpp
std::string CudaBackend::Decode(...) {
  auto start = std::chrono::high_resolution_clock::now();

  // Call llama.cpp decode (uses FlashAttention internally if available)
  auto result = llama_backend_->Decode(...);

  auto end = std::chrono::high_resolution_clock::now();
  double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();

  // Record metrics
  if (capabilities_.supports_flash_attention) {
    metrics::RecordFlashAttentionExecutionTime(duration_ms);
    metrics::RecordFlashAttentionTokensProcessed(num_tokens_generated);
  }

  return result;
}
```

### Phase 3: Optimization (Week 4-6)

**Goal:** Make FlashAttention run faster in InferFlux

**Optimization Opportunities:**

1. **Reduce calling overhead:**
```cpp
// Instead of:
for (int i = 0; i < num_tokens; ++i) {
  llama_decode(context, token);  // Individual calls
}

// Do this:
llama_batch batch = llama_batch_init(num_tokens);
llama_decode_batch(context, batch);  // Batched call
```

2. **Prefetch next batch:**
```cpp
// While GPU processes batch N, prepare batch N+1 on CPU
std::future<void> prefetch = std::async(std::launch::async, [&]() {
  PrepareNextBatch();
});

ProcessCurrentBatch();
prefetch.wait();
```

3. **Optimize memory allocations:**
```cpp
// Reuse memory across batches
class FlashAttentionMemoryPool {
  void* qkv_memory_;  // Pre-allocated QKV memory
  size_t pool_size_;

public:
  FlashAttentionMemoryPool(size_t max_batch, size_t max_seq_len);
  void* GetQKVMemory();  // Fast allocation
};
```

4. **Tune for Ada RTX 4000:**
```yaml
# config/server.cuda.yaml
runtime:
  cuda:
    flash_attention:
      enabled: true
      # Ada RTX 4000 has 20GB VRAM
      max_batch_size: 32  # Adjust based on model size
      max_seq_len: 8192   # Ada can handle this
      # Use FP16 for better performance
      use_fp16: true
```

### Phase 4: Benchmarking (Week 7-8)

**Goal:** Measure real performance improvements

**Benchmark Script:**

```bash
#!/bin/bash
# scripts/benchmark_flash_attention.sh

MODEL="${1:-tinyllama}"
BATCH_SIZES=(1 2 4 8 16)
SEQ_LENS=(512 1024 2048 4096)

echo "FlashAttention Benchmark on Ada RTX 4000"
echo "Model: $MODEL"
echo "=========================================="

for bs in "${BATCH_SIZES[@]}"; do
  for sl in "${SEQ_LENS[@]}"; do
    echo ""
    echo "Batch Size: $bs, Seq Len: $sl"

    # Run benchmark with FlashAttention enabled
    INFERFLUX_CUDA_FLASH_ATTENTION=fa2 \
    ./build/inferfluxd \
      --config config/server.cuda.yaml \
      --model "$MODEL" \
      --batch-size "$bs" \
      --seq-len "$sl" \
      --benchmark-tokens 10000

    # Run benchmark with standard attention
    INFERFLUX_CUDA_FLASH_ATTENTION=standard \
    ./build/inferfluxd \
      --config config/server.cuda.yaml \
      --model "$MODEL" \
      --batch-size "$bs" \
      --seq-len "$sl" \
      --benchmark-tokens 10000
  done
done
```

**Metrics to Collect:**
- Tokens per second (throughput)
- Time to first token (TTFT)
- Memory usage (VRAM)
- GPU utilization (%)
- Energy consumption (W)

**Expected Results:**
| Scenario | Standard Attention | FlashAttention-2 | Speedup |
|----------|-------------------|------------------|---------|
| BS=1, SL=512 | 50 tok/s | 75 tok/s | 1.5x |
| BS=1, SL=2048 | 30 tok/s | 60 tok/s | 2.0x |
| BS=8, SL=2048 | 100 tok/s | 200 tok/s | 2.0x |
| BS=16, SL=4096 | 80 tok/s | 180 tok/s | 2.25x |

*These are estimates - your actual results may vary*

## Testing FlashAttention Right Now

### Quick Test (5 minutes):

```bash
# 1. Build with CUDA
./scripts/build.sh

# 2. Start server with FlashAttention logging
INFERFLUX_CUDA_FLASH_ATTENTION=fa2 \
INFERFLUX_LOG_LEVEL=debug \
./build/inferfluxd --config config/server.cuda.yaml

# 3. In another terminal, make a request
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dev-key-123" \
  -d '{
    "prompt": "Hello, world!",
    "max_tokens": 100,
    "model": "tinyllama"
  }'

# 4. Check logs for FlashAttention usage
# Look for: "FlashAttention-2: YES" or "Using flash attention"
```

### Profiling with Nsight Systems:

```bash
# Profile the server
nsys profile --stats=true \
  --output=flash_attention_profile \
  ./build/inferfluxd --config config/server.cuda.yaml

# View results
nsys stats flash_attention_profile.qdrep

# Open GUI (optional)
nsys-ui flash_attention_profile.qdrep
```

## What Makes FlashAttention Fast?

### 1. Memory Tiling

Instead of loading the entire Q, K, V matrices into GPU memory:

```
Standard Attention:
[Q: 4096x4096] [K: 4096x4096] [V: 4096x4096] → Uses all VRAM

FlashAttention-2:
[Q: 128x128] [K: 128x128] [V: 128x128] → Fits in shared memory
Repeat for all tiles → Much faster!
```

### 2. Online Softmax

Standard attention computes softmax in two passes:
```
Pass 1: Compute exp(QK^T) → Store in memory
Pass 2: Normalize and multiply by V
```

FlashAttention computes in one pass:
```
Single Pass: Compute and normalize on-the-fly
→ Half the memory accesses!
```

### 3. GQA Optimization

For models like Llama 2/3 that use GQA:

```
MHA (Multi-Head Attention):
32 heads → 32 K/V matrices

GQA (Grouped-Query Attention):
32 heads → 8 K/V matrices (shared across groups)
→ 4x less memory for K/V!
```

## Troubleshooting

### FlashAttention not being used?

**Check 1: Verify GPU support**
```bash
# Run this command
nvidia-smi --query-gpu=name,compute_cap --format=csv

# Should show:
# Ada RTX 4000, 8.9  # Compute capability 8.9 = Ada (supports FA2)
```

**Check 2: Verify llama.cpp build**
```bash
# Check if llama.cpp was built with CUDA
ldd ./build/inferfluxd | grep cuda

# Should show:
# libcudart.so.12  # CUDA runtime
# libggml-cuda.so  # llama.cpp CUDA backend
```

**Check 3: Enable logging**
```bash
INFERFLUX_LOG_LEVEL=debug \
./build/inferfluxd --config config/server.cuda.yaml 2>&1 | grep -i flash
```

### Poor performance?

**Possible causes:**
1. Small batch sizes (< 4) → FlashAttention overhead dominates
2. Short sequences (< 512) → Standard attention is competitive
3. Memory fragmentation → Restart server
4. Wrong kernel selected → Check kernel selection logic

## Next Steps

1. ✅ **This week:** Read llama.cpp FlashAttention code
2. ✅ **Week 2:** Integrate metrics and monitoring
3. ✅ **Week 3:** Optimize calling path
4. ✅ **Week 4-5:** Benchmark and tune for Ada RTX 4000
5. ✅ **Week 6-8:** Advanced optimizations (prefetch, memory pool)

## Resources

- [FlashAttention-2 Paper](https://arxiv.org/abs/2307.08691)
- [FlashAttention-3 Paper](https://arxiv.org/abs/2407.16863) - For understanding Hopper optimizations
- [llama.cpp CUDA Implementation](https://github.com/ggerganov/llama.cpp/tree/master/ggml/src/ggml-cuda)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

## Summary

You don't need to write FlashAttention from scratch. llama.cpp already has excellent implementations optimized for your Ada RTX 4000. Your job is to:

1. **Integrate it properly** into InferFlux's architecture
2. **Add monitoring** to understand when it's being used
3. **Optimize the path** to reduce overhead
4. **Benchmark and tune** for your specific hardware

This is a much more achievable path for a solo developer, and you'll learn a lot about GPU optimization along the way!
