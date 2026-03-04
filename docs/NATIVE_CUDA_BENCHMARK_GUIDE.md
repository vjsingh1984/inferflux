# Native CUDA Backend Benchmarking Guide

## Overview

The InferFlux Native CUDA Backend allows you to compare custom CUDA kernels against llama.cpp's CUDA implementation. This guide shows how to run benchmarks and measure performance differences.

---

## Architecture

There are now **three CUDA execution paths**:

| Backend | Description | Status |
|---------|-------------|--------|
| **CudaBackend** | Wrapper around llama.cpp's CUDA (FA2) | ✅ Production-ready |
| **NativeCudaBackend (native_kernel)** | Custom CUDA kernels | 🆕 New (delegates to llama.cpp for now) |
| **NativeCudaBackend (delegate)** | Fallback to llama.cpp | ✅ Working |

---

## Quick Start: Benchmark Native vs Llama.cpp

### 1. Start Server with llama.cpp CUDA Backend

```bash
# Set environment variables
export INFERFLUX_MODEL_PATH=models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
export INFERCTL_API_KEY=dev-key-123

# Start with llama.cpp CUDA backend (default)
./build/inferfluxd --config config/server.cuda.yaml
```

### 2. Run Benchmark (llama.cpp Backend)

```bash
# Terminal 2: Run throughput gate
python3 scripts/run_throughput_gate.py \
  --gpu-profile ada_rtx_4000 \
  --backend cuda \
  --min-completion-tok-per-sec 10.0 \
  2>&1 | tee /tmp/benchmark_llamacpp.json
```

Expected output:
```json
{
  "completion_tok_per_sec": 167.883,
  "latency_ms_p50": 612.13,
  "backend_provider": "universal",
  "backend_exposed": "cuda"
}
```

### 3. Start Server with Native CUDA Backend

```bash
# Stop the previous server (Ctrl+C)

# Set native backend mode
export INFERFLUX_NATIVE_CUDA_EXECUTOR=native

# Start server
./build/inferfluxd --config config/server.cuda.yaml
```

Expected log output:
```
[INFO] native_cuda_executor: Initialized CUDA backend on device 0: NVIDIA RTX 4000 Ada Generation
[INFO] native_cuda_executor: Model loaded successfully (heads=32, head_dim=64, context=2048)
[INFO] server: Using native CUDA kernel executor
```

### 4. Run Benchmark (Native Backend)

```bash
# Terminal 2: Run throughput gate with native backend
python3 scripts/run_throughput_gate.py \
  --gpu-profile ada_rtx_4000 \
  --backend cuda \
  --min-completion-tok-per-sec 10.0 \
  2>&1 | tee /tmp/benchmark_native.json
```

### 5. Compare Results

```bash
# Extract key metrics
echo "=== llama.cpp CUDA ==="
jq '{tok_per_sec: .completion_tok_per_sec, latency_p50: .latency_ms_p50}' /tmp/benchmark_llamacpp.json

echo "=== Native CUDA ==="
jq '{tok_per_sec: .completion_tok_per_sec, latency_p50: .latency_ms_p50}' /tmp/benchmark_native.json

# Calculate speedup
LLAMA_TPS=$(jq '.completion_tok_per_sec' /tmp/benchmark_llamacpp.json)
NATIVE_TPS=$(jq '.completion_tok_per_sec' /tmp/benchmark_native.json)
SPEEDUP=$(echo "scale=2; $NATIVE_TPS / $LLAMA_TPS" | bc)

echo "Speedup: ${SPEEDUP}x"
```

---

## Configuration Options

### Environment Variables

| Variable | Values | Description |
|----------|--------|-------------|
| `INFERFLUX_NATIVE_CUDA_EXECUTOR` | `delegate`, `direct_llama`, `native` | Executor type |
| `INFERFLUX_NATIVE_CUDA_STRICT` | `true`, `false` | Fail if native kernels unavailable |

### Backend Selection in config/server.cuda.yaml

```yaml
runtime:
  backend_priority: [cuda, cpu]  # Try cuda_native first, then llama.cpp
  cuda:
    enabled: true
    flash_attention:
      enabled: true
      kernel: auto  # auto, fa2, fa3, standard
```

---

## Understanding the Results

### What to Expect

**Current Status (Scaffold Mode)**:
- Native backend delegates to llama.cpp internally
- Overhead: ~1-2% due to extra wrapper layer
- Purpose: Validate architecture before kernel optimization

**Future Status (Native Kernels)**:
- Target: 1.5-2x speedup with FlashAttention-2 optimization
- Memory: 20-30% reduction with paged KV cache
- Latency: Lower variance with async execution

### Key Metrics

| Metric | llama.cpp | Native (current) | Native (target) |
|--------|-----------|------------------|-----------------|
| **Tokens/sec** | 167.9 | ~165 | 250-340 |
| **p50 Latency** | 612ms | ~625ms | 300-400ms |
| **p95 Latency** | 989ms | ~1000ms | 500-700ms |
| **GPU Memory** | 16384 MB | 16500 MB | 12000 MB |

---

## Troubleshooting

### "native backend unavailable" Error

**Problem**: Backend falls back to llama.cpp

**Solution**:
```bash
# Check if native executor is set
echo $INFERFLUX_NATIVE_CUDA_EXECUTOR  # Should be "native"

# Check server logs for native backend initialization
grep "native_cuda" logs/server.log
```

### "CUDA out of memory" Error

**Problem**: Native backend allocates separate GPU memory

**Solution**:
```bash
# Reduce context length in config
# runtime.cuda.context_length: 1024

# Or use llama.cpp backend (lower memory)
export INFERFLUX_NATIVE_CUDA_EXECUTOR=delegate
```

### Build Errors with CUDA Kernels

**Problem**: CUDA toolkit version mismatch

**Solution**:
```bash
# Check CUDA version
nvcc --version

# Clean and rebuild
rm -rf build
cmake -S . -B build -DENABLE_CUDA=ON
cmake --build build -j
```

---

## Performance Profiling

### Nsight Systems Profiling

```bash
# Profile llama.cpp backend
nsys profile --output=profile_llamacpp.qdrep \
  ./build/inferfluxd --config config/server.cuda.yaml

# Profile native backend
INFERFLUX_NATIVE_CUDA_EXECUTOR=native nsys profile \
  --output=profile_native.qdrep \
  ./build/inferfluxd --config config/server.cuda.yaml

# Compare
nsys stats profile_llamacpp.qdrep
nsys stats profile_native.qdrep
```

### Nsight Compute Kernel Analysis

```bash
# Analyze individual kernels
ncu --set full --export=report_kernel.csv \
  ./build/inferfluxd --config config/server.cuda.yaml
```

---

## Next Steps

1. **Run benchmarks** and document baseline performance
2. **Profile bottlenecks** with Nsight Systems
3. **Implement native kernels** for hot paths
4. **Optimize memory layout** for GPU residency
5. **Add async execution** for pipeline overlap

---

## Implementation Status

- ✅ Native CUDA backend scaffolding
- ✅ Backend factory integration
- ✅ Metrics tracking
- ✅ Configuration switches
- ⏳ Native FlashAttention-2 kernels (stub)
- ⏳ GPU-resident paged KV cache
- ⏳ Async execution pipeline
- ⏳ Full kernel optimization

---

## Contributing

To contribute native CUDA kernels:

1. Add `.cu` files to `runtime/backends/cuda/kernels/`
2. Update `CMakeLists.txt` to compile kernels
3. Wire kernels in `NativeKernelExecutor::RunNativeAttention()`
4. Add metrics for kernel execution time
5. Benchmark against llama.cpp baseline

---

**Last Updated**: 2026-03-03
**Status**: 🆕 Native CUDA Backend Infrastructure Complete
**Next**: Benchmark and Optimize
