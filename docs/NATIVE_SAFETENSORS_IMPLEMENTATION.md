# Native CUDA Safetensors Implementation

## Status: Complete (Model Loading)

The native CUDA backend now supports loading models in safetensors format, completely independent from llama.cpp.

## What Works

### 1. Safetensors Binary Format Parser
- **File**: `runtime/backends/cuda/native/safetensors_parser.cpp`
- **Format Structure**:
  - [8 bytes: metadata length (little-endian uint64)]
  - [N bytes: JSON metadata]
  - [aligned: tensor data]

- **Capabilities**:
  - Memory-mapped file parsing
  - Flat JSON structure (tensors at top level)
  - dtype support: BF16, F16, F32, I8/I16/I32/I64, U8/U16/U32/U64, BOOL
  - Automatic 8-byte alignment for tensor data
  - Offset-based tensor access

### 2. SafetensorsLoader
- **File**: `runtime/backends/cuda/native_kernel_executor.cpp`
- **Features**:
  - Multi-shard model loading (model.safetensors.index.json)
  - Config.json parsing (Qwen2, LLaMA, etc.)
  - Model architecture detection (hidden_size, num_layers, num_heads, etc.)
  - Tensor registry with CPU memory mapping

### 3. Backend Integration
- **Files**: `scheduler/single_model_router.cpp`, `runtime/backends/backend_factory.cpp`
- **Changes**:
  - `BackendSupportsModelFormat()` now returns true for native CUDA + safetensors
  - Path resolution for native CUDA safetensors models
  - Model format routing: safetensors → native CUDA backend

### 4. NativeKernelExecutor
- **File**: `runtime/backends/cuda/native_kernel_executor.cpp`
- **Status**: Scaffold complete, kernels not implemented
- **What's Working**:
  - CUDA initialization (streams, device selection)
  - Model loading (config.json + safetensors shards)
  - Tensor parsing (434 tensors for Qwen2.5-3B)
- **What's Missing**:
  - GPU memory upload
  - Native inference kernels
  - Execution pipeline

## How to Use

### Configuration
```yaml
models:
  - id: qwen2.5-3b-native
    path: models/qwen2.5-3b-instruct-safetensors  # Directory, not file
    format: safetensors
    backend: cuda
    default: true

runtime:
  backend_exposure:
    prefer_native: true  # Use native executor
    allow_universal_fallback: false
```

### Environment Variables
```bash
INFERFLUX_NATIVE_CUDA_EXECUTOR=native  # Use NativeKernelExecutor
INFERFLUX_MODEL_PATH=models/qwen2.5-3b-instruct-safetensors
```

### Start Server
```bash
INFERFLUX_NATIVE_CUDA_EXECUTOR=native \
INFERFLUX_MODEL_PATH=models/qwen2.5-3b-instruct-safetensors \
./build/inferfluxd --config config/server.cuda.qwen14b.native.yaml
```

## Test Results

### Qwen2.5-3B-Instruct (Safetensors FP16)
- **Model Size**: 5.8 GB (2 shards)
- **Tensors Loaded**: 434 tensors (170 from first shard)
- **Architecture**: Qwen2, 36 layers, 16 heads, 2048 hidden size
- **Load Status**: ✅ SUCCESS
- **Backend**: Native CUDA (provider=native)
- **Format**: safetensors

### Example Logs
```
[INFO] safetensors_loader: Model config: qwen2, hidden_size=2048, num_layers=36, num_heads=16, head_dim=128
[INFO] safetensors_loader: Found 2 shard files
[INFO] safetensors_parser: Parsing file: model-00001-of-00002.safetensors (size: 3784 MB)
[INFO] safetensors_parser: Header: metadata_size=30200 bytes
[INFO] safetensors_parser: Successfully parsed 170 tensors
[INFO] safetensors_loader: Model loaded successfully: 434 tensors
[INFO] native_kernel_executor: Native CUDA model loaded successfully
```

## Next Steps (Not Implemented Yet)

### 1. GPU Memory Upload
```cpp
// In NativeKernelExecutor::UploadToGPU()
// - Allocate d_weights_ buffer
// - cudaMemcpyAsync for each tensor
// - Free CPU memory after upload
```

### 2. Native Inference Kernels
- **Attention**: FlashAttention-2 or PagedAttention
- **FFN Layers**: GEMM with CUTLASS/cuBLAS
- **RoPE**: Rotary Position Embedding
- **LayerNorm**: RMS Layer Normalization

### 3. Inference Pipeline
- **Prefill Phase**: Process all prompt tokens
- **Decode Phase**: Generate tokens one-by-one
- **Batching**: Continuous batching support
- **Sampling**: Token sampling (greedy, nucleus, etc.)

### 4. Performance Optimization
- **PagedAttention**: vLLM-style attention with paged KV cache
- **Speculative Decoding**: Draft model + validator
- **Quantization**: INT8/INT4 weight quantization
- **Flash Attention**: Memory-efficient attention

## Architecture Comparison

### llama.cpp CUDA Backend
- GGUF format only
- Optimized CUDA graphs
- FlashAttention-2 working
- Mature, production-ready

### Native CUDA Backend (Current)
- Safetensors format
- Model loading working
- Kernels not implemented
- Research/development phase

### Target Architecture (vLLM-style)
```
┌─────────────────────────────────────┐
│     NativeKernelExecutor            │
├─────────────────────────────────────┤
│  SafetensorsLoader (complete)       │
│  ├─ Parse config.json               │
│  ├─ Parse safetensors shards        │
│  └─ Upload to GPU                   │
├─────────────────────────────────────┤
│  Native Inference (TODO)            │
│  ├─ Prefill kernels                 │
│  ├─ Decode kernels                  │
│  ├─ PagedAttention                  │
│  └─ FFN (CUTLASS/cuBLAS)            │
├─────────────────────────────────────┤
│  Scheduler Integration              │
│  └─ Continuous batching             │
└─────────────────────────────────────┘
```

## Key Files

| File | Purpose | Status |
|------|---------|--------|
| `runtime/backends/cuda/native/safetensors_parser.h` | Binary format parser | ✅ Complete |
| `runtime/backends/cuda/native/safetensors_parser.cpp` | Parser implementation | ✅ Complete |
| `runtime/backends/cuda/native_kernel_executor.h` | Native executor interface | ✅ Complete |
| `runtime/backends/cuda/native_kernel_executor.cpp` | Executor implementation | 🟡 Scaffold |
| `scheduler/single_model_router.cpp` | Backend routing | ✅ Updated |
| `runtime/backends/backend_factory.cpp` | Backend factory | ✅ Working |

## Testing

### Unit Tests
```bash
# Add test in tests/unit/test_safetensors_parser.cpp
TEST_CASE("SafetensorsParser::Parse") {
    SafetensorsParser parser("models/qwen2.5-3b-instruct-safetensors/model-00001-of-00002.safetensors");
    REQUIRE(parser.Parse());

    auto* tensor = parser.GetTensor("model.embed_tokens.weight");
    REQUIRE(tensor != nullptr);
    REQUIRE(tensor->dtype == "BF16");
    REQUIRE(tensor->shape[0] == 151936);
    REQUIRE(tensor->shape[1] == 2048);
}
```

### Integration Tests
```bash
# Start server with native backend
INFERFLUX_NATIVE_CUDA_EXECUTOR=native \
./build/inferfluxd --config config/server.cuda.qwen14b.native.yaml

# Verify model loaded
curl http://localhost:8081/v1/models \
  -H "Authorization: Bearer dev-key-123"
```

## Performance Expectations

Once native kernels are implemented:
- **Target**: Match or exceed llama.cpp CUDA performance
- **Advantages**:
  - No GGUF conversion step
  - Direct HuggingFace model support
  - Custom optimization opportunities
  - PagedAttention for large batches

## References

- [Safetensors Format Spec](https://huggingface.co/docs/safetensors/index.html)
- [vLLM Architecture](https://docs.vllm.ai/en/latest/architecture/)
- [FlashAttention-2 Paper](https://arxiv.org/abs/2307.08691)
- [CUTLASS Library](https://github.com/NVIDIA/cutlass)

## Date: 2026-03-03
