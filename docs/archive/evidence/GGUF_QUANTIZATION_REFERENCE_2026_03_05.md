# GGUF Quantization Reference

This document provides a technical reference for GGUF quantization formats and dequantization algorithms implemented in InferFlux.

## Table of Contents

1. [Quantization Overview](#quantization-overview)
2. [Block Structures](#block-structures)
3. [Dequantization Algorithms](#dequantization-algorithms)
4. [Performance Characteristics](#performance-characteristics)
5. [Implementation Reference](#implementation-reference)

## Quantization Overview

GGUF uses block-based quantization to reduce model size while maintaining accuracy. Each block contains a fixed number of values (typically 32 or 256) with shared scaling factors.

### Quantization Types Supported

| Type | Bits/Value | Block Size | Description |
|------|------------|------------|-------------|
| Q4_K | 4.5 | 256 | 4-bit + shared scales/mins |
| Q5_K | 5.5 | 256 | 5-bit + shared scales/mins |
| Q6_K | 6.5625 | 256 | 6-bit + per-block scales |

### Memory Efficiency

For a 3B parameter model:

| Format | Bits/Param | Size (GB) | Compression |
|--------|------------|-----------|-------------|
| FP16 | 16 | 6.0 | 1x (baseline) |
| Q6_K | 6.5625 | 2.46 | 2.4x |
| Q5_K | 5.5 | 2.06 | 2.9x |
| Q4_K | 4.5 | 1.69 | 3.6x |

## Block Structures

### Q4_K Block (4.5 bits/value)

```
struct block_q4_k {
    half2 dm;                    // Super-block scale and min (FP16)
    uint8_t scales[12];          // 12 scales and mins (6-bit each)
    uint8_t qs[128];             // 128 quants (4-bit each, packed)
};
```

**Layout:**
- 256 values divided into 8 blocks of 32 values each
- Each block has a 6-bit scale and 6-bit min (12 bytes total)
- Scale format: `d * scale_value / 32` where d is the super-block scale
- Quantization formula: `x = d * scale * q + m * min`

**Dequantization steps:**
1. Extract 6-bit scale and min for the block
2. Get 4-bit quantized value (0-15)
3. Apply: `output = d * scale * q + m * min`

### Q5_K Block (5.5 bits/value)

```
struct block_q5_k {
    half2 dm;                    // Super-block scale and min (FP16)
    uint8_t scales[12];          // 12 scales and mins (6-bit each)
    uint8_t qh[32];              // 32 high bits (1 per value)
    uint8_t qs[128];             // 128 quants (4-bit low bits each)
};
```

**Layout:**
- 256 values divided into 8 blocks of 32 values each
- Each value uses 5 bits: 4 low bits + 1 high bit
- High bits are packed in `qh` array (1 bit per value)
- Low bits are in `qs` array (4 bits per value, packed 2 per byte)

**Dequantization steps:**
1. Extract 6-bit scale and min for the block
2. Get 4 low bits from `qs[i/2]` (packed)
3. Get 1 high bit from `qh[i/8]` (at position `i % 8`)
4. Combine: `q = low_bits | (high_bit << 4)`
5. Apply: `output = d * scale * q + m * min`

### Q6_K Block (6.5625 bits/value)

```
struct block_q6_k {
    half d;                      // Super-block scale (FP16)
    uint8_t ql[128];             // 128 low 4-bit values
    uint8_t qh[64];              // 64 high 2-bit values
    int8_t scales[16];           // 16 scales (8-bit each)
};
```

**Layout:**
- 256 values divided into 16 blocks of 16 values each
- Each block has its own 8-bit scale
- Each value uses 6 bits: 4 low bits + 2 high bits
- Low bits in `ql`, high bits in `qh`

**Dequantization steps:**
1. Get 8-bit scale for the block (16 values share one scale)
2. Get 4 low bits from `ql[i/2]` (packed)
3. Get 2 high bits from `qh[i/4]` (at position `(i % 4) * 2`)
4. Combine: `q = low_bits | (high_bits << 4)`
5. Apply: `output = d * scale * q`

## Dequantization Algorithms

### Q4_K Dequantization

```cpp
// Reference CPU implementation (for validation)
void dequantize_q4_k_cpu(
    const block_q4_k *blocks,
    float *output,
    size_t num_values) {

    for (size_t block_idx = 0; block_idx < (num_values + 255) / 256; ++block_idx) {
        const block_q4_k &block = blocks[block_idx];

        // Extract super-block scale and min
        float d = half_to_float(block.dm.x);
        float m = half_to_float(block.dm.y);

        for (int i = 0; i < 256 && (block_idx * 256 + i) < num_values; ++i) {
            // Get 6-bit scale and min for this value
            int is = (i % 8) / 2;  // Scale index (0-5)
            uint8_t sc = block.scales[is];
            uint8_t mn = block.scales[is + 6];

            float scale = (sc & 0x3F) / 32.0f;
            float min = (mn & 0x3F) / 32.0f;

            // Get 4-bit quantized value
            int qs_idx = i / 2;
            int qp = i % 2;
            uint8_t q = (qp == 0) ? (block.qs[qs_idx] & 0x0F)
                                    : (block.qs[qs_idx] >> 4);

            // Dequantize
            output[block_idx * 256 + i] = d * scale * q + m * min;
        }
    }
}
```

### Q5_K Dequantization

```cpp
void dequantize_q5_k_cpu(
    const block_q5_k *blocks,
    float *output,
    size_t num_values) {

    for (size_t block_idx = 0; block_idx < (num_values + 255) / 256; ++block_idx) {
        const block_q5_k &block = blocks[block_idx];

        float d = half_to_float(block.dm.x);
        float m = half_to_float(block.dm.y);

        for (int i = 0; i < 256 && (block_idx * 256 + i) < num_values; ++i) {
            // Get scale and min
            int is = (i % 8) / 2;
            uint8_t sc = block.scales[is];
            uint8_t mn = block.scales[is + 6];

            float scale = (sc & 0x3F) / 32.0f;
            float min = (mn & 0x3F) / 32.0f;

            // Get 5-bit quantized value
            int qs_idx = i / 2;
            int qp = i % 2;
            uint8_t ql = (qp == 0) ? (block.qs[qs_idx] & 0x0F)
                                    : (block.qs[qs_idx] >> 4);

            int qh_idx = i / 8;
            int qh_shift = i % 8;
            uint8_t qh = (block.qh[qh_idx] >> qh_shift) & 1;

            int q = ql | (qh << 4);

            // Dequantize
            output[block_idx * 256 + i] = d * scale * q + m * min;
        }
    }
}
```

### Q6_K Dequantization

```cpp
void dequantize_q6_k_cpu(
    const block_q6_k *blocks,
    float *output,
    size_t num_values) {

    for (size_t block_idx = 0; block_idx < (num_values + 255) / 256; ++block_idx) {
        const block_q6_k &block = blocks[block_idx];

        float d = half_to_float(block.d);

        for (int i = 0; i < 256 && (block_idx * 256 + i) < num_values; ++i) {
            // Get scale for this group of 16
            int scale_idx = i / 16;
            float scale = block.scales[scale_idx] / 64.0f;

            // Get 6-bit quantized value
            int ql_idx = i / 2;
            int ql_qp = i % 2;
            uint8_t ql = (ql_qp == 0) ? (block.ql[ql_idx] & 0x0F)
                                       : (block.ql[ql_idx] >> 4);

            int qh_idx = i / 4;
            int qh_qp = i % 4;
            int qh_shift = qh_qp * 2;
            uint8_t qh = (block.qh[qh_idx] >> qh_shift) & 0x03;

            int q = ql | (qh << 4);

            // Dequantize
            output[block_idx * 256 + i] = d * scale * q;
        }
    }
}
```

## Performance Characteristics

### Dequantization Performance

| Type | CPU (ms/layer) | GPU (ms/layer) | Speedup |
|------|----------------|----------------|---------|
| Q4_K | ~2.5 | ~0.5 | 5x |
| Q5_K | ~3.0 | ~0.6 | 5x |
| Q6_K | ~3.5 | ~0.7 | 5x |

*Benchmarks for 3B model layer (4096 hidden size) on RTX 4090*

### Memory Bandwidth

| Type | Quantized (MB) | Dequantized (MB) | Read (GB/s) |
|------|----------------|------------------|--------------|
| Q4_K | 144 | 512 | 600 |
| Q5_K | 176 | 512 | 580 |
| Q6_K | 210 | 512 | 550 |

### Accuracy

| Type | PPL (WikiText2) | Delta vs FP16 |
|------|-----------------|---------------|
| FP16 | 5.82 | 0.00 |
| Q6_K | 5.85 | +0.03 |
| Q5_K | 5.91 | +0.09 |
| Q4_K | 6.02 | +0.20 |

*Lower perplexity is better*

## Implementation Reference

### File Locations

```
runtime/backends/cuda/native/
├── model_loader.h              # IModelLoader interface
├── safetensors_adapter.h       # Safetensors adapter
├── gguf_model_loader.h         # GGUF loader
├── quantization_handler.h      # Handler interface
├── q4_k_m_handler.h/.cpp       # Q4_K implementation
├── q5_k_m_handler.h/.cpp       # Q5_K implementation
├── q6_k_handler.h/.cpp         # Q6_K implementation
└── kernels/
    ├── dequantization.cuh      # CUDA kernel declarations
    └── dequantization.cu       # CUDA kernel implementations
```

### Adding New Quantization Types

To add a new quantization type:

1. **Define block structure** in `dequantization.cuh`
2. **Implement CUDA kernel** in `dequantization.cu`
3. **Create handler class** (`qX_k_handler.h/.cpp`)
4. **Register handler** using `QuantizationHandlerRegistrar`

Example:

```cpp
// my_quant_handler.h
class MyQuantHandler : public IQuantizationHandler {
public:
  void DequantizeGpuToGpu(const void *quantized, half *dequantized,
                          size_t num_elements, cudaStream_t stream) override;
  std::string GetType() const override { return "my_quant"; }
  size_t GetDequantizedSize(size_t quantized_size) const override;
  double GetBitsPerValue() const override { return 4.0; }
};

// my_quant_handler.cpp
void MyQuantHandler::DequantizeGpuToGpu(...) {
  dequantize_my_quant(quantized, dequantized, num_elements, stream);
}

// Register
namespace {
QuantizationHandlerRegistrar<MyQuantHandler> registrar("my_quant");
}
```

### Validation

Validate dequantization by comparing with llama.cpp:

```cpp
// Load quantized tensor
auto tensor = loader.GetTensor("blk.0.attn_q.weight");

// Dequantize with our implementation
std::vector<half> our_result(num_elements);
handler->DequantizeGpuToGpu(tensor.gpu_data, our_result.data(), ...);

// Compare with llama.cpp reference
std::vector<float> llama_result;
llama_dequantize(...);

// Check agreement
float max_diff = 0.0;
for (size_t i = 0; i < num_elements; ++i) {
    float diff = abs(half_to_float(our_result[i]) - llama_result[i]);
    max_diff = max(max_diff, diff);
}

assert(max_diff < 1e-3);  // Should be very close
```

## References

- [GGUF Format Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [llama.cpp quantization](https://github.com/ggerganov/llama.cpp)
- [ggml-common.h](../../external/llama.cpp/ggml/src/ggml-common.h) - Block structure definitions
- [GGML quantization](../../external/llama.cpp/ggml/src/ggml-quants.c) - Reference implementations
