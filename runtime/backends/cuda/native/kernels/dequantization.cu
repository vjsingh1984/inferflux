#include "runtime/backends/cuda/native/kernels/quant_common.cuh"

namespace inferflux {
namespace runtime {
namespace cuda {
namespace native {

//==============================================================================
// Q4_K Dequantization Kernel
//==============================================================================

// Uses shared dequant_q4k_element() from quant_common.cuh.
// Each thread dequantizes one 256-element super-block.
__global__ void dequantize_q4_k_kernel(const block_q4_k *quantized,
                                       half *dequantized, size_t num_elements) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int num_blocks = (num_elements + QK_K - 1) / QK_K;

  if (idx >= num_blocks) {
    return;
  }

  const block_q4_k &b = quantized[idx];
  int start = idx * QK_K;

  float d = __half2float(__ushort_as_half(b.d));
  float dmin = __half2float(__ushort_as_half(b.dmin));

  for (int sb = 0; sb < 8; ++sb) {
    for (int e = 0; e < 32; ++e) {
      int out_idx = start + sb * 32 + e;
      if (out_idx < (int)num_elements) {
        dequantized[out_idx] =
            __float2half(dequant_q4k_element(b, d, dmin, sb, e));
      }
    }
  }
}

cudaError_t dequantize_q4_k(const void *quantized, half *dequantized,
                            size_t num_elements, cudaStream_t stream) {

  dim3 grid = calc_dequant_grid(num_elements);
  dim3 block = calc_dequant_block();

  dequantize_q4_k_kernel<<<grid, block, 0, stream>>>(
      static_cast<const block_q4_k *>(quantized), dequantized, num_elements);

  return cudaGetLastError();
}

//==============================================================================
// Q5_K Dequantization Kernel
//==============================================================================

// Uses shared dequant_q5k_element() from quant_common.cuh.
__global__ void dequantize_q5_k_kernel(const block_q5_k *quantized,
                                       half *dequantized, size_t num_elements) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int num_blocks = (num_elements + QK_K - 1) / QK_K;

  if (idx >= num_blocks) {
    return;
  }

  const block_q5_k &b = quantized[idx];
  int start = idx * QK_K;

  float d = __half2float(__ushort_as_half(b.d));
  float dmin = __half2float(__ushort_as_half(b.dmin));

  for (int sb = 0; sb < 8; ++sb) {
    for (int e = 0; e < 32; ++e) {
      int out_idx = start + sb * 32 + e;
      if (out_idx < (int)num_elements) {
        dequantized[out_idx] =
            __float2half(dequant_q5k_element(b, d, dmin, sb, e));
      }
    }
  }
}

cudaError_t dequantize_q5_k(const void *quantized, half *dequantized,
                            size_t num_elements, cudaStream_t stream) {

  dim3 grid = calc_dequant_grid(num_elements);
  dim3 block = calc_dequant_block();

  dequantize_q5_k_kernel<<<grid, block, 0, stream>>>(
      static_cast<const block_q5_k *>(quantized), dequantized, num_elements);

  return cudaGetLastError();
}

//==============================================================================
// Q6_K Dequantization Kernel
//==============================================================================

// Uses shared dequant_q6k_element() from quant_common.cuh.
// This ensures identical bit-extraction math as the fused GEMV/GEMM kernels.
__global__ void dequantize_q6_k_kernel(const block_q6_k *quantized,
                                       half *dequantized, size_t num_elements) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int num_blocks = (num_elements + QK_K - 1) / QK_K;

  if (idx >= num_blocks) {
    return;
  }

  const block_q6_k &b = quantized[idx];
  int start = idx * QK_K;

  float d = __half2float(__ushort_as_half(b.d));

  for (int step = 0; step < 8; ++step) {
    int g = step / 4;
    int sub = step % 4;
    for (int e = 0; e < 32; ++e) {
      int out_idx = start + g * 128 + sub * 32 + e;
      if (out_idx < (int)num_elements) {
        dequantized[out_idx] =
            __float2half(dequant_q6k_element(b, d, g, sub, e));
      }
    }
  }
}

cudaError_t dequantize_q6_k(const void *quantized, half *dequantized,
                            size_t num_elements, cudaStream_t stream) {

  dim3 grid = calc_dequant_grid(num_elements);
  dim3 block = calc_dequant_block();

  dequantize_q6_k_kernel<<<grid, block, 0, stream>>>(
      static_cast<const block_q6_k *>(quantized), dequantized, num_elements);

  return cudaGetLastError();
}

//==============================================================================
// Q8_0 Dequantization Kernel
//==============================================================================

// Uses shared dequant_q8_0_element() from quant_common.cuh.
__global__ void dequantize_q8_0_kernel(const block_q8_0 *quantized,
                                       half *dequantized, size_t num_elements) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int num_blocks = (num_elements + QK8_0 - 1) / QK8_0;

  if (idx >= num_blocks) {
    return;
  }

  const block_q8_0 &block = quantized[idx];
  int start = idx * QK8_0;

  float d = __half2float(__ushort_as_half(block.d));

  for (int i = 0; i < QK8_0 && (start + i) < (int)num_elements; ++i) {
    dequantized[start + i] =
        __float2half(dequant_q8_0_element(block, d, i));
  }
}

cudaError_t dequantize_q8_0(const void *quantized, half *dequantized,
                            size_t num_elements, cudaStream_t stream) {

  int num_blocks = (num_elements + QK8_0 - 1) / QK8_0;
  dim3 grid((num_blocks + 255) / 256);
  dim3 block(256);

  dequantize_q8_0_kernel<<<grid, block, 0, stream>>>(
      static_cast<const block_q8_0 *>(quantized), dequantized, num_elements);

  return cudaGetLastError();
}

//==============================================================================
// Q8_K Dequantization Kernel
//==============================================================================

// Uses shared dequant_q8k_element() from quant_common.cuh.
__global__ void dequantize_q8_k_kernel(const block_q8_k *quantized,
                                       half *dequantized, size_t num_elements) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int num_blocks = (num_elements + QK_K - 1) / QK_K;

  if (idx >= num_blocks) {
    return;
  }

  const block_q8_k &block = quantized[idx];
  int start = idx * QK_K;

  float d = block.d;

  for (int i = 0; i < QK_K && (start + i) < (int)num_elements; ++i) {
    dequantized[start + i] =
        __float2half(dequant_q8k_element(block, d, i));
  }
}

cudaError_t dequantize_q8_k(const void *quantized, half *dequantized,
                            size_t num_elements, cudaStream_t stream) {

  dim3 grid = calc_dequant_grid(num_elements);
  dim3 block = calc_dequant_block();

  dequantize_q8_k_kernel<<<grid, block, 0, stream>>>(
      static_cast<const block_q8_k *>(quantized), dequantized, num_elements);

  return cudaGetLastError();
}

} // namespace native
} // namespace cuda
} // namespace runtime
} // namespace inferflux
