#include "runtime/backends/cuda/native/kernels/dequantization.cuh"

namespace inferflux {
namespace runtime {
namespace cuda {
namespace native {

//==============================================================================
// Helper Functions
//==============================================================================

// Matches llama.cpp get_scale_min_k4: extract 6-bit scale and min for Q4_K/Q5_K
__device__ inline void get_scale_min_k4(int j, const unsigned char *q,
                                        unsigned char *d, unsigned char *m) {
  if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

//==============================================================================
// Q4_K Dequantization Kernel
//==============================================================================

// Matches llama.cpp dequantize_row_q4_K exactly.
// Layout: 4 groups of 64 elements. Each group has 2 sub-blocks sharing 32 qs
// bytes. Sub-block 2i uses low nibbles, sub-block 2i+1 uses high nibbles.
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

  const unsigned char *qs = b.qs;
  int is = 0;

  for (int j = 0; j < QK_K; j += 64) {
    unsigned char sc, m;
    get_scale_min_k4(is + 0, b.scales, &sc, &m);
    float d1 = d * sc;
    float m1 = dmin * m;
    get_scale_min_k4(is + 1, b.scales, &sc, &m);
    float d2 = d * sc;
    float m2 = dmin * m;

    for (int l = 0; l < 32; ++l) {
      int out_idx = start + j + l;
      if (out_idx < (int)num_elements) {
        dequantized[out_idx] = __float2half(d1 * (qs[l] & 0xF) - m1);
      }
    }
    for (int l = 0; l < 32; ++l) {
      int out_idx = start + j + 32 + l;
      if (out_idx < (int)num_elements) {
        dequantized[out_idx] = __float2half(d2 * (qs[l] >> 4) - m2);
      }
    }
    qs += 32;
    is += 2;
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

// Matches llama.cpp dequantize_row_q5_K exactly.
// Same paired low/high nibble layout as Q4_K but with an extra high bit per
// element stored in qh[].
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

  const unsigned char *ql = b.qs;
  const unsigned char *qh = b.qh;
  int is = 0;
  unsigned char u1 = 1, u2 = 2;

  for (int j = 0; j < QK_K; j += 64) {
    unsigned char sc, m;
    get_scale_min_k4(is + 0, b.scales, &sc, &m);
    float d1 = d * sc;
    float m1 = dmin * m;
    get_scale_min_k4(is + 1, b.scales, &sc, &m);
    float d2 = d * sc;
    float m2 = dmin * m;

    for (int l = 0; l < 32; ++l) {
      int out_idx = start + j + l;
      if (out_idx < (int)num_elements) {
        int q_val = (ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0);
        dequantized[out_idx] = __float2half(d1 * q_val - m1);
      }
    }
    for (int l = 0; l < 32; ++l) {
      int out_idx = start + j + 32 + l;
      if (out_idx < (int)num_elements) {
        int q_val = (ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0);
        dequantized[out_idx] = __float2half(d2 * q_val - m2);
      }
    }
    ql += 32;
    is += 2;
    u1 <<= 2;
    u2 <<= 2;
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

// Matches llama.cpp dequantize_row_q6_K exactly.
// Layout: 2 groups of 128, each with 4 sub-groups of 32 sharing ql/qh arrays.
// Sub 0: low nibble of ql[base..+31],    qh bits 0-1
// Sub 1: low nibble of ql[base+32..+63], qh bits 2-3
// Sub 2: high nibble of ql[base..+31],   qh bits 4-5
// Sub 3: high nibble of ql[base+32..+63],qh bits 6-7
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

  const unsigned char *ql = b.ql;
  const unsigned char *qh = b.qh;
  const char *sc = (const char *)b.scales;

  for (int n = 0; n < QK_K; n += 128) {
    for (int l = 0; l < 32; ++l) {
      int is = l / 16;
      int q1 = (int)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
      int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
      int q3 = (int)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
      int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

      int out0 = start + n + l;
      int out1 = start + n + 32 + l;
      int out2 = start + n + 64 + l;
      int out3 = start + n + 96 + l;

      if (out0 < (int)num_elements)
        dequantized[out0] = __float2half(d * sc[is + 0] * q1);
      if (out1 < (int)num_elements)
        dequantized[out1] = __float2half(d * sc[is + 2] * q2);
      if (out2 < (int)num_elements)
        dequantized[out2] = __float2half(d * sc[is + 4] * q3);
      if (out3 < (int)num_elements)
        dequantized[out3] = __float2half(d * sc[is + 6] * q4);
    }
    ql += 64;
    qh += 32;
    sc += 8;
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

__global__ void dequantize_q8_0_kernel(const block_q8_0 *quantized,
                                       half *dequantized, size_t num_elements) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int num_blocks = (num_elements + QK8_0 - 1) / QK8_0;

  if (idx >= num_blocks) {
    return;
  }

  const block_q8_0 &block = quantized[idx];

  int start = idx * QK8_0;
  int end = min(start + QK8_0, (int)num_elements);

  // Convert scale from uint16 to half
  half d = __ushort_as_half(block.d);

  // Dequantize 32 values: x = qs[i] * d
  for (int i = 0; i < QK8_0 && (start + i) < (int)num_elements; ++i) {
    float x = (float)block.qs[i] * __half2float(d);
    dequantized[start + i] = __float2half(x);
  }
}

cudaError_t dequantize_q8_0(const void *quantized, half *dequantized,
                            size_t num_elements, cudaStream_t stream) {

  // Q8_0 uses 32-value blocks (not 256 like K-series)
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

  // Dequantize 256 values: x = qs[i] * d
  for (int i = 0; i < QK_K && (start + i) < (int)num_elements; ++i) {
    float x = (float)block.qs[i] * d;
    dequantized[start + i] = __float2half(x);
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
