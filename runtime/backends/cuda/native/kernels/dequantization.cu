#include "runtime/backends/cuda/native/kernels/dequantization.cuh"

namespace inferflux {
namespace runtime {
namespace cuda {
namespace native {

//==============================================================================
// Helper Functions
//==============================================================================

// Get scales and mins for Q4_K/Q5_K (6-bit quantized)
__device__ inline void get_scale_min_qk(
    const unsigned char *scales,
    int i,
    half &d,
    half &m,
    const unsigned short *dm_raw) {
  // Extract 6-bit scales and mins
  // 12 scales packed into 12 bytes (each 6 bits)
  // Each scale/min pair represents one of 8 blocks

  int is = (i % 8) / 2; // Which scale index (0-5)

  unsigned char sc = scales[is];
  unsigned char mn = scales[is + 6];

  // Unpack 6-bit values
  float scale_f = (sc & 0x3F) * (1.0f / 32.0f);
  float min_f = (mn & 0x3F) * (1.0f / 32.0f);

  // Convert raw uint16 to half and apply
  half d_raw = __ushort_as_half(dm_raw[0]);
  half m_raw = __ushort_as_half(dm_raw[1]);

  // Apply super-block scales
  d = __float2half(__half2float(d_raw) * scale_f);
  m = __float2half(__half2float(m_raw) * min_f);
}

//==============================================================================
// Q4_K Dequantization Kernel
//==============================================================================

__global__ void dequantize_q4_k_kernel(
    const block_q4_k *quantized,
    half *dequantized,
    size_t num_elements) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int num_blocks = (num_elements + QK_K - 1) / QK_K;

  if (idx >= num_blocks) {
    return;
  }

  const block_q4_k &block = quantized[idx];

  int start = idx * QK_K;
  int end = min(start + QK_K, (int)num_elements);

  // Dequantize 256 values
  for (int i = 0; i < QK_K && (start + i) < (int)num_elements; ++i) {
    half d, m;
    const unsigned short dm_raw[2] = {block.d, block.dmin};
    get_scale_min_qk(block.scales, i, d, m, dm_raw);

    // Get 4-bit quantized value
    int ib = i / 8; // Which byte in qs[]
    int qs_idx = i / 2;
    int qp = i % 2;

    unsigned char q = block.qs[qs_idx];
    int q_val = (qp == 0) ? (q & 0x0F) : (q >> 4);

    // Dequantize: x = d * q + m
    float x = __half2float(d) * q_val + __half2float(m);
    dequantized[start + i] = __float2half(x);
  }
}

cudaError_t dequantize_q4_k(
    const void *quantized,
    half *dequantized,
    size_t num_elements,
    cudaStream_t stream) {

  dim3 grid = calc_dequant_grid(num_elements);
  dim3 block = calc_dequant_block();

  dequantize_q4_k_kernel<<<grid, block, 0, stream>>>(
      static_cast<const block_q4_k *>(quantized),
      dequantized,
      num_elements);

  return cudaGetLastError();
}

//==============================================================================
// Q5_K Dequantization Kernel
//==============================================================================

__global__ void dequantize_q5_k_kernel(
    const block_q5_k *quantized,
    half *dequantized,
    size_t num_elements) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int num_blocks = (num_elements + QK_K - 1) / QK_K;

  if (idx >= num_blocks) {
    return;
  }

  const block_q5_k &block = quantized[idx];

  int start = idx * QK_K;
  int end = min(start + QK_K, (int)num_elements);

  // Dequantize 256 values
  for (int i = 0; i < QK_K && (start + i) < (int)num_elements; ++i) {
    half d, m;
    const unsigned short dm_raw[2] = {block.d, block.dmin};
    get_scale_min_qk(block.scales, i, d, m, dm_raw);

    // Get 5-bit quantized value (4 low bits + 1 high bit)
    int qs_idx = i / 2;
    int qp = i % 2;

    unsigned char ql = block.qs[qs_idx];
    int q_low = (qp == 0) ? (ql & 0x0F) : (ql >> 4);

    int qh_idx = i / 8;
    int qh_shift = i % 8;
    int q_high = (block.qh[qh_idx] >> qh_shift) & 1;

    int q_val = q_low | (q_high << 4);

    // Dequantize: x = d * q + m
    float x = __half2float(d) * q_val + __half2float(m);
    dequantized[start + i] = __float2half(x);
  }
}

cudaError_t dequantize_q5_k(
    const void *quantized,
    half *dequantized,
    size_t num_elements,
    cudaStream_t stream) {

  dim3 grid = calc_dequant_grid(num_elements);
  dim3 block = calc_dequant_block();

  dequantize_q5_k_kernel<<<grid, block, 0, stream>>>(
      static_cast<const block_q5_k *>(quantized),
      dequantized,
      num_elements);

  return cudaGetLastError();
}

//==============================================================================
// Q6_K Dequantization Kernel
//==============================================================================

__global__ void dequantize_q6_k_kernel(
    const block_q6_k *quantized,
    half *dequantized,
    size_t num_elements) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int num_blocks = (num_elements + QK_K - 1) / QK_K;

  if (idx >= num_blocks) {
    return;
  }

  const block_q6_k &block = quantized[idx];

  int start = idx * QK_K;
  int end = min(start + QK_K, (int)num_elements);

  half d = __ushort_as_half(block.d);

  // Dequantize 256 values
  for (int i = 0; i < QK_K && (start + i) < (int)num_elements; ++i) {
    // Get scale for this group of 16 values
    int scale_idx = i / 16;
    half scale = __float2half(block.scales[scale_idx] * (1.0f / 64.0f));

    // Get 6-bit quantized value (4 low bits + 2 high bits)
    int ql_idx = i / 2;
    int ql_qp = i % 2;
    unsigned char ql = block.ql[ql_idx];
    int q_low = (ql_qp == 0) ? (ql & 0x0F) : (ql >> 4);

    int qh_idx = i / 4;
    int qh_qp = i % 4;
    int qh_shift = qh_qp * 2;
    unsigned char qh = block.qh[qh_idx];
    int q_high = (qh >> qh_shift) & 0x03;

    int q_val = q_low | (q_high << 4);

    // Dequantize: x = d * scale * q
    float x = __half2float(d) * __half2float(scale) * q_val;
    dequantized[start + i] = __float2half(x);
  }
}

cudaError_t dequantize_q6_k(
    const void *quantized,
    half *dequantized,
    size_t num_elements,
    cudaStream_t stream) {

  dim3 grid = calc_dequant_grid(num_elements);
  dim3 block = calc_dequant_block();

  dequantize_q6_k_kernel<<<grid, block, 0, stream>>>(
      static_cast<const block_q6_k *>(quantized),
      dequantized,
      num_elements);

  return cudaGetLastError();
}

//==============================================================================
// Q8_0 Dequantization Kernel
//==============================================================================

__global__ void dequantize_q8_0_kernel(
    const block_q8_0 *quantized,
    half *dequantized,
    size_t num_elements) {

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

cudaError_t dequantize_q8_0(
    const void *quantized,
    half *dequantized,
    size_t num_elements,
    cudaStream_t stream) {

  // Q8_0 uses 32-value blocks (not 256 like K-series)
  int num_blocks = (num_elements + QK8_0 - 1) / QK8_0;
  dim3 grid((num_blocks + 255) / 256);
  dim3 block(256);

  dequantize_q8_0_kernel<<<grid, block, 0, stream>>>(
      static_cast<const block_q8_0 *>(quantized),
      dequantized,
      num_elements);

  return cudaGetLastError();
}

} // namespace native
} // namespace cuda
} // namespace runtime
} // namespace inferflux
