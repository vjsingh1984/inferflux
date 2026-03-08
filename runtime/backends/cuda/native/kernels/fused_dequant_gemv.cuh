#pragma once

#include "runtime/backends/cuda/native/kernels/quant_common.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace inferflux {
namespace runtime {
namespace cuda {
namespace native {

// Number of warps per thread block for GEMV kernels.
// x vector is cooperatively loaded into shared memory by all threads in the
// block, then each warp computes one output element reading x from smem.
constexpr int kGemvWarpsPerBlock = 8;
constexpr int kGemvThreadsPerBlock = kGemvWarpsPerBlock * 32;

__device__ __forceinline__ int Dp4aS8(int a, int b, int c) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610)
  return __dp4a(a, b, c);
#else
  const auto *ap = reinterpret_cast<const signed char *>(&a);
  const auto *bp = reinterpret_cast<const signed char *>(&b);
  return c + static_cast<int>(ap[0]) * static_cast<int>(bp[0]) +
         static_cast<int>(ap[1]) * static_cast<int>(bp[1]) +
         static_cast<int>(ap[2]) * static_cast<int>(bp[2]) +
         static_cast<int>(ap[3]) * static_cast<int>(bp[3]);
#endif
}

// Vectorized half→float shared memory load using half2.
// Loads 2 half values (4 bytes) per thread per iteration → 2x bandwidth.
// K must be even (guaranteed: Q4_K/Q6_K/Q8_K use K%256==0, Q8_0 uses K%32==0).
__device__ __forceinline__ void
LoadHalfToSmem(const half *__restrict__ src, float *__restrict__ dst, int K,
               int tid) {
  const int K2 = K / 2;
  for (int i = tid; i < K2; i += kGemvThreadsPerBlock) {
    __half2 h2 = reinterpret_cast<const __half2 *>(src)[i];
    float2 f2 = __half22float2(h2);
    dst[2 * i] = f2.x;
    dst[2 * i + 1] = f2.y;
  }
}

// Vectorized half→float load with sum-of-squares accumulation (for RmsNorm).
__device__ __forceinline__ float
LoadHalfToSmemSumSq(const half *__restrict__ src, float *__restrict__ dst,
                    int K, int tid) {
  float local_sum_sq = 0.0f;
  const int K2 = K / 2;
  for (int i = tid; i < K2; i += kGemvThreadsPerBlock) {
    __half2 h2 = reinterpret_cast<const __half2 *>(src)[i];
    float2 f2 = __half22float2(h2);
    local_sum_sq += f2.x * f2.x + f2.y * f2.y;
    dst[2 * i] = f2.x;
    dst[2 * i + 1] = f2.y;
  }
  return local_sum_sq;
}

// Vectorized half→float load with max-abs accumulation (for dp4a quantization).
__device__ __forceinline__ float
LoadHalfToSmemMaxAbs(const half *__restrict__ src, float *__restrict__ dst,
                     int K, int tid) {
  float local_max = 0.0f;
  const int K2 = K / 2;
  for (int i = tid; i < K2; i += kGemvThreadsPerBlock) {
    __half2 h2 = reinterpret_cast<const __half2 *>(src)[i];
    float2 f2 = __half22float2(h2);
    local_max = fmaxf(local_max, fmaxf(fabsf(f2.x), fabsf(f2.y)));
    dst[2 * i] = f2.x;
    dst[2 * i + 1] = f2.y;
  }
  return local_max;
}

// Vectorized in-place RmsNorm application with half2 norm weight loads.
__device__ __forceinline__ void
ApplyNormInPlace(float *__restrict__ sx, const half *__restrict__ norm_weight,
                 float rms, int K, int tid) {
  const int K2 = K / 2;
  for (int i = tid; i < K2; i += kGemvThreadsPerBlock) {
    __half2 nw = reinterpret_cast<const __half2 *>(norm_weight)[i];
    float2 nf = __half22float2(nw);
    sx[2 * i] = sx[2 * i] * rms * nf.x;
    sx[2 * i + 1] = sx[2 * i + 1] * rms * nf.y;
  }
}

// Vectorized in-place RmsNorm application with max-abs tracking (for dp4a).
__device__ __forceinline__ float
ApplyNormInPlaceMaxAbs(float *__restrict__ sx,
                       const half *__restrict__ norm_weight, float rms, int K,
                       int tid) {
  float local_max = 0.0f;
  const int K2 = K / 2;
  for (int i = tid; i < K2; i += kGemvThreadsPerBlock) {
    __half2 nw = reinterpret_cast<const __half2 *>(norm_weight)[i];
    float2 nf = __half22float2(nw);
    float v0 = sx[2 * i] * rms * nf.x;
    float v1 = sx[2 * i + 1] * rms * nf.y;
    sx[2 * i] = v0;
    sx[2 * i + 1] = v1;
    local_max = fmaxf(local_max, fmaxf(fabsf(v0), fabsf(v1)));
  }
  return local_max;
}

//==============================================================================
// Q4_K Fused Dequant-GEMV (shared-memory x cache, multi-row via blockIdx.y)
//==============================================================================

// Grid: (ceil(N / kGemvWarpsPerBlock), M)
// Each block at (bx, row) loads x[row] into smem and computes output[row][cols].
//
// Inner loop: reads each qs byte once (both nibbles), vectorizes qs loads as
// int32 (4 bytes = 8 elements), and pre-extracts scale/min per sub-block pair.
__global__ void fused_dequant_gemv_q4k(const block_q4_k *__restrict__ weight,
                                       const half *__restrict__ x,
                                       half *__restrict__ output, int N,
                                       int K) {
  extern __shared__ float sx[];

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row = blockIdx.y; // Input row index (0..M-1)

  // Load x[row] into shared memory (vectorized half2 loads)
  const half *x_row = x + row * K;
  LoadHalfToSmem(x_row, sx, K, tid);
  __syncthreads();

  if (out_idx >= N)
    return;

  const int num_blocks = K / QK_K;
  const block_q4_k *wrow = weight + out_idx * num_blocks;

  float acc = 0.0f;

  for (int blk = 0; blk < num_blocks; ++blk) {
    const block_q4_k &b = wrow[blk];

    const float d = __half2float(*reinterpret_cast<const half *>(&b.d));
    const float dmin = __half2float(*reinterpret_cast<const half *>(&b.dmin));

    // Process sub-blocks in pairs (0&1, 2&3, 4&5, 6&7).
    // Each pair shares the same qs byte: qs[pair*32+lane] holds both nibbles.
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
      const int sb_lo = pair * 2;
      const int sb_hi = pair * 2 + 1;

      // Extract scales and mins for both sub-blocks (once per pair)
      unsigned char sc_lo, m_lo, sc_hi, m_hi;
      get_scale_min_k4(sb_lo, b.scales, &sc_lo, &m_lo);
      get_scale_min_k4(sb_hi, b.scales, &sc_hi, &m_hi);

      // Pre-compute per-sub-block scale factors
      const float d_sc_lo = d * static_cast<float>(sc_lo);
      const float dm_m_lo = dmin * static_cast<float>(m_lo);
      const float d_sc_hi = d * static_cast<float>(sc_hi);
      const float dm_m_hi = dmin * static_cast<float>(m_hi);

      // Read qs byte once — low nibble is sb_lo, high nibble is sb_hi
      const unsigned char qbyte = b.qs[pair * 32 + lane];
      const float q_lo = static_cast<float>(qbyte & 0x0F);
      const float q_hi = static_cast<float>(qbyte >> 4);

      const int base = blk * QK_K + pair * 64;
      acc += (d_sc_lo * q_lo - dm_m_lo) * sx[base + lane];
      acc += (d_sc_hi * q_hi - dm_m_hi) * sx[base + 32 + lane];
    }
  }

  // Warp reduction
  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
  }

  if (lane == 0) {
    output[row * N + out_idx] = __float2half(acc);
  }
}

//==============================================================================
// Q6_K Fused Dequant-GEMV (shared-memory x cache, multi-row via blockIdx.y)
//==============================================================================

__global__ void fused_dequant_gemv_q6k(const block_q6_k *__restrict__ weight,
                                       const half *__restrict__ x,
                                       half *__restrict__ output, int N,
                                       int K) {
  extern __shared__ float sx[];

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row = blockIdx.y;

  const half *x_row = x + row * K;
  LoadHalfToSmem(x_row, sx, K, tid);
  __syncthreads();

  if (out_idx >= N)
    return;

  const int num_blocks = K / QK_K;
  const block_q6_k *wrow = weight + out_idx * num_blocks;

  float acc = 0.0f;

  for (int blk = 0; blk < num_blocks; ++blk) {
    const block_q6_k &b = wrow[blk];

    float d = __half2float(*reinterpret_cast<const half *>(&b.d));

    // Process in group pairs: subs 0&2 share ql byte (low/high nibble),
    // subs 1&3 share another ql byte. Read each byte once.
#pragma unroll
    for (int g = 0; g < 2; ++g) {
      // Pair subs 0 & 2 (same ql byte at g*64+lane)
      {
        const int ql_idx = g * 64 + lane;
        const unsigned char ql_byte = b.ql[ql_idx];
        const int qh_idx = g * 32 + lane;
        const unsigned char qh_byte = b.qh[qh_idx];

        const int ql0 = ql_byte & 0x0F;
        const int qh0 = (qh_byte >> 0) & 0x03;
        const int q0 = (ql0 | (qh0 << 4)) - 32;
        const int sc0 = g * 8 + 0 * 2 + lane / 16;
        acc += d * static_cast<float>(b.scales[sc0]) * static_cast<float>(q0) *
               sx[blk * QK_K + g * 128 + lane];

        const int ql2 = ql_byte >> 4;
        const int qh2 = (qh_byte >> 4) & 0x03;
        const int q2 = (ql2 | (qh2 << 4)) - 32;
        const int sc2 = g * 8 + 2 * 2 + lane / 16;
        acc += d * static_cast<float>(b.scales[sc2]) * static_cast<float>(q2) *
               sx[blk * QK_K + g * 128 + 64 + lane];
      }
      // Pair subs 1 & 3 (same ql byte at g*64+32+lane)
      {
        const int ql_idx = g * 64 + 32 + lane;
        const unsigned char ql_byte = b.ql[ql_idx];
        const int qh_idx = g * 32 + lane;
        const unsigned char qh_byte = b.qh[qh_idx];

        const int ql1 = ql_byte & 0x0F;
        const int qh1 = (qh_byte >> 2) & 0x03;
        const int q1 = (ql1 | (qh1 << 4)) - 32;
        const int sc1 = g * 8 + 1 * 2 + lane / 16;
        acc += d * static_cast<float>(b.scales[sc1]) * static_cast<float>(q1) *
               sx[blk * QK_K + g * 128 + 32 + lane];

        const int ql3 = ql_byte >> 4;
        const int qh3 = (qh_byte >> 6) & 0x03;
        const int q3 = (ql3 | (qh3 << 4)) - 32;
        const int sc3 = g * 8 + 3 * 2 + lane / 16;
        acc += d * static_cast<float>(b.scales[sc3]) * static_cast<float>(q3) *
               sx[blk * QK_K + g * 128 + 96 + lane];
      }
    }
  }

  // Warp reduction
  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
  }

  if (lane == 0) {
    output[row * N + out_idx] = __float2half(acc);
  }
}

//==============================================================================
// Q8_0 Fused Dequant-GEMV (shared-memory x cache, multi-row via blockIdx.y)
//==============================================================================

__global__ void fused_dequant_gemv_q8_0(const block_q8_0 *__restrict__ weight,
                                        const half *__restrict__ x,
                                        half *__restrict__ output, int N,
                                        int K) {
  extern __shared__ float sx[];

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row = blockIdx.y;

  const half *x_row = x + row * K;
  LoadHalfToSmem(x_row, sx, K, tid);
  __syncthreads();

  if (out_idx >= N)
    return;

  const int num_blocks = K / QK8_0;
  const block_q8_0 *wrow = weight + out_idx * num_blocks;

  float acc = 0.0f;

  for (int blk = 0; blk < num_blocks; ++blk) {
    const block_q8_0 &b = wrow[blk];
    float d = __half2float(*reinterpret_cast<const half *>(&b.d));

    int elem_idx = blk * QK8_0 + lane;
    float w_val = dequant_q8_0_element(b, d, lane);
    acc += w_val * sx[elem_idx];
  }

  // Warp reduction
  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
  }

  if (lane == 0) {
    output[row * N + out_idx] = __float2half(acc);
  }
}

//==============================================================================
// Q8_K Fused Dequant-GEMV (shared-memory x cache, multi-row via blockIdx.y)
//==============================================================================

__global__ void fused_dequant_gemv_q8k(const block_q8_k *__restrict__ weight,
                                       const half *__restrict__ x,
                                       half *__restrict__ output, int N,
                                       int K) {
  extern __shared__ float sx[];

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row = blockIdx.y;

  const half *x_row = x + row * K;
  LoadHalfToSmem(x_row, sx, K, tid);
  __syncthreads();

  if (out_idx >= N)
    return;

  const int num_blocks = K / QK_K;
  const block_q8_k *wrow = weight + out_idx * num_blocks;

  float acc = 0.0f;

  for (int blk = 0; blk < num_blocks; ++blk) {
    const block_q8_k &b = wrow[blk];
    float d = b.d;

    // 8 steps of 32 elements each = 256 elements per block
    for (int step = 0; step < 8; ++step) {
      int elem = step * 32 + lane;
      int elem_idx = blk * QK_K + elem;
      float w_val = dequant_q8k_element(b, d, elem);
      acc += w_val * sx[elem_idx];
    }
  }

  // Warp reduction
  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
  }

  if (lane == 0) {
    output[row * N + out_idx] = __float2half(acc);
  }
}

//==============================================================================
// Fused RmsNorm+GEMV Kernels
//
// These kernels compute RmsNorm inline during the shared memory loading phase,
// eliminating the standalone RmsNorm kernel and intermediate d_norm_out_ buffer.
// Each GEMV independently normalizes from the residual — the norm computation
// is cheap (~1% of GEMV compute) and amortized across 8 warps per block.
//
// Shared memory layout: sx[K] (floats) + warp_sums[kGemvWarpsPerBlock] (floats)
//==============================================================================

//==============================================================================
// Q4_K Fused RmsNorm+GEMV
//==============================================================================

__global__ void
fused_rmsnorm_gemv_q4k(const block_q4_k *__restrict__ weight,
                       const half *__restrict__ residual,
                       const half *__restrict__ norm_weight,
                       half *__restrict__ output, int N, int K, float eps) {
  extern __shared__ char smem_raw[];
  float *sx = reinterpret_cast<float *>(smem_raw);
  float *warp_sums = sx + K;

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row = blockIdx.y;

  // Phase 1: Load residual into smem (vectorized) and compute sum-of-squares
  const half *res_row = residual + row * K;
  float local_sum_sq = LoadHalfToSmemSumSq(res_row, sx, K, tid);

  // Warp-level shuffle reduction
  for (int offset = 16; offset > 0; offset >>= 1) {
    local_sum_sq += __shfl_down_sync(0xFFFFFFFF, local_sum_sq, offset);
  }
  if (lane == 0)
    warp_sums[warp_id] = local_sum_sq;
  __syncthreads();

  // Final reduction across warps (thread 0 only)
  float rms;
  if (tid == 0) {
    float total = 0.0f;
    for (int w = 0; w < kGemvWarpsPerBlock; ++w)
      total += warp_sums[w];
    warp_sums[0] = rsqrtf(total / static_cast<float>(K) + eps);
  }
  __syncthreads();
  rms = warp_sums[0];

  // Phase 2: Normalize in-place (vectorized half2 norm weight loads)
  ApplyNormInPlace(sx, norm_weight, rms, K, tid);
  __syncthreads();

  // Phase 3: Standard GEMV dot product
  if (out_idx >= N)
    return;

  const int num_blocks = K / QK_K;
  const block_q4_k *wrow = weight + out_idx * num_blocks;

  float acc = 0.0f;
  for (int blk = 0; blk < num_blocks; ++blk) {
    const block_q4_k &b = wrow[blk];
    const float d = __half2float(*reinterpret_cast<const half *>(&b.d));
    const float dmin = __half2float(*reinterpret_cast<const half *>(&b.dmin));

#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
      const int sb_lo = pair * 2;
      const int sb_hi = pair * 2 + 1;

      unsigned char sc_lo, m_lo, sc_hi, m_hi;
      get_scale_min_k4(sb_lo, b.scales, &sc_lo, &m_lo);
      get_scale_min_k4(sb_hi, b.scales, &sc_hi, &m_hi);

      const float d_sc_lo = d * static_cast<float>(sc_lo);
      const float dm_m_lo = dmin * static_cast<float>(m_lo);
      const float d_sc_hi = d * static_cast<float>(sc_hi);
      const float dm_m_hi = dmin * static_cast<float>(m_hi);

      const unsigned char qbyte = b.qs[pair * 32 + lane];
      const float q_lo = static_cast<float>(qbyte & 0x0F);
      const float q_hi = static_cast<float>(qbyte >> 4);

      const int base = blk * QK_K + pair * 64;
      acc += (d_sc_lo * q_lo - dm_m_lo) * sx[base + lane];
      acc += (d_sc_hi * q_hi - dm_m_hi) * sx[base + 32 + lane];
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
  }
  if (lane == 0) {
    output[row * N + out_idx] = __float2half(acc);
  }
}

//==============================================================================
// Q6_K Fused RmsNorm+GEMV
//==============================================================================

__global__ void
fused_rmsnorm_gemv_q6k(const block_q6_k *__restrict__ weight,
                       const half *__restrict__ residual,
                       const half *__restrict__ norm_weight,
                       half *__restrict__ output, int N, int K, float eps) {
  extern __shared__ char smem_raw[];
  float *sx = reinterpret_cast<float *>(smem_raw);
  float *warp_sums = sx + K;

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row = blockIdx.y;

  const half *res_row = residual + row * K;
  float local_sum_sq = LoadHalfToSmemSumSq(res_row, sx, K, tid);
  for (int offset = 16; offset > 0; offset >>= 1) {
    local_sum_sq += __shfl_down_sync(0xFFFFFFFF, local_sum_sq, offset);
  }
  if (lane == 0)
    warp_sums[warp_id] = local_sum_sq;
  __syncthreads();

  float rms;
  if (tid == 0) {
    float total = 0.0f;
    for (int w = 0; w < kGemvWarpsPerBlock; ++w)
      total += warp_sums[w];
    warp_sums[0] = rsqrtf(total / static_cast<float>(K) + eps);
  }
  __syncthreads();
  rms = warp_sums[0];

  ApplyNormInPlace(sx, norm_weight, rms, K, tid);
  __syncthreads();

  if (out_idx >= N)
    return;

  const int num_blocks = K / QK_K;
  const block_q6_k *wrow = weight + out_idx * num_blocks;

  float acc = 0.0f;
  for (int blk = 0; blk < num_blocks; ++blk) {
    const block_q6_k &b = wrow[blk];
    const float d = __half2float(*reinterpret_cast<const half *>(&b.d));

#pragma unroll
    for (int g = 0; g < 2; ++g) {
      {
        const int ql_idx = g * 64 + lane;
        const unsigned char ql_byte = b.ql[ql_idx];
        const int qh_idx = g * 32 + lane;
        const unsigned char qh_byte = b.qh[qh_idx];

        const int ql0 = ql_byte & 0x0F;
        const int qh0 = (qh_byte >> 0) & 0x03;
        const int q0 = (ql0 | (qh0 << 4)) - 32;
        const int sc0 = g * 8 + 0 * 2 + lane / 16;
        acc += d * static_cast<float>(b.scales[sc0]) * static_cast<float>(q0) *
               sx[blk * QK_K + g * 128 + lane];

        const int ql2 = ql_byte >> 4;
        const int qh2 = (qh_byte >> 4) & 0x03;
        const int q2 = (ql2 | (qh2 << 4)) - 32;
        const int sc2 = g * 8 + 2 * 2 + lane / 16;
        acc += d * static_cast<float>(b.scales[sc2]) * static_cast<float>(q2) *
               sx[blk * QK_K + g * 128 + 64 + lane];
      }
      {
        const int ql_idx = g * 64 + 32 + lane;
        const unsigned char ql_byte = b.ql[ql_idx];
        const int qh_idx = g * 32 + lane;
        const unsigned char qh_byte = b.qh[qh_idx];

        const int ql1 = ql_byte & 0x0F;
        const int qh1 = (qh_byte >> 2) & 0x03;
        const int q1 = (ql1 | (qh1 << 4)) - 32;
        const int sc1 = g * 8 + 1 * 2 + lane / 16;
        acc += d * static_cast<float>(b.scales[sc1]) * static_cast<float>(q1) *
               sx[blk * QK_K + g * 128 + 32 + lane];

        const int ql3 = ql_byte >> 4;
        const int qh3 = (qh_byte >> 6) & 0x03;
        const int q3 = (ql3 | (qh3 << 4)) - 32;
        const int sc3 = g * 8 + 3 * 2 + lane / 16;
        acc += d * static_cast<float>(b.scales[sc3]) * static_cast<float>(q3) *
               sx[blk * QK_K + g * 128 + 96 + lane];
      }
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
  }
  if (lane == 0) {
    output[row * N + out_idx] = __float2half(acc);
  }
}

//==============================================================================
// Q8_0 Fused RmsNorm+GEMV
//==============================================================================

__global__ void
fused_rmsnorm_gemv_q8_0(const block_q8_0 *__restrict__ weight,
                        const half *__restrict__ residual,
                        const half *__restrict__ norm_weight,
                        half *__restrict__ output, int N, int K, float eps) {
  extern __shared__ char smem_raw[];
  float *sx = reinterpret_cast<float *>(smem_raw);
  float *warp_sums = sx + K;

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row = blockIdx.y;

  const half *res_row = residual + row * K;
  float local_sum_sq = LoadHalfToSmemSumSq(res_row, sx, K, tid);
  for (int offset = 16; offset > 0; offset >>= 1) {
    local_sum_sq += __shfl_down_sync(0xFFFFFFFF, local_sum_sq, offset);
  }
  if (lane == 0)
    warp_sums[warp_id] = local_sum_sq;
  __syncthreads();

  float rms;
  if (tid == 0) {
    float total = 0.0f;
    for (int w = 0; w < kGemvWarpsPerBlock; ++w)
      total += warp_sums[w];
    warp_sums[0] = rsqrtf(total / static_cast<float>(K) + eps);
  }
  __syncthreads();
  rms = warp_sums[0];

  ApplyNormInPlace(sx, norm_weight, rms, K, tid);
  __syncthreads();

  if (out_idx >= N)
    return;

  const int num_blocks = K / QK8_0;
  const block_q8_0 *wrow = weight + out_idx * num_blocks;

  float acc = 0.0f;
  for (int blk = 0; blk < num_blocks; ++blk) {
    const block_q8_0 &b = wrow[blk];
    float d = __half2float(*reinterpret_cast<const half *>(&b.d));
    int elem_idx = blk * QK8_0 + lane;
    float w_val = dequant_q8_0_element(b, d, lane);
    acc += w_val * sx[elem_idx];
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
  }
  if (lane == 0) {
    output[row * N + out_idx] = __float2half(acc);
  }
}

//==============================================================================
// Q8_K Fused RmsNorm+GEMV
//==============================================================================

__global__ void
fused_rmsnorm_gemv_q8k(const block_q8_k *__restrict__ weight,
                       const half *__restrict__ residual,
                       const half *__restrict__ norm_weight,
                       half *__restrict__ output, int N, int K, float eps) {
  extern __shared__ char smem_raw[];
  float *sx = reinterpret_cast<float *>(smem_raw);
  float *warp_sums = sx + K;

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row = blockIdx.y;

  const half *res_row = residual + row * K;
  float local_sum_sq = LoadHalfToSmemSumSq(res_row, sx, K, tid);
  for (int offset = 16; offset > 0; offset >>= 1) {
    local_sum_sq += __shfl_down_sync(0xFFFFFFFF, local_sum_sq, offset);
  }
  if (lane == 0)
    warp_sums[warp_id] = local_sum_sq;
  __syncthreads();

  float rms;
  if (tid == 0) {
    float total = 0.0f;
    for (int w = 0; w < kGemvWarpsPerBlock; ++w)
      total += warp_sums[w];
    warp_sums[0] = rsqrtf(total / static_cast<float>(K) + eps);
  }
  __syncthreads();
  rms = warp_sums[0];

  ApplyNormInPlace(sx, norm_weight, rms, K, tid);
  __syncthreads();

  if (out_idx >= N)
    return;

  const int num_blocks = K / QK_K;
  const block_q8_k *wrow = weight + out_idx * num_blocks;

  float acc = 0.0f;
  for (int blk = 0; blk < num_blocks; ++blk) {
    const block_q8_k &b = wrow[blk];
    float d = b.d;
    for (int step = 0; step < 8; ++step) {
      int elem = step * 32 + lane;
      int elem_idx = blk * QK_K + elem;
      float w_val = dequant_q8k_element(b, d, elem);
      acc += w_val * sx[elem_idx];
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
  }
  if (lane == 0) {
    output[row * N + out_idx] = __float2half(acc);
  }
}

//==============================================================================
// __dp4a Int8 GEMV Kernels for Q8_0/Q8_K (SM 6.1+)
//
// Uses hardware int8x4 dot product (__dp4a) for Q8_0/Q8_K where weights are
// already int8. Activations are quantized to int8 in shared memory (per-row
// scale: scale = max|x| / 127), reducing smem from K*4 bytes to K bytes.
//==============================================================================

//==============================================================================
// Q8_0 dp4a GEMV
//==============================================================================

__global__ void
fused_dequant_gemv_q8_0_dp4a(const block_q8_0 *__restrict__ weight,
                             const half *__restrict__ x,
                             half *__restrict__ output, int N, int K) {
  extern __shared__ char smem_raw[];
  signed char *sx_int8 = reinterpret_cast<signed char *>(smem_raw);
  // Warp reduction data after int8 activations: [kGemvWarpsPerBlock + 1] floats
  // Index 0..7 for warp max reductions, index kGemvWarpsPerBlock for final scale
  float *warp_data = reinterpret_cast<float *>(smem_raw + K);

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row = blockIdx.y;

  const half *x_row = x + row * K;

  // Phase 1: Find per-row max|x| (vectorized half2 loads)
  float local_max = 0.0f;
  {
    const int K2 = K / 2;
    for (int i = tid; i < K2; i += kGemvThreadsPerBlock) {
      __half2 h2 = reinterpret_cast<const __half2 *>(x_row)[i];
      float2 f2 = __half22float2(h2);
      local_max = fmaxf(local_max, fmaxf(fabsf(f2.x), fabsf(f2.y)));
    }
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));
  }
  if (lane == 0)
    warp_data[warp_id] = local_max;
  __syncthreads();

  // Final max across warps
  float max_val;
  if (tid == 0) {
    max_val = 0.0f;
    for (int w = 0; w < kGemvWarpsPerBlock; ++w)
      max_val = fmaxf(max_val, warp_data[w]);
    warp_data[kGemvWarpsPerBlock] = max_val;
  }
  __syncthreads();
  max_val = warp_data[kGemvWarpsPerBlock];
  float inv_scale = (max_val > 0.0f) ? 127.0f / max_val : 0.0f;
  float activation_scale = max_val / 127.0f;

  // Phase 2: Quantize x to int8 in shared memory (vectorized half2 loads)
  {
    const int K2 = K / 2;
    for (int i = tid; i < K2; i += kGemvThreadsPerBlock) {
      __half2 h2 = reinterpret_cast<const __half2 *>(x_row)[i];
      float2 f2 = __half22float2(h2);
      sx_int8[2 * i] =
          static_cast<signed char>(__float2int_rn(f2.x * inv_scale));
      sx_int8[2 * i + 1] =
          static_cast<signed char>(__float2int_rn(f2.y * inv_scale));
    }
  }
  __syncthreads();

  if (out_idx >= N)
    return;

  // Phase 3: __dp4a dot product
  const int num_blocks = K / QK8_0;
  const block_q8_0 *wrow = weight + out_idx * num_blocks;

  float acc = 0.0f;
  for (int blk = lane; blk < num_blocks; blk += 32) {
    const block_q8_0 &b = wrow[blk];
    float d = __half2float(*reinterpret_cast<const half *>(&b.d));
    int int_acc = 0;
    for (int j = 0; j < QK8_0; j += 4) {
      int w4 = *reinterpret_cast<const int *>(&b.qs[j]);
      int x4 = *reinterpret_cast<const int *>(&sx_int8[blk * QK8_0 + j]);
      int_acc = Dp4aS8(w4, x4, int_acc);
    }
    acc += d * static_cast<float>(int_acc) * activation_scale;
  }

  // Warp reduction
  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
  }
  if (lane == 0) {
    output[row * N + out_idx] = __float2half(acc);
  }
}

//==============================================================================
// Q8_K dp4a GEMV
//==============================================================================

__global__ void
fused_dequant_gemv_q8k_dp4a(const block_q8_k *__restrict__ weight,
                            const half *__restrict__ x,
                            half *__restrict__ output, int N, int K) {
  extern __shared__ char smem_raw[];
  signed char *sx_int8 = reinterpret_cast<signed char *>(smem_raw);
  float *warp_data = reinterpret_cast<float *>(smem_raw + K);

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row = blockIdx.y;

  const half *x_row = x + row * K;

  // Phase 1: Find per-row max|x| (vectorized half2 loads)
  float local_max = 0.0f;
  {
    const int K2 = K / 2;
    for (int i = tid; i < K2; i += kGemvThreadsPerBlock) {
      __half2 h2 = reinterpret_cast<const __half2 *>(x_row)[i];
      float2 f2 = __half22float2(h2);
      local_max = fmaxf(local_max, fmaxf(fabsf(f2.x), fabsf(f2.y)));
    }
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));
  }
  if (lane == 0)
    warp_data[warp_id] = local_max;
  __syncthreads();

  float max_val;
  if (tid == 0) {
    max_val = 0.0f;
    for (int w = 0; w < kGemvWarpsPerBlock; ++w)
      max_val = fmaxf(max_val, warp_data[w]);
    warp_data[kGemvWarpsPerBlock] = max_val;
  }
  __syncthreads();
  max_val = warp_data[kGemvWarpsPerBlock];
  float inv_scale = (max_val > 0.0f) ? 127.0f / max_val : 0.0f;
  float activation_scale = max_val / 127.0f;

  // Phase 2: Quantize x to int8 (vectorized half2 loads)
  {
    const int K2 = K / 2;
    for (int i = tid; i < K2; i += kGemvThreadsPerBlock) {
      __half2 h2 = reinterpret_cast<const __half2 *>(x_row)[i];
      float2 f2 = __half22float2(h2);
      sx_int8[2 * i] =
          static_cast<signed char>(__float2int_rn(f2.x * inv_scale));
      sx_int8[2 * i + 1] =
          static_cast<signed char>(__float2int_rn(f2.y * inv_scale));
    }
  }
  __syncthreads();

  if (out_idx >= N)
    return;

  // Phase 3: __dp4a dot product over Q8_K blocks (256 elements each)
  const int num_blocks = K / QK_K;
  const block_q8_k *wrow = weight + out_idx * num_blocks;

  float acc = 0.0f;
  for (int blk = lane; blk < num_blocks; blk += 32) {
    const block_q8_k &b = wrow[blk];
    float d = b.d;
    int int_acc = 0;
    for (int j = 0; j < QK_K; j += 4) {
      int w4 = *reinterpret_cast<const int *>(&b.qs[j]);
      int x4 = *reinterpret_cast<const int *>(&sx_int8[blk * QK_K + j]);
      int_acc = Dp4aS8(w4, x4, int_acc);
    }
    acc += d * static_cast<float>(int_acc) * activation_scale;
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
  }
  if (lane == 0) {
    output[row * N + out_idx] = __float2half(acc);
  }
}

//==============================================================================
// Combined Fused RmsNorm+GEMV+dp4a Kernels (Q8_0/Q8_K on SM 6.1+)
//
// Maximum-optimization path: no intermediate buffers, hardware int8 dot
// products, minimal shared memory (K bytes for int8 activations).
//==============================================================================

//==============================================================================
// Q8_0 Fused RmsNorm+GEMV+dp4a
//==============================================================================

__global__ void
fused_rmsnorm_gemv_q8_0_dp4a(const block_q8_0 *__restrict__ weight,
                             const half *__restrict__ residual,
                             const half *__restrict__ norm_weight,
                             half *__restrict__ output, int N, int K,
                             float eps) {
  // Shared memory: K floats for FP32 activations (reused as K int8s after
  // quantization), plus workspace for warp reductions.
  // We need K*sizeof(float) during norm phase, then K*sizeof(char) during dp4a.
  // Use the larger allocation (K*sizeof(float)) and reinterpret after norm.
  extern __shared__ char smem_raw[];
  float *sx_fp32 = reinterpret_cast<float *>(smem_raw);
  float *warp_data = sx_fp32 + K;

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row = blockIdx.y;

  // Phase 1: Load residual → FP32 (vectorized), compute sum-of-squares
  const half *res_row = residual + row * K;
  float local_sum_sq = LoadHalfToSmemSumSq(res_row, sx_fp32, K, tid);
  for (int offset = 16; offset > 0; offset >>= 1) {
    local_sum_sq += __shfl_down_sync(0xFFFFFFFF, local_sum_sq, offset);
  }
  if (lane == 0)
    warp_data[warp_id] = local_sum_sq;
  __syncthreads();

  float rms;
  if (tid == 0) {
    float total = 0.0f;
    for (int w = 0; w < kGemvWarpsPerBlock; ++w)
      total += warp_data[w];
    warp_data[0] = rsqrtf(total / static_cast<float>(K) + eps);
  }
  __syncthreads();
  rms = warp_data[0];

  // Phase 2: Apply norm+weight (vectorized), find max|normalized|
  float local_max = ApplyNormInPlaceMaxAbs(sx_fp32, norm_weight, rms, K, tid);
  for (int offset = 16; offset > 0; offset >>= 1) {
    local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));
  }
  if (lane == 0)
    warp_data[warp_id] = local_max;
  __syncthreads();

  float max_val;
  if (tid == 0) {
    max_val = 0.0f;
    for (int w = 0; w < kGemvWarpsPerBlock; ++w)
      max_val = fmaxf(max_val, warp_data[w]);
    warp_data[0] = max_val;
  }
  __syncthreads();
  max_val = warp_data[0];
  float inv_scale = (max_val > 0.0f) ? 127.0f / max_val : 0.0f;
  float activation_scale = max_val / 127.0f;

  // Phase 3: Quantize normalized activations to int8 (reuse smem as int8)
  __syncthreads();
  signed char *sx_int8 = reinterpret_cast<signed char *>(smem_raw);
  for (int i = tid; i < K; i += kGemvThreadsPerBlock) {
    sx_int8[i] = static_cast<signed char>(__float2int_rn(sx_fp32[i] * inv_scale));
  }
  __syncthreads();

  if (out_idx >= N)
    return;

  // Phase 4: __dp4a dot product with int8 weights
  const int num_blocks = K / QK8_0;
  const block_q8_0 *wrow = weight + out_idx * num_blocks;

  float acc = 0.0f;
  for (int blk = lane; blk < num_blocks; blk += 32) {
    const block_q8_0 &b = wrow[blk];
    float d = __half2float(*reinterpret_cast<const half *>(&b.d));
    int int_acc = 0;
    for (int j = 0; j < QK8_0; j += 4) {
      int w4 = *reinterpret_cast<const int *>(&b.qs[j]);
      int x4 = *reinterpret_cast<const int *>(&sx_int8[blk * QK8_0 + j]);
      int_acc = Dp4aS8(w4, x4, int_acc);
    }
    acc += d * static_cast<float>(int_acc) * activation_scale;
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
  }
  if (lane == 0) {
    output[row * N + out_idx] = __float2half(acc);
  }
}

//==============================================================================
// Q8_K Fused RmsNorm+GEMV+dp4a
//==============================================================================

__global__ void
fused_rmsnorm_gemv_q8k_dp4a(const block_q8_k *__restrict__ weight,
                            const half *__restrict__ residual,
                            const half *__restrict__ norm_weight,
                            half *__restrict__ output, int N, int K,
                            float eps) {
  extern __shared__ char smem_raw[];
  float *sx_fp32 = reinterpret_cast<float *>(smem_raw);
  float *warp_data = sx_fp32 + K;

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row = blockIdx.y;

  const half *res_row = residual + row * K;
  float local_sum_sq = LoadHalfToSmemSumSq(res_row, sx_fp32, K, tid);
  for (int offset = 16; offset > 0; offset >>= 1) {
    local_sum_sq += __shfl_down_sync(0xFFFFFFFF, local_sum_sq, offset);
  }
  if (lane == 0)
    warp_data[warp_id] = local_sum_sq;
  __syncthreads();

  float rms;
  if (tid == 0) {
    float total = 0.0f;
    for (int w = 0; w < kGemvWarpsPerBlock; ++w)
      total += warp_data[w];
    warp_data[0] = rsqrtf(total / static_cast<float>(K) + eps);
  }
  __syncthreads();
  rms = warp_data[0];

  float local_max = ApplyNormInPlaceMaxAbs(sx_fp32, norm_weight, rms, K, tid);
  for (int offset = 16; offset > 0; offset >>= 1) {
    local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));
  }
  if (lane == 0)
    warp_data[warp_id] = local_max;
  __syncthreads();

  float max_val;
  if (tid == 0) {
    max_val = 0.0f;
    for (int w = 0; w < kGemvWarpsPerBlock; ++w)
      max_val = fmaxf(max_val, warp_data[w]);
    warp_data[0] = max_val;
  }
  __syncthreads();
  max_val = warp_data[0];
  float inv_scale = (max_val > 0.0f) ? 127.0f / max_val : 0.0f;
  float activation_scale = max_val / 127.0f;

  __syncthreads();
  signed char *sx_int8 = reinterpret_cast<signed char *>(smem_raw);
  for (int i = tid; i < K; i += kGemvThreadsPerBlock) {
    sx_int8[i] = static_cast<signed char>(__float2int_rn(sx_fp32[i] * inv_scale));
  }
  __syncthreads();

  if (out_idx >= N)
    return;

  const int num_blocks = K / QK_K;
  const block_q8_k *wrow = weight + out_idx * num_blocks;

  float acc = 0.0f;
  for (int blk = lane; blk < num_blocks; blk += 32) {
    const block_q8_k &b = wrow[blk];
    float d = b.d;
    int int_acc = 0;
    for (int j = 0; j < QK_K; j += 4) {
      int w4 = *reinterpret_cast<const int *>(&b.qs[j]);
      int x4 = *reinterpret_cast<const int *>(&sx_int8[blk * QK_K + j]);
      int_acc = Dp4aS8(w4, x4, int_acc);
    }
    acc += d * static_cast<float>(int_acc) * activation_scale;
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
  }
  if (lane == 0) {
    output[row * N + out_idx] = __float2half(acc);
  }
}

} // namespace native
} // namespace cuda
} // namespace runtime
} // namespace inferflux
