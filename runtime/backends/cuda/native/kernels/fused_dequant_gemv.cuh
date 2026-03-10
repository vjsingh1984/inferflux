#pragma once

#include "runtime/backends/cuda/native/kernels/quant_common.cuh"

#include <cstdint>
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

template <typename BlockType, int Outputs> struct PackedProjectionGroupParams {
  const BlockType *weights[Outputs];
  half *outputs[Outputs];
  int output_cols[Outputs];
};

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

// Vectorized saturated subtraction on 4 packed signed bytes: a[i] - b[i].
// Equivalent to __vsubss4 but with explicit fallback for portability.
__device__ __forceinline__ int Vsubss4(int a, int b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 300)
  return __vsubss4(a, b);
#else
  const auto *ap = reinterpret_cast<const signed char *>(&a);
  const auto *bp = reinterpret_cast<const signed char *>(&b);
  int result;
  auto *rp = reinterpret_cast<signed char *>(&result);
  for (int i = 0; i < 4; ++i) {
    int v = static_cast<int>(ap[i]) - static_cast<int>(bp[i]);
    rp[i] = static_cast<signed char>(max(-128, min(127, v)));
  }
  return result;
#endif
}

// Load 4 packed int8 values from memory.
// Use the unaligned form for GGUF/Q8_1 block payloads whose row stride is not
// 4-byte aligned (for example Q6_K = 210 bytes, Q8_0 = 34 bytes, Q8_1 = 36
// bytes). Use the aligned form for row-major packed activation buffers where
// K is a multiple of 4 and offsets are generated in 4-byte steps.
__device__ __forceinline__ int LoadPackedInt32Unaligned(const void *ptr) {
  int value;
  memcpy(&value, ptr, sizeof(value));
  return value;
}

__device__ __forceinline__ int LoadPackedInt32Aligned(const void *ptr) {
  return *reinterpret_cast<const int *>(ptr);
}

// Vectorized half→float shared memory load using half2.
// Loads 2 half values (4 bytes) per thread per iteration → 2x bandwidth.
// K must be even (guaranteed: Q4_K/Q6_K/Q8_K use K%256==0, Q8_0 uses K%32==0).
__device__ __forceinline__ void LoadHalfToSmem(const half *__restrict__ src,
                                               float *__restrict__ dst, int K,
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

// Vectorized half→half shared memory copy using half2 (FP16 smem strategy).
// Copies half values directly without float conversion — 2x less smem bandwidth
// than LoadHalfToSmem. Used for Q6_K where int8 dp4a causes precision loss.
__device__ __forceinline__ void LoadHalfToSmemFp16(const half *__restrict__ src,
                                                   half *__restrict__ dst,
                                                   int K, int tid) {
  const int K2 = K / 2;
  for (int i = tid; i < K2; i += kGemvThreadsPerBlock) {
    reinterpret_cast<__half2 *>(dst)[i] =
        reinterpret_cast<const __half2 *>(src)[i];
  }
}

// In-place FP32→FP16 conversion for aliased shared memory.
// Used by RmsNorm+GEMV kernels: norm phase uses FP32, dot product phase
// uses FP16. The FP16 data fits in the same smem (K*2 < K*4 bytes).
__device__ __forceinline__ void
ConvertFp32ToFp16InPlace(float *__restrict__ sx_fp32,
                         half *__restrict__ sx_fp16, int K, int tid) {
  for (int base = 0; base < K; base += kGemvThreadsPerBlock) {
    int idx = base + tid;
    float val = (idx < K) ? sx_fp32[idx] : 0.0f;
    __syncthreads();
    if (idx < K)
      sx_fp16[idx] = __float2half(val);
    __syncthreads();
  }
}

//==============================================================================
// Q4_K Fused Dequant-GEMV (shared-memory x cache, multi-row via blockIdx.y)
//==============================================================================

// Grid: (ceil(N / kGemvWarpsPerBlock), M)
// Each block at (bx, row) loads x[row] into smem and computes
// output[row][cols].
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

// Deprecated reference kernel:
// Retained to compare against the current Q6_K live path, but not dispatched
// by `FusedQuantGemm`. The active non-packed Q6_K path is
// `fused_dequant_gemv_q6k_fp16`, which trades this FP32-smem strategy for
// lower shared-memory pressure while preserving stable numerics.

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
// Q6_K FP16 Shared Memory GEMV
//
// Uses FP16 activations in shared memory instead of FP32:
//   - 2x less smem per block (K×2 vs K×4 bytes)
//   - Activations stored as native half — no quantization needed
//   - FP32 accumulation preserves precision in reduction
//   - For K≥5120: 1.5-2x occupancy improvement
//   - For K≤4096: ~40% smem bandwidth reduction, same occupancy
//
// FP16 has 10 bits of mantissa vs Q6_K's 6-bit weight precision,
// so activation precision is NOT the bottleneck (validated by PoC).
//==============================================================================

__global__ void
fused_dequant_gemv_q6k_fp16(const block_q6_k *__restrict__ weight,
                            const half *__restrict__ x,
                            half *__restrict__ output, int N, int K) {
  extern __shared__ half sx_fp16[];

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row = blockIdx.y;

  // Load activations directly as FP16 (vectorized half2 copy)
  const half *x_row = x + row * K;
  LoadHalfToSmemFp16(x_row, sx_fp16, K, tid);
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
      // Sub-patterns 0 & 2 share ql bytes
      {
        const unsigned char ql_byte = b.ql[g * 64 + lane];
        const unsigned char qh_byte = b.qh[g * 32 + lane];

        const int q0 = ((ql_byte & 0x0F) | (((qh_byte >> 0) & 0x03) << 4)) - 32;
        const int sc0 = g * 8 + 0 * 2 + lane / 16;
        acc += d * static_cast<float>(b.scales[sc0]) * static_cast<float>(q0) *
               __half2float(sx_fp16[blk * QK_K + g * 128 + lane]);

        const int q2 = ((ql_byte >> 4) | (((qh_byte >> 4) & 0x03) << 4)) - 32;
        const int sc2 = g * 8 + 2 * 2 + lane / 16;
        acc += d * static_cast<float>(b.scales[sc2]) * static_cast<float>(q2) *
               __half2float(sx_fp16[blk * QK_K + g * 128 + 64 + lane]);
      }
      // Sub-patterns 1 & 3 share ql bytes at offset +32
      {
        const unsigned char ql_byte = b.ql[g * 64 + 32 + lane];
        const unsigned char qh_byte = b.qh[g * 32 + lane];

        const int q1 = ((ql_byte & 0x0F) | (((qh_byte >> 2) & 0x03) << 4)) - 32;
        const int sc1 = g * 8 + 1 * 2 + lane / 16;
        acc += d * static_cast<float>(b.scales[sc1]) * static_cast<float>(q1) *
               __half2float(sx_fp16[blk * QK_K + g * 128 + 32 + lane]);

        const int q3 = ((ql_byte >> 4) | (((qh_byte >> 6) & 0x03) << 4)) - 32;
        const int sc3 = g * 8 + 3 * 2 + lane / 16;
        acc += d * static_cast<float>(b.scales[sc3]) * static_cast<float>(q3) *
               __half2float(sx_fp16[blk * QK_K + g * 128 + 96 + lane]);
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
// Q6_K dp4a GEMV (SM 6.1+)
//
// Q6_K values are 6-bit integers (-32..31) — fit perfectly in int8.
// Extraction: q6 = (ql_nibble | (qh_2bits << 4)) - 32
// Matches llama.cpp's vec_dot_q6_K_q8_1 approach.
//
// Work distribution: 32 lanes process Q6_K super-blocks (256 elements).
// Each lane reads int32 from ql (4 nibbles) + extracts qh bits.
// Per sub-pattern: 32 elements, with 2 scales (16 elements each).
//
// Shared memory: K bytes (int8 activations) + workspace floats.
//==============================================================================

// Live standard-path kernel on SM 6.1+.
// This keeps Q6_K aligned with the packed/global-activation strategy by using
// dp4a for the dot product once activations are quantized into int8.

__global__ void
fused_dequant_gemv_q6k_dp4a(const block_q6_k *__restrict__ weight,
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
    local_max =
        fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));
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

  // Phase 3: Vectorized dp4a dot product over Q6_K super-blocks
  //
  // Work distribution: 32 lanes, each handles 8 elements per super-block.
  // Lane mapping: sub_pair = lane/8, e_base = (lane%8)*4
  //   sub_pair 0: subs 0+2, g=0    sub_pair 1: subs 1+3, g=0
  //   sub_pair 2: subs 0+2, g=1    sub_pair 3: subs 1+3, g=1
  // Each lane reads 4 ql bytes + 4 qh bytes as packed int32, extracts via
  // __vsubss4 (vectorized -32), then __dp4a for 4-way int8 dot product.
  const int num_blocks = K / QK_K;
  const block_q6_k *wrow = weight + out_idx * num_blocks;

  const int sub_pair = lane >> 3;     // 0..3
  const int e_base = (lane & 7) << 2; // 0,4,8,...,28
  const int g = sub_pair >> 1;        // 0,0,1,1
  const int sub_base = sub_pair & 1;  // 0,1 (processes sub_base and sub_base+2)
  const int qh_shift_lo = sub_base * 2;    // 0 or 2
  const int qh_shift_hi = qh_shift_lo + 4; // 4 or 6
  const int sc_lo = g * 8 + sub_base * 2 + e_base / 16;
  const int sc_hi = g * 8 + (sub_base + 2) * 2 + e_base / 16;

  float acc = 0.0f;

  for (int blk = 0; blk < num_blocks; ++blk) {
    const block_q6_k &b = wrow[blk];
    const float d = __half2float(*reinterpret_cast<const half *>(&b.d));
    const int act_base = blk * QK_K;

    // Read 4 contiguous ql/qh bytes via memcpy (block_q6_k is 210 bytes,
    // so odd-numbered blocks are NOT 4-byte aligned in global memory).
    const int ql4 =
        LoadPackedInt32Unaligned(&b.ql[g * 64 + sub_base * 32 + e_base]);
    const int qh4 = LoadPackedInt32Unaligned(&b.qh[g * 32 + e_base]);

    // --- Low nibble (sub = sub_base): subs 0 or 1 ---
    int vl_lo = ql4 & 0x0F0F0F0F;
    int vh_lo = ((qh4 >> qh_shift_lo) << 4) & 0x30303030;
    int vi_lo = Vsubss4(vl_lo | vh_lo, 0x20202020);
    const int x_lo = LoadPackedInt32Aligned(
        &sx_int8[act_base + g * 128 + sub_base * 32 + e_base]);
    int dot_lo = Dp4aS8(vi_lo, x_lo, 0);

    // --- High nibble (sub = sub_base + 2): subs 2 or 3 ---
    int vl_hi = (ql4 >> 4) & 0x0F0F0F0F;
    int vh_hi = ((qh4 >> qh_shift_hi) << 4) & 0x30303030;
    int vi_hi = Vsubss4(vl_hi | vh_hi, 0x20202020);
    const int x_hi = LoadPackedInt32Aligned(
        &sx_int8[act_base + g * 128 + (sub_base + 2) * 32 + e_base]);
    int dot_hi = Dp4aS8(vi_hi, x_hi, 0);

    acc += d * activation_scale *
           (static_cast<float>(b.scales[sc_lo]) * static_cast<float>(dot_lo) +
            static_cast<float>(b.scales[sc_hi]) * static_cast<float>(dot_hi));
  }

  // Warp reduction
  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
  }
  if (lane == 0) {
    output[row * N + out_idx] = __float2half(acc);
  }
}

__global__ void
fused_dequant_gemv_q6k_dp4a_packed(const block_q6_k *__restrict__ weight,
                                   const int8_t *__restrict__ x_q,
                                   const float *__restrict__ row_scales,
                                   half *__restrict__ output, int N, int K) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row = blockIdx.y;
  if (out_idx >= N) {
    return;
  }

  const int num_blocks = K / QK_K;
  const block_q6_k *wrow = weight + out_idx * num_blocks;
  const int8_t *x_row = x_q + static_cast<size_t>(row) * K;
  const float activation_scale = row_scales[row];

  const int sub_pair = lane >> 3;
  const int e_base = (lane & 7) << 2;
  const int g = sub_pair >> 1;
  const int sub_base = sub_pair & 1;
  const int qh_shift_lo = sub_base * 2;
  const int qh_shift_hi = qh_shift_lo + 4;
  const int sc_lo = g * 8 + sub_base * 2 + e_base / 16;
  const int sc_hi = g * 8 + (sub_base + 2) * 2 + e_base / 16;

  float acc = 0.0f;
  for (int blk = 0; blk < num_blocks; ++blk) {
    const block_q6_k &b = wrow[blk];
    const float d = __half2float(*reinterpret_cast<const half *>(&b.d));
    const int act_base = blk * QK_K;

    const int ql4 =
        LoadPackedInt32Unaligned(&b.ql[g * 64 + sub_base * 32 + e_base]);
    const int qh4 = LoadPackedInt32Unaligned(&b.qh[g * 32 + e_base]);

    const int vl_lo = ql4 & 0x0F0F0F0F;
    const int vh_lo = ((qh4 >> qh_shift_lo) << 4) & 0x30303030;
    const int vi_lo = Vsubss4(vl_lo | vh_lo, 0x20202020);
    const int x_lo = LoadPackedInt32Aligned(
        &x_row[act_base + g * 128 + sub_base * 32 + e_base]);
    const int dot_lo = Dp4aS8(vi_lo, x_lo, 0);

    const int vl_hi = (ql4 >> 4) & 0x0F0F0F0F;
    const int vh_hi = ((qh4 >> qh_shift_hi) << 4) & 0x30303030;
    const int vi_hi = Vsubss4(vl_hi | vh_hi, 0x20202020);
    const int x_hi = LoadPackedInt32Aligned(
        &x_row[act_base + g * 128 + (sub_base + 2) * 32 + e_base]);
    const int dot_hi = Dp4aS8(vi_hi, x_hi, 0);

    acc += d * activation_scale *
           (static_cast<float>(b.scales[sc_lo]) * static_cast<float>(dot_lo) +
            static_cast<float>(b.scales[sc_hi]) * static_cast<float>(dot_hi));
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
  }
  if (lane == 0) {
    output[row * N + out_idx] = __float2half(acc);
  }
}

template <int Outputs>
__global__ void fused_dequant_gemv_q6k_dp4a_packed_group(
    PackedProjectionGroupParams<block_q6_k, Outputs> params,
    const int8_t *__restrict__ x_q, const float *__restrict__ row_scales,
    int K) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row = blockIdx.y;

  bool any_active = false;
#pragma unroll
  for (int i = 0; i < Outputs; ++i) {
    any_active |= (out_idx < params.output_cols[i]);
  }
  if (!any_active) {
    return;
  }

  const int num_blocks = K / QK_K;
  const int8_t *x_row = x_q + static_cast<size_t>(row) * K;
  const float activation_scale = row_scales[row];

  const int sub_pair = lane >> 3;
  const int e_base = (lane & 7) << 2;
  const int g = sub_pair >> 1;
  const int sub_base = sub_pair & 1;
  const int qh_shift_lo = sub_base * 2;
  const int qh_shift_hi = qh_shift_lo + 4;
  const int sc_lo = g * 8 + sub_base * 2 + e_base / 16;
  const int sc_hi = g * 8 + (sub_base + 2) * 2 + e_base / 16;

  float acc[Outputs] = {};
  for (int blk = 0; blk < num_blocks; ++blk) {
    const int act_base = blk * QK_K;
    const int x_lo = LoadPackedInt32Aligned(
        &x_row[act_base + g * 128 + sub_base * 32 + e_base]);
    const int x_hi = LoadPackedInt32Aligned(
        &x_row[act_base + g * 128 + (sub_base + 2) * 32 + e_base]);

#pragma unroll
    for (int i = 0; i < Outputs; ++i) {
      if (out_idx >= params.output_cols[i]) {
        continue;
      }
      const block_q6_k *wrow = params.weights[i] + out_idx * num_blocks;
      const block_q6_k &b = wrow[blk];
      const float d = __half2float(*reinterpret_cast<const half *>(&b.d));

      const int ql4 =
          LoadPackedInt32Unaligned(&b.ql[g * 64 + sub_base * 32 + e_base]);
      const int qh4 = LoadPackedInt32Unaligned(&b.qh[g * 32 + e_base]);

      const int vl_lo = ql4 & 0x0F0F0F0F;
      const int vh_lo = ((qh4 >> qh_shift_lo) << 4) & 0x30303030;
      const int vi_lo = Vsubss4(vl_lo | vh_lo, 0x20202020);
      const int dot_lo = Dp4aS8(vi_lo, x_lo, 0);

      const int vl_hi = (ql4 >> 4) & 0x0F0F0F0F;
      const int vh_hi = ((qh4 >> qh_shift_hi) << 4) & 0x30303030;
      const int vi_hi = Vsubss4(vl_hi | vh_hi, 0x20202020);
      const int dot_hi = Dp4aS8(vi_hi, x_hi, 0);

      acc[i] +=
          d * activation_scale *
          (static_cast<float>(b.scales[sc_lo]) * static_cast<float>(dot_lo) +
           static_cast<float>(b.scales[sc_hi]) * static_cast<float>(dot_hi));
    }
  }

#pragma unroll
  for (int i = 0; i < Outputs; ++i) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc[i] += __shfl_down_sync(0xFFFFFFFF, acc[i], offset);
    }
  }

  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < Outputs; ++i) {
      if (out_idx < params.output_cols[i]) {
        params.outputs[i][row * params.output_cols[i] + out_idx] =
            __float2half(acc[i]);
      }
    }
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
// eliminating the standalone RmsNorm kernel and intermediate d_norm_out_
// buffer. Each GEMV independently normalizes from the residual — the norm
// computation is cheap (~1% of GEMV compute) and amortized across 8 warps per
// block.
//
// Shared memory layout: sx[K] (floats) + warp_sums[kGemvWarpsPerBlock] (floats)
//==============================================================================

//==============================================================================
// Q4_K Fused RmsNorm+GEMV
//==============================================================================

__global__ void fused_rmsnorm_gemv_q4k(const block_q4_k *__restrict__ weight,
                                       const half *__restrict__ residual,
                                       const half *__restrict__ norm_weight,
                                       half *__restrict__ output, int N, int K,
                                       float eps) {
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

// Deprecated reference kernel:
// The current fused-RMS Q6_K path is `fused_rmsnorm_gemv_q6k_fp16`. This
// FP32-smem version remains as a reference implementation only and is not
// wired into `GetRmsNormDispatchEntry`.

__global__ void fused_rmsnorm_gemv_q6k(const block_q6_k *__restrict__ weight,
                                       const half *__restrict__ residual,
                                       const half *__restrict__ norm_weight,
                                       half *__restrict__ output, int N, int K,
                                       float eps) {
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
// Q6_K Fused RmsNorm+GEMV with FP16 Shared Memory
//
// RmsNorm phase: load residual → FP32, compute norm, apply weights (FP32)
// Conversion: FP32 → FP16 in-place (aliased smem)
// Dot product: read FP16 from smem, FP32 accumulate
//
// Smem: K*sizeof(float) (FP32 phase dominates; reused as FP16 after norm)
//==============================================================================

__global__ void fused_rmsnorm_gemv_q6k_fp16(
    const block_q6_k *__restrict__ weight, const half *__restrict__ residual,
    const half *__restrict__ norm_weight, half *__restrict__ output, int N,
    int K, float eps) {
  extern __shared__ char smem_raw[];
  float *sx_fp32 = reinterpret_cast<float *>(smem_raw);
  float *warp_sums = sx_fp32 + K;

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row = blockIdx.y;

  // Phase 1: Load residual → FP32, compute sum-of-squares
  const half *res_row = residual + row * K;
  float local_sum_sq = LoadHalfToSmemSumSq(res_row, sx_fp32, K, tid);
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

  // Phase 2: Apply norm + weight in FP32
  ApplyNormInPlace(sx_fp32, norm_weight, rms, K, tid);
  __syncthreads();

  // Phase 3: Convert FP32 → FP16 in-place (aliased smem)
  half *sx_fp16 = reinterpret_cast<half *>(smem_raw);
  ConvertFp32ToFp16InPlace(sx_fp32, sx_fp16, K, tid);

  if (out_idx >= N)
    return;

  // Phase 4: Q6_K dot product with FP16 activations, FP32 accumulate
  const int num_blocks = K / QK_K;
  const block_q6_k *wrow = weight + out_idx * num_blocks;

  float acc = 0.0f;

  for (int blk = 0; blk < num_blocks; ++blk) {
    const block_q6_k &b = wrow[blk];
    const float d = __half2float(*reinterpret_cast<const half *>(&b.d));

#pragma unroll
    for (int g = 0; g < 2; ++g) {
      {
        const unsigned char ql_byte = b.ql[g * 64 + lane];
        const unsigned char qh_byte = b.qh[g * 32 + lane];

        const int q0 = ((ql_byte & 0x0F) | (((qh_byte >> 0) & 0x03) << 4)) - 32;
        const int sc0 = g * 8 + 0 * 2 + lane / 16;
        acc += d * static_cast<float>(b.scales[sc0]) * static_cast<float>(q0) *
               __half2float(sx_fp16[blk * QK_K + g * 128 + lane]);

        const int q2 = ((ql_byte >> 4) | (((qh_byte >> 4) & 0x03) << 4)) - 32;
        const int sc2 = g * 8 + 2 * 2 + lane / 16;
        acc += d * static_cast<float>(b.scales[sc2]) * static_cast<float>(q2) *
               __half2float(sx_fp16[blk * QK_K + g * 128 + 64 + lane]);
      }
      {
        const unsigned char ql_byte = b.ql[g * 64 + 32 + lane];
        const unsigned char qh_byte = b.qh[g * 32 + lane];

        const int q1 = ((ql_byte & 0x0F) | (((qh_byte >> 2) & 0x03) << 4)) - 32;
        const int sc1 = g * 8 + 1 * 2 + lane / 16;
        acc += d * static_cast<float>(b.scales[sc1]) * static_cast<float>(q1) *
               __half2float(sx_fp16[blk * QK_K + g * 128 + 32 + lane]);

        const int q3 = ((ql_byte >> 4) | (((qh_byte >> 6) & 0x03) << 4)) - 32;
        const int sc3 = g * 8 + 3 * 2 + lane / 16;
        acc += d * static_cast<float>(b.scales[sc3]) * static_cast<float>(q3) *
               __half2float(sx_fp16[blk * QK_K + g * 128 + 96 + lane]);
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

__global__ void fused_rmsnorm_gemv_q8_0(const block_q8_0 *__restrict__ weight,
                                        const half *__restrict__ residual,
                                        const half *__restrict__ norm_weight,
                                        half *__restrict__ output, int N, int K,
                                        float eps) {
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

__global__ void fused_rmsnorm_gemv_q8k(const block_q8_k *__restrict__ weight,
                                       const half *__restrict__ residual,
                                       const half *__restrict__ norm_weight,
                                       half *__restrict__ output, int N, int K,
                                       float eps) {
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
// Q4_K dp4a GEMV (SM 6.1+)
//
// Matches llama.cpp's vec_dot_q4_K_q8_1 approach:
// - Activations quantized to int8 with global scale (max|x| / 127)
// - Q4 nibbles packed as int32, dot product via __dp4a
// - Activation sums for dmin*m correction via dp4a(0x01010101, x, 0)
//
// Shared memory layout: [K int8 activations] [kGemvWarpsPerBlock+1 floats]
//
// Work distribution: 32 lanes, each handles 4 qs bytes (8 elements) per
// super-block. pair = lane >> 3, offs = (lane & 7) * 4.
//==============================================================================

__global__ void
fused_dequant_gemv_q4k_dp4a(const block_q4_k *__restrict__ weight,
                            const half *__restrict__ x,
                            half *__restrict__ output, int N, int K) {
  extern __shared__ char smem_raw[];
  const int num_blocks = K / QK_K;
  signed char *sx_int8 = reinterpret_cast<signed char *>(smem_raw);
  // Per-super-block activation scales stored after int8 activations
  float *blk_scales = reinterpret_cast<float *>(smem_raw + K);

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row = blockIdx.y;

  const half *x_row = x + row * K;
  // Warp reduction workspace (reuses blk_scales area temporarily)
  float *warp_data = blk_scales;

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
    local_max =
        fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));
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

  // Phase 3: dp4a dot product over Q4_K super-blocks (256 elements each)
  const block_q4_k *wrow = weight + out_idx * num_blocks;

  // Each lane handles one pair of sub-blocks (4 qs bytes = 8 elements)
  const int pair = lane >> 3;      // 0..3
  const int offs = (lane & 7) * 4; // 0, 4, 8, ..., 28

  float acc = 0.0f;

  for (int blk = 0; blk < num_blocks; ++blk) {
    const block_q4_k &b = wrow[blk];

    const float d = __half2float(*reinterpret_cast<const half *>(&b.d));
    const float dmin = __half2float(*reinterpret_cast<const half *>(&b.dmin));

    // Extract scales and mins for both sub-blocks in this pair
    unsigned char sc_lo, m_lo, sc_hi, m_hi;
    get_scale_min_k4(pair * 2, b.scales, &sc_lo, &m_lo);
    get_scale_min_k4(pair * 2 + 1, b.scales, &sc_hi, &m_hi);

    const float d_sc_lo = d * static_cast<float>(sc_lo);
    const float dm_m_lo = dmin * static_cast<float>(m_lo);
    const float d_sc_hi = d * static_cast<float>(sc_hi);
    const float dm_m_hi = dmin * static_cast<float>(m_hi);

    // Read 4 qs bytes as int32 → extract low and high nibbles
    int qs4 = *reinterpret_cast<const int *>(&b.qs[pair * 32 + offs]);
    int q_lo4 = qs4 & 0x0F0F0F0F;
    int q_hi4 = (qs4 >> 4) & 0x0F0F0F0F;

    // Read 4 activation int8 values for each sub-block
    const int base = blk * QK_K + pair * 64;
    int x_lo4 = *reinterpret_cast<const int *>(&sx_int8[base + offs]);
    int x_hi4 = *reinterpret_cast<const int *>(&sx_int8[base + 32 + offs]);

    // dp4a: weighted dot product (q4 × activation)
    int dot_lo = Dp4aS8(q_lo4, x_lo4, 0);
    int dot_hi = Dp4aS8(q_hi4, x_hi4, 0);

    // dp4a: activation sums for dmin*m correction
    int sum_lo = Dp4aS8(0x01010101, x_lo4, 0);
    int sum_hi = Dp4aS8(0x01010101, x_hi4, 0);

    acc += activation_scale * (d_sc_lo * static_cast<float>(dot_lo) -
                               dm_m_lo * static_cast<float>(sum_lo) +
                               d_sc_hi * static_cast<float>(dot_hi) -
                               dm_m_hi * static_cast<float>(sum_hi));
  }

  // Warp reduction
  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
  }
  if (lane == 0) {
    output[row * N + out_idx] = __float2half(acc);
  }
}

__global__ void
fused_dequant_gemv_q4k_dp4a_packed(const block_q4_k *__restrict__ weight,
                                   const int8_t *__restrict__ x_q,
                                   const float *__restrict__ row_scales,
                                   half *__restrict__ output, int N, int K) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row = blockIdx.y;
  if (out_idx >= N) {
    return;
  }

  const int num_blocks = K / QK_K;
  const block_q4_k *wrow = weight + out_idx * num_blocks;
  const int8_t *x_row = x_q + static_cast<size_t>(row) * K;
  const float activation_scale = row_scales[row];
  const int pair = lane >> 3;
  const int offs = (lane & 7) * 4;

  float acc = 0.0f;
  for (int blk = 0; blk < num_blocks; ++blk) {
    const block_q4_k &b = wrow[blk];
    const float d = __half2float(*reinterpret_cast<const half *>(&b.d));
    const float dmin = __half2float(*reinterpret_cast<const half *>(&b.dmin));

    unsigned char sc_lo, m_lo, sc_hi, m_hi;
    get_scale_min_k4(pair * 2, b.scales, &sc_lo, &m_lo);
    get_scale_min_k4(pair * 2 + 1, b.scales, &sc_hi, &m_hi);

    const float d_sc_lo = d * static_cast<float>(sc_lo);
    const float dm_m_lo = dmin * static_cast<float>(m_lo);
    const float d_sc_hi = d * static_cast<float>(sc_hi);
    const float dm_m_hi = dmin * static_cast<float>(m_hi);

    const int qs4 = *reinterpret_cast<const int *>(&b.qs[pair * 32 + offs]);
    const int q_lo4 = qs4 & 0x0F0F0F0F;
    const int q_hi4 = (qs4 >> 4) & 0x0F0F0F0F;

    const int base = blk * QK_K + pair * 64;
    const int x_lo4 = *reinterpret_cast<const int *>(&x_row[base + offs]);
    const int x_hi4 = *reinterpret_cast<const int *>(&x_row[base + 32 + offs]);

    const int dot_lo = Dp4aS8(q_lo4, x_lo4, 0);
    const int dot_hi = Dp4aS8(q_hi4, x_hi4, 0);
    const int sum_lo = Dp4aS8(0x01010101, x_lo4, 0);
    const int sum_hi = Dp4aS8(0x01010101, x_hi4, 0);

    acc += activation_scale * (d_sc_lo * static_cast<float>(dot_lo) -
                               dm_m_lo * static_cast<float>(sum_lo) +
                               d_sc_hi * static_cast<float>(dot_hi) -
                               dm_m_hi * static_cast<float>(sum_hi));
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
  }
  if (lane == 0) {
    output[row * N + out_idx] = __float2half(acc);
  }
}

template <int Outputs>
__global__ void fused_dequant_gemv_q4k_dp4a_packed_group(
    PackedProjectionGroupParams<block_q4_k, Outputs> params,
    const int8_t *__restrict__ x_q, const float *__restrict__ row_scales,
    int K) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row = blockIdx.y;

  bool any_active = false;
#pragma unroll
  for (int i = 0; i < Outputs; ++i) {
    any_active |= (out_idx < params.output_cols[i]);
  }
  if (!any_active) {
    return;
  }

  const int num_blocks = K / QK_K;
  const int8_t *x_row = x_q + static_cast<size_t>(row) * K;
  const float activation_scale = row_scales[row];
  const int pair = lane >> 3;
  const int offs = (lane & 7) * 4;

  float acc[Outputs] = {};
  for (int blk = 0; blk < num_blocks; ++blk) {
    const int base = blk * QK_K + pair * 64;
    const int x_lo4 = *reinterpret_cast<const int *>(&x_row[base + offs]);
    const int x_hi4 = *reinterpret_cast<const int *>(&x_row[base + 32 + offs]);

#pragma unroll
    for (int i = 0; i < Outputs; ++i) {
      if (out_idx >= params.output_cols[i]) {
        continue;
      }
      const block_q4_k *wrow = params.weights[i] + out_idx * num_blocks;
      const block_q4_k &b = wrow[blk];
      const float d = __half2float(*reinterpret_cast<const half *>(&b.d));
      const float dmin = __half2float(*reinterpret_cast<const half *>(&b.dmin));

      unsigned char sc_lo, m_lo, sc_hi, m_hi;
      get_scale_min_k4(pair * 2, b.scales, &sc_lo, &m_lo);
      get_scale_min_k4(pair * 2 + 1, b.scales, &sc_hi, &m_hi);

      const float d_sc_lo = d * static_cast<float>(sc_lo);
      const float dm_m_lo = dmin * static_cast<float>(m_lo);
      const float d_sc_hi = d * static_cast<float>(sc_hi);
      const float dm_m_hi = dmin * static_cast<float>(m_hi);

      const int qs4 = *reinterpret_cast<const int *>(&b.qs[pair * 32 + offs]);
      const int q_lo4 = qs4 & 0x0F0F0F0F;
      const int q_hi4 = (qs4 >> 4) & 0x0F0F0F0F;

      const int dot_lo = Dp4aS8(q_lo4, x_lo4, 0);
      const int dot_hi = Dp4aS8(q_hi4, x_hi4, 0);
      const int sum_lo = Dp4aS8(0x01010101, x_lo4, 0);
      const int sum_hi = Dp4aS8(0x01010101, x_hi4, 0);

      acc[i] += activation_scale * (d_sc_lo * static_cast<float>(dot_lo) -
                                    dm_m_lo * static_cast<float>(sum_lo) +
                                    d_sc_hi * static_cast<float>(dot_hi) -
                                    dm_m_hi * static_cast<float>(sum_hi));
    }
  }

#pragma unroll
  for (int i = 0; i < Outputs; ++i) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc[i] += __shfl_down_sync(0xFFFFFFFF, acc[i], offset);
    }
  }

  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < Outputs; ++i) {
      if (out_idx < params.output_cols[i]) {
        params.outputs[i][row * params.output_cols[i] + out_idx] =
            __float2half(acc[i]);
      }
    }
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
  // Index 0..7 for warp max reductions, index kGemvWarpsPerBlock for final
  // scale
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
    local_max =
        fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));
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
      // block_q8_0 is 34 bytes, so odd-numbered rows are not 4-byte aligned.
      const int w4 = LoadPackedInt32Unaligned(&b.qs[j]);
      const int x4 = LoadPackedInt32Aligned(&sx_int8[blk * QK8_0 + j]);
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

__global__ void
fused_dequant_gemv_q8_0_dp4a_packed(const block_q8_0 *__restrict__ weight,
                                    const int8_t *__restrict__ x_q,
                                    const float *__restrict__ row_scales,
                                    half *__restrict__ output, int N, int K) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row = blockIdx.y;
  if (out_idx >= N) {
    return;
  }

  const int num_blocks = K / QK8_0;
  const block_q8_0 *wrow = weight + out_idx * num_blocks;
  const int8_t *x_row = x_q + static_cast<size_t>(row) * K;
  const float activation_scale = row_scales[row];

  float acc = 0.0f;
  for (int blk = lane; blk < num_blocks; blk += 32) {
    const block_q8_0 &b = wrow[blk];
    const float d = __half2float(*reinterpret_cast<const half *>(&b.d));
    int int_acc = 0;
    for (int j = 0; j < QK8_0; j += 4) {
      const int w4 = LoadPackedInt32Unaligned(&b.qs[j]);
      const int x4 = LoadPackedInt32Aligned(&x_row[blk * QK8_0 + j]);
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

template <int Outputs>
__global__ void fused_dequant_gemv_q8_0_dp4a_packed_group(
    PackedProjectionGroupParams<block_q8_0, Outputs> params,
    const int8_t *__restrict__ x_q, const float *__restrict__ row_scales,
    int K) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row = blockIdx.y;

  bool any_active = false;
#pragma unroll
  for (int i = 0; i < Outputs; ++i) {
    any_active |= (out_idx < params.output_cols[i]);
  }
  if (!any_active) {
    return;
  }

  const int num_blocks = K / QK8_0;
  const int8_t *x_row = x_q + static_cast<size_t>(row) * K;
  const float activation_scale = row_scales[row];

  float acc[Outputs] = {};
  for (int blk = lane; blk < num_blocks; blk += 32) {
#pragma unroll
    for (int i = 0; i < Outputs; ++i) {
      if (out_idx >= params.output_cols[i]) {
        continue;
      }
      const block_q8_0 *wrow = params.weights[i] + out_idx * num_blocks;
      const block_q8_0 &b = wrow[blk];
      const float d = __half2float(*reinterpret_cast<const half *>(&b.d));
      int int_acc = 0;
      for (int j = 0; j < QK8_0; j += 4) {
        const int w4 = LoadPackedInt32Unaligned(&b.qs[j]);
        const int x4 = LoadPackedInt32Aligned(&x_row[blk * QK8_0 + j]);
        int_acc = Dp4aS8(w4, x4, int_acc);
      }
      acc[i] += d * static_cast<float>(int_acc) * activation_scale;
    }
  }

#pragma unroll
  for (int i = 0; i < Outputs; ++i) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc[i] += __shfl_down_sync(0xFFFFFFFF, acc[i], offset);
    }
  }

  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < Outputs; ++i) {
      if (out_idx < params.output_cols[i]) {
        params.outputs[i][row * params.output_cols[i] + out_idx] =
            __float2half(acc[i]);
      }
    }
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
    local_max =
        fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));
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

__global__ void
fused_dequant_gemv_q8k_dp4a_packed(const block_q8_k *__restrict__ weight,
                                   const int8_t *__restrict__ x_q,
                                   const float *__restrict__ row_scales,
                                   half *__restrict__ output, int N, int K) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row = blockIdx.y;
  if (out_idx >= N) {
    return;
  }

  const int num_blocks = K / QK_K;
  const block_q8_k *wrow = weight + out_idx * num_blocks;
  const int8_t *x_row = x_q + static_cast<size_t>(row) * K;
  const float activation_scale = row_scales[row];

  float acc = 0.0f;
  for (int blk = lane; blk < num_blocks; blk += 32) {
    const block_q8_k &b = wrow[blk];
    const float d = b.d;
    int int_acc = 0;
    for (int j = 0; j < QK_K; j += 4) {
      const int w4 = *reinterpret_cast<const int *>(&b.qs[j]);
      const int x4 = *reinterpret_cast<const int *>(&x_row[blk * QK_K + j]);
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

template <int Outputs>
__global__ void fused_dequant_gemv_q8k_dp4a_packed_group(
    PackedProjectionGroupParams<block_q8_k, Outputs> params,
    const int8_t *__restrict__ x_q, const float *__restrict__ row_scales,
    int K) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row = blockIdx.y;

  bool any_active = false;
#pragma unroll
  for (int i = 0; i < Outputs; ++i) {
    any_active |= (out_idx < params.output_cols[i]);
  }
  if (!any_active) {
    return;
  }

  const int num_blocks = K / QK_K;
  const int8_t *x_row = x_q + static_cast<size_t>(row) * K;
  const float activation_scale = row_scales[row];

  float acc[Outputs] = {};
  for (int blk = lane; blk < num_blocks; blk += 32) {
#pragma unroll
    for (int i = 0; i < Outputs; ++i) {
      if (out_idx >= params.output_cols[i]) {
        continue;
      }
      const block_q8_k *wrow = params.weights[i] + out_idx * num_blocks;
      const block_q8_k &b = wrow[blk];
      const float d = b.d;
      int int_acc = 0;
      for (int j = 0; j < QK_K; j += 4) {
        const int w4 = *reinterpret_cast<const int *>(&b.qs[j]);
        const int x4 = *reinterpret_cast<const int *>(&x_row[blk * QK_K + j]);
        int_acc = Dp4aS8(w4, x4, int_acc);
      }
      acc[i] += d * static_cast<float>(int_acc) * activation_scale;
    }
  }

#pragma unroll
  for (int i = 0; i < Outputs; ++i) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc[i] += __shfl_down_sync(0xFFFFFFFF, acc[i], offset);
    }
  }

  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < Outputs; ++i) {
      if (out_idx < params.output_cols[i]) {
        params.outputs[i][row * params.output_cols[i] + out_idx] =
            __float2half(acc[i]);
      }
    }
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

__global__ void fused_rmsnorm_gemv_q8_0_dp4a(
    const block_q8_0 *__restrict__ weight, const half *__restrict__ residual,
    const half *__restrict__ norm_weight, half *__restrict__ output, int N,
    int K, float eps) {
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
    local_max =
        fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));
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

  // Phase 3: Quantize FP32→int8 in chunks to avoid cross-warp smem race
  // (FP32 and int8 alias the same memory; writing int8[X] corrupts FP32[X/4])
  signed char *sx_int8 = reinterpret_cast<signed char *>(smem_raw);
  for (int base = 0; base < K; base += kGemvThreadsPerBlock) {
    int idx = base + tid;
    float val = (idx < K) ? sx_fp32[idx] * inv_scale : 0.0f;
    __syncthreads();
    if (idx < K)
      sx_int8[idx] = static_cast<signed char>(__float2int_rn(val));
    __syncthreads();
  }

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
      const int w4 = LoadPackedInt32Unaligned(&b.qs[j]);
      const int x4 = LoadPackedInt32Aligned(&sx_int8[blk * QK8_0 + j]);
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

__global__ void fused_rmsnorm_gemv_q8k_dp4a(
    const block_q8_k *__restrict__ weight, const half *__restrict__ residual,
    const half *__restrict__ norm_weight, half *__restrict__ output, int N,
    int K, float eps) {
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
    local_max =
        fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));
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

  // Quantize FP32→int8 in chunks to avoid cross-warp smem race
  signed char *sx_int8 = reinterpret_cast<signed char *>(smem_raw);
  for (int base = 0; base < K; base += kGemvThreadsPerBlock) {
    int idx = base + tid;
    float val = (idx < K) ? sx_fp32[idx] * inv_scale : 0.0f;
    __syncthreads();
    if (idx < K)
      sx_int8[idx] = static_cast<signed char>(__float2int_rn(val));
    __syncthreads();
  }

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

//==============================================================================
// Q4_K Fused RmsNorm+GEMV+dp4a (SM 6.1+)
//
// Combined norm+quantize+dp4a path for Q4_K: no intermediate buffers,
// hardware int8 dot products with Q4 nibble extraction.
//==============================================================================

__global__ void fused_rmsnorm_gemv_q4k_dp4a(
    const block_q4_k *__restrict__ weight, const half *__restrict__ residual,
    const half *__restrict__ norm_weight, half *__restrict__ output, int N,
    int K, float eps) {
  extern __shared__ char smem_raw[];
  float *sx_fp32 = reinterpret_cast<float *>(smem_raw);
  float *warp_data = sx_fp32 + K;

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row = blockIdx.y;

  // Phase 1: Load residual → FP32, compute sum-of-squares
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

  // Phase 2: Apply norm+weight, find max|normalized|
  float local_max = ApplyNormInPlaceMaxAbs(sx_fp32, norm_weight, rms, K, tid);
  for (int offset = 16; offset > 0; offset >>= 1) {
    local_max =
        fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));
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

  // Phase 3: Quantize FP32→int8 in chunks to avoid cross-warp smem race
  signed char *sx_int8 = reinterpret_cast<signed char *>(smem_raw);
  for (int base = 0; base < K; base += kGemvThreadsPerBlock) {
    int idx = base + tid;
    float val = (idx < K) ? sx_fp32[idx] * inv_scale : 0.0f;
    __syncthreads();
    if (idx < K)
      sx_int8[idx] = static_cast<signed char>(__float2int_rn(val));
    __syncthreads();
  }

  if (out_idx >= N)
    return;

  // Phase 4: dp4a dot product over Q4_K super-blocks
  const int num_blocks = K / QK_K;
  const block_q4_k *wrow = weight + out_idx * num_blocks;

  const int pair = lane >> 3;
  const int offs = (lane & 7) * 4;

  float acc = 0.0f;

  for (int blk = 0; blk < num_blocks; ++blk) {
    const block_q4_k &b = wrow[blk];

    const float d = __half2float(*reinterpret_cast<const half *>(&b.d));
    const float dmin = __half2float(*reinterpret_cast<const half *>(&b.dmin));

    unsigned char sc_lo, m_lo, sc_hi, m_hi;
    get_scale_min_k4(pair * 2, b.scales, &sc_lo, &m_lo);
    get_scale_min_k4(pair * 2 + 1, b.scales, &sc_hi, &m_hi);

    const float d_sc_lo = d * static_cast<float>(sc_lo);
    const float dm_m_lo = dmin * static_cast<float>(m_lo);
    const float d_sc_hi = d * static_cast<float>(sc_hi);
    const float dm_m_hi = dmin * static_cast<float>(m_hi);

    int qs4 = *reinterpret_cast<const int *>(&b.qs[pair * 32 + offs]);
    int q_lo4 = qs4 & 0x0F0F0F0F;
    int q_hi4 = (qs4 >> 4) & 0x0F0F0F0F;

    const int base = blk * QK_K + pair * 64;
    int x_lo4 = *reinterpret_cast<const int *>(&sx_int8[base + offs]);
    int x_hi4 = *reinterpret_cast<const int *>(&sx_int8[base + 32 + offs]);

    int dot_lo = Dp4aS8(q_lo4, x_lo4, 0);
    int dot_hi = Dp4aS8(q_hi4, x_hi4, 0);
    int sum_lo = Dp4aS8(0x01010101, x_lo4, 0);
    int sum_hi = Dp4aS8(0x01010101, x_hi4, 0);

    acc += activation_scale * (d_sc_lo * static_cast<float>(dot_lo) -
                               dm_m_lo * static_cast<float>(sum_lo) +
                               d_sc_hi * static_cast<float>(dot_hi) -
                               dm_m_hi * static_cast<float>(sum_hi));
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
  }
  if (lane == 0) {
    output[row * N + out_idx] = __float2half(acc);
  }
}

//==============================================================================
// Q6_K Fused RmsNorm+GEMV+dp4a (SM 6.1+)
//
// Combined norm+quantize+scalar-int8 path for Q6_K: no intermediate buffers,
// int8 activations (4x smem savings vs FP32), and Q6_K bit extraction.
//==============================================================================

// Live fused-RMS kernel on SM 6.1+.
// The FP16 shared-memory fused-RMS variant remains as the compatibility path on
// older or non-dp4a devices.

__global__ void fused_rmsnorm_gemv_q6k_dp4a(
    const block_q6_k *__restrict__ weight, const half *__restrict__ residual,
    const half *__restrict__ norm_weight, half *__restrict__ output, int N,
    int K, float eps) {
  extern __shared__ char smem_raw[];
  float *sx_fp32 = reinterpret_cast<float *>(smem_raw);
  float *warp_data = sx_fp32 + K;

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row = blockIdx.y;

  // Phase 1: Load residual → FP32, compute sum-of-squares
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

  // Phase 2: Apply norm+weight, find max|normalized|
  float local_max = ApplyNormInPlaceMaxAbs(sx_fp32, norm_weight, rms, K, tid);
  for (int offset = 16; offset > 0; offset >>= 1) {
    local_max =
        fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));
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

  // Phase 3: Quantize FP32→int8 in chunks to avoid cross-warp smem race
  signed char *sx_int8 = reinterpret_cast<signed char *>(smem_raw);
  for (int base = 0; base < K; base += kGemvThreadsPerBlock) {
    int idx = base + tid;
    float val = (idx < K) ? sx_fp32[idx] * inv_scale : 0.0f;
    __syncthreads();
    if (idx < K)
      sx_int8[idx] = static_cast<signed char>(__float2int_rn(val));
    __syncthreads();
  }

  if (out_idx >= N)
    return;

  // Phase 4: Vectorized dp4a dot product (same as fused_dequant_gemv_q6k_dp4a)
  const int num_blocks = K / QK_K;
  const block_q6_k *wrow = weight + out_idx * num_blocks;

  const int sub_pair = lane >> 3;
  const int e_base = (lane & 7) << 2;
  const int g = sub_pair >> 1;
  const int sub_base = sub_pair & 1;
  const int qh_shift_lo = sub_base * 2;
  const int qh_shift_hi = qh_shift_lo + 4;
  const int sc_lo = g * 8 + sub_base * 2 + e_base / 16;
  const int sc_hi = g * 8 + (sub_base + 2) * 2 + e_base / 16;

  float acc = 0.0f;

  for (int blk = 0; blk < num_blocks; ++blk) {
    const block_q6_k &b = wrow[blk];
    const float d = __half2float(*reinterpret_cast<const half *>(&b.d));
    const int act_base = blk * QK_K;

    // Safe unaligned reads (block_q6_k is 210 bytes, not 4-byte aligned)
    int ql4, qh4;
    memcpy(&ql4, &b.ql[g * 64 + sub_base * 32 + e_base], 4);
    memcpy(&qh4, &b.qh[g * 32 + e_base], 4);

    int vl_lo = ql4 & 0x0F0F0F0F;
    int vh_lo = ((qh4 >> qh_shift_lo) << 4) & 0x30303030;
    int vi_lo = Vsubss4(vl_lo | vh_lo, 0x20202020);
    int x_lo = *reinterpret_cast<const int *>(
        &sx_int8[act_base + g * 128 + sub_base * 32 + e_base]);
    int dot_lo = Dp4aS8(vi_lo, x_lo, 0);

    int vl_hi = (ql4 >> 4) & 0x0F0F0F0F;
    int vh_hi = ((qh4 >> qh_shift_hi) << 4) & 0x30303030;
    int vi_hi = Vsubss4(vl_hi | vh_hi, 0x20202020);
    int x_hi = *reinterpret_cast<const int *>(
        &sx_int8[act_base + g * 128 + (sub_base + 2) * 32 + e_base]);
    int dot_hi = Dp4aS8(vi_hi, x_hi, 0);

    acc += d * activation_scale *
           (static_cast<float>(b.scales[sc_lo]) * static_cast<float>(dot_lo) +
            static_cast<float>(b.scales[sc_hi]) * static_cast<float>(dot_hi));
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
  }
  if (lane == 0) {
    output[row * N + out_idx] = __float2half(acc);
  }
}

//==============================================================================
// Q8_1 Activation Quantization Kernels
//
// Quantize FP16 activations to block_q8_1 format:
//   - Per-32-element block scale (d = max|x| / 127)
//   - Pre-computed d*sum(qs) for Q4_K dmin correction
//   - Stored in global memory, reused across sibling projections via L2 cache
//==============================================================================

// Standalone FP16 → Q8_1 quantization (for O proj, down proj inputs)
// Grid: (M,)  Block: (kGemvThreadsPerBlock,)
__global__ void quantize_row_q8_1_kernel(const half *__restrict__ input,
                                         block_q8_1 *__restrict__ output,
                                         int K) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int num_blocks = K / QK8_1;

  const half *x_row = input + row * K;
  block_q8_1 *out_row = output + row * num_blocks;

  for (int blk0 = warp_id * 2; blk0 < num_blocks;
       blk0 += kGemvWarpsPerBlock * 2) {
    const int blk1 = blk0 + 1;
    const bool has_blk1 = blk1 < num_blocks;

    float val0 = __half2float(x_row[blk0 * QK8_1 + lane]);
    float val1 =
        has_blk1 ? __half2float(x_row[blk1 * QK8_1 + lane]) : 0.0f;

    float amax0 = fabsf(val0);
    float amax1 = fabsf(val1);
    for (int offset = 16; offset > 0; offset >>= 1) {
      amax0 = fmaxf(amax0, __shfl_xor_sync(0xFFFFFFFF, amax0, offset));
      amax1 = fmaxf(amax1, __shfl_xor_sync(0xFFFFFFFF, amax1, offset));
    }

    float d0 = amax0 / 127.0f;
    float inv_d0 = (amax0 > 0.0f) ? 127.0f / amax0 : 0.0f;
    int q0 = __float2int_rn(val0 * inv_d0);
    q0 = max(-128, min(127, q0));
    int sum0 = q0;

    float d1 = amax1 / 127.0f;
    float inv_d1 = (amax1 > 0.0f) ? 127.0f / amax1 : 0.0f;
    int q1 = has_blk1 ? __float2int_rn(val1 * inv_d1) : 0;
    q1 = max(-128, min(127, q1));
    int sum1 = q1;

    for (int offset = 16; offset > 0; offset >>= 1) {
      sum0 += __shfl_xor_sync(0xFFFFFFFF, sum0, offset);
      sum1 += __shfl_xor_sync(0xFFFFFFFF, sum1, offset);
    }

    out_row[blk0].qs[lane] = static_cast<int8_t>(q0);
    if (lane == 0) {
      out_row[blk0].ds = __halves2half2(
          __float2half(d0), __float2half(d0 * static_cast<float>(sum0)));
    }
    if (has_blk1) {
      out_row[blk1].qs[lane] = static_cast<int8_t>(q1);
      if (lane == 0) {
        out_row[blk1].ds = __halves2half2(
            __float2half(d1), __float2half(d1 * static_cast<float>(sum1)));
      }
    }
  }
}

__global__ void silu_mul_quantize_q8_1_kernel(const half *__restrict__ gate,
                                              const half *__restrict__ up,
                                              block_q8_1 *__restrict__ output,
                                              int K) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int num_blocks = K / QK8_1;

  const half *gate_row = gate + row * K;
  const half *up_row = up + row * K;
  block_q8_1 *out_row = output + row * num_blocks;

  for (int blk0 = warp_id * 2; blk0 < num_blocks;
       blk0 += kGemvWarpsPerBlock * 2) {
    const int blk1 = blk0 + 1;
    const bool has_blk1 = blk1 < num_blocks;
    const int idx0 = blk0 * QK8_1 + lane;
    const float g0 = __half2float(gate_row[idx0]);
    const float u0 = __half2float(up_row[idx0]);
    const float val0 = (g0 / (1.0f + expf(-g0))) * u0;

    float val1 = 0.0f;
    if (has_blk1) {
      const int idx1 = blk1 * QK8_1 + lane;
      const float g1 = __half2float(gate_row[idx1]);
      const float u1 = __half2float(up_row[idx1]);
      val1 = (g1 / (1.0f + expf(-g1))) * u1;
    }

    float amax0 = fabsf(val0);
    float amax1 = fabsf(val1);
    for (int offset = 16; offset > 0; offset >>= 1) {
      amax0 = fmaxf(amax0, __shfl_xor_sync(0xFFFFFFFF, amax0, offset));
      amax1 = fmaxf(amax1, __shfl_xor_sync(0xFFFFFFFF, amax1, offset));
    }

    float d0 = amax0 / 127.0f;
    float inv_d0 = (amax0 > 0.0f) ? 127.0f / amax0 : 0.0f;
    int q0 = __float2int_rn(val0 * inv_d0);
    q0 = max(-128, min(127, q0));
    int sum0 = q0;

    float d1 = amax1 / 127.0f;
    float inv_d1 = (amax1 > 0.0f) ? 127.0f / amax1 : 0.0f;
    int q1 = has_blk1 ? __float2int_rn(val1 * inv_d1) : 0;
    q1 = max(-128, min(127, q1));
    int sum1 = q1;

    for (int offset = 16; offset > 0; offset >>= 1) {
      sum0 += __shfl_xor_sync(0xFFFFFFFF, sum0, offset);
      sum1 += __shfl_xor_sync(0xFFFFFFFF, sum1, offset);
    }

    out_row[blk0].qs[lane] = static_cast<int8_t>(q0);
    if (lane == 0) {
      out_row[blk0].ds = __halves2half2(
          __float2half(d0), __float2half(d0 * static_cast<float>(sum0)));
    }
    if (has_blk1) {
      out_row[blk1].qs[lane] = static_cast<int8_t>(q1);
      if (lane == 0) {
        out_row[blk1].ds = __halves2half2(
            __float2half(d1), __float2half(d1 * static_cast<float>(sum1)));
      }
    }
  }
}

// Fused RmsNorm + Q8_1 quantization: residual → normalized → Q8_1 blocks
// Eliminates standalone RmsNorm kernel while producing Q8_1 for reuse.
// Grid: (M,)  Block: (kGemvThreadsPerBlock,)
// Smem: K * sizeof(float) + kGemvWarpsPerBlock * sizeof(float)
__global__ void fused_rmsnorm_quantize_q8_1_kernel(
    const half *__restrict__ residual, const half *__restrict__ norm_weight,
    block_q8_1 *__restrict__ output, int K, float eps) {
  extern __shared__ char smem_raw[];
  float *sx = reinterpret_cast<float *>(smem_raw);
  float *warp_sums = sx + K;

  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;

  // Phase 1: Load residual → FP32 smem, compute sum-of-squares
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

  // Phase 2: Apply norm weights in-place (vectorized half2 norm loads)
  ApplyNormInPlace(sx, norm_weight, rms, K, tid);
  __syncthreads();

  // Phase 3: Quantize normalized values to Q8_1 blocks
  const int num_blocks = K / QK8_1;
  block_q8_1 *out_row = output + row * num_blocks;

  for (int blk0 = warp_id * 2; blk0 < num_blocks;
       blk0 += kGemvWarpsPerBlock * 2) {
    const int blk1 = blk0 + 1;
    const bool has_blk1 = blk1 < num_blocks;

    float val0 = sx[blk0 * QK8_1 + lane];
    float val1 = has_blk1 ? sx[blk1 * QK8_1 + lane] : 0.0f;

    float amax0 = fabsf(val0);
    float amax1 = fabsf(val1);
    for (int offset = 16; offset > 0; offset >>= 1) {
      amax0 = fmaxf(amax0, __shfl_xor_sync(0xFFFFFFFF, amax0, offset));
      amax1 = fmaxf(amax1, __shfl_xor_sync(0xFFFFFFFF, amax1, offset));
    }

    float d0 = amax0 / 127.0f;
    float inv_d0 = (amax0 > 0.0f) ? 127.0f / amax0 : 0.0f;
    int q0 = __float2int_rn(val0 * inv_d0);
    q0 = max(-128, min(127, q0));
    int sum0 = q0;

    float d1 = amax1 / 127.0f;
    float inv_d1 = (amax1 > 0.0f) ? 127.0f / amax1 : 0.0f;
    int q1 = has_blk1 ? __float2int_rn(val1 * inv_d1) : 0;
    q1 = max(-128, min(127, q1));
    int sum1 = q1;

    for (int offset = 16; offset > 0; offset >>= 1) {
      sum0 += __shfl_xor_sync(0xFFFFFFFF, sum0, offset);
      sum1 += __shfl_xor_sync(0xFFFFFFFF, sum1, offset);
    }

    out_row[blk0].qs[lane] = static_cast<int8_t>(q0);
    if (lane == 0) {
      out_row[blk0].ds = __halves2half2(
          __float2half(d0), __float2half(d0 * static_cast<float>(sum0)));
    }
    if (has_blk1) {
      out_row[blk1].qs[lane] = static_cast<int8_t>(q1);
      if (lane == 0) {
        out_row[blk1].ds = __halves2half2(
            __float2half(d1), __float2half(d1 * static_cast<float>(sum1)));
      }
    }
  }
}

//==============================================================================
// Q8_1 Activation GEMV Kernels
//
// Read pre-quantized Q8_1 activations from global memory (L2 cached).
// No shared memory needed for activations — all smem freed for occupancy.
// Per-block activation scales provide better precision than per-row.
//==============================================================================

//==============================================================================
// Q4_K × Q8_1 GEMV
//
// Inner loop matches llama.cpp vec_dot_q4_K_q8_1:
// - dp4a for Q4 nibbles × Q8_1 int8 activations
// - Pre-computed s field eliminates dp4a(0x01010101, x, 0) sum for dmin
//==============================================================================

__global__ void
fused_dequant_gemv_q4k_q8_1(const block_q4_k *__restrict__ weight,
                            const block_q8_1 *__restrict__ act_q8_1,
                            half *__restrict__ output, int N, int K) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row = blockIdx.y;
  if (out_idx >= N)
    return;

  const int num_super_blocks = K / QK_K;
  const block_q4_k *wrow = weight + out_idx * num_super_blocks;
  const int num_q8_per_row = K / QK8_1;
  const block_q8_1 *a_row = act_q8_1 + row * num_q8_per_row;

  const int pair = lane >> 3;
  const int offs = (lane & 7) * 4;

  float acc = 0.0f;

  for (int blk = 0; blk < num_super_blocks; ++blk) {
    const block_q4_k &b = wrow[blk];
    const float d = __half2float(*reinterpret_cast<const half *>(&b.d));
    const float dmin = __half2float(*reinterpret_cast<const half *>(&b.dmin));

    unsigned char sc_lo, m_lo, sc_hi, m_hi;
    get_scale_min_k4(pair * 2, b.scales, &sc_lo, &m_lo);
    get_scale_min_k4(pair * 2 + 1, b.scales, &sc_hi, &m_hi);

    const float d_sc_lo = d * static_cast<float>(sc_lo);
    const float dm_m_lo = dmin * static_cast<float>(m_lo);
    const float d_sc_hi = d * static_cast<float>(sc_hi);
    const float dm_m_hi = dmin * static_cast<float>(m_hi);

    // Q4 nibbles as int32
    int qs4 = *reinterpret_cast<const int *>(&b.qs[pair * 32 + offs]);
    int q_lo4 = qs4 & 0x0F0F0F0F;
    int q_hi4 = (qs4 >> 4) & 0x0F0F0F0F;

    // Q8_1 blocks: 8 per Q4_K super-block, pair maps to blocks pair*2, pair*2+1
    const block_q8_1 &a_lo = a_row[blk * 8 + pair * 2];
    const block_q8_1 &a_hi = a_row[blk * 8 + pair * 2 + 1];

    int x_lo4;
    int x_hi4;
    memcpy(&x_lo4, &a_lo.qs[offs], sizeof(x_lo4));
    memcpy(&x_hi4, &a_hi.qs[offs], sizeof(x_hi4));

    float d8_lo = __half2float(__low2half(a_lo.ds));
    float d8_hi = __half2float(__low2half(a_hi.ds));

    int dot_lo = Dp4aS8(q_lo4, x_lo4, 0);
    int dot_hi = Dp4aS8(q_hi4, x_hi4, 0);

    acc += d_sc_lo * d8_lo * static_cast<float>(dot_lo) +
           d_sc_hi * d8_hi * static_cast<float>(dot_hi);

    // dmin correction: pre-computed s = d*sum(qs) from Q8_1 block
    // Only first lane per 8-lane group adds (warp reduction sums correctly)
    if ((lane & 7) == 0) {
      float s_lo = __half2float(__high2half(a_lo.ds));
      float s_hi = __half2float(__high2half(a_hi.ds));
      acc -= dm_m_lo * s_lo + dm_m_hi * s_hi;
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
  }
  if (lane == 0) {
    output[row * N + out_idx] = __float2half(acc);
  }
}

template <int NumSuperBlocks>
__global__ void fused_dequant_gemv_q4k_q8_1_fixed_blocks(
    const block_q4_k *__restrict__ weight,
    const block_q8_1 *__restrict__ act_q8_1, half *__restrict__ output,
    int N) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row = blockIdx.y;
  if (out_idx >= N)
    return;

  const block_q4_k *wrow = weight + out_idx * NumSuperBlocks;
  const block_q8_1 *a_row = act_q8_1 + row * (NumSuperBlocks * (QK_K / QK8_1));

  const int pair = lane >> 3;
  const int offs = (lane & 7) * 4;

  float acc = 0.0f;

#pragma unroll
  for (int blk = 0; blk < NumSuperBlocks; ++blk) {
    const block_q4_k &b = wrow[blk];
    const float d = __half2float(*reinterpret_cast<const half *>(&b.d));
    const float dmin = __half2float(*reinterpret_cast<const half *>(&b.dmin));

    unsigned char sc_lo, m_lo, sc_hi, m_hi;
    get_scale_min_k4(pair * 2, b.scales, &sc_lo, &m_lo);
    get_scale_min_k4(pair * 2 + 1, b.scales, &sc_hi, &m_hi);

    const float d_sc_lo = d * static_cast<float>(sc_lo);
    const float dm_m_lo = dmin * static_cast<float>(m_lo);
    const float d_sc_hi = d * static_cast<float>(sc_hi);
    const float dm_m_hi = dmin * static_cast<float>(m_hi);

    const int qs4 = *reinterpret_cast<const int *>(&b.qs[pair * 32 + offs]);
    const int q_lo4 = qs4 & 0x0F0F0F0F;
    const int q_hi4 = (qs4 >> 4) & 0x0F0F0F0F;

    const block_q8_1 &a_lo = a_row[blk * 8 + pair * 2];
    const block_q8_1 &a_hi = a_row[blk * 8 + pair * 2 + 1];
    int x_lo4;
    int x_hi4;
    memcpy(&x_lo4, &a_lo.qs[offs], sizeof(x_lo4));
    memcpy(&x_hi4, &a_hi.qs[offs], sizeof(x_hi4));
    const float d8_lo = __half2float(__low2half(a_lo.ds));
    const float d8_hi = __half2float(__low2half(a_hi.ds));

    const int dot_lo = Dp4aS8(q_lo4, x_lo4, 0);
    const int dot_hi = Dp4aS8(q_hi4, x_hi4, 0);

    acc += d_sc_lo * d8_lo * static_cast<float>(dot_lo) +
           d_sc_hi * d8_hi * static_cast<float>(dot_hi);
    if ((lane & 7) == 0) {
      const float s_lo = __half2float(__high2half(a_lo.ds));
      const float s_hi = __half2float(__high2half(a_hi.ds));
      acc -= dm_m_lo * s_lo + dm_m_hi * s_hi;
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
  }
  if (lane == 0) {
    output[row * N + out_idx] = __float2half(acc);
  }
}

__global__ void
fused_dequant_gemv_q4k_q8_1_rowpair(const block_q4_k *__restrict__ weight,
                                    const block_q8_1 *__restrict__ act_q8_1,
                                    half *__restrict__ output, int N, int K,
                                    int total_rows) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row0 = blockIdx.y * 2;
  const int row1 = row0 + 1;
  if (out_idx >= N || row0 >= total_rows)
    return;

  const bool has_row1 = row1 < total_rows;
  const int num_super_blocks = K / QK_K;
  const block_q4_k *wrow = weight + out_idx * num_super_blocks;
  const int num_q8_per_row = K / QK8_1;
  const block_q8_1 *a_row0 = act_q8_1 + row0 * num_q8_per_row;
  const block_q8_1 *a_row1 = has_row1 ? (act_q8_1 + row1 * num_q8_per_row)
                                      : nullptr;

  const int pair = lane >> 3;
  const int offs = (lane & 7) * 4;

  float acc0 = 0.0f;
  float acc1 = 0.0f;

  for (int blk = 0; blk < num_super_blocks; ++blk) {
    const block_q4_k &b = wrow[blk];
    const float d = __half2float(*reinterpret_cast<const half *>(&b.d));
    const float dmin = __half2float(*reinterpret_cast<const half *>(&b.dmin));

    unsigned char sc_lo, m_lo, sc_hi, m_hi;
    get_scale_min_k4(pair * 2, b.scales, &sc_lo, &m_lo);
    get_scale_min_k4(pair * 2 + 1, b.scales, &sc_hi, &m_hi);

    const float d_sc_lo = d * static_cast<float>(sc_lo);
    const float dm_m_lo = dmin * static_cast<float>(m_lo);
    const float d_sc_hi = d * static_cast<float>(sc_hi);
    const float dm_m_hi = dmin * static_cast<float>(m_hi);

    const int qs4 = *reinterpret_cast<const int *>(&b.qs[pair * 32 + offs]);
    const int q_lo4 = qs4 & 0x0F0F0F0F;
    const int q_hi4 = (qs4 >> 4) & 0x0F0F0F0F;

    const block_q8_1 &a0_lo = a_row0[blk * 8 + pair * 2];
    const block_q8_1 &a0_hi = a_row0[blk * 8 + pair * 2 + 1];
    int x0_lo4;
    int x0_hi4;
    memcpy(&x0_lo4, &a0_lo.qs[offs], sizeof(x0_lo4));
    memcpy(&x0_hi4, &a0_hi.qs[offs], sizeof(x0_hi4));
    const float d8_0_lo = __half2float(__low2half(a0_lo.ds));
    const float d8_0_hi = __half2float(__low2half(a0_hi.ds));
    const int dot0_lo = Dp4aS8(q_lo4, x0_lo4, 0);
    const int dot0_hi = Dp4aS8(q_hi4, x0_hi4, 0);

    acc0 += d_sc_lo * d8_0_lo * static_cast<float>(dot0_lo) +
            d_sc_hi * d8_0_hi * static_cast<float>(dot0_hi);
    if ((lane & 7) == 0) {
      const float s0_lo = __half2float(__high2half(a0_lo.ds));
      const float s0_hi = __half2float(__high2half(a0_hi.ds));
      acc0 -= dm_m_lo * s0_lo + dm_m_hi * s0_hi;
    }

    if (has_row1) {
      const block_q8_1 &a1_lo = a_row1[blk * 8 + pair * 2];
      const block_q8_1 &a1_hi = a_row1[blk * 8 + pair * 2 + 1];
      int x1_lo4;
      int x1_hi4;
      memcpy(&x1_lo4, &a1_lo.qs[offs], sizeof(x1_lo4));
      memcpy(&x1_hi4, &a1_hi.qs[offs], sizeof(x1_hi4));
      const float d8_1_lo = __half2float(__low2half(a1_lo.ds));
      const float d8_1_hi = __half2float(__low2half(a1_hi.ds));
      const int dot1_lo = Dp4aS8(q_lo4, x1_lo4, 0);
      const int dot1_hi = Dp4aS8(q_hi4, x1_hi4, 0);

      acc1 += d_sc_lo * d8_1_lo * static_cast<float>(dot1_lo) +
              d_sc_hi * d8_1_hi * static_cast<float>(dot1_hi);
      if ((lane & 7) == 0) {
        const float s1_lo = __half2float(__high2half(a1_lo.ds));
        const float s1_hi = __half2float(__high2half(a1_hi.ds));
        acc1 -= dm_m_lo * s1_lo + dm_m_hi * s1_hi;
      }
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    acc0 += __shfl_down_sync(0xFFFFFFFF, acc0, offset);
    acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
  }
  if (lane == 0) {
    output[row0 * N + out_idx] = __float2half(acc0);
    if (has_row1) {
      output[row1 * N + out_idx] = __float2half(acc1);
    }
  }
}

__global__ void
fused_dequant_gemv_q4k_q8_1_rowquad(const block_q4_k *__restrict__ weight,
                                    const block_q8_1 *__restrict__ act_q8_1,
                                    half *__restrict__ output, int N, int K,
                                    int total_rows) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row_base = blockIdx.y * 4;
  if (out_idx >= N || row_base >= total_rows)
    return;

  const int num_rows = min(4, total_rows - row_base);
  const int num_super_blocks = K / QK_K;
  const block_q4_k *wrow = weight + out_idx * num_super_blocks;
  const int num_q8_per_row = K / QK8_1;
  const block_q8_1 *a_rows[4] = {
      act_q8_1 + row_base * num_q8_per_row,
      num_rows > 1 ? (act_q8_1 + (row_base + 1) * num_q8_per_row) : nullptr,
      num_rows > 2 ? (act_q8_1 + (row_base + 2) * num_q8_per_row) : nullptr,
      num_rows > 3 ? (act_q8_1 + (row_base + 3) * num_q8_per_row) : nullptr,
  };

  const int pair = lane >> 3;
  const int offs = (lane & 7) * 4;

  float acc[4] = {};

  for (int blk = 0; blk < num_super_blocks; ++blk) {
    const block_q4_k &b = wrow[blk];
    const float d = __half2float(*reinterpret_cast<const half *>(&b.d));
    const float dmin = __half2float(*reinterpret_cast<const half *>(&b.dmin));

    unsigned char sc_lo, m_lo, sc_hi, m_hi;
    get_scale_min_k4(pair * 2, b.scales, &sc_lo, &m_lo);
    get_scale_min_k4(pair * 2 + 1, b.scales, &sc_hi, &m_hi);

    const float d_sc_lo = d * static_cast<float>(sc_lo);
    const float dm_m_lo = dmin * static_cast<float>(m_lo);
    const float d_sc_hi = d * static_cast<float>(sc_hi);
    const float dm_m_hi = dmin * static_cast<float>(m_hi);

    const int qs4 = *reinterpret_cast<const int *>(&b.qs[pair * 32 + offs]);
    const int q_lo4 = qs4 & 0x0F0F0F0F;
    const int q_hi4 = (qs4 >> 4) & 0x0F0F0F0F;

#pragma unroll
    for (int r = 0; r < 4; ++r) {
      if (r >= num_rows)
        continue;

      const block_q8_1 &a_lo = a_rows[r][blk * 8 + pair * 2];
      const block_q8_1 &a_hi = a_rows[r][blk * 8 + pair * 2 + 1];
      int x_lo4;
      int x_hi4;
      memcpy(&x_lo4, &a_lo.qs[offs], sizeof(x_lo4));
      memcpy(&x_hi4, &a_hi.qs[offs], sizeof(x_hi4));
      const float d8_lo = __half2float(__low2half(a_lo.ds));
      const float d8_hi = __half2float(__low2half(a_hi.ds));
      const int dot_lo = Dp4aS8(q_lo4, x_lo4, 0);
      const int dot_hi = Dp4aS8(q_hi4, x_hi4, 0);

      acc[r] += d_sc_lo * d8_lo * static_cast<float>(dot_lo) +
                d_sc_hi * d8_hi * static_cast<float>(dot_hi);
      if ((lane & 7) == 0) {
        const float s_lo = __half2float(__high2half(a_lo.ds));
        const float s_hi = __half2float(__high2half(a_hi.ds));
        acc[r] -= dm_m_lo * s_lo + dm_m_hi * s_hi;
      }
    }
  }

#pragma unroll
  for (int r = 0; r < 4; ++r) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc[r] += __shfl_down_sync(0xFFFFFFFF, acc[r], offset);
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      if (r < num_rows) {
        output[(row_base + r) * N + out_idx] = __float2half(acc[r]);
      }
    }
  }
}

template <int Outputs>
__global__ void fused_dequant_gemv_q4k_q8_1_group(
    PackedProjectionGroupParams<block_q4_k, Outputs> params,
    const block_q8_1 *__restrict__ act_q8_1, int K) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row = blockIdx.y;

  bool any_active = false;
#pragma unroll
  for (int i = 0; i < Outputs; ++i) {
    any_active |= (out_idx < params.output_cols[i]);
  }
  if (!any_active)
    return;

  const int num_super_blocks = K / QK_K;
  const int num_q8_per_row = K / QK8_1;
  const block_q8_1 *a_row = act_q8_1 + row * num_q8_per_row;

  const int pair = lane >> 3;
  const int offs = (lane & 7) * 4;

  float acc[Outputs] = {};

  for (int blk = 0; blk < num_super_blocks; ++blk) {
    const block_q8_1 &a_lo = a_row[blk * 8 + pair * 2];
    const block_q8_1 &a_hi = a_row[blk * 8 + pair * 2 + 1];
    int x_lo4;
    int x_hi4;
    memcpy(&x_lo4, &a_lo.qs[offs], sizeof(x_lo4));
    memcpy(&x_hi4, &a_hi.qs[offs], sizeof(x_hi4));
    float d8_lo = __half2float(__low2half(a_lo.ds));
    float d8_hi = __half2float(__low2half(a_hi.ds));
    float s_lo = __half2float(__high2half(a_lo.ds));
    float s_hi = __half2float(__high2half(a_hi.ds));

#pragma unroll
    for (int i = 0; i < Outputs; ++i) {
      if (out_idx >= params.output_cols[i])
        continue;

      const block_q4_k *wrow = params.weights[i] + out_idx * num_super_blocks;
      const block_q4_k &b = wrow[blk];
      const float d = __half2float(*reinterpret_cast<const half *>(&b.d));
      const float dmin = __half2float(*reinterpret_cast<const half *>(&b.dmin));

      unsigned char sc_lo, m_lo, sc_hi, m_hi;
      get_scale_min_k4(pair * 2, b.scales, &sc_lo, &m_lo);
      get_scale_min_k4(pair * 2 + 1, b.scales, &sc_hi, &m_hi);

      int qs4 = *reinterpret_cast<const int *>(&b.qs[pair * 32 + offs]);
      int q_lo4 = qs4 & 0x0F0F0F0F;
      int q_hi4 = (qs4 >> 4) & 0x0F0F0F0F;

      int dot_lo = Dp4aS8(q_lo4, x_lo4, 0);
      int dot_hi = Dp4aS8(q_hi4, x_hi4, 0);

      acc[i] +=
          d * static_cast<float>(sc_lo) * d8_lo * static_cast<float>(dot_lo) +
          d * static_cast<float>(sc_hi) * d8_hi * static_cast<float>(dot_hi);

      if ((lane & 7) == 0) {
        acc[i] -= dmin * static_cast<float>(m_lo) * s_lo +
                  dmin * static_cast<float>(m_hi) * s_hi;
      }
    }
  }

#pragma unroll
  for (int i = 0; i < Outputs; ++i) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc[i] += __shfl_down_sync(0xFFFFFFFF, acc[i], offset);
    }
  }

  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < Outputs; ++i) {
      if (out_idx < params.output_cols[i]) {
        params.outputs[i][row * params.output_cols[i] + out_idx] =
            __float2half(acc[i]);
      }
    }
  }
}

template <int Outputs, int NumSuperBlocks>
__global__ void fused_dequant_gemv_q4k_q8_1_group_fixed_blocks(
    PackedProjectionGroupParams<block_q4_k, Outputs> params,
    const block_q8_1 *__restrict__ act_q8_1, int K) {
  (void)K;
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row = blockIdx.y;

  bool any_active = false;
#pragma unroll
  for (int i = 0; i < Outputs; ++i) {
    any_active |= (out_idx < params.output_cols[i]);
  }
  if (!any_active)
    return;

  const int num_q8_per_row = NumSuperBlocks * (QK_K / QK8_1);
  const block_q8_1 *a_row = act_q8_1 + row * num_q8_per_row;

  const int pair = lane >> 3;
  const int offs = (lane & 7) * 4;

  float acc[Outputs] = {};

#pragma unroll
  for (int blk = 0; blk < NumSuperBlocks; ++blk) {
    const block_q8_1 &a_lo = a_row[blk * 8 + pair * 2];
    const block_q8_1 &a_hi = a_row[blk * 8 + pair * 2 + 1];
    int x_lo4;
    int x_hi4;
    memcpy(&x_lo4, &a_lo.qs[offs], sizeof(x_lo4));
    memcpy(&x_hi4, &a_hi.qs[offs], sizeof(x_hi4));
    const float d8_lo = __half2float(__low2half(a_lo.ds));
    const float d8_hi = __half2float(__low2half(a_hi.ds));
    const float s_lo = __half2float(__high2half(a_lo.ds));
    const float s_hi = __half2float(__high2half(a_hi.ds));

#pragma unroll
    for (int i = 0; i < Outputs; ++i) {
      if (out_idx >= params.output_cols[i])
        continue;

      const block_q4_k *wrow = params.weights[i] + out_idx * NumSuperBlocks;
      const block_q4_k &b = wrow[blk];
      const float d = __half2float(*reinterpret_cast<const half *>(&b.d));
      const float dmin = __half2float(*reinterpret_cast<const half *>(&b.dmin));

      unsigned char sc_lo, m_lo, sc_hi, m_hi;
      get_scale_min_k4(pair * 2, b.scales, &sc_lo, &m_lo);
      get_scale_min_k4(pair * 2 + 1, b.scales, &sc_hi, &m_hi);

      const int qs4 = *reinterpret_cast<const int *>(&b.qs[pair * 32 + offs]);
      const int q_lo4 = qs4 & 0x0F0F0F0F;
      const int q_hi4 = (qs4 >> 4) & 0x0F0F0F0F;
      const int dot_lo = Dp4aS8(q_lo4, x_lo4, 0);
      const int dot_hi = Dp4aS8(q_hi4, x_hi4, 0);

      acc[i] +=
          d * static_cast<float>(sc_lo) * d8_lo * static_cast<float>(dot_lo) +
          d * static_cast<float>(sc_hi) * d8_hi * static_cast<float>(dot_hi);

      if ((lane & 7) == 0) {
        acc[i] -= dmin * static_cast<float>(m_lo) * s_lo +
                  dmin * static_cast<float>(m_hi) * s_hi;
      }
    }
  }

#pragma unroll
  for (int i = 0; i < Outputs; ++i) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc[i] += __shfl_down_sync(0xFFFFFFFF, acc[i], offset);
    }
  }

  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < Outputs; ++i) {
      if (out_idx < params.output_cols[i]) {
        params.outputs[i][row * params.output_cols[i] + out_idx] =
            __float2half(acc[i]);
      }
    }
  }
}

template <int Outputs>
__global__ void fused_dequant_gemv_q4k_q8_1_group_rowquad(
    PackedProjectionGroupParams<block_q4_k, Outputs> params,
    const block_q8_1 *__restrict__ act_q8_1, int K, int total_rows) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row_base = blockIdx.y * 4;
  if (row_base >= total_rows)
    return;

  bool any_active = false;
#pragma unroll
  for (int i = 0; i < Outputs; ++i) {
    any_active |= (out_idx < params.output_cols[i]);
  }
  if (!any_active)
    return;

  const int num_rows = min(4, total_rows - row_base);
  const int num_super_blocks = K / QK_K;
  const int num_q8_per_row = K / QK8_1;
  const block_q8_1 *a_rows[4] = {
      act_q8_1 + row_base * num_q8_per_row,
      num_rows > 1 ? (act_q8_1 + (row_base + 1) * num_q8_per_row) : nullptr,
      num_rows > 2 ? (act_q8_1 + (row_base + 2) * num_q8_per_row) : nullptr,
      num_rows > 3 ? (act_q8_1 + (row_base + 3) * num_q8_per_row) : nullptr,
  };

  const int pair = lane >> 3;
  const int offs = (lane & 7) * 4;

  float acc[4][Outputs] = {};

  for (int blk = 0; blk < num_super_blocks; ++blk) {
#pragma unroll
    for (int i = 0; i < Outputs; ++i) {
      if (out_idx >= params.output_cols[i])
        continue;

      const block_q4_k *wrow = params.weights[i] + out_idx * num_super_blocks;
      const block_q4_k &b = wrow[blk];
      const float d = __half2float(*reinterpret_cast<const half *>(&b.d));
      const float dmin = __half2float(*reinterpret_cast<const half *>(&b.dmin));

      unsigned char sc_lo, m_lo, sc_hi, m_hi;
      get_scale_min_k4(pair * 2, b.scales, &sc_lo, &m_lo);
      get_scale_min_k4(pair * 2 + 1, b.scales, &sc_hi, &m_hi);

      const float d_sc_lo = d * static_cast<float>(sc_lo);
      const float dm_m_lo = dmin * static_cast<float>(m_lo);
      const float d_sc_hi = d * static_cast<float>(sc_hi);
      const float dm_m_hi = dmin * static_cast<float>(m_hi);

      const int qs4 = *reinterpret_cast<const int *>(&b.qs[pair * 32 + offs]);
      const int q_lo4 = qs4 & 0x0F0F0F0F;
      const int q_hi4 = (qs4 >> 4) & 0x0F0F0F0F;

#pragma unroll
      for (int r = 0; r < 4; ++r) {
        if (r >= num_rows)
          continue;

        const block_q8_1 &a_lo = a_rows[r][blk * 8 + pair * 2];
        const block_q8_1 &a_hi = a_rows[r][blk * 8 + pair * 2 + 1];
        int x_lo4;
        int x_hi4;
        memcpy(&x_lo4, &a_lo.qs[offs], sizeof(x_lo4));
        memcpy(&x_hi4, &a_hi.qs[offs], sizeof(x_hi4));
        const float d8_lo = __half2float(__low2half(a_lo.ds));
        const float d8_hi = __half2float(__low2half(a_hi.ds));
        const int dot_lo = Dp4aS8(q_lo4, x_lo4, 0);
        const int dot_hi = Dp4aS8(q_hi4, x_hi4, 0);

        acc[r][i] += d_sc_lo * d8_lo * static_cast<float>(dot_lo) +
                     d_sc_hi * d8_hi * static_cast<float>(dot_hi);
        if ((lane & 7) == 0) {
          const float s_lo = __half2float(__high2half(a_lo.ds));
          const float s_hi = __half2float(__high2half(a_hi.ds));
          acc[r][i] -= dm_m_lo * s_lo + dm_m_hi * s_hi;
        }
      }
    }
  }

#pragma unroll
  for (int r = 0; r < 4; ++r) {
#pragma unroll
    for (int i = 0; i < Outputs; ++i) {
      for (int offset = 16; offset > 0; offset >>= 1) {
        acc[r][i] += __shfl_down_sync(0xFFFFFFFF, acc[r][i], offset);
      }
    }
  }

  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      if (r >= num_rows)
        continue;
#pragma unroll
      for (int i = 0; i < Outputs; ++i) {
        if (out_idx < params.output_cols[i]) {
          params.outputs[i][(row_base + r) * params.output_cols[i] + out_idx] =
              __float2half(acc[r][i]);
        }
      }
    }
  }
}

template <int Outputs, int NumSuperBlocks>
__global__ void fused_dequant_gemv_q4k_q8_1_group_rowpair_fixed_blocks(
    PackedProjectionGroupParams<block_q4_k, Outputs> params,
    const block_q8_1 *__restrict__ act_q8_1, int K, int total_rows) {
  (void)K;
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row0 = blockIdx.y * 2;
  const int row1 = row0 + 1;
  if (row0 >= total_rows)
    return;

  const bool has_row1 = row1 < total_rows;
  bool any_active = false;
#pragma unroll
  for (int i = 0; i < Outputs; ++i) {
    any_active |= (out_idx < params.output_cols[i]);
  }
  if (!any_active)
    return;

  const int num_q8_per_row = NumSuperBlocks * (QK_K / QK8_1);
  const block_q8_1 *a_row0 = act_q8_1 + row0 * num_q8_per_row;
  const block_q8_1 *a_row1 = has_row1 ? (act_q8_1 + row1 * num_q8_per_row)
                                      : nullptr;

  const int pair = lane >> 3;
  const int offs = (lane & 7) * 4;

  float acc_row0[Outputs] = {};
  float acc_row1[Outputs] = {};

#pragma unroll
  for (int blk = 0; blk < NumSuperBlocks; ++blk) {
    const block_q8_1 &a0_lo = a_row0[blk * 8 + pair * 2];
    const block_q8_1 &a0_hi = a_row0[blk * 8 + pair * 2 + 1];
    int x0_lo4;
    int x0_hi4;
    memcpy(&x0_lo4, &a0_lo.qs[offs], sizeof(x0_lo4));
    memcpy(&x0_hi4, &a0_hi.qs[offs], sizeof(x0_hi4));
    const float d8_0_lo = __half2float(__low2half(a0_lo.ds));
    const float d8_0_hi = __half2float(__low2half(a0_hi.ds));
    const float s0_lo = __half2float(__high2half(a0_lo.ds));
    const float s0_hi = __half2float(__high2half(a0_hi.ds));

    int x1_lo4 = 0;
    int x1_hi4 = 0;
    float d8_1_lo = 0.0f;
    float d8_1_hi = 0.0f;
    float s1_lo = 0.0f;
    float s1_hi = 0.0f;
    if (has_row1) {
      const block_q8_1 &a1_lo = a_row1[blk * 8 + pair * 2];
      const block_q8_1 &a1_hi = a_row1[blk * 8 + pair * 2 + 1];
      memcpy(&x1_lo4, &a1_lo.qs[offs], sizeof(x1_lo4));
      memcpy(&x1_hi4, &a1_hi.qs[offs], sizeof(x1_hi4));
      d8_1_lo = __half2float(__low2half(a1_lo.ds));
      d8_1_hi = __half2float(__low2half(a1_hi.ds));
      s1_lo = __half2float(__high2half(a1_lo.ds));
      s1_hi = __half2float(__high2half(a1_hi.ds));
    }

#pragma unroll
    for (int i = 0; i < Outputs; ++i) {
      if (out_idx >= params.output_cols[i])
        continue;

      const block_q4_k *wrow = params.weights[i] + out_idx * NumSuperBlocks;
      const block_q4_k &b = wrow[blk];
      const float d = __half2float(*reinterpret_cast<const half *>(&b.d));
      const float dmin = __half2float(*reinterpret_cast<const half *>(&b.dmin));

      unsigned char sc_lo, m_lo, sc_hi, m_hi;
      get_scale_min_k4(pair * 2, b.scales, &sc_lo, &m_lo);
      get_scale_min_k4(pair * 2 + 1, b.scales, &sc_hi, &m_hi);

      const float d_sc_lo = d * static_cast<float>(sc_lo);
      const float dm_m_lo = dmin * static_cast<float>(m_lo);
      const float d_sc_hi = d * static_cast<float>(sc_hi);
      const float dm_m_hi = dmin * static_cast<float>(m_hi);

      const int qs4 = *reinterpret_cast<const int *>(&b.qs[pair * 32 + offs]);
      const int q_lo4 = qs4 & 0x0F0F0F0F;
      const int q_hi4 = (qs4 >> 4) & 0x0F0F0F0F;

      const int dot0_lo = Dp4aS8(q_lo4, x0_lo4, 0);
      const int dot0_hi = Dp4aS8(q_hi4, x0_hi4, 0);

      acc_row0[i] += d_sc_lo * d8_0_lo * static_cast<float>(dot0_lo) +
                     d_sc_hi * d8_0_hi * static_cast<float>(dot0_hi);
      if ((lane & 7) == 0) {
        acc_row0[i] -= dm_m_lo * s0_lo + dm_m_hi * s0_hi;
      }

      if (has_row1) {
        const int dot1_lo = Dp4aS8(q_lo4, x1_lo4, 0);
        const int dot1_hi = Dp4aS8(q_hi4, x1_hi4, 0);
        acc_row1[i] += d_sc_lo * d8_1_lo * static_cast<float>(dot1_lo) +
                       d_sc_hi * d8_1_hi * static_cast<float>(dot1_hi);
        if ((lane & 7) == 0) {
          acc_row1[i] -= dm_m_lo * s1_lo + dm_m_hi * s1_hi;
        }
      }
    }
  }

#pragma unroll
  for (int i = 0; i < Outputs; ++i) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc_row0[i] += __shfl_down_sync(0xFFFFFFFF, acc_row0[i], offset);
      acc_row1[i] += __shfl_down_sync(0xFFFFFFFF, acc_row1[i], offset);
    }
  }

  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < Outputs; ++i) {
      if (out_idx < params.output_cols[i]) {
        params.outputs[i][row0 * params.output_cols[i] + out_idx] =
            __float2half(acc_row0[i]);
        if (has_row1) {
          params.outputs[i][row1 * params.output_cols[i] + out_idx] =
              __float2half(acc_row1[i]);
        }
      }
    }
  }
}

// Experimental grouped-triple row-pair kernel.
// Kept for rework and future profiling, but disabled by default in dispatch
// after real-model decode benchmarks showed corrupted output on RTX 4000 Ada.
template <int Outputs>
__global__ void fused_dequant_gemv_q4k_q8_1_group_rowpair(
    PackedProjectionGroupParams<block_q4_k, Outputs> params,
    const block_q8_1 *__restrict__ act_q8_1, int K, int total_rows) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row0 = blockIdx.y * 2;
  const int row1 = row0 + 1;
  if (row0 >= total_rows)
    return;

  const bool has_row1 = row1 < total_rows;
  bool any_active = false;
#pragma unroll
  for (int i = 0; i < Outputs; ++i) {
    any_active |= (out_idx < params.output_cols[i]);
  }
  if (!any_active)
    return;

  const int num_super_blocks = K / QK_K;
  const int num_q8_per_row = K / QK8_1;
  const block_q8_1 *a_row0 = act_q8_1 + row0 * num_q8_per_row;
  const block_q8_1 *a_row1 = has_row1 ? (act_q8_1 + row1 * num_q8_per_row)
                                      : nullptr;

  const int pair = lane >> 3;
  const int offs = (lane & 7) * 4;

  float acc_row0[Outputs] = {};
  float acc_row1[Outputs] = {};

  for (int blk = 0; blk < num_super_blocks; ++blk) {
    const block_q8_1 &a0_lo = a_row0[blk * 8 + pair * 2];
    const block_q8_1 &a0_hi = a_row0[blk * 8 + pair * 2 + 1];
    int x0_lo4;
    int x0_hi4;
    memcpy(&x0_lo4, &a0_lo.qs[offs], sizeof(x0_lo4));
    memcpy(&x0_hi4, &a0_hi.qs[offs], sizeof(x0_hi4));
    const float d8_0_lo = __half2float(__low2half(a0_lo.ds));
    const float d8_0_hi = __half2float(__low2half(a0_hi.ds));
    const float s0_lo = __half2float(__high2half(a0_lo.ds));
    const float s0_hi = __half2float(__high2half(a0_hi.ds));

    int x1_lo4 = 0;
    int x1_hi4 = 0;
    float d8_1_lo = 0.0f;
    float d8_1_hi = 0.0f;
    float s1_lo = 0.0f;
    float s1_hi = 0.0f;
    if (has_row1) {
      const block_q8_1 &a1_lo = a_row1[blk * 8 + pair * 2];
      const block_q8_1 &a1_hi = a_row1[blk * 8 + pair * 2 + 1];
      memcpy(&x1_lo4, &a1_lo.qs[offs], sizeof(x1_lo4));
      memcpy(&x1_hi4, &a1_hi.qs[offs], sizeof(x1_hi4));
      d8_1_lo = __half2float(__low2half(a1_lo.ds));
      d8_1_hi = __half2float(__low2half(a1_hi.ds));
      s1_lo = __half2float(__high2half(a1_lo.ds));
      s1_hi = __half2float(__high2half(a1_hi.ds));
    }

#pragma unroll
    for (int i = 0; i < Outputs; ++i) {
      if (out_idx >= params.output_cols[i])
        continue;

      const block_q4_k *wrow = params.weights[i] + out_idx * num_super_blocks;
      const block_q4_k &b = wrow[blk];
      const float d = __half2float(*reinterpret_cast<const half *>(&b.d));
      const float dmin = __half2float(*reinterpret_cast<const half *>(&b.dmin));

      unsigned char sc_lo, m_lo, sc_hi, m_hi;
      get_scale_min_k4(pair * 2, b.scales, &sc_lo, &m_lo);
      get_scale_min_k4(pair * 2 + 1, b.scales, &sc_hi, &m_hi);

      const float d_sc_lo = d * static_cast<float>(sc_lo);
      const float dm_m_lo = dmin * static_cast<float>(m_lo);
      const float d_sc_hi = d * static_cast<float>(sc_hi);
      const float dm_m_hi = dmin * static_cast<float>(m_hi);

      const int qs4 = *reinterpret_cast<const int *>(&b.qs[pair * 32 + offs]);
      const int q_lo4 = qs4 & 0x0F0F0F0F;
      const int q_hi4 = (qs4 >> 4) & 0x0F0F0F0F;

      const int dot0_lo = Dp4aS8(q_lo4, x0_lo4, 0);
      const int dot0_hi = Dp4aS8(q_hi4, x0_hi4, 0);
      acc_row0[i] += d_sc_lo * d8_0_lo * static_cast<float>(dot0_lo) +
                     d_sc_hi * d8_0_hi * static_cast<float>(dot0_hi);
      if ((lane & 7) == 0) {
        acc_row0[i] -= dm_m_lo * s0_lo + dm_m_hi * s0_hi;
      }

      if (has_row1) {
        const int dot1_lo = Dp4aS8(q_lo4, x1_lo4, 0);
        const int dot1_hi = Dp4aS8(q_hi4, x1_hi4, 0);
        acc_row1[i] += d_sc_lo * d8_1_lo * static_cast<float>(dot1_lo) +
                       d_sc_hi * d8_1_hi * static_cast<float>(dot1_hi);
        if ((lane & 7) == 0) {
          acc_row1[i] -= dm_m_lo * s1_lo + dm_m_hi * s1_hi;
        }
      }
    }
  }

#pragma unroll
  for (int i = 0; i < Outputs; ++i) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc_row0[i] += __shfl_down_sync(0xFFFFFFFF, acc_row0[i], offset);
      acc_row1[i] += __shfl_down_sync(0xFFFFFFFF, acc_row1[i], offset);
    }
  }

  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < Outputs; ++i) {
      if (out_idx < params.output_cols[i]) {
        params.outputs[i][row0 * params.output_cols[i] + out_idx] =
            __float2half(acc_row0[i]);
        if (has_row1) {
          params.outputs[i][row1 * params.output_cols[i] + out_idx] =
              __float2half(acc_row1[i]);
        }
      }
    }
  }
}

//==============================================================================
// Q6_K × Q8_1 GEMV
//==============================================================================

__global__ void
fused_dequant_gemv_q6k_q8_1(const block_q6_k *__restrict__ weight,
                            const block_q8_1 *__restrict__ act_q8_1,
                            half *__restrict__ output, int N, int K) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row = blockIdx.y;
  if (out_idx >= N)
    return;

  const int num_super_blocks = K / QK_K;
  const block_q6_k *wrow = weight + out_idx * num_super_blocks;
  const int num_q8_per_row = K / QK8_1;
  const block_q8_1 *a_row = act_q8_1 + row * num_q8_per_row;

  const int sub_pair = lane >> 3;
  const int e_base = (lane & 7) << 2;
  const int g = sub_pair >> 1;
  const int sub_base = sub_pair & 1;
  const int qh_shift_lo = sub_base * 2;
  const int qh_shift_hi = qh_shift_lo + 4;
  const int sc_lo = g * 8 + sub_base * 2 + e_base / 16;
  const int sc_hi = g * 8 + (sub_base + 2) * 2 + e_base / 16;

  float acc = 0.0f;

  for (int blk = 0; blk < num_super_blocks; ++blk) {
    const block_q6_k &b = wrow[blk];
    const float d = __half2float(*reinterpret_cast<const half *>(&b.d));

    const int ql4 =
        LoadPackedInt32Unaligned(&b.ql[g * 64 + sub_base * 32 + e_base]);
    const int qh4 = LoadPackedInt32Unaligned(&b.qh[g * 32 + e_base]);

    // Q8_1 block indices: g*4+sub_base for low, g*4+sub_base+2 for high
    const block_q8_1 &a_lo = a_row[blk * 8 + g * 4 + sub_base];
    const block_q8_1 &a_hi = a_row[blk * 8 + g * 4 + sub_base + 2];

    const int x_lo = LoadPackedInt32Unaligned(&a_lo.qs[e_base]);
    const int x_hi = LoadPackedInt32Unaligned(&a_hi.qs[e_base]);

    float d8_lo = __half2float(__low2half(a_lo.ds));
    float d8_hi = __half2float(__low2half(a_hi.ds));

    int vl_lo = ql4 & 0x0F0F0F0F;
    int vh_lo = ((qh4 >> qh_shift_lo) << 4) & 0x30303030;
    int vi_lo = Vsubss4(vl_lo | vh_lo, 0x20202020);
    int dot_lo = Dp4aS8(vi_lo, x_lo, 0);

    int vl_hi = (ql4 >> 4) & 0x0F0F0F0F;
    int vh_hi = ((qh4 >> qh_shift_hi) << 4) & 0x30303030;
    int vi_hi = Vsubss4(vl_hi | vh_hi, 0x20202020);
    int dot_hi = Dp4aS8(vi_hi, x_hi, 0);

    acc += d * (static_cast<float>(b.scales[sc_lo]) * d8_lo *
                    static_cast<float>(dot_lo) +
                static_cast<float>(b.scales[sc_hi]) * d8_hi *
                    static_cast<float>(dot_hi));
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
  }
  if (lane == 0) {
    output[row * N + out_idx] = __float2half(acc);
  }
}

template <int NumSuperBlocks>
__global__ void fused_dequant_gemv_q6k_q8_1_fixed_blocks(
    const block_q6_k *__restrict__ weight,
    const block_q8_1 *__restrict__ act_q8_1, half *__restrict__ output,
    int N) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row = blockIdx.y;
  if (out_idx >= N)
    return;

  const block_q6_k *wrow = weight + out_idx * NumSuperBlocks;
  const block_q8_1 *a_row = act_q8_1 + row * (NumSuperBlocks * (QK_K / QK8_1));

  const int sub_pair = lane >> 3;
  const int e_base = (lane & 7) << 2;
  const int g = sub_pair >> 1;
  const int sub_base = sub_pair & 1;
  const int qh_shift_lo = sub_base * 2;
  const int qh_shift_hi = qh_shift_lo + 4;
  const int sc_lo = g * 8 + sub_base * 2 + e_base / 16;
  const int sc_hi = g * 8 + (sub_base + 2) * 2 + e_base / 16;

  float acc = 0.0f;

#pragma unroll
  for (int blk = 0; blk < NumSuperBlocks; ++blk) {
    const block_q6_k &b = wrow[blk];
    const float d = __half2float(*reinterpret_cast<const half *>(&b.d));

    const int ql4 =
        LoadPackedInt32Unaligned(&b.ql[g * 64 + sub_base * 32 + e_base]);
    const int qh4 = LoadPackedInt32Unaligned(&b.qh[g * 32 + e_base]);

    const block_q8_1 &a_lo = a_row[blk * 8 + g * 4 + sub_base];
    const block_q8_1 &a_hi = a_row[blk * 8 + g * 4 + sub_base + 2];

    const int x_lo = LoadPackedInt32Unaligned(&a_lo.qs[e_base]);
    const int x_hi = LoadPackedInt32Unaligned(&a_hi.qs[e_base]);
    const float d8_lo = __half2float(__low2half(a_lo.ds));
    const float d8_hi = __half2float(__low2half(a_hi.ds));

    const int vl_lo = ql4 & 0x0F0F0F0F;
    const int vh_lo = ((qh4 >> qh_shift_lo) << 4) & 0x30303030;
    const int vi_lo = Vsubss4(vl_lo | vh_lo, 0x20202020);
    const int dot_lo = Dp4aS8(vi_lo, x_lo, 0);

    const int vl_hi = (ql4 >> 4) & 0x0F0F0F0F;
    const int vh_hi = ((qh4 >> qh_shift_hi) << 4) & 0x30303030;
    const int vi_hi = Vsubss4(vl_hi | vh_hi, 0x20202020);
    const int dot_hi = Dp4aS8(vi_hi, x_hi, 0);

    acc += d * (static_cast<float>(b.scales[sc_lo]) * d8_lo *
                    static_cast<float>(dot_lo) +
                static_cast<float>(b.scales[sc_hi]) * d8_hi *
                    static_cast<float>(dot_hi));
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
  }
  if (lane == 0) {
    output[row * N + out_idx] = __float2half(acc);
  }
}

__global__ void
fused_dequant_gemv_q6k_q8_1_rowpair(const block_q6_k *__restrict__ weight,
                                    const block_q8_1 *__restrict__ act_q8_1,
                                    half *__restrict__ output, int N, int K,
                                    int total_rows) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row0 = blockIdx.y * 2;
  const int row1 = row0 + 1;
  if (out_idx >= N || row0 >= total_rows)
    return;

  const bool has_row1 = row1 < total_rows;
  const int num_super_blocks = K / QK_K;
  const block_q6_k *wrow = weight + out_idx * num_super_blocks;
  const int num_q8_per_row = K / QK8_1;
  const block_q8_1 *a_row0 = act_q8_1 + row0 * num_q8_per_row;
  const block_q8_1 *a_row1 = has_row1 ? (act_q8_1 + row1 * num_q8_per_row)
                                      : nullptr;

  const int sub_pair = lane >> 3;
  const int e_base = (lane & 7) << 2;
  const int g = sub_pair >> 1;
  const int sub_base = sub_pair & 1;
  const int qh_shift_lo = sub_base * 2;
  const int qh_shift_hi = qh_shift_lo + 4;
  const int sc_lo = g * 8 + sub_base * 2 + e_base / 16;
  const int sc_hi = g * 8 + (sub_base + 2) * 2 + e_base / 16;

  float acc0 = 0.0f;
  float acc1 = 0.0f;

  for (int blk = 0; blk < num_super_blocks; ++blk) {
    const block_q6_k &b = wrow[blk];
    const float d = __half2float(*reinterpret_cast<const half *>(&b.d));
    const int ql4 =
        LoadPackedInt32Unaligned(&b.ql[g * 64 + sub_base * 32 + e_base]);
    const int qh4 = LoadPackedInt32Unaligned(&b.qh[g * 32 + e_base]);

    const int vl_lo = ql4 & 0x0F0F0F0F;
    const int vh_lo = ((qh4 >> qh_shift_lo) << 4) & 0x30303030;
    const int vi_lo = Vsubss4(vl_lo | vh_lo, 0x20202020);
    const int vl_hi = (ql4 >> 4) & 0x0F0F0F0F;
    const int vh_hi = ((qh4 >> qh_shift_hi) << 4) & 0x30303030;
    const int vi_hi = Vsubss4(vl_hi | vh_hi, 0x20202020);

    const block_q8_1 &a0_lo = a_row0[blk * 8 + g * 4 + sub_base];
    const block_q8_1 &a0_hi = a_row0[blk * 8 + g * 4 + sub_base + 2];
    const int x0_lo = LoadPackedInt32Unaligned(&a0_lo.qs[e_base]);
    const int x0_hi = LoadPackedInt32Unaligned(&a0_hi.qs[e_base]);
    const float d8_0_lo = __half2float(__low2half(a0_lo.ds));
    const float d8_0_hi = __half2float(__low2half(a0_hi.ds));
    const int dot0_lo = Dp4aS8(vi_lo, x0_lo, 0);
    const int dot0_hi = Dp4aS8(vi_hi, x0_hi, 0);
    acc0 += d * (static_cast<float>(b.scales[sc_lo]) * d8_0_lo *
                     static_cast<float>(dot0_lo) +
                 static_cast<float>(b.scales[sc_hi]) * d8_0_hi *
                     static_cast<float>(dot0_hi));

    if (has_row1) {
      const block_q8_1 &a1_lo = a_row1[blk * 8 + g * 4 + sub_base];
      const block_q8_1 &a1_hi = a_row1[blk * 8 + g * 4 + sub_base + 2];
      const int x1_lo = LoadPackedInt32Unaligned(&a1_lo.qs[e_base]);
      const int x1_hi = LoadPackedInt32Unaligned(&a1_hi.qs[e_base]);
      const float d8_1_lo = __half2float(__low2half(a1_lo.ds));
      const float d8_1_hi = __half2float(__low2half(a1_hi.ds));
      const int dot1_lo = Dp4aS8(vi_lo, x1_lo, 0);
      const int dot1_hi = Dp4aS8(vi_hi, x1_hi, 0);
      acc1 += d * (static_cast<float>(b.scales[sc_lo]) * d8_1_lo *
                       static_cast<float>(dot1_lo) +
                   static_cast<float>(b.scales[sc_hi]) * d8_1_hi *
                       static_cast<float>(dot1_hi));
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    acc0 += __shfl_down_sync(0xFFFFFFFF, acc0, offset);
    acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
  }
  if (lane == 0) {
    output[row0 * N + out_idx] = __float2half(acc0);
    if (has_row1) {
      output[row1 * N + out_idx] = __float2half(acc1);
    }
  }
}

__global__ void
fused_dequant_gemv_q6k_q8_1_rowquad(const block_q6_k *__restrict__ weight,
                                    const block_q8_1 *__restrict__ act_q8_1,
                                    half *__restrict__ output, int N, int K,
                                    int total_rows) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row_base = blockIdx.y * 4;
  if (out_idx >= N || row_base >= total_rows)
    return;

  const int num_rows = min(4, total_rows - row_base);
  const int num_super_blocks = K / QK_K;
  const block_q6_k *wrow = weight + out_idx * num_super_blocks;
  const int num_q8_per_row = K / QK8_1;
  const block_q8_1 *a_rows[4] = {
      act_q8_1 + row_base * num_q8_per_row,
      num_rows > 1 ? (act_q8_1 + (row_base + 1) * num_q8_per_row) : nullptr,
      num_rows > 2 ? (act_q8_1 + (row_base + 2) * num_q8_per_row) : nullptr,
      num_rows > 3 ? (act_q8_1 + (row_base + 3) * num_q8_per_row) : nullptr,
  };

  const int sub_pair = lane >> 3;
  const int e_base = (lane & 7) << 2;
  const int g = sub_pair >> 1;
  const int sub_base = sub_pair & 1;
  const int qh_shift_lo = sub_base * 2;
  const int qh_shift_hi = qh_shift_lo + 4;
  const int sc_lo = g * 8 + sub_base * 2 + e_base / 16;
  const int sc_hi = g * 8 + (sub_base + 2) * 2 + e_base / 16;

  float acc[4] = {};

  for (int blk = 0; blk < num_super_blocks; ++blk) {
    const block_q6_k &b = wrow[blk];
    const float d = __half2float(*reinterpret_cast<const half *>(&b.d));
    const int ql4 =
        LoadPackedInt32Unaligned(&b.ql[g * 64 + sub_base * 32 + e_base]);
    const int qh4 = LoadPackedInt32Unaligned(&b.qh[g * 32 + e_base]);

    const int vl_lo = ql4 & 0x0F0F0F0F;
    const int vh_lo = ((qh4 >> qh_shift_lo) << 4) & 0x30303030;
    const int vi_lo = Vsubss4(vl_lo | vh_lo, 0x20202020);
    const int vl_hi = (ql4 >> 4) & 0x0F0F0F0F;
    const int vh_hi = ((qh4 >> qh_shift_hi) << 4) & 0x30303030;
    const int vi_hi = Vsubss4(vl_hi | vh_hi, 0x20202020);

#pragma unroll
    for (int r = 0; r < 4; ++r) {
      if (r >= num_rows)
        continue;

      const block_q8_1 &a_lo = a_rows[r][blk * 8 + g * 4 + sub_base];
      const block_q8_1 &a_hi = a_rows[r][blk * 8 + g * 4 + sub_base + 2];
      const int x_lo = LoadPackedInt32Unaligned(&a_lo.qs[e_base]);
      const int x_hi = LoadPackedInt32Unaligned(&a_hi.qs[e_base]);
      const float d8_lo = __half2float(__low2half(a_lo.ds));
      const float d8_hi = __half2float(__low2half(a_hi.ds));
      const int dot_lo = Dp4aS8(vi_lo, x_lo, 0);
      const int dot_hi = Dp4aS8(vi_hi, x_hi, 0);

      acc[r] += d * (static_cast<float>(b.scales[sc_lo]) * d8_lo *
                         static_cast<float>(dot_lo) +
                     static_cast<float>(b.scales[sc_hi]) * d8_hi *
                         static_cast<float>(dot_hi));
    }
  }

#pragma unroll
  for (int r = 0; r < 4; ++r) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc[r] += __shfl_down_sync(0xFFFFFFFF, acc[r], offset);
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      if (r < num_rows) {
        output[(row_base + r) * N + out_idx] = __float2half(acc[r]);
      }
    }
  }
}

template <int Outputs>
__global__ void fused_dequant_gemv_q6k_q8_1_group(
    PackedProjectionGroupParams<block_q6_k, Outputs> params,
    const block_q8_1 *__restrict__ act_q8_1, int K) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row = blockIdx.y;

  bool any_active = false;
#pragma unroll
  for (int i = 0; i < Outputs; ++i) {
    any_active |= (out_idx < params.output_cols[i]);
  }
  if (!any_active)
    return;

  const int num_super_blocks = K / QK_K;
  const int num_q8_per_row = K / QK8_1;
  const block_q8_1 *a_row = act_q8_1 + row * num_q8_per_row;

  const int sub_pair = lane >> 3;
  const int e_base = (lane & 7) << 2;
  const int g = sub_pair >> 1;
  const int sub_base = sub_pair & 1;
  const int qh_shift_lo = sub_base * 2;
  const int qh_shift_hi = qh_shift_lo + 4;
  const int sc_lo = g * 8 + sub_base * 2 + e_base / 16;
  const int sc_hi = g * 8 + (sub_base + 2) * 2 + e_base / 16;

  float acc[Outputs] = {};

  for (int blk = 0; blk < num_super_blocks; ++blk) {
    const block_q8_1 &a_lo = a_row[blk * 8 + g * 4 + sub_base];
    const block_q8_1 &a_hi = a_row[blk * 8 + g * 4 + sub_base + 2];
    const int x_lo = LoadPackedInt32Unaligned(&a_lo.qs[e_base]);
    const int x_hi = LoadPackedInt32Unaligned(&a_hi.qs[e_base]);
    float d8_lo = __half2float(__low2half(a_lo.ds));
    float d8_hi = __half2float(__low2half(a_hi.ds));

#pragma unroll
    for (int i = 0; i < Outputs; ++i) {
      if (out_idx >= params.output_cols[i])
        continue;

      const block_q6_k *wrow = params.weights[i] + out_idx * num_super_blocks;
      const block_q6_k &b = wrow[blk];
      const float d = __half2float(*reinterpret_cast<const half *>(&b.d));

      const int ql4 =
          LoadPackedInt32Unaligned(&b.ql[g * 64 + sub_base * 32 + e_base]);
      const int qh4 = LoadPackedInt32Unaligned(&b.qh[g * 32 + e_base]);

      int vl_lo = ql4 & 0x0F0F0F0F;
      int vh_lo = ((qh4 >> qh_shift_lo) << 4) & 0x30303030;
      int vi_lo = Vsubss4(vl_lo | vh_lo, 0x20202020);
      int dot_lo = Dp4aS8(vi_lo, x_lo, 0);

      int vl_hi = (ql4 >> 4) & 0x0F0F0F0F;
      int vh_hi = ((qh4 >> qh_shift_hi) << 4) & 0x30303030;
      int vi_hi = Vsubss4(vl_hi | vh_hi, 0x20202020);
      int dot_hi = Dp4aS8(vi_hi, x_hi, 0);

      acc[i] += d * (static_cast<float>(b.scales[sc_lo]) * d8_lo *
                         static_cast<float>(dot_lo) +
                     static_cast<float>(b.scales[sc_hi]) * d8_hi *
                         static_cast<float>(dot_hi));
    }
  }

#pragma unroll
  for (int i = 0; i < Outputs; ++i) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc[i] += __shfl_down_sync(0xFFFFFFFF, acc[i], offset);
    }
  }

  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < Outputs; ++i) {
      if (out_idx < params.output_cols[i]) {
        params.outputs[i][row * params.output_cols[i] + out_idx] =
            __float2half(acc[i]);
      }
    }
  }
}

template <int Outputs>
__global__ void fused_dequant_gemv_q6k_q8_1_group_rowquad(
    PackedProjectionGroupParams<block_q6_k, Outputs> params,
    const block_q8_1 *__restrict__ act_q8_1, int K, int total_rows) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row_base = blockIdx.y * 4;
  if (row_base >= total_rows)
    return;

  bool any_active = false;
#pragma unroll
  for (int i = 0; i < Outputs; ++i) {
    any_active |= (out_idx < params.output_cols[i]);
  }
  if (!any_active)
    return;

  const int num_rows = min(4, total_rows - row_base);
  const int num_super_blocks = K / QK_K;
  const int num_q8_per_row = K / QK8_1;
  const block_q8_1 *a_rows[4] = {
      act_q8_1 + row_base * num_q8_per_row,
      num_rows > 1 ? (act_q8_1 + (row_base + 1) * num_q8_per_row) : nullptr,
      num_rows > 2 ? (act_q8_1 + (row_base + 2) * num_q8_per_row) : nullptr,
      num_rows > 3 ? (act_q8_1 + (row_base + 3) * num_q8_per_row) : nullptr,
  };

  const int sub_pair = lane >> 3;
  const int e_base = (lane & 7) << 2;
  const int g = sub_pair >> 1;
  const int sub_base = sub_pair & 1;
  const int qh_shift_lo = sub_base * 2;
  const int qh_shift_hi = qh_shift_lo + 4;
  const int sc_lo = g * 8 + sub_base * 2 + e_base / 16;
  const int sc_hi = g * 8 + (sub_base + 2) * 2 + e_base / 16;

  float acc[4][Outputs] = {};

  for (int blk = 0; blk < num_super_blocks; ++blk) {
#pragma unroll
    for (int i = 0; i < Outputs; ++i) {
      if (out_idx >= params.output_cols[i])
        continue;

      const block_q6_k *wrow = params.weights[i] + out_idx * num_super_blocks;
      const block_q6_k &b = wrow[blk];
      const float d = __half2float(*reinterpret_cast<const half *>(&b.d));
      const int ql4 =
          LoadPackedInt32Unaligned(&b.ql[g * 64 + sub_base * 32 + e_base]);
      const int qh4 = LoadPackedInt32Unaligned(&b.qh[g * 32 + e_base]);

      const int vl_lo = ql4 & 0x0F0F0F0F;
      const int vh_lo = ((qh4 >> qh_shift_lo) << 4) & 0x30303030;
      const int vi_lo = Vsubss4(vl_lo | vh_lo, 0x20202020);
      const int vl_hi = (ql4 >> 4) & 0x0F0F0F0F;
      const int vh_hi = ((qh4 >> qh_shift_hi) << 4) & 0x30303030;
      const int vi_hi = Vsubss4(vl_hi | vh_hi, 0x20202020);

#pragma unroll
      for (int r = 0; r < 4; ++r) {
        if (r >= num_rows)
          continue;

        const block_q8_1 &a_lo = a_rows[r][blk * 8 + g * 4 + sub_base];
        const block_q8_1 &a_hi = a_rows[r][blk * 8 + g * 4 + sub_base + 2];
        const int x_lo = LoadPackedInt32Unaligned(&a_lo.qs[e_base]);
        const int x_hi = LoadPackedInt32Unaligned(&a_hi.qs[e_base]);
        const float d8_lo = __half2float(__low2half(a_lo.ds));
        const float d8_hi = __half2float(__low2half(a_hi.ds));
        const int dot_lo = Dp4aS8(vi_lo, x_lo, 0);
        const int dot_hi = Dp4aS8(vi_hi, x_hi, 0);

        acc[r][i] += d * (static_cast<float>(b.scales[sc_lo]) * d8_lo *
                              static_cast<float>(dot_lo) +
                          static_cast<float>(b.scales[sc_hi]) * d8_hi *
                              static_cast<float>(dot_hi));
      }
    }
  }

#pragma unroll
  for (int r = 0; r < 4; ++r) {
#pragma unroll
    for (int i = 0; i < Outputs; ++i) {
      for (int offset = 16; offset > 0; offset >>= 1) {
        acc[r][i] += __shfl_down_sync(0xFFFFFFFF, acc[r][i], offset);
      }
    }
  }

  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      if (r >= num_rows)
        continue;
#pragma unroll
      for (int i = 0; i < Outputs; ++i) {
        if (out_idx < params.output_cols[i]) {
          params.outputs[i][(row_base + r) * params.output_cols[i] + out_idx] =
              __float2half(acc[r][i]);
        }
      }
    }
  }
}

// Experimental grouped-triple row-pair kernel.
// Kept for rework and future profiling, but disabled by default in dispatch
// after real-model decode benchmarks showed corrupted output on RTX 4000 Ada.
template <int Outputs>
__global__ void fused_dequant_gemv_q6k_q8_1_group_rowpair(
    PackedProjectionGroupParams<block_q6_k, Outputs> params,
    const block_q8_1 *__restrict__ act_q8_1, int K, int total_rows) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row0 = blockIdx.y * 2;
  const int row1 = row0 + 1;
  if (row0 >= total_rows)
    return;

  const bool has_row1 = row1 < total_rows;
  bool any_active = false;
#pragma unroll
  for (int i = 0; i < Outputs; ++i) {
    any_active |= (out_idx < params.output_cols[i]);
  }
  if (!any_active)
    return;

  const int num_super_blocks = K / QK_K;
  const int num_q8_per_row = K / QK8_1;
  const block_q8_1 *a_row0 = act_q8_1 + row0 * num_q8_per_row;
  const block_q8_1 *a_row1 = has_row1 ? (act_q8_1 + row1 * num_q8_per_row)
                                      : nullptr;

  const int sub_pair = lane >> 3;
  const int e_base = (lane & 7) << 2;
  const int g = sub_pair >> 1;
  const int sub_base = sub_pair & 1;
  const int qh_shift_lo = sub_base * 2;
  const int qh_shift_hi = qh_shift_lo + 4;
  const int sc_lo = g * 8 + sub_base * 2 + e_base / 16;
  const int sc_hi = g * 8 + (sub_base + 2) * 2 + e_base / 16;

  float acc_row0[Outputs] = {};
  float acc_row1[Outputs] = {};

  for (int blk = 0; blk < num_super_blocks; ++blk) {
    const block_q8_1 &a0_lo = a_row0[blk * 8 + g * 4 + sub_base];
    const block_q8_1 &a0_hi = a_row0[blk * 8 + g * 4 + sub_base + 2];
    const int x0_lo = LoadPackedInt32Unaligned(&a0_lo.qs[e_base]);
    const int x0_hi = LoadPackedInt32Unaligned(&a0_hi.qs[e_base]);
    const float d8_0_lo = __half2float(__low2half(a0_lo.ds));
    const float d8_0_hi = __half2float(__low2half(a0_hi.ds));

    int x1_lo = 0;
    int x1_hi = 0;
    float d8_1_lo = 0.0f;
    float d8_1_hi = 0.0f;
    if (has_row1) {
      const block_q8_1 &a1_lo = a_row1[blk * 8 + g * 4 + sub_base];
      const block_q8_1 &a1_hi = a_row1[blk * 8 + g * 4 + sub_base + 2];
      x1_lo = LoadPackedInt32Unaligned(&a1_lo.qs[e_base]);
      x1_hi = LoadPackedInt32Unaligned(&a1_hi.qs[e_base]);
      d8_1_lo = __half2float(__low2half(a1_lo.ds));
      d8_1_hi = __half2float(__low2half(a1_hi.ds));
    }

#pragma unroll
    for (int i = 0; i < Outputs; ++i) {
      if (out_idx >= params.output_cols[i])
        continue;

      const block_q6_k *wrow = params.weights[i] + out_idx * num_super_blocks;
      const block_q6_k &b = wrow[blk];
      const float d = __half2float(*reinterpret_cast<const half *>(&b.d));
      const int ql4 =
          LoadPackedInt32Unaligned(&b.ql[g * 64 + sub_base * 32 + e_base]);
      const int qh4 = LoadPackedInt32Unaligned(&b.qh[g * 32 + e_base]);

      const int vl_lo = ql4 & 0x0F0F0F0F;
      const int vh_lo = ((qh4 >> qh_shift_lo) << 4) & 0x30303030;
      const int vi_lo = Vsubss4(vl_lo | vh_lo, 0x20202020);
      const int vl_hi = (ql4 >> 4) & 0x0F0F0F0F;
      const int vh_hi = ((qh4 >> qh_shift_hi) << 4) & 0x30303030;
      const int vi_hi = Vsubss4(vl_hi | vh_hi, 0x20202020);

      const int dot0_lo = Dp4aS8(vi_lo, x0_lo, 0);
      const int dot0_hi = Dp4aS8(vi_hi, x0_hi, 0);
      acc_row0[i] += d * (static_cast<float>(b.scales[sc_lo]) * d8_0_lo *
                              static_cast<float>(dot0_lo) +
                          static_cast<float>(b.scales[sc_hi]) * d8_0_hi *
                              static_cast<float>(dot0_hi));

      if (has_row1) {
        const int dot1_lo = Dp4aS8(vi_lo, x1_lo, 0);
        const int dot1_hi = Dp4aS8(vi_hi, x1_hi, 0);
        acc_row1[i] += d * (static_cast<float>(b.scales[sc_lo]) * d8_1_lo *
                                static_cast<float>(dot1_lo) +
                            static_cast<float>(b.scales[sc_hi]) * d8_1_hi *
                                static_cast<float>(dot1_hi));
      }
    }
  }

#pragma unroll
  for (int i = 0; i < Outputs; ++i) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc_row0[i] += __shfl_down_sync(0xFFFFFFFF, acc_row0[i], offset);
      acc_row1[i] += __shfl_down_sync(0xFFFFFFFF, acc_row1[i], offset);
    }
  }

  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < Outputs; ++i) {
      if (out_idx < params.output_cols[i]) {
        params.outputs[i][row0 * params.output_cols[i] + out_idx] =
            __float2half(acc_row0[i]);
        if (has_row1) {
          params.outputs[i][row1 * params.output_cols[i] + out_idx] =
              __float2half(acc_row1[i]);
        }
      }
    }
  }
}

//==============================================================================
// Q8_0 × Q8_1 GEMV
//==============================================================================

__global__ void
fused_dequant_gemv_q8_0_q8_1(const block_q8_0 *__restrict__ weight,
                             const block_q8_1 *__restrict__ act_q8_1,
                             half *__restrict__ output, int N, int K) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row = blockIdx.y;
  if (out_idx >= N)
    return;

  // Q8_0 and Q8_1 both have 32-element blocks: 1:1 mapping
  const int num_blocks = K / QK8_0;
  const block_q8_0 *wrow = weight + out_idx * num_blocks;
  const block_q8_1 *a_row = act_q8_1 + row * num_blocks;

  float acc = 0.0f;
  for (int blk = lane; blk < num_blocks; blk += 32) {
    const block_q8_0 &b = wrow[blk];
    const float d_w = __half2float(*reinterpret_cast<const half *>(&b.d));
    const block_q8_1 &a = a_row[blk];
    const float d_a = __half2float(__low2half(a.ds));

    int int_acc = 0;
    for (int j = 0; j < QK8_0; j += 4) {
      const int w4 = LoadPackedInt32Unaligned(&b.qs[j]);
      const int x4 = LoadPackedInt32Unaligned(&a.qs[j]);
      int_acc = Dp4aS8(w4, x4, int_acc);
    }
    acc += d_w * d_a * static_cast<float>(int_acc);
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
  }
  if (lane == 0) {
    output[row * N + out_idx] = __float2half(acc);
  }
}

template <int Outputs>
__global__ void fused_dequant_gemv_q8_0_q8_1_group(
    PackedProjectionGroupParams<block_q8_0, Outputs> params,
    const block_q8_1 *__restrict__ act_q8_1, int K) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row = blockIdx.y;

  bool any_active = false;
#pragma unroll
  for (int i = 0; i < Outputs; ++i) {
    any_active |= (out_idx < params.output_cols[i]);
  }
  if (!any_active)
    return;

  const int num_blocks = K / QK8_0;
  const block_q8_1 *a_row = act_q8_1 + row * num_blocks;

  float acc[Outputs] = {};
  for (int blk = lane; blk < num_blocks; blk += 32) {
    const block_q8_1 &a = a_row[blk];
    const float d_a = __half2float(__low2half(a.ds));

#pragma unroll
    for (int i = 0; i < Outputs; ++i) {
      if (out_idx >= params.output_cols[i])
        continue;

      const block_q8_0 *wrow = params.weights[i] + out_idx * num_blocks;
      const block_q8_0 &b = wrow[blk];
      const float d_w = __half2float(*reinterpret_cast<const half *>(&b.d));
      int int_acc = 0;
      for (int j = 0; j < QK8_0; j += 4) {
        const int w4 = LoadPackedInt32Unaligned(&b.qs[j]);
        const int x4 = LoadPackedInt32Unaligned(&a.qs[j]);
        int_acc = Dp4aS8(w4, x4, int_acc);
      }
      acc[i] += d_w * d_a * static_cast<float>(int_acc);
    }
  }

#pragma unroll
  for (int i = 0; i < Outputs; ++i) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc[i] += __shfl_down_sync(0xFFFFFFFF, acc[i], offset);
    }
  }

  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < Outputs; ++i) {
      if (out_idx < params.output_cols[i]) {
        params.outputs[i][row * params.output_cols[i] + out_idx] =
            __float2half(acc[i]);
      }
    }
  }
}

//==============================================================================
// Q8_K × Q8_1 GEMV
//==============================================================================

__global__ void
fused_dequant_gemv_q8k_q8_1(const block_q8_k *__restrict__ weight,
                            const block_q8_1 *__restrict__ act_q8_1,
                            half *__restrict__ output, int N, int K) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row = blockIdx.y;
  if (out_idx >= N)
    return;

  // Q8_K super-block = 256 elements = 8 Q8_1 blocks
  const int num_super_blocks = K / QK_K;
  const block_q8_k *wrow = weight + out_idx * num_super_blocks;
  const int num_q8_per_row = K / QK8_1;
  const block_q8_1 *a_row = act_q8_1 + row * num_q8_per_row;

  float acc = 0.0f;
  for (int blk = lane; blk < num_super_blocks; blk += 32) {
    const block_q8_k &b = wrow[blk];
    const float d_w = b.d;

    for (int sub = 0; sub < 8; ++sub) {
      const block_q8_1 &a = a_row[blk * 8 + sub];
      const float d_a = __half2float(__low2half(a.ds));

      int int_acc = 0;
      for (int j = 0; j < QK8_1; j += 4) {
        int w4 = *reinterpret_cast<const int *>(&b.qs[sub * QK8_1 + j]);
        int x4;
        memcpy(&x4, &a.qs[j], sizeof(x4));
        int_acc = Dp4aS8(w4, x4, int_acc);
      }
      acc += d_w * d_a * static_cast<float>(int_acc);
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
  }
  if (lane == 0) {
    output[row * N + out_idx] = __float2half(acc);
  }
}

template <int Outputs>
__global__ void fused_dequant_gemv_q8k_q8_1_group(
    PackedProjectionGroupParams<block_q8_k, Outputs> params,
    const block_q8_1 *__restrict__ act_q8_1, int K) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;
  const int row = blockIdx.y;

  bool any_active = false;
#pragma unroll
  for (int i = 0; i < Outputs; ++i) {
    any_active |= (out_idx < params.output_cols[i]);
  }
  if (!any_active)
    return;

  const int num_super_blocks = K / QK_K;
  const int num_q8_per_row = K / QK8_1;
  const block_q8_1 *a_row = act_q8_1 + row * num_q8_per_row;

  float acc[Outputs] = {};
  for (int blk = lane; blk < num_super_blocks; blk += 32) {
    for (int sub = 0; sub < 8; ++sub) {
      const block_q8_1 &a = a_row[blk * 8 + sub];
      const float d_a = __half2float(__low2half(a.ds));

#pragma unroll
      for (int i = 0; i < Outputs; ++i) {
        if (out_idx >= params.output_cols[i])
          continue;

        const block_q8_k *wrow = params.weights[i] + out_idx * num_super_blocks;
        const block_q8_k &b = wrow[blk];
        const float d_w = b.d;
        int int_acc = 0;
        for (int j = 0; j < QK8_1; j += 4) {
          int w4 = *reinterpret_cast<const int *>(&b.qs[sub * QK8_1 + j]);
          int x4;
          memcpy(&x4, &a.qs[j], sizeof(x4));
          int_acc = Dp4aS8(w4, x4, int_acc);
        }
        acc[i] += d_w * d_a * static_cast<float>(int_acc);
      }
    }
  }

#pragma unroll
  for (int i = 0; i < Outputs; ++i) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc[i] += __shfl_down_sync(0xFFFFFFFF, acc[i], offset);
    }
  }

  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < Outputs; ++i) {
      if (out_idx < params.output_cols[i]) {
        params.outputs[i][row * params.output_cols[i] + out_idx] =
            __float2half(acc[i]);
      }
    }
  }
}

//==============================================================================
// Column-pair Q4_K × Q8_1 GEMV
//
// Each warp computes TWO output columns instead of one, reusing the same Q8_1
// activation data. This halves grid.x and improves instruction-level
// parallelism by interleaving independent accumulations.
//==============================================================================

__global__ void
fused_dequant_gemv_q4k_q8_1_colpair(const block_q4_k *__restrict__ weight,
                                    const block_q8_1 *__restrict__ act_q8_1,
                                    half *__restrict__ output, int N, int K) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx0 = (blockIdx.x * kGemvWarpsPerBlock + warp_id) * 2;
  const int out_idx1 = out_idx0 + 1;
  const int row = blockIdx.y;
  if (out_idx0 >= N)
    return;

  const bool has_col1 = out_idx1 < N;
  const int num_super_blocks = K / QK_K;
  const block_q4_k *wrow0 = weight + out_idx0 * num_super_blocks;
  const block_q4_k *wrow1 =
      has_col1 ? (weight + out_idx1 * num_super_blocks) : wrow0;
  const int num_q8_per_row = K / QK8_1;
  const block_q8_1 *a_row = act_q8_1 + row * num_q8_per_row;

  const int pair = lane >> 3;
  const int offs = (lane & 7) * 4;

  float acc0 = 0.0f;
  float acc1 = 0.0f;

  for (int blk = 0; blk < num_super_blocks; ++blk) {
    // Load Q8_1 activations once (shared between both output columns)
    const block_q8_1 &a_lo = a_row[blk * 8 + pair * 2];
    const block_q8_1 &a_hi = a_row[blk * 8 + pair * 2 + 1];
    int x_lo4, x_hi4;
    memcpy(&x_lo4, &a_lo.qs[offs], sizeof(x_lo4));
    memcpy(&x_hi4, &a_hi.qs[offs], sizeof(x_hi4));
    const float d8_lo = __half2float(__low2half(a_lo.ds));
    const float d8_hi = __half2float(__low2half(a_hi.ds));

    // Pre-load dmin correction terms (same for both weight rows)
    float s_lo = 0.0f, s_hi = 0.0f;
    if ((lane & 7) == 0) {
      s_lo = __half2float(__high2half(a_lo.ds));
      s_hi = __half2float(__high2half(a_hi.ds));
    }

    // Output column 0
    {
      const block_q4_k &b = wrow0[blk];
      const float d = __half2float(*reinterpret_cast<const half *>(&b.d));
      const float dmin = __half2float(*reinterpret_cast<const half *>(&b.dmin));

      unsigned char sc_lo, m_lo, sc_hi, m_hi;
      get_scale_min_k4(pair * 2, b.scales, &sc_lo, &m_lo);
      get_scale_min_k4(pair * 2 + 1, b.scales, &sc_hi, &m_hi);

      int qs4 = *reinterpret_cast<const int *>(&b.qs[pair * 32 + offs]);
      int q_lo4 = qs4 & 0x0F0F0F0F;
      int q_hi4 = (qs4 >> 4) & 0x0F0F0F0F;

      acc0 += d * static_cast<float>(sc_lo) * d8_lo *
                  static_cast<float>(Dp4aS8(q_lo4, x_lo4, 0)) +
              d * static_cast<float>(sc_hi) * d8_hi *
                  static_cast<float>(Dp4aS8(q_hi4, x_hi4, 0));
      if ((lane & 7) == 0) {
        acc0 -= dmin * static_cast<float>(m_lo) * s_lo +
                dmin * static_cast<float>(m_hi) * s_hi;
      }
    }

    // Output column 1
    if (has_col1) {
      const block_q4_k &b = wrow1[blk];
      const float d = __half2float(*reinterpret_cast<const half *>(&b.d));
      const float dmin = __half2float(*reinterpret_cast<const half *>(&b.dmin));

      unsigned char sc_lo, m_lo, sc_hi, m_hi;
      get_scale_min_k4(pair * 2, b.scales, &sc_lo, &m_lo);
      get_scale_min_k4(pair * 2 + 1, b.scales, &sc_hi, &m_hi);

      int qs4 = *reinterpret_cast<const int *>(&b.qs[pair * 32 + offs]);
      int q_lo4 = qs4 & 0x0F0F0F0F;
      int q_hi4 = (qs4 >> 4) & 0x0F0F0F0F;

      acc1 += d * static_cast<float>(sc_lo) * d8_lo *
                  static_cast<float>(Dp4aS8(q_lo4, x_lo4, 0)) +
              d * static_cast<float>(sc_hi) * d8_hi *
                  static_cast<float>(Dp4aS8(q_hi4, x_hi4, 0));
      if ((lane & 7) == 0) {
        acc1 -= dmin * static_cast<float>(m_lo) * s_lo +
                dmin * static_cast<float>(m_hi) * s_hi;
      }
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    acc0 += __shfl_down_sync(0xFFFFFFFF, acc0, offset);
    acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
  }
  if (lane == 0) {
    output[row * N + out_idx0] = __float2half(acc0);
    if (has_col1) {
      output[row * N + out_idx1] = __float2half(acc1);
    }
  }
}

//==============================================================================
// Column-pair Q6_K × Q8_1 GEMV
//==============================================================================

__global__ void
fused_dequant_gemv_q6k_q8_1_colpair(const block_q6_k *__restrict__ weight,
                                    const block_q8_1 *__restrict__ act_q8_1,
                                    half *__restrict__ output, int N, int K) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx0 = (blockIdx.x * kGemvWarpsPerBlock + warp_id) * 2;
  const int out_idx1 = out_idx0 + 1;
  const int row = blockIdx.y;
  if (out_idx0 >= N)
    return;

  const bool has_col1 = out_idx1 < N;
  const int num_super_blocks = K / QK_K;
  const block_q6_k *wrow0 = weight + out_idx0 * num_super_blocks;
  const block_q6_k *wrow1 =
      has_col1 ? (weight + out_idx1 * num_super_blocks) : wrow0;
  const int num_q8_per_row = K / QK8_1;
  const block_q8_1 *a_row = act_q8_1 + row * num_q8_per_row;

  const int sub_pair = lane >> 3;
  const int e_base = (lane & 7) << 2;
  const int g = sub_pair >> 1;
  const int sub_base = sub_pair & 1;
  const int qh_shift_lo = sub_base * 2;
  const int qh_shift_hi = qh_shift_lo + 4;
  const int sc_lo = g * 8 + sub_base * 2 + e_base / 16;
  const int sc_hi = g * 8 + (sub_base + 2) * 2 + e_base / 16;

  float acc0 = 0.0f;
  float acc1 = 0.0f;

  for (int blk = 0; blk < num_super_blocks; ++blk) {
    // Load Q8_1 activations once (shared between both output columns)
    const block_q8_1 &a_lo = a_row[blk * 8 + g * 4 + sub_base];
    const block_q8_1 &a_hi = a_row[blk * 8 + g * 4 + sub_base + 2];
    const int x_lo = LoadPackedInt32Unaligned(&a_lo.qs[e_base]);
    const int x_hi = LoadPackedInt32Unaligned(&a_hi.qs[e_base]);
    const float d8_lo = __half2float(__low2half(a_lo.ds));
    const float d8_hi = __half2float(__low2half(a_hi.ds));

    // Output column 0
    {
      const block_q6_k &b = wrow0[blk];
      const float d = __half2float(*reinterpret_cast<const half *>(&b.d));

      const int ql4 =
          LoadPackedInt32Unaligned(&b.ql[g * 64 + sub_base * 32 + e_base]);
      const int qh4 = LoadPackedInt32Unaligned(&b.qh[g * 32 + e_base]);

      int vl_lo = ql4 & 0x0F0F0F0F;
      int vh_lo = ((qh4 >> qh_shift_lo) << 4) & 0x30303030;
      int vi_lo = Vsubss4(vl_lo | vh_lo, 0x20202020);
      int dot_lo = Dp4aS8(vi_lo, x_lo, 0);

      int vl_hi = (ql4 >> 4) & 0x0F0F0F0F;
      int vh_hi = ((qh4 >> qh_shift_hi) << 4) & 0x30303030;
      int vi_hi = Vsubss4(vl_hi | vh_hi, 0x20202020);
      int dot_hi = Dp4aS8(vi_hi, x_hi, 0);

      acc0 += d * (static_cast<float>(b.scales[sc_lo]) * d8_lo *
                       static_cast<float>(dot_lo) +
                   static_cast<float>(b.scales[sc_hi]) * d8_hi *
                       static_cast<float>(dot_hi));
    }

    // Output column 1
    if (has_col1) {
      const block_q6_k &b = wrow1[blk];
      const float d = __half2float(*reinterpret_cast<const half *>(&b.d));

      const int ql4 =
          LoadPackedInt32Unaligned(&b.ql[g * 64 + sub_base * 32 + e_base]);
      const int qh4 = LoadPackedInt32Unaligned(&b.qh[g * 32 + e_base]);

      int vl_lo = ql4 & 0x0F0F0F0F;
      int vh_lo = ((qh4 >> qh_shift_lo) << 4) & 0x30303030;
      int vi_lo = Vsubss4(vl_lo | vh_lo, 0x20202020);
      int dot_lo = Dp4aS8(vi_lo, x_lo, 0);

      int vl_hi = (ql4 >> 4) & 0x0F0F0F0F;
      int vh_hi = ((qh4 >> qh_shift_hi) << 4) & 0x30303030;
      int vi_hi = Vsubss4(vl_hi | vh_hi, 0x20202020);
      int dot_hi = Dp4aS8(vi_hi, x_hi, 0);

      acc1 += d * (static_cast<float>(b.scales[sc_lo]) * d8_lo *
                       static_cast<float>(dot_lo) +
                   static_cast<float>(b.scales[sc_hi]) * d8_hi *
                       static_cast<float>(dot_hi));
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    acc0 += __shfl_down_sync(0xFFFFFFFF, acc0, offset);
    acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
  }
  if (lane == 0) {
    output[row * N + out_idx0] = __float2half(acc0);
    if (has_col1) {
      output[row * N + out_idx1] = __float2half(acc1);
    }
  }
}

} // namespace native
} // namespace cuda
} // namespace runtime
} // namespace inferflux
