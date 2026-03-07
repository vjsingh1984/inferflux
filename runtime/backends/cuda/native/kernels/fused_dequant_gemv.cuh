#pragma once

#include "runtime/backends/cuda/native/kernels/dequantization.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace inferflux {
namespace runtime {
namespace cuda {
namespace native {

// Number of warps per thread block for GEMV kernels
constexpr int kGemvWarpsPerBlock = 8;
constexpr int kGemvThreadsPerBlock = kGemvWarpsPerBlock * 32;

//==============================================================================
// Q4_K Fused Dequant-GEMV
//==============================================================================

// Each warp computes one output element.
// For weight matrix W[N,K] (row-major) and input x[K]:
//   output[n] = sum_k( dequant(W[n,k]) * x[k] )
//
// Q4_K super-blocks: 256 elements each, with 8 sub-blocks of 32 elements.
// Each sub-block has a 6-bit scale and 6-bit min packed in scales[12].
__global__ void fused_dequant_gemv_q4k(const block_q4_k *__restrict__ weight,
                                       const half *__restrict__ x,
                                       half *__restrict__ output, int N,
                                       int K) {
  int out_idx = blockIdx.x * kGemvWarpsPerBlock + (threadIdx.x >> 5);
  if (out_idx >= N)
    return;

  int lane = threadIdx.x & 31;
  int num_blocks = K / QK_K; // Number of super-blocks per row

  const block_q4_k *row = weight + out_idx * num_blocks;

  float acc = 0.0f;

  for (int blk = 0; blk < num_blocks; ++blk) {
    const block_q4_k &b = row[blk];

    // Super-block scales (FP16 stored as uint16)
    float d = __half2float(*reinterpret_cast<const half *>(&b.d));
    float dmin = __half2float(*reinterpret_cast<const half *>(&b.dmin));

    // Q4_K layout: 256 elements = 4 groups of 64.
    // Each group of 64 has two sub-blocks sharing 32 qs bytes:
    //   sub-block 2i:   low nibble of qs[i*32..(i+1)*32-1]
    //   sub-block 2i+1: high nibble of qs[i*32..(i+1)*32-1]
    // Scale index = sub-block index (0-7).
    // With 32 lanes, each lane handles one element per sub-block.
    for (int sb = 0; sb < 8; ++sb) {
      int elem_idx = blk * QK_K + sb * 32 + lane;

      // Decode 6-bit scale and min (matches llama.cpp get_scale_min_k4)
      unsigned char sc, m;
      if (sb < 4) {
        sc = b.scales[sb] & 63;
        m = b.scales[sb + 4] & 63;
      } else {
        sc = (b.scales[sb + 4] & 0xF) | ((b.scales[sb - 4] >> 6) << 4);
        m = (b.scales[sb + 4] >> 4) | ((b.scales[sb] >> 6) << 4);
      }

      // Each pair of sub-blocks shares 32 qs bytes.
      // Even sub-block → low nibble, odd sub-block → high nibble.
      int qs_byte_idx = (sb / 2) * 32 + lane;
      unsigned char qbyte = b.qs[qs_byte_idx];
      int q = (sb & 1) ? (qbyte >> 4) : (qbyte & 0x0F);

      float w_val = d * static_cast<float>(sc) * static_cast<float>(q) -
                    dmin * static_cast<float>(m);

      float x_val = __half2float(x[elem_idx]);
      acc += w_val * x_val;
    }
  }

  // Warp reduction
  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
  }

  if (lane == 0) {
    output[out_idx] = __float2half(acc);
  }
}

//==============================================================================
// Q6_K Fused Dequant-GEMV
//==============================================================================

// Q6_K: 256 elements per super-block, 16 sub-blocks of 16 elements each.
// 6-bit quants: lower 4 bits in ql[], upper 2 bits in qh[].
// 8-bit scales in scales[16].
__global__ void fused_dequant_gemv_q6k(const block_q6_k *__restrict__ weight,
                                       const half *__restrict__ x,
                                       half *__restrict__ output, int N,
                                       int K) {
  int out_idx = blockIdx.x * kGemvWarpsPerBlock + (threadIdx.x >> 5);
  if (out_idx >= N)
    return;

  int lane = threadIdx.x & 31;
  int num_blocks = K / QK_K;

  const block_q6_k *row = weight + out_idx * num_blocks;

  float acc = 0.0f;

  for (int blk = 0; blk < num_blocks; ++blk) {
    const block_q6_k &b = row[blk];

    float d = __half2float(*reinterpret_cast<const half *>(&b.d));

    // Q6_K layout: 256 elements = 2 groups of 128.
    // Each group of 128 has 4 sub-groups of 32 sharing ql/qh arrays:
    //   sub 0: low nibble of ql[base..base+31],     qh bits 0-1
    //   sub 1: low nibble of ql[base+32..base+63],  qh bits 2-3
    //   sub 2: high nibble of ql[base..base+31],    qh bits 4-5
    //   sub 3: high nibble of ql[base+32..base+63], qh bits 6-7
    // (matches llama.cpp dequantize_row_q6_K)
    for (int step = 0; step < 8; ++step) {
      int g = step / 4;   // group (0 or 1)
      int sub = step % 4; // sub-pattern within group
      int elem_idx = blk * QK_K + g * 128 + sub * 32 + lane;

      // ql: sub 0,2 use ql[g*64+lane], sub 1,3 use ql[g*64+32+lane]
      // sub 0,1 use low nibble, sub 2,3 use high nibble
      int ql_idx = g * 64 + ((sub & 1) ? 32 : 0) + lane;
      unsigned char ql_byte = b.ql[ql_idx];
      int ql_val = (sub >= 2) ? (ql_byte >> 4) : (ql_byte & 0x0F);

      // qh: all 4 subs share qh[g*32+lane], each shifted by sub*2 bits
      int qh_idx = g * 32 + lane;
      int qh_val = (b.qh[qh_idx] >> (sub * 2)) & 0x03;

      // Combine to 6-bit value centered at 32
      int q = (ql_val | (qh_val << 4)) - 32;

      // Scale: 2 scales per sub (one per 16 lanes)
      int scale_idx = g * 8 + sub * 2 + lane / 16;
      float scale = static_cast<float>(b.scales[scale_idx]);

      float w_val = d * scale * static_cast<float>(q);
      float x_val = __half2float(x[elem_idx]);
      acc += w_val * x_val;
    }
  }

  // Warp reduction
  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
  }

  if (lane == 0) {
    output[out_idx] = __float2half(acc);
  }
}

//==============================================================================
// Q8_0 Fused Dequant-GEMV
//==============================================================================

// Q8_0: simplest format. 32 elements per block, 8-bit signed quants, 1 scale.
// dequantized[i] = qs[i] * d
__global__ void fused_dequant_gemv_q8_0(const block_q8_0 *__restrict__ weight,
                                        const half *__restrict__ x,
                                        half *__restrict__ output, int N,
                                        int K) {
  int out_idx = blockIdx.x * kGemvWarpsPerBlock + (threadIdx.x >> 5);
  if (out_idx >= N)
    return;

  int lane = threadIdx.x & 31;
  int num_blocks = K / QK8_0;

  const block_q8_0 *row = weight + out_idx * num_blocks;

  float acc = 0.0f;

  for (int blk = 0; blk < num_blocks; ++blk) {
    const block_q8_0 &b = row[blk];

    float d = __half2float(*reinterpret_cast<const half *>(&b.d));

    // Each lane handles one of 32 elements in this block
    int elem_idx = blk * QK8_0 + lane;
    float w_val = d * static_cast<float>(b.qs[lane]);
    float x_val = __half2float(x[elem_idx]);
    acc += w_val * x_val;
  }

  // Warp reduction
  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
  }

  if (lane == 0) {
    output[out_idx] = __float2half(acc);
  }
}

//==============================================================================
// Q8_K Fused Dequant-GEMV
//==============================================================================

// Q8_K: 256 elements per super-block, float scale, 8-bit signed quants.
// dequantized[i] = qs[i] * d
// Same formula as Q8_0 but with 256-element blocks instead of 32.
__global__ void fused_dequant_gemv_q8k(const block_q8_k *__restrict__ weight,
                                       const half *__restrict__ x,
                                       half *__restrict__ output, int N,
                                       int K) {
  int out_idx = blockIdx.x * kGemvWarpsPerBlock + (threadIdx.x >> 5);
  if (out_idx >= N)
    return;

  int lane = threadIdx.x & 31;
  int num_blocks = K / QK_K;

  const block_q8_k *row = weight + out_idx * num_blocks;

  float acc = 0.0f;

  for (int blk = 0; blk < num_blocks; ++blk) {
    const block_q8_k &b = row[blk];

    float d = b.d;

    // Each thread handles 8 elements (256 / 32 = 8 per lane)
    for (int step = 0; step < 8; ++step) {
      int elem = step * 32 + lane;
      int elem_idx = blk * QK_K + elem;

      float w_val = d * static_cast<float>(b.qs[elem]);
      float x_val = __half2float(x[elem_idx]);
      acc += w_val * x_val;
    }
  }

  // Warp reduction
  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
  }

  if (lane == 0) {
    output[out_idx] = __float2half(acc);
  }
}

} // namespace native
} // namespace cuda
} // namespace runtime
} // namespace inferflux
