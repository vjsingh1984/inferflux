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

//==============================================================================
// Q4_K Fused Dequant-GEMV (shared-memory x cache)
//==============================================================================

__global__ void fused_dequant_gemv_q4k(const block_q4_k *__restrict__ weight,
                                       const half *__restrict__ x,
                                       half *__restrict__ output, int N,
                                       int K) {
  extern __shared__ float sx[];

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x * kGemvWarpsPerBlock + warp_id;

  // Cooperatively load x into shared memory (all threads participate)
  for (int i = tid; i < K; i += kGemvThreadsPerBlock) {
    sx[i] = __half2float(x[i]);
  }
  __syncthreads();

  if (out_idx >= N)
    return;

  const int num_blocks = K / QK_K;
  const block_q4_k *row = weight + out_idx * num_blocks;

  float acc = 0.0f;

  for (int blk = 0; blk < num_blocks; ++blk) {
    const block_q4_k &b = row[blk];

    float d = __half2float(*reinterpret_cast<const half *>(&b.d));
    float dmin = __half2float(*reinterpret_cast<const half *>(&b.dmin));

    for (int sb = 0; sb < 8; ++sb) {
      int elem_idx = blk * QK_K + sb * 32 + lane;
      float w_val = dequant_q4k_element(b, d, dmin, sb, lane);
      acc += w_val * sx[elem_idx];
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
// Q6_K Fused Dequant-GEMV (shared-memory x cache)
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

  for (int i = tid; i < K; i += kGemvThreadsPerBlock) {
    sx[i] = __half2float(x[i]);
  }
  __syncthreads();

  if (out_idx >= N)
    return;

  const int num_blocks = K / QK_K;
  const block_q6_k *row = weight + out_idx * num_blocks;

  float acc = 0.0f;

  for (int blk = 0; blk < num_blocks; ++blk) {
    const block_q6_k &b = row[blk];

    float d = __half2float(*reinterpret_cast<const half *>(&b.d));

    for (int step = 0; step < 8; ++step) {
      int g = step / 4;
      int sub = step % 4;
      int elem_idx = blk * QK_K + g * 128 + sub * 32 + lane;

      float w_val = dequant_q6k_element(b, d, g, sub, lane);
      acc += w_val * sx[elem_idx];
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
// Q8_0 Fused Dequant-GEMV (shared-memory x cache)
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

  for (int i = tid; i < K; i += kGemvThreadsPerBlock) {
    sx[i] = __half2float(x[i]);
  }
  __syncthreads();

  if (out_idx >= N)
    return;

  const int num_blocks = K / QK8_0;
  const block_q8_0 *row = weight + out_idx * num_blocks;

  float acc = 0.0f;

  for (int blk = 0; blk < num_blocks; ++blk) {
    const block_q8_0 &b = row[blk];
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
    output[out_idx] = __float2half(acc);
  }
}

//==============================================================================
// Q8_K Fused Dequant-GEMV (shared-memory x cache)
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

  for (int i = tid; i < K; i += kGemvThreadsPerBlock) {
    sx[i] = __half2float(x[i]);
  }
  __syncthreads();

  if (out_idx >= N)
    return;

  const int num_blocks = K / QK_K;
  const block_q8_k *row = weight + out_idx * num_blocks;

  float acc = 0.0f;

  for (int blk = 0; blk < num_blocks; ++blk) {
    const block_q8_k &b = row[blk];
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
    output[out_idx] = __float2half(acc);
  }
}

} // namespace native
} // namespace cuda
} // namespace runtime
} // namespace inferflux
