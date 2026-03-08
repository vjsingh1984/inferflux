#pragma once

#include "runtime/backends/cuda/native/kernels/quant_common.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace inferflux {
namespace runtime {
namespace cuda {
namespace native {

// Maximum M the tiled GEMM kernels can handle (shared memory + register limit).
// The adaptive dispatch threshold may be lower on GPUs with fast tensor cores.
constexpr int kFusedGemmMaxM = 32;

// Tile dimensions for small-batch GEMM
constexpr int kTileN = 8; // Output columns per thread block
constexpr int kSmallBatchThreads = 256;

//==============================================================================
// Q4_K Fused Dequant-GEMM (small M)
//==============================================================================

// Uses shared dequant_q4k_element() from quant_common.cuh.
__global__ void fused_dequant_gemm_q4k(const block_q4_k *__restrict__ weight,
                                       const half *__restrict__ X,
                                       half *__restrict__ output, int M, int N,
                                       int K) {
  int col_start = blockIdx.x * kTileN;
  int tid = threadIdx.x;

  __shared__ float smem_acc[kFusedGemmMaxM][kTileN];

  if (tid < kFusedGemmMaxM * kTileN) {
    int r = tid / kTileN;
    int c = tid % kTileN;
    if (r < M)
      smem_acc[r][c] = 0.0f;
  }
  __syncthreads();

  int num_blocks_k = K / QK_K;

  for (int col_off = 0; col_off < kTileN && (col_start + col_off) < N;
       ++col_off) {
    int out_col = col_start + col_off;
    const block_q4_k *row = weight + out_col * num_blocks_k;

    float local_acc[kFusedGemmMaxM] = {};

    for (int blk = tid; blk < num_blocks_k; blk += kSmallBatchThreads) {
      const block_q4_k &b = row[blk];
      float d = __half2float(*reinterpret_cast<const half *>(&b.d));
      float dmin = __half2float(*reinterpret_cast<const half *>(&b.dmin));

      for (int sb = 0; sb < 8; ++sb) {
        for (int e = 0; e < 32; ++e) {
          int k_idx = blk * QK_K + sb * 32 + e;
          float w_val = dequant_q4k_element(b, d, dmin, sb, e);

          for (int row_idx = 0; row_idx < M; ++row_idx) {
            float x_val = __half2float(X[row_idx * K + k_idx]);
            local_acc[row_idx] += w_val * x_val;
          }
        }
      }
    }

    for (int row_idx = 0; row_idx < M; ++row_idx) {
      atomicAdd(&smem_acc[row_idx][col_off], local_acc[row_idx]);
    }
  }

  __syncthreads();

  if (tid < M * kTileN) {
    int r = tid / kTileN;
    int c = tid % kTileN;
    int out_col = col_start + c;
    if (out_col < N) {
      output[r * N + out_col] = __float2half(smem_acc[r][c]);
    }
  }
}

//==============================================================================
// Q6_K Fused Dequant-GEMM (small M)
//==============================================================================

// Uses shared dequant_q6k_element() from quant_common.cuh.
__global__ void fused_dequant_gemm_q6k(const block_q6_k *__restrict__ weight,
                                       const half *__restrict__ X,
                                       half *__restrict__ output, int M, int N,
                                       int K) {
  int col_start = blockIdx.x * kTileN;
  int tid = threadIdx.x;

  __shared__ float smem_acc[kFusedGemmMaxM][kTileN];

  if (tid < kFusedGemmMaxM * kTileN) {
    int r = tid / kTileN;
    int c = tid % kTileN;
    if (r < M)
      smem_acc[r][c] = 0.0f;
  }
  __syncthreads();

  int num_blocks_k = K / QK_K;

  for (int col_off = 0; col_off < kTileN && (col_start + col_off) < N;
       ++col_off) {
    int out_col = col_start + col_off;
    const block_q6_k *row = weight + out_col * num_blocks_k;

    float local_acc[kFusedGemmMaxM] = {};

    for (int blk = tid; blk < num_blocks_k; blk += kSmallBatchThreads) {
      const block_q6_k &b = row[blk];
      float d = __half2float(*reinterpret_cast<const half *>(&b.d));

      for (int step = 0; step < 8; ++step) {
        int g = step / 4;
        int sub = step % 4;

        for (int e = 0; e < 32; ++e) {
          int k_idx = blk * QK_K + g * 128 + sub * 32 + e;
          float w_val = dequant_q6k_element(b, d, g, sub, e);

          for (int row_idx = 0; row_idx < M; ++row_idx) {
            float x_val = __half2float(X[row_idx * K + k_idx]);
            local_acc[row_idx] += w_val * x_val;
          }
        }
      }
    }

    for (int row_idx = 0; row_idx < M; ++row_idx) {
      atomicAdd(&smem_acc[row_idx][col_off], local_acc[row_idx]);
    }
  }

  __syncthreads();

  if (tid < M * kTileN) {
    int r = tid / kTileN;
    int c = tid % kTileN;
    int out_col = col_start + c;
    if (out_col < N) {
      output[r * N + out_col] = __float2half(smem_acc[r][c]);
    }
  }
}

//==============================================================================
// Q8_0 Fused Dequant-GEMM (small M)
//==============================================================================

// Uses shared dequant_q8_0_element() from quant_common.cuh.
__global__ void fused_dequant_gemm_q8_0(const block_q8_0 *__restrict__ weight,
                                        const half *__restrict__ X,
                                        half *__restrict__ output, int M, int N,
                                        int K) {
  int col_start = blockIdx.x * kTileN;
  int tid = threadIdx.x;

  __shared__ float smem_acc[kFusedGemmMaxM][kTileN];

  if (tid < kFusedGemmMaxM * kTileN) {
    int r = tid / kTileN;
    int c = tid % kTileN;
    if (r < M)
      smem_acc[r][c] = 0.0f;
  }
  __syncthreads();

  int num_blocks_k = K / QK8_0;

  for (int col_off = 0; col_off < kTileN && (col_start + col_off) < N;
       ++col_off) {
    int out_col = col_start + col_off;
    const block_q8_0 *row = weight + out_col * num_blocks_k;

    float local_acc[kFusedGemmMaxM] = {};

    for (int blk = tid; blk < num_blocks_k; blk += kSmallBatchThreads) {
      const block_q8_0 &b = row[blk];
      float d = __half2float(*reinterpret_cast<const half *>(&b.d));

      for (int e = 0; e < QK8_0; ++e) {
        int k_idx = blk * QK8_0 + e;
        float w_val = dequant_q8_0_element(b, d, e);

        for (int row_idx = 0; row_idx < M; ++row_idx) {
          float x_val = __half2float(X[row_idx * K + k_idx]);
          local_acc[row_idx] += w_val * x_val;
        }
      }
    }

    for (int row_idx = 0; row_idx < M; ++row_idx) {
      atomicAdd(&smem_acc[row_idx][col_off], local_acc[row_idx]);
    }
  }

  __syncthreads();

  if (tid < M * kTileN) {
    int r = tid / kTileN;
    int c = tid % kTileN;
    int out_col = col_start + c;
    if (out_col < N) {
      output[r * N + out_col] = __float2half(smem_acc[r][c]);
    }
  }
}

//==============================================================================
// Q8_K Fused Dequant-GEMM (small M)
//==============================================================================

// Uses shared dequant_q8k_element() from quant_common.cuh.
__global__ void fused_dequant_gemm_q8k(const block_q8_k *__restrict__ weight,
                                       const half *__restrict__ X,
                                       half *__restrict__ output, int M, int N,
                                       int K) {
  int col_start = blockIdx.x * kTileN;
  int tid = threadIdx.x;

  __shared__ float smem_acc[kFusedGemmMaxM][kTileN];

  if (tid < kFusedGemmMaxM * kTileN) {
    int r = tid / kTileN;
    int c = tid % kTileN;
    if (r < M)
      smem_acc[r][c] = 0.0f;
  }
  __syncthreads();

  int num_blocks_k = K / QK_K;

  for (int col_off = 0; col_off < kTileN && (col_start + col_off) < N;
       ++col_off) {
    int out_col = col_start + col_off;
    const block_q8_k *row = weight + out_col * num_blocks_k;

    float local_acc[kFusedGemmMaxM] = {};

    for (int blk = tid; blk < num_blocks_k; blk += kSmallBatchThreads) {
      const block_q8_k &b = row[blk];
      float d = b.d;

      for (int step = 0; step < 8; ++step) {
        for (int e = 0; e < 32; ++e) {
          int elem = step * 32 + e;
          int k_idx = blk * QK_K + elem;
          float w_val = dequant_q8k_element(b, d, elem);

          for (int row_idx = 0; row_idx < M; ++row_idx) {
            float x_val = __half2float(X[row_idx * K + k_idx]);
            local_acc[row_idx] += w_val * x_val;
          }
        }
      }
    }

    for (int row_idx = 0; row_idx < M; ++row_idx) {
      atomicAdd(&smem_acc[row_idx][col_off], local_acc[row_idx]);
    }
  }

  __syncthreads();

  if (tid < M * kTileN) {
    int r = tid / kTileN;
    int c = tid % kTileN;
    int out_col = col_start + c;
    if (out_col < N) {
      output[r * N + out_col] = __float2half(smem_acc[r][c]);
    }
  }
}

} // namespace native
} // namespace cuda
} // namespace runtime
} // namespace inferflux
