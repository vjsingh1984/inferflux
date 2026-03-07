#pragma once

#include "runtime/backends/cuda/native/kernels/dequantization.cuh"

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

// For M=2..8 activations:
// Grid: (ceil(N / kTileN), 1, 1)
// Block: (kSmallBatchThreads, 1, 1)
//
// Each thread block computes a [M, kTileN] output tile.
// Threads cooperatively iterate over K dimension, loading and dequantizing
// Q4_K blocks, then computing partial products for all M rows.
__global__ void fused_dequant_gemm_q4k(const block_q4_k *__restrict__ weight,
                                       const half *__restrict__ X,
                                       half *__restrict__ output, int M, int N,
                                       int K) {
  // Each thread block handles kTileN output columns
  int col_start = blockIdx.x * kTileN;
  int tid = threadIdx.x;

  // Shared memory for accumulation: [kFusedGemmMaxM][kTileN]
  __shared__ float smem_acc[kFusedGemmMaxM][kTileN];

  // Initialize accumulators
  if (tid < kFusedGemmMaxM * kTileN) {
    int r = tid / kTileN;
    int c = tid % kTileN;
    if (r < M)
      smem_acc[r][c] = 0.0f;
  }
  __syncthreads();

  int num_blocks_k = K / QK_K;

  // Each thread iterates over a subset of K-dimension super-blocks
  for (int col_off = 0; col_off < kTileN && (col_start + col_off) < N;
       ++col_off) {
    int out_col = col_start + col_off;
    const block_q4_k *row = weight + out_col * num_blocks_k;

    float local_acc[kFusedGemmMaxM] = {};

    for (int blk = tid; blk < num_blocks_k; blk += kSmallBatchThreads) {
      const block_q4_k &b = row[blk];
      float d = __half2float(*reinterpret_cast<const half *>(&b.d));
      float dmin = __half2float(*reinterpret_cast<const half *>(&b.dmin));

      // Process all 256 elements in this super-block.
      // Q4_K layout: 4 groups of 64, each group has 2 sub-blocks
      // sharing 32 qs bytes (low then high nibble).
      for (int sb = 0; sb < 8; ++sb) {
        unsigned char sc, m_val;
        if (sb < 4) {
          sc = b.scales[sb] & 63;
          m_val = b.scales[sb + 4] & 63;
        } else {
          sc = (b.scales[sb + 4] & 0xF) | ((b.scales[sb - 4] >> 6) << 4);
          m_val = (b.scales[sb + 4] >> 4) | ((b.scales[sb] >> 6) << 4);
        }

        for (int e = 0; e < 32; ++e) {
          int k_idx = blk * QK_K + sb * 32 + e;

          // Paired sub-blocks share 32 qs bytes
          int qs_byte_idx = (sb / 2) * 32 + e;
          unsigned char qbyte = b.qs[qs_byte_idx];
          int q = (sb & 1) ? (qbyte >> 4) : (qbyte & 0x0F);

          float w_val = d * static_cast<float>(sc) * static_cast<float>(q) -
                        dmin * static_cast<float>(m_val);

          for (int row_idx = 0; row_idx < M; ++row_idx) {
            float x_val = __half2float(X[row_idx * K + k_idx]);
            local_acc[row_idx] += w_val * x_val;
          }
        }
      }
    }

    // Reduce local accumulators to shared memory
    for (int row_idx = 0; row_idx < M; ++row_idx) {
      atomicAdd(&smem_acc[row_idx][col_off], local_acc[row_idx]);
    }
  }

  __syncthreads();

  // Write output
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

          int ql_idx = g * 64 + ((sub & 1) ? 32 : 0) + e;
          unsigned char ql_byte = b.ql[ql_idx];
          int ql_val = (sub >= 2) ? (ql_byte >> 4) : (ql_byte & 0x0F);

          int qh_idx = g * 32 + e;
          int qh_val = (b.qh[qh_idx] >> (sub * 2)) & 0x03;

          int q = (ql_val | (qh_val << 4)) - 32;

          int scale_idx = g * 8 + sub * 2 + e / 16;
          float scale = static_cast<float>(b.scales[scale_idx]);

          float w_val = d * scale * static_cast<float>(q);

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
        float w_val = d * static_cast<float>(b.qs[e]);

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
          float w_val = d * static_cast<float>(b.qs[elem]);

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
