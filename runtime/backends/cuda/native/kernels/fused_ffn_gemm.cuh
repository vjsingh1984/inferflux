#pragma once

#include "runtime/backends/cuda/native/kernels/quant_common.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace inferflux {
namespace runtime {
namespace cuda {
namespace native {

//------------------------------------------------------------------------------
// SwiGLU activation function: SiLU(x) * y
//------------------------------------------------------------------------------
__device__ __forceinline__ float SwiGLU(float gate, float up) {
  const float silu = gate * (1.0f / (1.0f + expf(-gate)));
  return silu * up;
}

__device__ __forceinline__ float WarpReduceSum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xFFFFFFFF, value, offset);
  }
  return __shfl_sync(0xFFFFFFFF, value, 0);
}

template <typename BlockType>
__device__ __forceinline__ float DotQ4KRowFromSmem(
    const BlockType *__restrict__ wrow, const float *__restrict__ activation,
    int K, int lane) {
  float acc = 0.0f;
  constexpr int QK = QK_K;
  const int num_blocks = K / QK;

  for (int blk = 0; blk < num_blocks; ++blk) {
    const BlockType &b = wrow[blk];
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

      const int base = blk * QK + pair * 64;
      acc += (d_sc_lo * q_lo - dm_m_lo) * activation[base + lane];
      acc += (d_sc_hi * q_hi - dm_m_hi) * activation[base + 32 + lane];
    }
  }

  return WarpReduceSum(acc);
}

//==============================================================================
// Fused FFN GEMV Kernel for Q4_K Quantization
//==============================================================================
//
// One CTA owns:
//   - one batch row
//   - an output tile of hidden rows
//
// The CTA computes a tile of activated intermediate values once, stores them in
// shared memory, and reuses them across the whole output tile. This avoids the
// pathological "recompute gate/up for every output row" behavior of the initial
// bring-up kernel while keeping the implementation narrow and parity-testable.
//

template <typename BlockType>
__global__ void FusedFFNGemmQ4K(
    const BlockType *__restrict__ gate_weight,
    const BlockType *__restrict__ up_weight,
    const BlockType *__restrict__ down_weight,
    const half *__restrict__ activation,
    half *__restrict__ output,
    int N_inter,
    int N_hidden,
    int M,
    int K
) {
  constexpr int kOutputTile = kGemvWarpsPerBlock;
  constexpr int kIntermediateTile = 32;

  extern __shared__ float shared[];
  float *s_activation = shared;
  float *s_activated = shared + K;

  const int tid = threadIdx.x;
  const int row = blockIdx.y;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
  const int out_base = blockIdx.x * kOutputTile;
  const int out_idx = out_base + warp_id;

  if (row >= M) {
    return;
  }

  // Load activation into shared memory
  const half *act_row = activation + row * K;
  LoadHalfToSmem(act_row, s_activation, K, tid);
  __syncthreads();

  float down_acc = 0.0f;
  constexpr int QK = QK_K;
  const int num_gate_blocks = K / QK;
  const int num_down_blocks = N_inter / QK;

  for (int inter_base = 0; inter_base < N_inter;
       inter_base += kIntermediateTile) {
#pragma unroll
    for (int pass = 0; pass < (kIntermediateTile / kOutputTile); ++pass) {
      const int tile_inter = pass * kOutputTile + warp_id;
      const int inter_idx = inter_base + tile_inter;
      float activated = 0.0f;

      if (inter_idx < N_inter) {
        const BlockType *gate_wrow = gate_weight + inter_idx * num_gate_blocks;
        const BlockType *up_wrow = up_weight + inter_idx * num_gate_blocks;
        const float gate_acc =
            DotQ4KRowFromSmem(gate_wrow, s_activation, K, lane);
        const float up_acc =
            DotQ4KRowFromSmem(up_wrow, s_activation, K, lane);
        if (lane == 0) {
          activated = SwiGLU(gate_acc, up_acc);
        }
      }

      if (lane == 0) {
        s_activated[tile_inter] = activated;
      }
    }
    __syncthreads();

    if (out_idx < N_hidden) {
      const BlockType *down_wrow = down_weight + out_idx * num_down_blocks;
      float partial = 0.0f;
      const int inter_idx = inter_base + lane;
      if (inter_idx < N_inter) {
        const int down_blk = inter_idx / QK;
        const int in_block = inter_idx % QK;
        const int sb = in_block / 32;
        const int e = in_block % 32;
        const BlockType &db = down_wrow[down_blk];
        const float dd = __half2float(*reinterpret_cast<const half *>(&db.d));
        const float ddmin =
            __half2float(*reinterpret_cast<const half *>(&db.dmin));
        const float down_w = dequant_q4k_element(db, dd, ddmin, sb, e);
        partial = s_activated[lane] * down_w;
      }

      down_acc += WarpReduceSum(partial);
    }
    __syncthreads();
  }

  if (out_idx < N_hidden && lane == 0) {
    output[row * N_hidden + out_idx] = __float2half(down_acc);
  }
}

} // namespace native
} // namespace cuda
} // namespace runtime
} // namespace inferflux
