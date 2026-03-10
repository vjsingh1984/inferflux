#pragma once

#include "runtime/backends/cuda/native/kernels/fused_dequant_gemv.cuh"

namespace inferflux {
namespace runtime {
namespace cuda {
namespace native {

// ============================================================================
// Cooperative-Warp GEMV v2
//
// All warps in a block collaborate on ONE output element, each processing a
// strided subset of super-blocks.  This gives coherent L2 access (one weight
// row per block) instead of the v1 column-major pattern where 8 warps access
// 8 unrelated weight rows.
//
// Block: 128 threads = 4 warps × 32 lanes  (kGemvWarpsPerBlockV2)
// Grid:  dim3(N, 1, M)  — one block per output element per batch row
// Smem:  float warp_sums[kGemvWarpsPerBlockV2]  (16 bytes)
// ============================================================================

constexpr int kGemvWarpsPerBlockV2 = 4;
constexpr int kGemvThreadsPerBlockV2 = kGemvWarpsPerBlockV2 * 32; // 128

// ============================================================================
// Q4_K × Q8_1 cooperative kernel
// ============================================================================

__global__ void
fused_dequant_gemv_q4k_q8_1_v2(const block_q4_k *__restrict__ weight,
                                const block_q8_1 *__restrict__ act_q8_1,
                                half *__restrict__ output, int N, int K) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x;
  const int row = blockIdx.z;
  if (out_idx >= N)
    return;

  const int num_super_blocks = K / QK_K;
  const block_q4_k *wrow = weight + out_idx * num_super_blocks;
  const int num_q8_per_row = K / QK8_1;
  const block_q8_1 *a_row = act_q8_1 + row * num_q8_per_row;

  const int pair = lane >> 3;
  const int offs = (lane & 7) * 4;

  float acc = 0.0f;

  // First lane in each 8-lane group index for scale broadcast
  const int group_leader = lane & ~7;

  // Cooperative: each warp handles strided super-blocks
  for (int blk = warp_id; blk < num_super_blocks; blk += kGemvWarpsPerBlockV2) {
    const block_q4_k &b = wrow[blk];
    const float d = __half2float(*reinterpret_cast<const half *>(&b.d));
    const float dmin = __half2float(*reinterpret_cast<const half *>(&b.dmin));

    // Scale extraction: only group leader computes, broadcast via shuffle.
    // Reduces get_scale_min_k4 calls from 32 to 4 per warp per super-block.
    unsigned char sc_lo, m_lo, sc_hi, m_hi;
    if ((lane & 7) == 0) {
      get_scale_min_k4(pair * 2, b.scales, &sc_lo, &m_lo);
      get_scale_min_k4(pair * 2 + 1, b.scales, &sc_hi, &m_hi);
    }
    sc_lo = __shfl_sync(0xFFFFFFFF, sc_lo, group_leader);
    m_lo = __shfl_sync(0xFFFFFFFF, m_lo, group_leader);
    sc_hi = __shfl_sync(0xFFFFFFFF, sc_hi, group_leader);
    m_hi = __shfl_sync(0xFFFFFFFF, m_hi, group_leader);

    const float d_sc_lo = d * static_cast<float>(sc_lo);
    const float dm_m_lo = dmin * static_cast<float>(m_lo);
    const float d_sc_hi = d * static_cast<float>(sc_hi);
    const float dm_m_hi = dmin * static_cast<float>(m_hi);

    int qs4 = *reinterpret_cast<const int *>(&b.qs[pair * 32 + offs]);
    int q_lo4 = qs4 & 0x0F0F0F0F;
    int q_hi4 = (qs4 >> 4) & 0x0F0F0F0F;

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

    if ((lane & 7) == 0) {
      float s_lo = __half2float(__high2half(a_lo.ds));
      float s_hi = __half2float(__high2half(a_hi.ds));
      acc -= dm_m_lo * s_lo + dm_m_hi * s_hi;
    }
  }

  // Intra-warp reduction
  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
  }

  // Cross-warp reduction via shared memory
  __shared__ float warp_sums[kGemvWarpsPerBlockV2];
  if (lane == 0)
    warp_sums[warp_id] = acc;
  __syncthreads();
  if (tid == 0) {
    float sum = warp_sums[0];
    for (int w = 1; w < kGemvWarpsPerBlockV2; ++w)
      sum += warp_sums[w];
    output[row * N + out_idx] = __float2half(sum);
  }
}

// ============================================================================
// Q6_K × Q8_1 cooperative kernel
// ============================================================================

__global__ void
fused_dequant_gemv_q6k_q8_1_v2(const block_q6_k *__restrict__ weight,
                                const block_q8_1 *__restrict__ act_q8_1,
                                half *__restrict__ output, int N, int K) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x;
  const int row = blockIdx.z;
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

  for (int blk = warp_id; blk < num_super_blocks; blk += kGemvWarpsPerBlockV2) {
    const block_q6_k &b = wrow[blk];
    const float d = __half2float(*reinterpret_cast<const half *>(&b.d));

    const int ql4 =
        LoadPackedInt32Unaligned(&b.ql[g * 64 + sub_base * 32 + e_base]);
    const int qh4 = LoadPackedInt32Unaligned(&b.qh[g * 32 + e_base]);

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

  // Intra-warp reduction
  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
  }

  // Cross-warp reduction
  __shared__ float warp_sums[kGemvWarpsPerBlockV2];
  if (lane == 0)
    warp_sums[warp_id] = acc;
  __syncthreads();
  if (tid == 0) {
    float sum = warp_sums[0];
    for (int w = 1; w < kGemvWarpsPerBlockV2; ++w)
      sum += warp_sums[w];
    output[row * N + out_idx] = __float2half(sum);
  }
}

// ============================================================================
// Q8_0 × Q8_1 cooperative kernel
// ============================================================================

__global__ void
fused_dequant_gemv_q8_0_q8_1_v2(const block_q8_0 *__restrict__ weight,
                                 const block_q8_1 *__restrict__ act_q8_1,
                                 half *__restrict__ output, int N, int K) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x;
  const int row = blockIdx.z;
  if (out_idx >= N)
    return;

  // Q8_0 and Q8_1 both have 32-element blocks: 1:1 mapping
  const int num_blocks = K / QK8_0;
  const block_q8_0 *wrow = weight + out_idx * num_blocks;
  const block_q8_1 *a_row = act_q8_1 + row * num_blocks;

  float acc = 0.0f;
  // Each warp iterates over strided blocks; within each block each lane
  // handles the full 32-element dot product (same as v1).
  for (int blk = warp_id * 32 + lane; blk < num_blocks;
       blk += kGemvWarpsPerBlockV2 * 32) {
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

  // Intra-warp reduction
  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
  }

  // Cross-warp reduction
  __shared__ float warp_sums[kGemvWarpsPerBlockV2];
  if (lane == 0)
    warp_sums[warp_id] = acc;
  __syncthreads();
  if (tid == 0) {
    float sum = warp_sums[0];
    for (int w = 1; w < kGemvWarpsPerBlockV2; ++w)
      sum += warp_sums[w];
    output[row * N + out_idx] = __float2half(sum);
  }
}

// ============================================================================
// Q8_K × Q8_1 cooperative kernel
// ============================================================================

__global__ void
fused_dequant_gemv_q8k_q8_1_v2(const block_q8_k *__restrict__ weight,
                                const block_q8_1 *__restrict__ act_q8_1,
                                half *__restrict__ output, int N, int K) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x;
  const int row = blockIdx.z;
  if (out_idx >= N)
    return;

  // Q8_K super-block = 256 elements = 8 Q8_1 blocks
  const int num_super_blocks = K / QK_K;
  const block_q8_k *wrow = weight + out_idx * num_super_blocks;
  const int num_q8_per_row = K / QK8_1;
  const block_q8_1 *a_row = act_q8_1 + row * num_q8_per_row;

  float acc = 0.0f;
  // v1 distributes blocks across lanes; v2 distributes across warps too
  for (int blk = warp_id * 32 + lane; blk < num_super_blocks;
       blk += kGemvWarpsPerBlockV2 * 32) {
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

  // Intra-warp reduction
  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
  }

  // Cross-warp reduction
  __shared__ float warp_sums[kGemvWarpsPerBlockV2];
  if (lane == 0)
    warp_sums[warp_id] = acc;
  __syncthreads();
  if (tid == 0) {
    float sum = warp_sums[0];
    for (int w = 1; w < kGemvWarpsPerBlockV2; ++w)
      sum += warp_sums[w];
    output[row * N + out_idx] = __float2half(sum);
  }
}

// ============================================================================
// Q4_K × Q8_1 cooperative grouped kernel (2 or 3 outputs)
//
// All warps collaborate on ONE output column index, processing Outputs weight
// rows.  Activations are loaded once per super-block and reused for all
// outputs (same benefit as v1 grouped kernels, plus L2 locality).
// ============================================================================

template <int Outputs>
__global__ void fused_dequant_gemv_q4k_q8_1_v2_group(
    PackedProjectionGroupParams<block_q4_k, Outputs> params,
    const block_q8_1 *__restrict__ act_q8_1, int K) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x;
  const int row = blockIdx.z;

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
  const int group_leader = lane & ~7;

  float acc[Outputs] = {};

  for (int blk = warp_id; blk < num_super_blocks; blk += kGemvWarpsPerBlockV2) {
    // Load activation once, reuse across all outputs
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
      if ((lane & 7) == 0) {
        get_scale_min_k4(pair * 2, b.scales, &sc_lo, &m_lo);
        get_scale_min_k4(pair * 2 + 1, b.scales, &sc_hi, &m_hi);
      }
      sc_lo = __shfl_sync(0xFFFFFFFF, sc_lo, group_leader);
      m_lo = __shfl_sync(0xFFFFFFFF, m_lo, group_leader);
      sc_hi = __shfl_sync(0xFFFFFFFF, sc_hi, group_leader);
      m_hi = __shfl_sync(0xFFFFFFFF, m_hi, group_leader);

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

  // Intra-warp reduction for each output
#pragma unroll
  for (int i = 0; i < Outputs; ++i) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc[i] += __shfl_down_sync(0xFFFFFFFF, acc[i], offset);
    }
  }

  // Cross-warp reduction
  __shared__ float warp_sums[kGemvWarpsPerBlockV2 * Outputs];
  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < Outputs; ++i) {
      warp_sums[i * kGemvWarpsPerBlockV2 + warp_id] = acc[i];
    }
  }
  __syncthreads();
  if (tid == 0) {
#pragma unroll
    for (int i = 0; i < Outputs; ++i) {
      if (out_idx < params.output_cols[i]) {
        float sum = 0.0f;
        for (int w = 0; w < kGemvWarpsPerBlockV2; ++w)
          sum += warp_sums[i * kGemvWarpsPerBlockV2 + w];
        params.outputs[i][row * params.output_cols[i] + out_idx] =
            __float2half(sum);
      }
    }
  }
}

// ============================================================================
// Q6_K × Q8_1 cooperative grouped kernel (2 or 3 outputs)
// ============================================================================

template <int Outputs>
__global__ void fused_dequant_gemv_q6k_q8_1_v2_group(
    PackedProjectionGroupParams<block_q6_k, Outputs> params,
    const block_q8_1 *__restrict__ act_q8_1, int K) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x;
  const int row = blockIdx.z;

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

  for (int blk = warp_id; blk < num_super_blocks; blk += kGemvWarpsPerBlockV2) {
    // Load activation once
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

  // Intra-warp reduction
#pragma unroll
  for (int i = 0; i < Outputs; ++i) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc[i] += __shfl_down_sync(0xFFFFFFFF, acc[i], offset);
    }
  }

  // Cross-warp reduction
  __shared__ float warp_sums[kGemvWarpsPerBlockV2 * Outputs];
  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < Outputs; ++i) {
      warp_sums[i * kGemvWarpsPerBlockV2 + warp_id] = acc[i];
    }
  }
  __syncthreads();
  if (tid == 0) {
#pragma unroll
    for (int i = 0; i < Outputs; ++i) {
      if (out_idx < params.output_cols[i]) {
        float sum = 0.0f;
        for (int w = 0; w < kGemvWarpsPerBlockV2; ++w)
          sum += warp_sums[i * kGemvWarpsPerBlockV2 + w];
        params.outputs[i][row * params.output_cols[i] + out_idx] =
            __float2half(sum);
      }
    }
  }
}

// ============================================================================
// Q4_K × Q8_1 cooperative row-pair kernel (M=2-3)
//
// All 4 warps cooperate on K, but each block handles 2 input rows by sharing
// weight loads across both activation rows.
// ============================================================================

__global__ void fused_dequant_gemv_q4k_q8_1_v2_rowpair(
    const block_q4_k *__restrict__ weight,
    const block_q8_1 *__restrict__ act_q8_1, half *__restrict__ output, int N,
    int K, int total_rows) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x;
  const int row0 = blockIdx.z * 2;
  const int row1 = row0 + 1;
  if (out_idx >= N || row0 >= total_rows)
    return;

  const bool has_row1 = row1 < total_rows;

  const int num_super_blocks = K / QK_K;
  const block_q4_k *wrow = weight + out_idx * num_super_blocks;
  const int num_q8_per_row = K / QK8_1;
  const block_q8_1 *a_row0 = act_q8_1 + row0 * num_q8_per_row;
  const block_q8_1 *a_row1 =
      has_row1 ? (act_q8_1 + row1 * num_q8_per_row) : nullptr;

  const int pair = lane >> 3;
  const int offs = (lane & 7) * 4;
  const int group_leader = lane & ~7;

  float acc0 = 0.0f;
  float acc1 = 0.0f;

  for (int blk = warp_id; blk < num_super_blocks; blk += kGemvWarpsPerBlockV2) {
    const block_q4_k &b = wrow[blk];
    const float d = __half2float(*reinterpret_cast<const half *>(&b.d));
    const float dmin = __half2float(*reinterpret_cast<const half *>(&b.dmin));

    unsigned char sc_lo, m_lo, sc_hi, m_hi;
    if ((lane & 7) == 0) {
      get_scale_min_k4(pair * 2, b.scales, &sc_lo, &m_lo);
      get_scale_min_k4(pair * 2 + 1, b.scales, &sc_hi, &m_hi);
    }
    sc_lo = __shfl_sync(0xFFFFFFFF, sc_lo, group_leader);
    m_lo = __shfl_sync(0xFFFFFFFF, m_lo, group_leader);
    sc_hi = __shfl_sync(0xFFFFFFFF, sc_hi, group_leader);
    m_hi = __shfl_sync(0xFFFFFFFF, m_hi, group_leader);

    const float d_sc_lo = d * static_cast<float>(sc_lo);
    const float dm_m_lo = dmin * static_cast<float>(m_lo);
    const float d_sc_hi = d * static_cast<float>(sc_hi);
    const float dm_m_hi = dmin * static_cast<float>(m_hi);

    int qs4 = *reinterpret_cast<const int *>(&b.qs[pair * 32 + offs]);
    int q_lo4 = qs4 & 0x0F0F0F0F;
    int q_hi4 = (qs4 >> 4) & 0x0F0F0F0F;

    // Row 0
    {
      const block_q8_1 &a_lo = a_row0[blk * 8 + pair * 2];
      const block_q8_1 &a_hi = a_row0[blk * 8 + pair * 2 + 1];
      int x_lo4, x_hi4;
      memcpy(&x_lo4, &a_lo.qs[offs], sizeof(x_lo4));
      memcpy(&x_hi4, &a_hi.qs[offs], sizeof(x_hi4));
      float d8_lo = __half2float(__low2half(a_lo.ds));
      float d8_hi = __half2float(__low2half(a_hi.ds));
      acc0 += d_sc_lo * d8_lo * static_cast<float>(Dp4aS8(q_lo4, x_lo4, 0)) +
              d_sc_hi * d8_hi * static_cast<float>(Dp4aS8(q_hi4, x_hi4, 0));
      if ((lane & 7) == 0) {
        acc0 -= dm_m_lo * __half2float(__high2half(a_lo.ds)) +
                dm_m_hi * __half2float(__high2half(a_hi.ds));
      }
    }

    // Row 1
    if (has_row1) {
      const block_q8_1 &a_lo = a_row1[blk * 8 + pair * 2];
      const block_q8_1 &a_hi = a_row1[blk * 8 + pair * 2 + 1];
      int x_lo4, x_hi4;
      memcpy(&x_lo4, &a_lo.qs[offs], sizeof(x_lo4));
      memcpy(&x_hi4, &a_hi.qs[offs], sizeof(x_hi4));
      float d8_lo = __half2float(__low2half(a_lo.ds));
      float d8_hi = __half2float(__low2half(a_hi.ds));
      acc1 += d_sc_lo * d8_lo * static_cast<float>(Dp4aS8(q_lo4, x_lo4, 0)) +
              d_sc_hi * d8_hi * static_cast<float>(Dp4aS8(q_hi4, x_hi4, 0));
      if ((lane & 7) == 0) {
        acc1 -= dm_m_lo * __half2float(__high2half(a_lo.ds)) +
                dm_m_hi * __half2float(__high2half(a_hi.ds));
      }
    }
  }

  // Intra-warp reduction
  for (int offset = 16; offset > 0; offset >>= 1) {
    acc0 += __shfl_down_sync(0xFFFFFFFF, acc0, offset);
    acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
  }

  // Cross-warp reduction
  __shared__ float warp_sums[kGemvWarpsPerBlockV2 * 2];
  if (lane == 0) {
    warp_sums[warp_id] = acc0;
    warp_sums[kGemvWarpsPerBlockV2 + warp_id] = acc1;
  }
  __syncthreads();
  if (tid == 0) {
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    for (int w = 0; w < kGemvWarpsPerBlockV2; ++w) {
      sum0 += warp_sums[w];
      sum1 += warp_sums[kGemvWarpsPerBlockV2 + w];
    }
    output[row0 * N + out_idx] = __float2half(sum0);
    if (has_row1) {
      output[row1 * N + out_idx] = __float2half(sum1);
    }
  }
}

// ============================================================================
// Q6_K × Q8_1 cooperative row-pair kernel (M=2-3)
// ============================================================================

__global__ void fused_dequant_gemv_q6k_q8_1_v2_rowpair(
    const block_q6_k *__restrict__ weight,
    const block_q8_1 *__restrict__ act_q8_1, half *__restrict__ output, int N,
    int K, int total_rows) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int out_idx = blockIdx.x;
  const int row0 = blockIdx.z * 2;
  const int row1 = row0 + 1;
  if (out_idx >= N || row0 >= total_rows)
    return;

  const bool has_row1 = row1 < total_rows;

  const int num_super_blocks = K / QK_K;
  const block_q6_k *wrow = weight + out_idx * num_super_blocks;
  const int num_q8_per_row = K / QK8_1;
  const block_q8_1 *a_row0 = act_q8_1 + row0 * num_q8_per_row;
  const block_q8_1 *a_row1 =
      has_row1 ? (act_q8_1 + row1 * num_q8_per_row) : nullptr;

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

  for (int blk = warp_id; blk < num_super_blocks; blk += kGemvWarpsPerBlockV2) {
    const block_q6_k &b = wrow[blk];
    const float d = __half2float(*reinterpret_cast<const half *>(&b.d));

    const int ql4 =
        LoadPackedInt32Unaligned(&b.ql[g * 64 + sub_base * 32 + e_base]);
    const int qh4 = LoadPackedInt32Unaligned(&b.qh[g * 32 + e_base]);

    int vl_lo = ql4 & 0x0F0F0F0F;
    int vh_lo = ((qh4 >> qh_shift_lo) << 4) & 0x30303030;
    int vi_lo = Vsubss4(vl_lo | vh_lo, 0x20202020);

    int vl_hi = (ql4 >> 4) & 0x0F0F0F0F;
    int vh_hi = ((qh4 >> qh_shift_hi) << 4) & 0x30303030;
    int vi_hi = Vsubss4(vl_hi | vh_hi, 0x20202020);

    // Row 0
    {
      const block_q8_1 &a_lo = a_row0[blk * 8 + g * 4 + sub_base];
      const block_q8_1 &a_hi = a_row0[blk * 8 + g * 4 + sub_base + 2];
      const int x_lo = LoadPackedInt32Unaligned(&a_lo.qs[e_base]);
      const int x_hi = LoadPackedInt32Unaligned(&a_hi.qs[e_base]);
      float d8_lo = __half2float(__low2half(a_lo.ds));
      float d8_hi = __half2float(__low2half(a_hi.ds));
      acc0 += d * (static_cast<float>(b.scales[sc_lo]) * d8_lo *
                        static_cast<float>(Dp4aS8(vi_lo, x_lo, 0)) +
                    static_cast<float>(b.scales[sc_hi]) * d8_hi *
                        static_cast<float>(Dp4aS8(vi_hi, x_hi, 0)));
    }

    // Row 1
    if (has_row1) {
      const block_q8_1 &a_lo = a_row1[blk * 8 + g * 4 + sub_base];
      const block_q8_1 &a_hi = a_row1[blk * 8 + g * 4 + sub_base + 2];
      const int x_lo = LoadPackedInt32Unaligned(&a_lo.qs[e_base]);
      const int x_hi = LoadPackedInt32Unaligned(&a_hi.qs[e_base]);
      float d8_lo = __half2float(__low2half(a_lo.ds));
      float d8_hi = __half2float(__low2half(a_hi.ds));
      acc1 += d * (static_cast<float>(b.scales[sc_lo]) * d8_lo *
                        static_cast<float>(Dp4aS8(vi_lo, x_lo, 0)) +
                    static_cast<float>(b.scales[sc_hi]) * d8_hi *
                        static_cast<float>(Dp4aS8(vi_hi, x_hi, 0)));
    }
  }

  // Intra-warp reduction
  for (int offset = 16; offset > 0; offset >>= 1) {
    acc0 += __shfl_down_sync(0xFFFFFFFF, acc0, offset);
    acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
  }

  // Cross-warp reduction
  __shared__ float warp_sums[kGemvWarpsPerBlockV2 * 2];
  if (lane == 0) {
    warp_sums[warp_id] = acc0;
    warp_sums[kGemvWarpsPerBlockV2 + warp_id] = acc1;
  }
  __syncthreads();
  if (tid == 0) {
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    for (int w = 0; w < kGemvWarpsPerBlockV2; ++w) {
      sum0 += warp_sums[w];
      sum1 += warp_sums[kGemvWarpsPerBlockV2 + w];
    }
    output[row0 * N + out_idx] = __float2half(sum0);
    if (has_row1) {
      output[row1 * N + out_idx] = __float2half(sum1);
    }
  }
}

} // namespace native
} // namespace cuda
} // namespace runtime
} // namespace inferflux
