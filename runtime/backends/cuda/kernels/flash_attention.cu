#include "flash_attention.cuh"
#include "runtime/backends/cuda/common/dtype_traits.cuh"
#include <cooperative_groups.h>
#include <cmath>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;
namespace inferflux {
namespace cuda_kernel {

constexpr int WARP_SIZE = 32;

// ============================================================================
// Device helpers
// ============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}

// ============================================================================
// Standard Scaled Dot-Product Attention (FP32, kept for reference)
// ============================================================================

__global__ void ScaledDotProductAttentionKernel(
    const float* __restrict__ Q, const float* __restrict__ K,
    const float* __restrict__ V, float* __restrict__ O, int batch_size,
    int num_heads, int seq_len, int head_dim, float scale) {
  const int batch_idx = blockIdx.z;
  const int head_idx = blockIdx.y;
  const int seq_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (batch_idx >= batch_size || head_idx >= num_heads || seq_idx >= seq_len)
    return;

  const size_t batch_offset = batch_idx * num_heads * seq_len * head_dim;
  const size_t head_offset = head_idx * seq_len * head_dim;
  const size_t seq_offset = seq_idx * head_dim;

  const float* q_ptr = Q + batch_offset + head_offset + seq_offset;
  float* o_ptr = O + batch_offset + head_offset + seq_offset;

  extern __shared__ float attention_scores[];

  float max_score = -INFINITY;
  float sum_exp = 0.0f;

  for (int k_idx = 0; k_idx < seq_len; k_idx++) {
    const float* k_ptr = K + batch_offset + head_offset + k_idx * head_dim;
    float dot = 0.0f;
    for (int d = 0; d < head_dim; d++) {
      dot += q_ptr[d] * k_ptr[d];
    }
    float score = dot * scale;
    if (score > max_score) {
      sum_exp = expf(max_score - score) * sum_exp + 1.0f;
      max_score = score;
    } else {
      sum_exp += expf(score - max_score);
    }
    attention_scores[k_idx] = score;
  }

  for (int d = 0; d < head_dim; d++) {
    float acc = 0.0f;
    for (int k_idx = 0; k_idx < seq_len; k_idx++) {
      float weight = expf(attention_scores[k_idx] - max_score) / sum_exp;
      const float* v_ptr = V + batch_offset + head_offset + k_idx * head_dim;
      acc += weight * v_ptr[d];
    }
    o_ptr[d] = acc;
  }
}

// ============================================================================
// FlashAttention-2 (FP32, simplified tiled)
// ============================================================================

__global__ void FlashAttention2Kernel(const float* __restrict__ Q,
                                      const float* __restrict__ K,
                                      const float* __restrict__ V,
                                      float* __restrict__ O, int batch_size,
                                      int num_heads, int seq_len, int head_dim,
                                      float scale) {
  const int tid = threadIdx.x;
  const int batch_idx = blockIdx.z;
  const int head_idx = blockIdx.y;
  const int seq_start = blockIdx.x * blockDim.x;

  if (batch_idx >= batch_size || head_idx >= num_heads || seq_start >= seq_len)
    return;

  const int seq_idx = seq_start + tid;
  const size_t batch_offset = batch_idx * num_heads * seq_len * head_dim;
  const size_t head_offset = head_idx * seq_len * head_dim;

  float row_max = -INFINITY;
  float row_sum = 0.0f;
  float O_acc[256] = {0.0f};

  const int TILE_SIZE = blockDim.x;
  for (int j = 0; j < seq_len; j += TILE_SIZE) {
    __syncthreads();
    for (int k_idx = j; k_idx < min(j + TILE_SIZE, seq_len); k_idx++) {
      if (seq_idx >= seq_len) continue;

      const float* q_ptr = Q + batch_offset + head_offset + seq_idx * head_dim;
      const float* k_ptr = K + batch_offset + head_offset + k_idx * head_dim;

      float qk = 0.0f;
      for (int d = 0; d < head_dim; d++) {
        qk += q_ptr[d] * k_ptr[d];
      }
      float score = qk * scale;

      float prev_max = row_max;
      row_max = fmaxf(row_max, score);
      row_sum = row_sum * expf(prev_max - row_max) + expf(score - row_max);

      const float* v_ptr = V + batch_offset + head_offset + k_idx * head_dim;
      float weight = expf(score - row_max);
      for (int d = 0; d < head_dim; d++) {
        O_acc[d] += weight * v_ptr[d];
      }
    }
  }

  if (seq_idx < seq_len) {
    float* o_ptr = O + batch_offset + head_offset + seq_idx * head_dim;
    for (int d = 0; d < head_dim; d++) {
      o_ptr[d] = O_acc[d] / row_sum;
    }
  }
}

// ============================================================================
// FlashAttention-2 FP16 with Causal Mask + GQA (legacy non-templated)
// ============================================================================

__global__ void FlashAttention2FP16KernelV2(
    const half* __restrict__ Q, const half* __restrict__ K,
    const half* __restrict__ V, half* __restrict__ O, int query_len,
    int kv_len, int num_heads, int num_kv_heads, int head_dim, float scale,
    bool causal) {
  const int batch_idx = blockIdx.z;
  const int head_idx = blockIdx.y;
  const int q_pos = blockIdx.x;
  const int d = threadIdx.x;

  if (q_pos >= query_len || d >= head_dim) return;

  const int kv_head_ratio =
      (num_kv_heads > 0) ? (num_heads / num_kv_heads) : 1;
  const int kv_head_idx = head_idx / kv_head_ratio;

  const size_t q_base =
      ((size_t)batch_idx * num_heads + head_idx) * query_len * head_dim;
  const size_t kv_base =
      ((size_t)batch_idx * num_kv_heads + kv_head_idx) * kv_len * head_dim;
  const size_t o_base =
      ((size_t)batch_idx * num_heads + head_idx) * query_len * head_dim;

  float q_val = __half2float(Q[q_base + (size_t)q_pos * head_dim + d]);
  const int causal_limit = causal ? (kv_len - query_len + q_pos + 1) : kv_len;

  extern __shared__ float smem[];

  float row_max = -INFINITY;
  float row_sum = 0.0f;
  float o_acc = 0.0f;

  for (int kv_pos = 0; kv_pos < causal_limit; kv_pos++) {
    float k_val =
        __half2float(K[kv_base + (size_t)kv_pos * head_dim + d]);
    float partial_dot = q_val * k_val;

    smem[d] = partial_dot;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (d < s && d + s < head_dim) {
        smem[d] += smem[d + s];
      }
      __syncthreads();
    }

    float score = smem[0] * scale;

    float prev_max = row_max;
    float new_max = fmaxf(row_max, score);
    float rescale = expf(prev_max - new_max);
    float exp_score = expf(score - new_max);

    o_acc = o_acc * rescale +
            exp_score *
                __half2float(V[kv_base + (size_t)kv_pos * head_dim + d]);

    row_sum = row_sum * rescale + exp_score;
    row_max = new_max;
  }

  if (row_sum > 0.0f) {
    O[o_base + (size_t)q_pos * head_dim + d] =
        __float2half(o_acc / row_sum);
  } else {
    O[o_base + (size_t)q_pos * head_dim + d] = __float2half(0.0f);
  }
}

// ============================================================================
// Templated FlashAttention-2 kernel (BF16 + FP16)
// ============================================================================

template <typename T>
__global__ void FlashAttention2TypedKernel(
    const T* __restrict__ Q, const T* __restrict__ K,
    const T* __restrict__ V, T* __restrict__ O, int query_len,
    int kv_len, int num_heads, int num_kv_heads, int head_dim, float scale,
    bool causal) {
  const int batch_idx = blockIdx.z;
  const int head_idx = blockIdx.y;
  const int q_pos = blockIdx.x;
  const int d = threadIdx.x;

  if (q_pos >= query_len || d >= head_dim) return;

  const int kv_head_ratio =
      (num_kv_heads > 0) ? (num_heads / num_kv_heads) : 1;
  const int kv_head_idx = head_idx / kv_head_ratio;

  // SHD (Sequence, Head, Dim) layout — matches GEMM output and KV cache.
  // Q/O: [batch, query_len, num_heads, head_dim]
  // K/V: [batch, kv_len, num_kv_heads, head_dim]
  const int q_stride = num_heads * head_dim;    // per-position stride for Q/O
  const int kv_stride = num_kv_heads * head_dim; // per-position stride for K/V

  const size_t q_offset =
      (size_t)batch_idx * query_len * q_stride +
      (size_t)q_pos * q_stride + head_idx * head_dim + d;
  const size_t o_offset =
      (size_t)batch_idx * query_len * q_stride +
      (size_t)q_pos * q_stride + head_idx * head_dim + d;

  float q_val = DtypeTraits<T>::to_float(Q[q_offset]);
  const int causal_limit = causal ? (kv_len - query_len + q_pos + 1) : kv_len;

  extern __shared__ float smem[];

  float row_max = -INFINITY;
  float row_sum = 0.0f;
  float o_acc = 0.0f;

  for (int kv_pos = 0; kv_pos < causal_limit; kv_pos++) {
    size_t kv_offset =
        (size_t)batch_idx * kv_len * kv_stride +
        (size_t)kv_pos * kv_stride + kv_head_idx * head_dim + d;
    float k_val = DtypeTraits<T>::to_float(K[kv_offset]);
    float partial_dot = q_val * k_val;

    smem[d] = partial_dot;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (d < s && d + s < head_dim) {
        smem[d] += smem[d + s];
      }
      __syncthreads();
    }

    float score = smem[0] * scale;

    float prev_max = row_max;
    float new_max = fmaxf(row_max, score);
    float rescale = expf(prev_max - new_max);
    float exp_score = expf(score - new_max);

    size_t v_offset =
        (size_t)batch_idx * kv_len * kv_stride +
        (size_t)kv_pos * kv_stride + kv_head_idx * head_dim + d;
    o_acc = o_acc * rescale +
            exp_score * DtypeTraits<T>::to_float(V[v_offset]);

    row_sum = row_sum * rescale + exp_score;
    row_max = new_max;
  }

  if (row_sum > 0.0f) {
    O[o_offset] = DtypeTraits<T>::from_float(o_acc / row_sum);
  } else {
    O[o_offset] = DtypeTraits<T>::from_float(0.0f);
  }
}

// ============================================================================
// Host wrappers
// ============================================================================

cudaError_t ScaledDotProductAttention(const float* d_Q, const float* d_K,
                                      const float* d_V, float* d_O,
                                      int batch_size, int num_heads,
                                      int seq_len, int head_dim,
                                      cudaStream_t stream) {
  const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
  const int threads_per_block = 256;
  const int blocks_x = (seq_len + threads_per_block - 1) / threads_per_block;
  const dim3 grid(blocks_x, num_heads, batch_size);
  const int smem_size = seq_len * sizeof(float);

  ScaledDotProductAttentionKernel<<<grid, threads_per_block, smem_size,
                                    stream>>>(d_Q, d_K, d_V, d_O, batch_size,
                                              num_heads, seq_len, head_dim,
                                              scale);
  return cudaGetLastError();
}

cudaError_t FlashAttention2(const float* d_Q, const float* d_K,
                            const float* d_V, float* d_O, int batch_size,
                            int num_heads, int seq_len, int head_dim,
                            cudaStream_t stream) {
  const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
  const int threads_per_block = 256;
  const int blocks_x = (seq_len + threads_per_block - 1) / threads_per_block;
  const dim3 grid(blocks_x, num_heads, batch_size);

  FlashAttention2Kernel<<<grid, threads_per_block, 0, stream>>>(
      d_Q, d_K, d_V, d_O, batch_size, num_heads, seq_len, head_dim, scale);
  return cudaGetLastError();
}

cudaError_t FlashAttention2FP16(const half* Q, const half* K, const half* V,
                                half* O, int batch_size, int query_len,
                                int kv_len, int num_heads, int num_kv_heads,
                                int head_dim, float scale, bool causal,
                                cudaStream_t stream) {
  int threads = head_dim;
  int t = 1;
  while (t < threads) t <<= 1;
  threads = min(t, 1024);

  dim3 grid(query_len, num_heads, batch_size);
  int smem = threads * sizeof(float);

  FlashAttention2FP16KernelV2<<<grid, threads, smem, stream>>>(
      Q, K, V, O, query_len, kv_len, num_heads, num_kv_heads, head_dim, scale,
      causal);
  return cudaGetLastError();
}

// Templated FlashAttention-2 host wrapper
template <typename T>
cudaError_t FlashAttention2Typed(const T* Q, const T* K, const T* V, T* O,
                                 int batch_size, int query_len, int kv_len,
                                 int num_heads, int num_kv_heads, int head_dim,
                                 float scale, bool causal,
                                 cudaStream_t stream) {
  int threads = head_dim;
  int t = 1;
  while (t < threads) t <<= 1;
  threads = min(t, 1024);

  dim3 grid(query_len, num_heads, batch_size);
  int smem = threads * sizeof(float);

  FlashAttention2TypedKernel<T><<<grid, threads, smem, stream>>>(
      Q, K, V, O, query_len, kv_len, num_heads, num_kv_heads, head_dim, scale,
      causal);
  return cudaGetLastError();
}

// Explicit instantiations
template cudaError_t FlashAttention2Typed<half>(
    const half*, const half*, const half*, half*, int, int, int, int, int, int,
    float, bool, cudaStream_t);
template cudaError_t FlashAttention2Typed<__nv_bfloat16>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    __nv_bfloat16*, int, int, int, int, int, int, float, bool, cudaStream_t);

}  // namespace cuda_kernel
}  // namespace inferflux
