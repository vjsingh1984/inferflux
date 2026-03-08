#include "flash_attention.cuh"
#include "runtime/backends/cuda/common/dtype_traits.cuh"
#include <cmath>
#include <cuda_runtime.h>

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
// Tiled FlashAttention-2 kernel (BF16 + FP16)
//
// Properly tiled implementation with:
// - K/V tiles loaded into shared memory (amortized global reads)
// - Warp-shuffle dot product reduction (no per-position barriers)
// - Batched cross-warp reduction (1 barrier per tile, not per KV position)
// - Online softmax with rescaling
//
// Grid: (query_len, num_heads, batch_size)
// Block: next_pow2(head_dim) threads
// Shared memory: (2 * Bc * d + num_warps * Bc + Bc) * sizeof(float)
//
// For kv_len=1024, d=128: 128 barriers vs 8192 in the scalar version.
// ============================================================================

constexpr int FA2_TILE_KV = 32;

template <typename T>
__global__ void FlashAttention2TypedKernel(
    const T *__restrict__ Q, const T *__restrict__ K, const T *__restrict__ V,
    T *__restrict__ O, int query_len, int kv_len, int num_heads,
    int num_kv_heads, int head_dim, float scale, bool causal) {
  const int batch_idx = blockIdx.z;
  const int head_idx = blockIdx.y;
  const int q_pos = blockIdx.x;

  if (q_pos >= query_len)
    return;

  const int d = threadIdx.x;
  const int num_threads = blockDim.x;
  const int warp_id = d / 32;
  const int lane = d & 31;
  const int num_warps = num_threads / 32;

  const int kv_head_ratio = (num_kv_heads > 0) ? (num_heads / num_kv_heads) : 1;
  const int kv_head_idx = head_idx / kv_head_ratio;

  // SHD layout strides
  const int q_stride = num_heads * head_dim;
  const int kv_stride = num_kv_heads * head_dim;

  // Load this thread's Q dimension into register
  float q_reg = 0.0f;
  if (d < head_dim) {
    size_t q_offset = (size_t)batch_idx * query_len * q_stride +
                      (size_t)q_pos * q_stride + head_idx * head_dim + d;
    q_reg = DtypeTraits<T>::to_float(Q[q_offset]);
  }

  const int causal_limit = causal ? (kv_len - query_len + q_pos + 1) : kv_len;

  // Dynamic shared memory layout:
  //   s_k:         [FA2_TILE_KV * head_dim] floats — K tile
  //   s_v:         [FA2_TILE_KV * head_dim] floats — V tile
  //   s_warp_dots: [num_warps * FA2_TILE_KV] floats — per-warp partial sums
  //   s_scores:    [FA2_TILE_KV] floats — final attention scores
  extern __shared__ float smem[];
  float *s_k = smem;
  float *s_v = smem + FA2_TILE_KV * head_dim;
  float *s_warp_dots = smem + 2 * FA2_TILE_KV * head_dim;
  float *s_scores = s_warp_dots + num_warps * FA2_TILE_KV;

  // KV base pointer (this batch, this kv_head)
  const size_t kv_base =
      (size_t)batch_idx * kv_len * kv_stride + kv_head_idx * head_dim;

  // Online softmax state
  float row_max = -INFINITY;
  float row_sum = 0.0f;
  float o_acc = 0.0f;

  // Tile loop over KV positions
  for (int kv_start = 0; kv_start < causal_limit; kv_start += FA2_TILE_KV) {
    int tile_len = min(FA2_TILE_KV, causal_limit - kv_start);
    int total_elements = tile_len * head_dim;

    // Phase 1: Cooperatively load K and V tiles into shared memory.
    // Access is coalesced: consecutive threads load consecutive dimensions.
    for (int i = d; i < total_elements; i += num_threads) {
      int t = i / head_dim;
      int dim = i % head_dim;
      size_t kv_offset = kv_base + (size_t)(kv_start + t) * kv_stride + dim;
      s_k[i] = DtypeTraits<T>::to_float(K[kv_offset]);
      s_v[i] = DtypeTraits<T>::to_float(V[kv_offset]);
    }
    __syncthreads(); // BARRIER 1: tiles loaded

    // Phase 2: Compute tile_len dot products using warp-level reduction.
    // Each thread multiplies its Q dimension with the corresponding K
    // dimension, then warp shuffle reduces across 32 lanes. Lane 0 writes to
    // shared memory.
    for (int t = 0; t < tile_len; t++) {
      float partial = (d < head_dim) ? q_reg * s_k[t * head_dim + d] : 0.0f;
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1)
        partial += __shfl_down_sync(0xFFFFFFFF, partial, offset);
      if (lane == 0)
        s_warp_dots[warp_id * FA2_TILE_KV + t] = partial;
    }
    __syncthreads(); // BARRIER 2: warp partials ready

    // Phase 3: Cross-warp reduction — sum partial dot products across warps.
    // First tile_len threads each reduce one score.
    if (d < tile_len) {
      float dot_sum = 0.0f;
      for (int w = 0; w < num_warps; w++)
        dot_sum += s_warp_dots[w * FA2_TILE_KV + d];
      s_scores[d] = dot_sum * scale;
    }
    __syncthreads(); // BARRIER 3: scores ready

    // Phase 4: Online softmax + V accumulation.
    // All threads read the same scores and independently update their
    // o_acc dimension. Since all threads see identical scores, they compute
    // identical row_max/row_sum — no broadcast needed.
    for (int t = 0; t < tile_len; t++) {
      float score = s_scores[t];
      float new_max = fmaxf(row_max, score);
      float rescale = expf(row_max - new_max);
      float exp_w = expf(score - new_max);

      if (d < head_dim)
        o_acc = o_acc * rescale + exp_w * s_v[t * head_dim + d];

      row_sum = row_sum * rescale + exp_w;
      row_max = new_max;
    }
    __syncthreads(); // BARRIER 4: safe to overwrite smem in next iteration
  }

  // Write output
  if (d < head_dim) {
    size_t o_offset = (size_t)batch_idx * query_len * q_stride +
                      (size_t)q_pos * q_stride + head_idx * head_dim + d;
    O[o_offset] =
        DtypeTraits<T>::from_float((row_sum > 0.0f) ? (o_acc / row_sum) : 0.0f);
  }
}

// ============================================================================
// Host wrappers
// ============================================================================

// Templated FlashAttention-2 host wrapper (tiled)
template <typename T>
cudaError_t FlashAttention2Typed(const T *Q, const T *K, const T *V, T *O,
                                 int batch_size, int query_len, int kv_len,
                                 int num_heads, int num_kv_heads, int head_dim,
                                 float scale, bool causal,
                                 cudaStream_t stream) {
  // Thread count = next power of 2 >= head_dim
  int threads = 1;
  while (threads < head_dim)
    threads <<= 1;
  threads = min(threads, 1024);
  int num_warps = threads / 32;

  // Shared memory: K tile + V tile + warp partials + scores
  int smem =
      (2 * FA2_TILE_KV * head_dim + num_warps * FA2_TILE_KV + FA2_TILE_KV) *
      sizeof(float);

  dim3 grid(query_len, num_heads, batch_size);

  FlashAttention2TypedKernel<T><<<grid, threads, smem, stream>>>(
      Q, K, V, O, query_len, kv_len, num_heads, num_kv_heads, head_dim, scale,
      causal);
  return cudaGetLastError();
}

// Explicit instantiations
template cudaError_t FlashAttention2Typed<half>(const half *, const half *,
                                                const half *, half *, int, int,
                                                int, int, int, int, float, bool,
                                                cudaStream_t);
template cudaError_t FlashAttention2Typed<__nv_bfloat16>(
    const __nv_bfloat16 *, const __nv_bfloat16 *, const __nv_bfloat16 *,
    __nv_bfloat16 *, int, int, int, int, int, int, float, bool, cudaStream_t);

// ============================================================================
// FlashDecodeMultiSeq: Batched decode attention for B sequences with
// variable KV cache lengths. Each query has query_len=1.
//
// Grid: (batch_size, num_heads)
// Block: next_pow2(head_dim) threads
// Each block handles one (batch, head) pair.
// ============================================================================

template <typename T>
__global__ void FlashDecodeMultiSeqKernel(
    const T *__restrict__ Q, const T *const *__restrict__ K_ptrs,
    const T *const *__restrict__ V_ptrs, T *__restrict__ O,
    const int *__restrict__ kv_lens, int num_heads, int num_kv_heads,
    int head_dim, float scale) {
  const int b = blockIdx.x;
  const int head_idx = blockIdx.y;

  const int d = threadIdx.x;
  const int num_threads = blockDim.x;
  const int warp_id = d / 32;
  const int lane = d & 31;
  const int num_warps = num_threads / 32;

  const int kv_len = kv_lens[b];
  if (kv_len <= 0)
    return;

  const int kv_head_ratio =
      (num_kv_heads > 0) ? (num_heads / num_kv_heads) : 1;
  const int kv_head_idx = head_idx / kv_head_ratio;

  // Q is [B, num_heads * head_dim]
  float q_reg = 0.0f;
  if (d < head_dim) {
    q_reg = DtypeTraits<T>::to_float(
        Q[b * num_heads * head_dim + head_idx * head_dim + d]);
  }

  // K/V pointers for this batch element: [kv_len, num_kv_heads * head_dim]
  const T *K = K_ptrs[b];
  const T *V = V_ptrs[b];
  const int kv_stride = num_kv_heads * head_dim;

  // Shared memory layout: same as FA2
  extern __shared__ float smem[];
  float *s_k = smem;
  float *s_v = smem + FA2_TILE_KV * head_dim;
  float *s_warp_dots = smem + 2 * FA2_TILE_KV * head_dim;
  float *s_scores = s_warp_dots + num_warps * FA2_TILE_KV;

  float row_max = -INFINITY;
  float row_sum = 0.0f;
  float o_acc = 0.0f;

  for (int kv_start = 0; kv_start < kv_len; kv_start += FA2_TILE_KV) {
    int tile_len = min(FA2_TILE_KV, kv_len - kv_start);
    int total_elements = tile_len * head_dim;

    // Load K and V tiles
    for (int i = d; i < total_elements; i += num_threads) {
      int t = i / head_dim;
      int dim = i % head_dim;
      size_t kv_offset =
          (size_t)(kv_start + t) * kv_stride + kv_head_idx * head_dim + dim;
      s_k[i] = DtypeTraits<T>::to_float(K[kv_offset]);
      s_v[i] = DtypeTraits<T>::to_float(V[kv_offset]);
    }
    __syncthreads();

    // Dot products with warp reduction
    for (int t = 0; t < tile_len; t++) {
      float partial = (d < head_dim) ? q_reg * s_k[t * head_dim + d] : 0.0f;
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1)
        partial += __shfl_down_sync(0xFFFFFFFF, partial, offset);
      if (lane == 0)
        s_warp_dots[warp_id * FA2_TILE_KV + t] = partial;
    }
    __syncthreads();

    // Cross-warp reduction
    if (d < tile_len) {
      float dot_sum = 0.0f;
      for (int w = 0; w < num_warps; w++)
        dot_sum += s_warp_dots[w * FA2_TILE_KV + d];
      s_scores[d] = dot_sum * scale;
    }
    __syncthreads();

    // Online softmax + V accumulation
    for (int t = 0; t < tile_len; t++) {
      float score = s_scores[t];
      float new_max = fmaxf(row_max, score);
      float rescale = expf(row_max - new_max);
      float exp_w = expf(score - new_max);

      if (d < head_dim)
        o_acc = o_acc * rescale + exp_w * s_v[t * head_dim + d];

      row_sum = row_sum * rescale + exp_w;
      row_max = new_max;
    }
    __syncthreads();
  }

  // Write output: O is [B, num_heads * head_dim]
  if (d < head_dim) {
    O[b * num_heads * head_dim + head_idx * head_dim + d] =
        DtypeTraits<T>::from_float((row_sum > 0.0f) ? (o_acc / row_sum) : 0.0f);
  }
}

template <typename T>
cudaError_t FlashDecodeMultiSeq(const T *Q, const T *const *d_k_ptrs,
                                const T *const *d_v_ptrs, T *O,
                                const int *d_kv_lens, int batch_size,
                                int num_heads, int num_kv_heads, int head_dim,
                                float scale, cudaStream_t stream) {
  int threads = 1;
  while (threads < head_dim)
    threads <<= 1;
  threads = min(threads, 1024);
  int num_warps = threads / 32;

  int smem =
      (2 * FA2_TILE_KV * head_dim + num_warps * FA2_TILE_KV + FA2_TILE_KV) *
      sizeof(float);

  dim3 grid(batch_size, num_heads);

  FlashDecodeMultiSeqKernel<T><<<grid, threads, smem, stream>>>(
      Q, d_k_ptrs, d_v_ptrs, O, d_kv_lens, num_heads, num_kv_heads, head_dim,
      scale);
  return cudaGetLastError();
}

template cudaError_t FlashDecodeMultiSeq<half>(const half *, const half *const *,
                                               const half *const *, half *,
                                               const int *, int, int, int, int,
                                               float, cudaStream_t);
template cudaError_t FlashDecodeMultiSeq<__nv_bfloat16>(
    const __nv_bfloat16 *, const __nv_bfloat16 *const *,
    const __nv_bfloat16 *const *, __nv_bfloat16 *, const int *, int, int, int,
    int, float, cudaStream_t);

} // namespace cuda_kernel
} // namespace inferflux
