#pragma once

#include "runtime/backends/cuda/common/dtype_traits.cuh"

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace inferflux {
namespace cuda_kernel {

// ============================================================================
// FusedRoPEKvAppendStrided: Fuse BatchedRoPE + BatchedKvAppendStrided
//
// Applies RoPE rotation to Q and K, then scatter-writes K and V into the
// strided KV cache — all in a single kernel launch.
//
// Thread layout: 256 threads/block, grid covers Q pairs + K pairs + V copies.
// For K threads: RoPE rotation → scatter-write to kv_buffer
// For V threads: direct copy to KV cache (no rotation)
// For Q threads: RoPE rotation in-place (no cache write)
//
// Saves 2 kernel launches per layer (BatchedRoPE + BatchedKvAppendStrided → 1).
// ============================================================================

template <typename T>
__global__ void FusedRoPEKvAppendStridedKernel(
    T *__restrict__ q,                    // [B, num_heads * head_dim]
    T *__restrict__ k_new,                // [B, num_kv_heads * head_dim]
    const T *__restrict__ v_new,          // [B, kv_dim]
    T *__restrict__ kv_buffer,            // KV cache
    const int *__restrict__ d_seq_ids,    // [B] sequence IDs
    const int *__restrict__ d_n_past,     // [B] positions
    int layer, int batch_size, int num_heads, int num_kv_heads, int head_dim,
    int kv_dim, size_t slot_stride, size_t layer_stride, size_t kv_stride,
    float freq_base, int rope_type) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int half_dim = head_dim / 2;
  const int q_pairs_per_seq = num_heads * half_dim;
  const int k_pairs_per_seq = num_kv_heads * half_dim;
  const int v_elems_per_seq = kv_dim;
  const int work_per_seq = q_pairs_per_seq + k_pairs_per_seq + v_elems_per_seq;
  const int total_work = batch_size * work_per_seq;

  if (idx >= total_work)
    return;

  const int b = idx / work_per_seq;
  const int local = idx % work_per_seq;
  const int position = d_n_past[b];

  if (local < q_pairs_per_seq) {
    // --- Q RoPE (in-place, no cache write) ---
    const int pair_in_q = local;
    const int pair_idx = pair_in_q % half_dim;
    const int head_idx = pair_in_q / half_dim;

    float freq = 1.0f / powf(freq_base,
                              2.0f * pair_idx / static_cast<float>(head_dim));
    float angle = position * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    int base = b * num_heads * head_dim + head_idx * head_dim;
    int i0, i1;
    if (rope_type == 2) {
      i0 = base + pair_idx;
      i1 = base + pair_idx + half_dim;
    } else {
      i0 = base + 2 * pair_idx;
      i1 = base + 2 * pair_idx + 1;
    }

    float v0 = DtypeTraits<T>::to_float(q[i0]);
    float v1 = DtypeTraits<T>::to_float(q[i1]);
    q[i0] = DtypeTraits<T>::from_float(v0 * cos_val - v1 * sin_val);
    q[i1] = DtypeTraits<T>::from_float(v0 * sin_val + v1 * cos_val);

  } else if (local < q_pairs_per_seq + k_pairs_per_seq) {
    // --- K RoPE + KV cache scatter-write ---
    const int pair_in_k = local - q_pairs_per_seq;
    const int pair_idx = pair_in_k % half_dim;
    const int head_idx = pair_in_k / half_dim;

    float freq = 1.0f / powf(freq_base,
                              2.0f * pair_idx / static_cast<float>(head_dim));
    float angle = position * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    int base = b * num_kv_heads * head_dim + head_idx * head_dim;
    int i0, i1;
    if (rope_type == 2) {
      i0 = base + pair_idx;
      i1 = base + pair_idx + half_dim;
    } else {
      i0 = base + 2 * pair_idx;
      i1 = base + 2 * pair_idx + 1;
    }

    float v0 = DtypeTraits<T>::to_float(k_new[i0]);
    float v1 = DtypeTraits<T>::to_float(k_new[i1]);
    T rotated0 = DtypeTraits<T>::from_float(v0 * cos_val - v1 * sin_val);
    T rotated1 = DtypeTraits<T>::from_float(v0 * sin_val + v1 * cos_val);

    // Write rotated K to k_new (for FlashAttention to read as q-local K)
    k_new[i0] = rotated0;
    k_new[i1] = rotated1;

    // Scatter-write rotated K to KV cache
    const size_t layer_offset = static_cast<size_t>(layer) * layer_stride;
    const size_t seq_offset =
        static_cast<size_t>(d_seq_ids[b]) * slot_stride;
    const size_t base_cache =
        seq_offset + layer_offset +
        static_cast<size_t>(d_n_past[b]) * kv_dim;

    // Map pair indices back to element indices for cache write
    int elem0, elem1;
    if (rope_type == 2) {
      elem0 = head_idx * head_dim + pair_idx;
      elem1 = head_idx * head_dim + pair_idx + half_dim;
    } else {
      elem0 = head_idx * head_dim + 2 * pair_idx;
      elem1 = head_idx * head_dim + 2 * pair_idx + 1;
    }
    kv_buffer[base_cache + static_cast<size_t>(elem0)] = rotated0;
    kv_buffer[base_cache + static_cast<size_t>(elem1)] = rotated1;

  } else {
    // --- V direct copy to KV cache ---
    const int v_local = local - q_pairs_per_seq - k_pairs_per_seq;
    const int d = v_local; // element index within kv_dim

    const size_t layer_offset = static_cast<size_t>(layer) * layer_stride;
    const size_t seq_offset =
        static_cast<size_t>(d_seq_ids[b]) * slot_stride;
    const size_t token_offset =
        static_cast<size_t>(d_n_past[b]) * kv_dim + static_cast<size_t>(d);
    const size_t v_offset =
        seq_offset + layer_offset + kv_stride + token_offset;

    kv_buffer[v_offset] = v_new[b * kv_dim + d];
  }
}

template <typename T>
cudaError_t FusedRoPEKvAppendStrided(
    T *q, T *k_new, const T *v_new, T *kv_buffer, const int *d_seq_ids,
    const int *d_n_past, int layer, int batch_size, int num_heads,
    int num_kv_heads, int head_dim, int kv_dim, size_t slot_stride,
    size_t layer_stride, size_t kv_stride, float freq_base, cudaStream_t stream,
    int rope_type = 0) {
  const int half_dim = head_dim / 2;
  const int work_per_seq =
      num_heads * half_dim + num_kv_heads * half_dim + kv_dim;
  const int total_work = batch_size * work_per_seq;
  const int threads = 256;
  const int blocks = (total_work + threads - 1) / threads;

  FusedRoPEKvAppendStridedKernel<T><<<blocks, threads, 0, stream>>>(
      q, k_new, v_new, kv_buffer, d_seq_ids, d_n_past, layer, batch_size,
      num_heads, num_kv_heads, head_dim, kv_dim, slot_stride, layer_stride,
      kv_stride, freq_base, rope_type);
  return cudaGetLastError();
}

} // namespace cuda_kernel
} // namespace inferflux
