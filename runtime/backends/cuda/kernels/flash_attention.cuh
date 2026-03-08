#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace inferflux {
namespace cuda_kernel {

// Tiled FlashAttention-2 with causal mask and GQA support.
template <typename T>
cudaError_t FlashAttention2Typed(const T *Q, const T *K, const T *V, T *O,
                                 int batch_size, int query_len, int kv_len,
                                 int num_heads, int num_kv_heads, int head_dim,
                                 float scale, bool causal,
                                 cudaStream_t stream = 0);

/**
 * FlashDecodeMultiSeq: Batched decode attention for B sequences with
 * different KV cache lengths. Each query has query_len=1.
 *
 * d_k_ptrs/d_v_ptrs: [B] device pointers to each sequence's K/V cache
 *   for the current layer (layout: [kv_len, num_kv_heads * head_dim]).
 * d_kv_lens: [B] per-sequence KV lengths on device.
 * Q layout: [B, num_heads * head_dim]
 * O layout: [B, num_heads * head_dim]
 */
template <typename T>
cudaError_t FlashDecodeMultiSeq(const T *Q, const T *const *d_k_ptrs,
                                const T *const *d_v_ptrs, T *O,
                                const int *d_kv_lens, int batch_size,
                                int num_heads, int num_kv_heads, int head_dim,
                                float scale, cudaStream_t stream = 0);

} // namespace cuda_kernel
} // namespace inferflux
