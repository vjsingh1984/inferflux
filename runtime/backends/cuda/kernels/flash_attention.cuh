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

// Batched FlashDecode: one kernel for B sequences with different K/V pointers
// and kv_lens. Decode-only (query_len=1 per sequence).
template <typename T>
cudaError_t FlashDecodeTyped(const T *Q, T *O, const T *const *d_K_ptrs,
                             const T *const *d_V_ptrs, const int *d_kv_lens,
                             int batch_size, int num_heads, int num_kv_heads,
                             int head_dim, float scale, cudaStream_t stream);

} // namespace cuda_kernel
} // namespace inferflux
