#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace inferflux {
namespace cuda_kernel {

// Standard Scaled Dot-Product Attention (FP32)
cudaError_t ScaledDotProductAttention(const float *d_Q, const float *d_K,
                                      const float *d_V, float *d_O,
                                      int batch_size, int num_heads,
                                      int seq_len, int head_dim,
                                      cudaStream_t stream = 0);

// FlashAttention-2 (FP32, simplified)
cudaError_t FlashAttention2(const float *d_Q, const float *d_K,
                            const float *d_V, float *d_O, int batch_size,
                            int num_heads, int seq_len, int head_dim,
                            cudaStream_t stream = 0);

/**
 * FlashAttention-2 FP16 with causal mask and GQA support.
 *
 * @param Q         [batch, num_heads, query_len, head_dim] FP16
 * @param K         [batch, num_kv_heads, kv_len, head_dim] FP16
 * @param V         [batch, num_kv_heads, kv_len, head_dim] FP16
 * @param O         [batch, num_heads, query_len, head_dim] FP16
 * @param query_len Length of Q sequence
 * @param kv_len    Length of K/V sequence (>= query_len with cache)
 * @param num_heads Number of Q attention heads
 * @param num_kv_heads Number of K/V heads (GQA when < num_heads)
 * @param head_dim  Dimension per head
 * @param scale     Softmax scale factor (typically 1/sqrt(head_dim))
 * @param causal    Apply causal mask
 */
cudaError_t FlashAttention2FP16(const half *Q, const half *K, const half *V,
                                half *O, int batch_size, int query_len,
                                int kv_len, int num_heads, int num_kv_heads,
                                int head_dim, float scale, bool causal,
                                cudaStream_t stream = 0);

// Templated FlashAttention-2 with causal mask and GQA support.
template <typename T>
cudaError_t FlashAttention2Typed(const T *Q, const T *K, const T *V, T *O,
                                 int batch_size, int query_len, int kv_len,
                                 int num_heads, int num_kv_heads, int head_dim,
                                 float scale, bool causal,
                                 cudaStream_t stream = 0);

} // namespace cuda_kernel
} // namespace inferflux
