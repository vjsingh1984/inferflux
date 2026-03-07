#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace inferflux {
namespace cuda_kernel {

// ============================================================================
// Templated kernel declarations (BF16 + FP16)
// ============================================================================

template <typename T>
cudaError_t RmsNorm(const T *input, const T *weight, T *output, int count,
                    int hidden_size, float eps, cudaStream_t stream);

// rope_type: 0=NORM (consecutive pairs), 2=NEOX (split-half pairs)
// freq_scale: frequency multiplier (default 1.0, <1.0 for extended context)
template <typename T>
cudaError_t RoPE(T *q, T *k, int seq_len, int num_heads, int num_kv_heads,
                 int head_dim, int n_past, float freq_base, cudaStream_t stream,
                 int rope_type = 0, float freq_scale = 1.0f);

// Batched RoPE: applies RoPE to B sequences (seq_len=1 each) with different
// n_past values. Q layout: [B, num_heads * head_dim], K: [B, num_kv_heads *
// head_dim]. d_n_past: device array of B ints.
template <typename T>
cudaError_t BatchRoPE(T *q, T *k, int batch_size, int num_heads,
                      int num_kv_heads, int head_dim, const int *d_n_past,
                      float freq_base, cudaStream_t stream, int rope_type = 0,
                      float freq_scale = 1.0f);

template <typename T>
cudaError_t SiluMul(const T *gate, const T *up, T *output, int count,
                    cudaStream_t stream);

template <typename T>
cudaError_t ResidualAdd(T *residual, const T *input, int count,
                        cudaStream_t stream);

template <typename T>
cudaError_t EmbeddingLookup(const T *table, const int *token_ids, T *output,
                            int seq_len, int hidden_size, cudaStream_t stream);

template <typename T>
cudaError_t HalfToFloat(const T *input, float *output, int count,
                        cudaStream_t stream);

/**
 * BiasAdd: output[i] += bias[i % bias_dim] for rows * bias_dim elements.
 * Used to add projection biases after GEMM (e.g., Q/K/V biases in Qwen2).
 */
template <typename T>
cudaError_t BiasAdd(T *output, const T *bias, int rows, int bias_dim,
                    cudaStream_t stream);

/**
 * BatchKvAppend: scatter K/V vectors from [B, kv_dim] into per-sequence cache
 * slots in a single kernel launch, replacing B x 2 cudaMemcpyAsync D2D calls.
 */
template <typename T>
cudaError_t BatchKvAppend(const T *k_new, const T *v_new, T *kv_buffer,
                          const int *d_seq_ids, const int *d_n_past,
                          int batch_size, int kv_dim, size_t slot_stride,
                          size_t layer_stride, size_t kv_stride, int layer,
                          cudaStream_t stream);

/**
 * LogitBias: logits[token_ids[i]] += biases[i] for num_biases entries.
 */
cudaError_t LogitBias(float *logits, const int *token_ids, const float *biases,
                      int num_biases, cudaStream_t stream);

/**
 * RepetitionPenalty: for each token in history,
 *   logits[tok] > 0 ? logits[tok] /= penalty : logits[tok] *= penalty
 */
cudaError_t RepetitionPenalty(float *logits, const int *history,
                              int history_len, float penalty,
                              cudaStream_t stream);

// ============================================================================
// Non-templated FP16 overloads (backward compatibility)
// ============================================================================

cudaError_t RmsNorm(const half *input, const half *weight, half *output,
                    int count, int hidden_size, float eps, cudaStream_t stream);

cudaError_t RoPE(half *q, half *k, int seq_len, int num_heads, int num_kv_heads,
                 int head_dim, int n_past, float freq_base, cudaStream_t stream,
                 int rope_type = 0, float freq_scale = 1.0f);

cudaError_t BatchRoPE(half *q, half *k, int batch_size, int num_heads,
                      int num_kv_heads, int head_dim, const int *d_n_past,
                      float freq_base, cudaStream_t stream, int rope_type = 0,
                      float freq_scale = 1.0f);

cudaError_t SiluMul(const half *gate, const half *up, half *output, int count,
                    cudaStream_t stream);

cudaError_t ResidualAdd(half *residual, const half *input, int count,
                        cudaStream_t stream);

cudaError_t EmbeddingLookup(const half *table, const int *token_ids,
                            half *output, int seq_len, int hidden_size,
                            cudaStream_t stream);

cudaError_t HalfToFloat(const half *input, float *output, int count,
                        cudaStream_t stream);

} // namespace cuda_kernel
} // namespace inferflux
