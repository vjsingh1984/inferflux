#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <curand.h>
#include <vector>

namespace inferflux {

/**
 * GPU-side sampling: temperature, top-k, top-p, multinomial.
 *
 * For greedy (temperature=0): argmax via parallel reduction.
 * For stochastic: softmax -> top-k -> top-p -> multinomial sample.
 */
class GpuSampler {
public:
  GpuSampler() = default;
  ~GpuSampler();

  GpuSampler(const GpuSampler &) = delete;
  GpuSampler &operator=(const GpuSampler &) = delete;

  /**
   * Initialize with vocab size and CUDA stream.
   */
  bool Initialize(int vocab_size, cudaStream_t stream);

  /**
   * Sample a token from logits.
   *
   * @param d_logits    [vocab_size] FP32 logits on device
   * @param temperature Temperature (0 = greedy)
   * @param top_k       Top-k filtering (0 = disabled)
   * @param top_p       Top-p / nucleus filtering (1.0 = disabled)
   * @param seed        Random seed (UINT32_MAX = auto)
   * @return Sampled token ID
   */
  int Sample(const float *d_logits, float temperature, int top_k, float top_p,
             uint32_t seed = UINT32_MAX);

  /**
   * Sample tokens from batched logits.
   *
   * @param d_logits      [batch_size * vocab_size] FP32 logits on device
   * @param batch_size    Number of sequences in batch
   * @param temperatures  Per-sequence temperature
   * @param top_ks        Per-sequence top-k
   * @param top_ps        Per-sequence top-p
   * @param out_tokens    Output: sampled token IDs
   */
  void SampleBatch(const float *d_logits, int batch_size,
                   const std::vector<float> &temperatures,
                   const std::vector<int> &top_ks,
                   const std::vector<float> &top_ps,
                   std::vector<int> *out_tokens);

  /**
   * Apply logit bias to a sequence's logits on GPU.
   *
   * @param d_logits    [vocab_size] FP32 logits on device (modified in-place)
   * @param token_ids   Token IDs to bias
   * @param biases      Bias values corresponding to token_ids
   */
  void ApplyLogitBias(float *d_logits, const std::vector<int> &token_ids,
                      const std::vector<float> &biases);

  /**
   * Apply repetition penalty to a sequence's logits on GPU.
   *
   * @param d_logits    [vocab_size] FP32 logits on device (modified in-place)
   * @param history     Token IDs in the generation history
   * @param penalty     Multiplicative penalty (1.0 = disabled)
   */
  void ApplyRepetitionPenalty(float *d_logits, const std::vector<int> &history,
                              float penalty);

private:
  int GreedyArgmax(const float *d_logits);
  int StochasticSample(const float *d_logits, float temperature, int top_k,
                       float top_p);

  cudaStream_t stream_{nullptr};
  int vocab_size_{0};

  // Scratch buffers
  float *d_probs_{nullptr}; // [vocab_size] probabilities
  int *d_indices_{nullptr}; // [vocab_size] sorted indices
  float *d_temp_{nullptr};  // [vocab_size] temp storage
  int *d_result_{nullptr};  // [1] sampled token ID (device)
  int h_result_{0};         // sampled token ID (host)

  // For argmax
  float *d_max_val_{nullptr}; // [1] max value
  int *d_max_idx_{nullptr};   // [1] max index

  // cuRAND
  curandGenerator_t rng_{nullptr};
  float *d_uniform_{nullptr}; // [1] random uniform
  bool rng_initialized_{false};

  // Batched argmax results (lazily allocated)
  int *d_batch_results_{nullptr};
  std::vector<int> h_batch_results_;

  // Logit bias / repetition penalty scratch (lazily allocated)
  int *d_bias_token_ids_{nullptr};
  float *d_bias_values_{nullptr};
  int *d_penalty_history_{nullptr};
  int bias_scratch_size_{0};
  int penalty_scratch_size_{0};
};

} // namespace inferflux
