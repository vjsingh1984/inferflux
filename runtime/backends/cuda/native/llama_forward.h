#pragma once

#include "runtime/backends/cuda/native/model_forward.h"

namespace inferflux {

/**
 * LlamaForwardTyped<T>: Llama/Qwen decoder-only transformer forward pass.
 *
 * Implements the standard pipeline:
 *   embed -> (rmsnorm -> attn -> residual -> rmsnorm -> ffn -> residual) x L
 *   -> rmsnorm -> lm_head -> logits
 */
template <typename T> class LlamaForwardTyped : public ModelForward {
public:
  LlamaForwardTyped() = default;
  ~LlamaForwardTyped() override;

  bool Initialize(const SafetensorsLoader::ModelConfig &config,
                  const WeightMap &weights, IKvCacheGpu *kv_cache,
                  CublasGemm *gemm, cudaStream_t stream) override;

  bool Forward(const std::vector<int> &token_ids, int n_past, int sequence_id,
               float *d_logits) override;

  bool BatchForward(const std::vector<int> &token_ids,
                    const std::vector<int> &n_past,
                    const std::vector<int> &sequence_ids, float *d_logits,
                    int batch_size) override;

  void SetStream(cudaStream_t stream) override;

  void FreeScratchBuffers() override;

  std::string ModelType() const override { return "llama"; }

  int VocabSize() const override { return vocab_size_; }

private:
  // Model config
  int hidden_size_{0};
  int num_layers_{0};
  int num_heads_{0};
  int num_kv_heads_{0};
  int head_dim_{0};
  int intermediate_size_{0};
  int vocab_size_{0};
  int max_seq_len_{0};
  int max_batch_size_{32};
  float rope_freq_base_{10000.0f};
  float rms_norm_eps_{1e-5f};
  int rope_type_{0}; // 0 = kNorm (consecutive), 2 = kNeox (split-half)

  // External references (not owned)
  const WeightMap *weights_{nullptr};
  IKvCacheGpu *kv_cache_{nullptr};
  CublasGemm *gemm_{nullptr};
  cudaStream_t stream_{nullptr};

  // Scratch buffers on GPU (pre-allocated for max_seq_len)
  T *d_hidden_{nullptr};
  T *d_residual_{nullptr};
  T *d_norm_out_{nullptr};
  T *d_q_{nullptr};
  T *d_k_new_{nullptr};
  T *d_v_new_{nullptr};
  T *d_attn_out_{nullptr};
  T *d_ffn_gate_{nullptr};
  T *d_ffn_up_{nullptr};
  T *d_ffn_down_{nullptr};
  int *d_token_ids_{nullptr};
  T *d_logits_typed_{nullptr};

  // Batch metadata buffers (pre-allocated for max_batch_size)
  int *d_batch_n_past_{nullptr};
  int *d_batch_seq_ids_{nullptr};
  int *d_batch_kv_lens_{nullptr};

  // Device pointer arrays for batched KV/attention (max_batch_size each)
  void *d_k_ptrs_{nullptr};        // T** on device
  void *d_v_ptrs_{nullptr};        // T** on device
  void *d_k_append_ptrs_{nullptr}; // T** on device
  void *d_v_append_ptrs_{nullptr}; // T** on device

  bool AllocateScratch();
};

// Backward-compatible alias
using LlamaForward = LlamaForwardTyped<half>;

} // namespace inferflux
