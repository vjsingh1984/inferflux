#pragma once

#include "runtime/backends/cuda/native/cublas_gemm.h"
#include "runtime/backends/cuda/native/kv_cache_gpu.h"
#include "runtime/backends/cuda/native/model_forward.h"
#include "runtime/backends/cuda/native/model_loader.h"
#include "runtime/backends/cuda/native/quantized_weight_map.h"
#include <memory>

namespace inferflux {

/**
 * @brief QuantizedForward - Forward pass for quantized GGUF models
 *
 * Implements the same transformer forward pass as LlamaForward but works
 * with QuantizedWeightMap to support GGUF quantized models.
 *
 * Key features:
 * - Works with both quantized and non-quantized models
 * - Lazy dequantization handled by QuantizedWeightMap
 * - Supports phase overlap with dual-stream execution
 * - Compatible with existing infrastructure
 */
class QuantizedForward : public ModelForward {
public:
  QuantizedForward() = default;
  ~QuantizedForward() override;

  /**
   * @brief Initialize with quantized weight map
   *
   * @param config Model info (from IModelLoader)
   * @param weights Quantized weight map (handles lazy dequantization)
   * @param kv_cache KV cache for attention
   * @param gemm cuBLAS GEMM interface
   * @param stream CUDA stream
   * @return true on success
   */
  bool Initialize(const ModelInfo &config, QuantizedWeightMap *weights,
                  IKvCacheGpu *kv_cache, CublasGemm *gemm, cudaStream_t stream);

  // ModelForward interface (compatibility shim)
  bool Initialize(const SafetensorsLoader::ModelConfig &config,
                  const WeightMap &weights, IKvCacheGpu *kv_cache,
                  CublasGemm *gemm, cudaStream_t stream) override;

  bool Forward(const std::vector<int> &token_ids, int n_past, int sequence_id,
               float *d_logits) override;

  bool BatchForward(
      const std::vector<int> &token_ids, const std::vector<int> &n_past,
      const std::vector<int> &sequence_ids, float *d_logits,
      int batch_size) override; // NOLINT(bugprone-easily-swappable-parameters)

  void SetStream(cudaStream_t stream) override;
  void FreeScratchBuffers() override;

  std::string ModelType() const override { return model_type_; }
  int VocabSize() const override { return vocab_size_; }

  /**
   * @brief Get quantization information
   */
  bool IsQuantized() const { return is_quantized_; }
  std::string GetQuantizationType() const { return quantization_type_; }

private:
  struct AttentionState {
    int n_past{0};
    int sequence_id{-1};
  };

  struct RmsNormParams {
    int size{0};
    float eps{0.0F};
  };

  struct RmsNormBuffers {
    const half *input{nullptr};
    const half *weight{nullptr};
    half *output{nullptr};
  };

  struct RoPEBuffers {
    half *q{nullptr};
    half *k{nullptr};
  };

  struct RoPEParams {
    int n_past{0};
    int layer{0};
  };

  void FreeScratchBuffersImpl();

  // Forward pass implementation
  bool RunForwardPass(const std::vector<int> &token_ids, int n_past,
                      int sequence_id, float *d_logits);

  // Attention computation
  bool ComputeAttention(int layer, const AttentionState &state);

  // Feed-forward network
  bool ComputeFFN(int layer);

  // Layer normalization
  bool ComputeRMSNorm(const RmsNormBuffers &buffers,
                      const RmsNormParams &params);

  // RoPE (Rotary Position Embedding)
  void ApplyRoPE(const RoPEBuffers &buffers, const RoPEParams &params);

  // Model config
  int hidden_size_{0};
  int num_layers_{0};
  int num_heads_{0};
  int num_kv_heads_{0};
  int head_dim_{0};
  int intermediate_size_{0};
  int vocab_size_{0};
  int max_seq_len_{2048};
  float rope_freq_base_{10000.0f};
  float rms_norm_eps_{1e-5f};
  std::string model_type_;

  // Quantization info
  bool is_quantized_{false};
  std::string quantization_type_;

  // External references (not owned)
  QuantizedWeightMap *weights_{nullptr};
  IKvCacheGpu *kv_cache_{nullptr};
  CublasGemm *gemm_{nullptr};
  cudaStream_t stream_{nullptr};

  // Scratch buffers on GPU
  half *d_hidden_{nullptr};
  half *d_residual_{nullptr};
  half *d_norm_out_{nullptr};
  half *d_q_{nullptr};
  half *d_k_new_{nullptr};
  half *d_v_new_{nullptr};
  half *d_attn_out_{nullptr};
  half *d_ffn_gate_{nullptr};
  half *d_ffn_up_{nullptr};
  half *d_ffn_down_{nullptr};
  int *d_token_ids_{nullptr};

  bool AllocateScratch();
};

/**
 * @brief Factory function for creating QuantizedForward
 *
 * @param model_type Model architecture type (llama, qwen2, etc.)
 * @return Unique pointer to QuantizedForward
 */
std::unique_ptr<QuantizedForward>
CreateQuantizedForward(const std::string &model_type);

} // namespace inferflux
