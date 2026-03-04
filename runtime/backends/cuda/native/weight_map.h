#pragma once

#include "runtime/backends/cuda/native_kernel_executor.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace inferflux {

/**
 * WeightMapTyped<T>: typed GPU pointer accessors for safetensors model weights.
 *
 * Wraps SafetensorsLoader to provide const T* accessors organized by
 * model architecture (Llama/Qwen tensor naming conventions).
 */
template <typename T> class WeightMapTyped {
public:
  WeightMapTyped() = default;

  bool Build(const SafetensorsLoader &loader,
             const SafetensorsLoader::ModelConfig &config);

  // --- Per-layer accessors ---
  const T *LayerQProj(int layer) const;
  const T *LayerKProj(int layer) const;
  const T *LayerVProj(int layer) const;
  const T *LayerOProj(int layer) const;
  const T *LayerInputNorm(int layer) const;
  const T *LayerPostAttnNorm(int layer) const;
  const T *LayerGateProj(int layer) const;
  const T *LayerUpProj(int layer) const;
  const T *LayerDownProj(int layer) const;

  // --- Per-layer bias accessors (nullptr if model has no biases) ---
  const T *LayerQProjBias(int layer) const;
  const T *LayerKProjBias(int layer) const;
  const T *LayerVProjBias(int layer) const;

  // --- Global accessors ---
  const T *EmbedTokens() const { return embed_tokens_; }
  const T *FinalNorm() const { return final_norm_; }
  const T *LmHead() const { return lm_head_; }

  int NumLayers() const { return num_layers_; }

private:
  struct LayerWeights {
    const T *q_proj{nullptr};
    const T *k_proj{nullptr};
    const T *v_proj{nullptr};
    const T *o_proj{nullptr};
    const T *input_norm{nullptr};
    const T *post_attn_norm{nullptr};
    const T *gate_proj{nullptr};
    const T *up_proj{nullptr};
    const T *down_proj{nullptr};
    // Optional biases (Qwen2 has q/k/v biases)
    const T *q_proj_bias{nullptr};
    const T *k_proj_bias{nullptr};
    const T *v_proj_bias{nullptr};
  };

  const T *Resolve(const SafetensorsLoader &loader,
                   const std::string &name) const;

  std::vector<LayerWeights> layers_;
  const T *embed_tokens_{nullptr};
  const T *final_norm_{nullptr};
  const T *lm_head_{nullptr};
  int num_layers_{0};
};

// Backward-compatible alias
using WeightMap = WeightMapTyped<half>;

} // namespace inferflux
