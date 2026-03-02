#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "runtime/backends/mlx/mlx_loader.h"

namespace inferflux {

struct MlxSamplerOutput {
  std::string text;
  std::vector<int32_t> ids;
  bool ok{false};
};

// ---------------------------------------------------------------------------
// MlxExecutionEngine
//
// Implements the LLaMA/Mistral transformer forward pass via the mlx-c API.
// Holds a per-layer KV cache that grows with each Step() call.
//
// Lifecycle:
//   Initialize()              — acquire GPU stream
//   LoadWeights(store, cfg)   — wire in weight tensors + model config
//   Reset()                   — clear KV cache (call before each new prompt)
//   Step(token_ids)           — forward pass, returns greedy-sampled token id
//   Shutdown()                — release GPU resources
// ---------------------------------------------------------------------------
class MlxExecutionEngine {
public:
  MlxExecutionEngine() = default;
  ~MlxExecutionEngine();

  bool Initialize();
  void Shutdown();

  // Wire materialised weights and model config into the engine.
  // `store` must outlive the engine.
  bool LoadWeights(const MlxWeightStore &store, const MlxModelConfig &cfg);

  // True after a successful LoadWeights() call.
  bool WeightsLoaded() const { return weights_ != nullptr; }

  // Clear the KV cache and reset sequence position.
  void Reset();

  // Run one forward pass over token_ids starting at position n_past_.
  // Returns the greedy-sampled next token ID, or -1 on error.
  int32_t Step(const std::vector<int32_t> &token_ids);

  // Current KV cache length (tokens processed so far).
  int NPast() const { return n_past_; }

private:
#ifdef INFERFLUX_HAS_MLX
  int32_t Forward(const std::vector<int32_t> &token_ids);

  const MlxWeightStore *weights_{nullptr};
  MlxModelConfig config_;
  mlx_stream stream_{};

  // Per-layer KV cache: [1, n_kv_heads, n_past, head_dim].
  // {} (empty ctx) when the slot has not been populated yet.
  std::vector<mlx_array> key_cache_;
  std::vector<mlx_array> val_cache_;

  int n_past_{0};
  bool initialized_{false};
#endif
};

} // namespace inferflux
