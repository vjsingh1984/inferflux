#pragma once

#include "runtime/backends/cuda/native/quantized_weight_map.h"

namespace inferflux {

/// Friend-based accessor for QuantizedWeightMap private members.
class WeightMapTestAccess {
public:
  explicit WeightMapTestAccess(QuantizedWeightMap &w) : w_(w) {}

  auto &scratch_buffer_elements() { return w_.scratch_buffer_elements_; }
  auto &embed_tokens() { return w_.embed_tokens_; }
  auto &final_norm() { return w_.final_norm_; }
  auto &lm_head() { return w_.lm_head_; }
  auto &embed_tokens_accessor() { return w_.embed_tokens_accessor; }
  auto &final_norm_accessor() { return w_.final_norm_accessor; }
  auto &lm_head_accessor() { return w_.lm_head_accessor; }

private:
  QuantizedWeightMap &w_;
};

} // namespace inferflux
