#pragma once

#include "model/tokenizer/simple_tokenizer.h"
#include "runtime/backends/cpu/cpu_backend.h"

#include <memory>
#include <vector>

namespace inferflux {

struct SpeculativeConfig {
  bool enabled{false};
  int max_prefill_tokens{256};
  std::string draft_model;
};

class SpeculativeDecoder {
 public:
  SpeculativeDecoder(SpeculativeConfig config,
                     std::shared_ptr<CPUDeviceContext> device,
                     SimpleTokenizer* tokenizer);

  bool Enabled() const { return config_.enabled; }
  SpeculativeConfig Config() const { return config_; }

  std::vector<int> Draft(const std::vector<int>& prompt_tokens, int max_new_tokens);

 private:
  SpeculativeConfig config_;
  std::shared_ptr<CPUDeviceContext> device_;
  SimpleTokenizer* tokenizer_;
};

}  // namespace inferflux
