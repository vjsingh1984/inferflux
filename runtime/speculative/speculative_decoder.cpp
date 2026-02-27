#include "runtime/speculative/speculative_decoder.h"

#include <algorithm>
#include <iostream>

namespace inferflux {

SpeculativeDecoder::SpeculativeDecoder(SpeculativeConfig config,
                                       std::shared_ptr<CPUDeviceContext> device,
                                       SimpleTokenizer* tokenizer,
                                       std::shared_ptr<LlamaCPUBackend> draft_backend)
    : config_(std::move(config)),
      device_(std::move(device)),
      tokenizer_(tokenizer),
      draft_backend_(std::move(draft_backend)) {}

std::vector<int> SpeculativeDecoder::Draft(const std::vector<int>& prompt_tokens, int max_new_tokens) {
  std::vector<int> prompt = prompt_tokens;
  if (config_.max_prefill_tokens > 0 &&
      static_cast<int>(prompt.size()) > config_.max_prefill_tokens) {
    prompt.resize(config_.max_prefill_tokens);
  }
  std::vector<int> speculative;
  if (draft_backend_ && draft_backend_->IsReady()) {
    auto draft_text = draft_backend_->Generate(tokenizer_->Decode(prompt_tokens),
                                              max_new_tokens);
    speculative = tokenizer_->Encode(draft_text);
  } else {
    speculative = device_->RunGreedyDecode(prompt);
  }
  if (static_cast<int>(speculative.size()) > max_new_tokens + static_cast<int>(prompt_tokens.size())) {
    speculative.resize(max_new_tokens + prompt_tokens.size());
  }
  std::cout << "[speculative] draft produced " << speculative.size() - prompt_tokens.size()
            << " tokens using config max_prefill=" << config_.max_prefill_tokens << std::endl;
  return speculative;
}

}  // namespace inferflux
