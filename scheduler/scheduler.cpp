#include "scheduler/scheduler.h"

#include <algorithm>
#include <iostream>

namespace inferflux {

Scheduler::Scheduler(SimpleTokenizer tokenizer,
                     std::shared_ptr<CPUDeviceContext> device,
                     std::shared_ptr<PagedKVCache> cache,
                     std::shared_ptr<LlamaCPUBackend> llama_backend,
                     std::shared_ptr<SpeculativeDecoder> speculative_decoder)
    : tokenizer_(std::move(tokenizer)),
      device_(std::move(device)),
      cache_(std::move(cache)),
      llama_backend_(std::move(llama_backend)),
      speculative_decoder_(std::move(speculative_decoder)) {}

GenerateResponse Scheduler::Generate(const GenerateRequest& request) {
  std::lock_guard<std::mutex> lock(mutex_);
  GenerateResponse response;
  auto prompt_tokens = tokenizer_.Encode(request.prompt);
  response.prompt_tokens = static_cast<int>(prompt_tokens.size());

  std::vector<int> speculative_tokens;
  if (speculative_decoder_ && speculative_decoder_->Enabled()) {
    speculative_tokens = speculative_decoder_->Draft(prompt_tokens, request.max_tokens);
  }

  if (llama_backend_ && llama_backend_->IsReady()) {
    if (!speculative_tokens.empty()) {
      speculative_tokens = speculative_decoder_->Validate(prompt_tokens, speculative_tokens,
                                                         request.max_tokens, llama_backend_);
      response.completion = tokenizer_.Decode(speculative_tokens);
    } else {
      auto text = llama_backend_->Generate(request.prompt, request.max_tokens);
      response.completion = text.empty() ? "[llama.cpp backend returned empty response]" : text;
    }
  } else {
    // Fallback stub that produces a human-readable reply.
    response.completion = "InferFlux (stub): I received your prompt \"" + request.prompt +
                          "\" and this is a placeholder response while the full backend initializes.";
  }

  auto completion_tokens = tokenizer_.Encode(response.completion);
  response.completion_tokens = static_cast<int>(completion_tokens.size());
  (void)cache_;      // reserved for future GPU implementations
  (void)device_;     // placeholder; device scheduling handled in future revisions
  return response;
}

}  // namespace inferflux
