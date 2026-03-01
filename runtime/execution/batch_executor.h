#pragma once

#include "model/tokenizer/simple_tokenizer.h"
#include "runtime/backends/cpu/llama_backend.h"
#include "runtime/device_context.h"
#include "runtime/kv_cache/paged_kv_cache.h"
#include "runtime/prefix_cache/prefix_cache.h"
#include "runtime/speculative/speculative_decoder.h"
#include "scheduler/model_router.h"
#include "scheduler/request_batch.h"
#include "scheduler/scheduler.h"

#include <memory>
#include <string>

namespace inferflux {

class BatchExecutor {
 public:
  BatchExecutor(SimpleTokenizer* tokenizer,
                std::shared_ptr<CPUDeviceContext> device,
                std::shared_ptr<PagedKVCache> cache,
                std::shared_ptr<ModelRouter> router,
                std::shared_ptr<SpeculativeDecoder> speculative_decoder,
                std::shared_ptr<PrefixCache> prefix_cache);

  std::vector<InferenceResult> ExecuteBatch(const RequestBatch& batch,
                                            const std::vector<std::shared_ptr<LlamaCPUBackend>>& backend_overrides);

 private:
  struct ExecutionOutcome {
    InferenceResult result;
    double prefill_ms{0};
    double decode_ms{0};
  };

  ExecutionOutcome ExecuteRequest(InferenceRequest& inference,
                                  std::shared_ptr<LlamaCPUBackend> backend_override);
  std::shared_ptr<LlamaCPUBackend> ResolveBackend(const std::string& requested_model,
                                                  std::string* resolved_id);

  SimpleTokenizer* tokenizer_;
  std::shared_ptr<CPUDeviceContext> device_;
  std::shared_ptr<PagedKVCache> cache_;
  std::shared_ptr<ModelRouter> router_;
  std::shared_ptr<SpeculativeDecoder> speculative_decoder_;
  std::shared_ptr<PrefixCache> prefix_cache_;
};

}  // namespace inferflux
