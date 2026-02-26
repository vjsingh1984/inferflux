#pragma once

#include "model/tokenizer/simple_tokenizer.h"
#include "runtime/backends/cpu/cpu_backend.h"
#include "runtime/backends/cpu/llama_backend.h"
#include "runtime/kv_cache/paged_kv_cache.h"

#include <memory>
#include <mutex>
#include <queue>
#include <string>

namespace inferflux {

struct GenerateRequest {
  std::string prompt;
  int max_tokens{64};
};

struct GenerateResponse {
  std::string completion;
  int prompt_tokens{0};
  int completion_tokens{0};
};

class Scheduler {
 public:
  Scheduler(SimpleTokenizer tokenizer,
            std::shared_ptr<CPUDeviceContext> device,
            std::shared_ptr<PagedKVCache> cache,
            std::shared_ptr<LlamaCPUBackend> llama_backend = nullptr);

  GenerateResponse Generate(const GenerateRequest& request);

 private:
  SimpleTokenizer tokenizer_;
  std::shared_ptr<CPUDeviceContext> device_;
  std::shared_ptr<PagedKVCache> cache_;
  std::shared_ptr<LlamaCPUBackend> llama_backend_;
  std::mutex mutex_;
};

}  // namespace inferflux
