#pragma once

#include "model/tokenizer/simple_tokenizer.h"
#include "runtime/backends/cpu/cpu_backend.h"
#include "runtime/speculative/speculative_decoder.h"
#include "runtime/kv_cache/paged_kv_cache.h"
#include "scheduler/model_router.h"
#include "scheduler/fairness_controller.h"
#include "scheduler/request_batch.h"
#include "runtime/prefix_cache/prefix_cache.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace inferflux {

class BatchExecutor;
struct SpeculativeStats {
  int total_chunks{0};
  int accepted_chunks{0};
  int reused_tokens{0};
};

struct InferenceResult {
  std::string completion;
  int prompt_tokens{0};
  int completion_tokens{0};
  bool no_backend{false};
  std::string model_id;
  SpeculativeStats speculative;
};

struct PendingRequest {
  InferenceRequest inference;
  std::promise<InferenceResult> promise;
  std::chrono::steady_clock::time_point enqueue_time;
  uint64_t sequence{0};
  int priority{0};
  std::shared_ptr<LlamaCPUBackend> resolved_backend;
};

class Scheduler {
 public:
  Scheduler(SimpleTokenizer tokenizer,
            std::shared_ptr<CPUDeviceContext> device,
            std::shared_ptr<PagedKVCache> cache,
            std::shared_ptr<ModelRouter> router = nullptr,
            std::shared_ptr<SpeculativeDecoder> speculative_decoder = nullptr,
            std::shared_ptr<PrefixCache> prefix_cache = nullptr,
            FairnessConfig fairness_config = {});
  ~Scheduler();

  InferenceResult Generate(InferenceRequest request);
  void UpdateFairnessConfig(const FairnessConfig& config);

  // Expose the router for use by the HTTP admin endpoint (/v1/models).
  ModelRouter* Router() const { return router_.get(); }

 private:
  struct BatchSelection {
    RequestBatch batch;
    std::vector<std::shared_ptr<PendingRequest>> pending;
    std::size_t total_tokens{0};
  };

  BatchSelection BuildBatchLocked();
  void WorkerLoop();
  void ProcessBatch(BatchSelection selection);
  void ApplyFairness(BatchSelection* selection);
  void UpdateQueueDepthLocked() const;
  void ResolveBackends(const std::vector<std::shared_ptr<PendingRequest>>& batch);

  SimpleTokenizer tokenizer_;
  std::shared_ptr<CPUDeviceContext> device_;
  std::shared_ptr<PagedKVCache> cache_;
  std::shared_ptr<ModelRouter> router_;
  std::shared_ptr<SpeculativeDecoder> speculative_decoder_;
  std::shared_ptr<PrefixCache> prefix_cache_;
  std::vector<std::shared_ptr<PendingRequest>> pending_;
  mutable std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  std::thread worker_;
  std::atomic<bool> stop_{false};
  std::atomic<uint64_t> next_sequence_{0};
  std::atomic<uint64_t> next_batch_id_{0};
  std::unique_ptr<BatchExecutor> executor_;
  FairnessController fairness_controller_;
  FairnessConfig fairness_config_;
};

}  // namespace inferflux
