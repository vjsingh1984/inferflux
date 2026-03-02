#pragma once

#include "model/tokenizer/simple_tokenizer.h"
#include "runtime/backends/cpu/cpu_backend.h"
#include "runtime/disaggregated/kv_channel.h"
#include "runtime/kv_cache/paged_kv_cache.h"
#include "runtime/logprob.h"
#include "runtime/prefix_cache/radix_prefix_cache.h"
#include "runtime/speculative/speculative_decoder.h"
#include "scheduler/fairness_controller.h"
#include "scheduler/model_router.h"
#include "scheduler/request_batch.h"

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

// Disaggregated prefill/decode configuration (§2.5).
struct DisaggregatedConfig {
  int prefill_pool_size{0}; // 0 = unified
  int decode_pool_size{0};  // 0 = unified
  std::shared_ptr<disaggregated::IKVTransport> kv_transport;
};

// Scheduler: handles request queuing, batch selection, and fairness.
class Scheduler {
public:
  struct Config {
    int max_batch_size{4};
    int max_batch_tokens{2048};
  };

  explicit Scheduler(
      SimpleTokenizer &tokenizer, std::shared_ptr<CPUDeviceContext> device,
      std::shared_ptr<PagedKVCache> cache, std::shared_ptr<ModelRouter> router,
      std::shared_ptr<SpeculativeDecoder> speculative_decoder = nullptr,
      std::shared_ptr<RadixPrefixCache> prefix_cache = nullptr,
      const FairnessConfig &fairness_config = {},
      const DisaggregatedConfig &disagg_config = {});
  ~Scheduler();

  // Non-blocking admission: adds request to queue and returns a future result.
  std::future<InferenceResult> Generate(InferenceRequest request);

  // Update fairness policy at runtime.
  void UpdateFairnessConfig(const FairnessConfig &config);

  // Status accessors.
  int QueueDepth() const;
  int LiveDecodeWorkers() const;
  int ConfiguredDecodeWorkers() const {
    return disagg_config_.decode_pool_size;
  }
  ModelRouter *Router() const { return router_.get(); }
  RadixPrefixCache *PrefixCache() const { return prefix_cache_.get(); }

  // Sequence slot allocator for §2.5 phased prefill/decode.
  // Slots are borrowed during Prefill() and returned after full request
  // completion. AllocSeqSlot() is called only from WorkerLoop (no lock needed).
  // FreeSeqSlot() may also be called from DecodeWorkerLoop or RadixPrefixCache
  // eviction — callers must hold queue_mutex_ when calling from a
  // non-worker-loop context.
  int AllocSeqSlot();
  void FreeSeqSlot(int slot);

private:
  struct PendingRequest {
    InferenceRequest inference;
    std::promise<InferenceResult> promise;
    std::chrono::steady_clock::time_point enqueue_time;
    int priority{0};
    int priority_level{0};
    uint64_t sequence{0};
    std::shared_ptr<LlamaCPUBackend> resolved_backend;
  };

  struct BatchSelection {
    RequestBatch batch;
    std::vector<std::shared_ptr<PendingRequest>> pending;
    std::size_t total_tokens{0};
  };

  void WorkerLoop();
  void ProcessBatch(BatchSelection selection);
  void DecodeWorkerLoop();
  void StartDecodeWorkers();
  void StopDecodeWorkers();
  void ApplyFairness(BatchSelection *selection);
  void UpdateQueueDepthLocked() const;
  void
  ResolveBackends(const std::vector<std::shared_ptr<PendingRequest>> &batch);

  BatchSelection BuildBatchLocked();

  // Swapping logic (§P1c).
  bool TrySwapIn(InferenceRequest &inf);
  bool TrySwapOut(InferenceRequest &inf);

  SimpleTokenizer &tokenizer_;
  std::shared_ptr<CPUDeviceContext> device_;
  std::shared_ptr<PagedKVCache> cache_;
  std::shared_ptr<ModelRouter> router_;
  std::shared_ptr<SpeculativeDecoder> speculative_decoder_;
  std::shared_ptr<RadixPrefixCache> prefix_cache_;
  std::vector<std::shared_ptr<PendingRequest>> pending_prefill_;
  std::vector<std::shared_ptr<PendingRequest>> pending_decode_;
  mutable std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  std::thread worker_;
  std::vector<std::thread> decode_workers_;
  bool use_decode_workers_{false};
  std::atomic<bool> stop_{false};
  std::unique_ptr<BatchExecutor> executor_;
  std::atomic<uint64_t> next_batch_id_{0};
  std::atomic<uint64_t> next_sequence_{0};
  FairnessController fairness_controller_;
  FairnessConfig fairness_config_;
  DisaggregatedConfig disagg_config_;

  // Sequence slot bookkeeping (§2.5).  Bitmask of 64 slots; 1 = free.
  // Using an atomic bitmask allows concurrent calls to AllocSeqSlot /
  // FreeSeqSlot without a data race on the underlying storage.
  std::atomic<uint64_t> seq_slots_free_{};
  std::atomic<int> live_decode_workers_{
      0}; // live thread count; 0 = no pool or all exited
};

} // namespace inferflux
