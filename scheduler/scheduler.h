#pragma once

#include "model/tokenizer/simple_tokenizer.h"
#include "runtime/backends/cpu/cpu_backend.h"
#include "runtime/speculative/speculative_decoder.h"
#include "runtime/kv_cache/paged_kv_cache.h"
#include "runtime/disaggregated/kv_channel.h"
#include "scheduler/model_router.h"
#include "scheduler/fairness_controller.h"
#include "scheduler/request_batch.h"
#include "runtime/prefix_cache/prefix_cache.h"
#include "runtime/prefix_cache/radix_prefix_cache.h"

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

struct DisaggregatedConfig {
  int prefill_pool_size{1};
  // decode_pool_size > 0 enables the decode worker pool and sets use_decode_workers_=true.
  // Default is 0 (decode runs on the WorkerLoop thread, matching pre-§2.5 behaviour).
  int decode_pool_size{0};
  // kv_transport is the channel over which prefill workers hand KV state to
  // decode workers.  KVChannel (in-process) is the default; ShmKVTransport
  // (POSIX SHM) is selected when INFERFLUX_KV_TRANSPORT=shm.
  std::shared_ptr<disaggregated::IKVTransport> kv_transport;
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
            std::shared_ptr<RadixPrefixCache> prefix_cache = nullptr,
            FairnessConfig fairness_config = {},
            DisaggregatedConfig disagg_config = {});
  ~Scheduler();

  InferenceResult Generate(InferenceRequest request);
  void UpdateFairnessConfig(const FairnessConfig& config);

  // Expose the router for use by the HTTP admin endpoint (/v1/models).
  ModelRouter* Router() const { return router_.get(); }

  // Number of DecodeWorkerLoop threads currently running.  Zero means either
  // decode_pool_size=0 or all workers have exited (degraded/stopped state).
  // HttpServer uses this for the decode-role /readyz gate so that readiness
  // reflects live thread health rather than a static startup flag.
  int LiveDecodeWorkers() const {
    return live_decode_workers_.load(std::memory_order_relaxed);
  }

  // Configured pool size from DisaggregatedConfig::decode_pool_size.
  // HttpServer compares LiveDecodeWorkers() == ConfiguredDecodeWorkers() so
  // that even a single worker crash flips /readyz to not_ready.
  int ConfiguredDecodeWorkers() const { return disagg_config_.decode_pool_size; }

 private:
  struct BatchSelection {
    RequestBatch batch;
    std::vector<std::shared_ptr<PendingRequest>> pending;
    std::size_t total_tokens{0};
  };

  BatchSelection BuildBatchLocked();
  void WorkerLoop();
  void ProcessBatch(BatchSelection selection);
  void DecodeWorkerLoop();
  void StartDecodeWorkers();
  void StopDecodeWorkers();
  void ApplyFairness(BatchSelection* selection);
  void UpdateQueueDepthLocked() const;
  void ResolveBackends(const std::vector<std::shared_ptr<PendingRequest>>& batch);

  // Sequence slot allocator for §2.5 phased prefill/decode.
  // Slots are borrowed during Prefill() and returned after full request completion.
  // AllocSeqSlot() is called only from WorkerLoop (no lock needed).
  // FreeSeqSlot() may also be called from DecodeWorkerLoop — callers must hold
  // queue_mutex_ when calling from a non-worker-loop context.
  int AllocSeqSlot();
  void FreeSeqSlot(int slot);

  SimpleTokenizer tokenizer_;
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
  std::atomic<uint64_t> next_sequence_{0};
  std::atomic<uint64_t> next_batch_id_{0};
  std::unique_ptr<BatchExecutor> executor_;
  FairnessController fairness_controller_;
  FairnessConfig fairness_config_;
  DisaggregatedConfig disagg_config_;
  std::vector<bool> seq_slots_free_;  // size = kMaxSequenceSlots; true = available
  std::atomic<int> live_decode_workers_{0};  // live thread count; 0 = no pool or all exited
};

}  // namespace inferflux
