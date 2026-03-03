#include "runtime/backends/cuda/native_cuda_executor.h"
#include "runtime/backends/cuda/native_kernel_executor.h"
#include "runtime/backends/common/batching_utils.h"

#include "server/logging/logger.h"
#include "server/metrics/metrics.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>

namespace inferflux {

namespace {

#ifndef INFERFLUX_USE_COMMON_BACKEND_TYPES
// Local implementations when feature flag is OFF
bool LocalIsPrefillLikeInput(const LlamaCPUBackend::UnifiedBatchInput &input) {
  return input.tokens.size() > 1;
}

bool LocalIsPrefillOnlyBatch(
    const std::vector<LlamaCPUBackend::UnifiedBatchInput> &inputs) {
  if (inputs.empty()) {
    return false;
  }
  for (const auto &input : inputs) {
    if (!LocalIsPrefillLikeInput(input)) {
      return false;
    }
  }
  return true;
}
#endif

bool IsDecodeLane(LlamaCPUBackend::UnifiedBatchLane lane) {
  return lane != LlamaCPUBackend::UnifiedBatchLane::kPrefill;
}

std::vector<LlamaCPUBackend::UnifiedBatchInput>
BuildSubset(const std::vector<LlamaCPUBackend::UnifiedBatchInput> &inputs,
            const std::vector<std::size_t> &indices) {
  std::vector<LlamaCPUBackend::UnifiedBatchInput> subset;
  subset.reserve(indices.size());
  for (std::size_t idx : indices) {
    subset.push_back(inputs[idx]);
  }
  return subset;
}

class DelegateCudaExecutor final : public NativeCudaExecutor {
public:
  DelegateCudaExecutor() : backend_(std::make_shared<CudaBackend>()) {}

  std::string Name() const override { return "delegate_cuda_backend"; }
  bool IsFallback() const override { return true; }
  const std::string &FallbackReason() const override {
    static const std::string reason =
        "native backend scaffold mode active; execution delegates to "
        "universal llama CUDA path";
    return reason;
  }

  bool LoadModel(const std::filesystem::path &model_path,
                 const LlamaBackendConfig &config) override {
    return backend_->LoadModel(model_path, config);
  }

  std::vector<UnifiedBatchOutput>
  ExecuteUnifiedBatch(const std::vector<UnifiedBatchInput> &inputs) override {
    return backend_->ExecuteUnifiedBatch(inputs);
  }

  bool SupportsAsyncUnifiedBatch() const override {
    return backend_->SupportsAsyncUnifiedBatch();
  }

  UnifiedBatchHandle
  SubmitUnifiedBatchAsync(const std::vector<UnifiedBatchInput> &inputs,
                          UnifiedBatchLane lane) override {
    return backend_->SubmitUnifiedBatchAsync(inputs, lane);
  }

  bool TryCollectUnifiedBatchAsync(
      UnifiedBatchHandle handle,
      std::vector<UnifiedBatchOutput> *outputs) override {
    return backend_->TryCollectUnifiedBatchAsync(handle, outputs);
  }

  std::shared_ptr<LlamaCPUBackend> BackendHandle() const override {
    return backend_;
  }

private:
  std::shared_ptr<CudaBackend> backend_;
};

class DirectLlamaCudaExecutor final : public NativeCudaExecutor {
public:
  DirectLlamaCudaExecutor() = default;
  ~DirectLlamaCudaExecutor() override { StopAsyncRuntime(); }

  std::string Name() const override { return "direct_llama_scaffold"; }
  bool IsFallback() const override { return true; }
  const std::string &FallbackReason() const override {
    static const std::string reason =
        "native backend direct scaffold active; using llama CUDA kernels "
        "without native kernels";
    return reason;
  }

  bool LoadModel(const std::filesystem::path &model_path,
                 const LlamaBackendConfig &config) override {
    StopAsyncRuntime();

    effective_config_ =
        TuneLlamaBackendConfig(LlamaBackendTarget::kCuda, config);
    model_path_ = model_path;
    decode_backend_ = std::make_shared<LlamaCPUBackend>();
    if (!decode_backend_->LoadModel(model_path, effective_config_)) {
      return false;
    }

    prefill_replica_enabled_ = false;
    prefill_backend_.reset();
    if (effective_config_.cuda_phase_overlap_prefill_replica) {
      LlamaBackendConfig replica_cfg = effective_config_;
      replica_cfg.cuda_phase_overlap_scaffold = false;
      replica_cfg.cuda_phase_overlap_prefill_replica = false;
      auto replica = std::make_shared<LlamaCPUBackend>();
      if (replica->LoadModel(model_path, replica_cfg)) {
        prefill_backend_ = std::move(replica);
        prefill_replica_enabled_ = true;
      } else {
        log::Warn("native_cuda_executor",
                  "direct_llama_scaffold failed to initialize prefill replica; "
                  "continuing in single-context mode");
      }
    }

    {
      std::lock_guard<std::mutex> lock(async_mutex_);
      async_runtime_enabled_ = effective_config_.cuda_phase_overlap_scaffold;
      next_async_handle_ = 1;
      completed_.clear();
      decode_queue_.clear();
      prefill_queue_.clear();
      worker_stop_ = false;
      worker_running_ = false;
    }
    {
      std::lock_guard<std::mutex> lock(fence_mutex_);
      next_fence_ticket_ = 1;
      sequence_pending_prefill_ticket_.clear();
      completed_fence_tickets_.clear();
    }

    if (async_runtime_enabled_) {
      StartAsyncRuntimeLocked();
    } else {
      GlobalMetrics().SetCudaLaneQueueDepth(/*decode_lane=*/true, 0);
      GlobalMetrics().SetCudaLaneQueueDepth(/*decode_lane=*/false, 0);
    }
    return true;
  }

  std::vector<UnifiedBatchOutput>
  ExecuteUnifiedBatch(const std::vector<UnifiedBatchInput> &inputs) override {
    if (!decode_backend_) {
      return {};
    }
    if (!effective_config_.cuda_phase_overlap_scaffold || inputs.size() < 2) {
      return decode_backend_->ExecuteUnifiedBatch(inputs);
    }

    std::vector<std::size_t> decode_indices;
    std::vector<std::size_t> prefill_indices;
    decode_indices.reserve(inputs.size());
    prefill_indices.reserve(inputs.size());

    std::size_t prefill_tokens = 0;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
#ifdef INFERFLUX_USE_COMMON_BACKEND_TYPES
      if (BatchAnalyzer::IsPrefillLikeInput(inputs[i])) {
#else
      if (LocalIsPrefillLikeInput(inputs[i])) {
#endif
        prefill_indices.push_back(i);
        prefill_tokens += inputs[i].tokens.size();
      } else {
        decode_indices.push_back(i);
      }
    }

    const std::size_t min_prefill_tokens = static_cast<std::size_t>(
        std::max(1, effective_config_.cuda_phase_overlap_min_prefill_tokens));
    if (decode_indices.empty() || prefill_indices.empty() ||
        prefill_tokens < min_prefill_tokens) {
      return decode_backend_->ExecuteUnifiedBatch(inputs);
    }

    std::vector<UnifiedBatchOutput> merged(inputs.size());

    const auto decode_out = decode_backend_->ExecuteUnifiedBatch(
        BuildSubset(inputs, decode_indices));
    for (std::size_t i = 0; i < decode_out.size() && i < decode_indices.size();
         ++i) {
      merged[decode_indices[i]] = decode_out[i];
    }

    auto prefill_target = (prefill_replica_enabled_ && prefill_backend_)
                              ? prefill_backend_
                              : decode_backend_;
    const auto prefill_out = prefill_target->ExecuteUnifiedBatch(
        BuildSubset(inputs, prefill_indices));
    for (std::size_t i = 0;
         i < prefill_out.size() && i < prefill_indices.size(); ++i) {
      merged[prefill_indices[i]] = prefill_out[i];
    }
    return merged;
  }

  bool SupportsAsyncUnifiedBatch() const override {
    std::lock_guard<std::mutex> lock(async_mutex_);
    return async_runtime_enabled_;
  }

  UnifiedBatchHandle
  SubmitUnifiedBatchAsync(const std::vector<UnifiedBatchInput> &inputs,
                          UnifiedBatchLane lane) override {
    if (inputs.empty()) {
      return 0;
    }

    {
      std::lock_guard<std::mutex> lock(async_mutex_);
      if (async_runtime_enabled_) {
        StartAsyncRuntimeLocked();
        const UnifiedBatchHandle handle = next_async_handle_++;
        QueuedBatch job;
        job.handle = handle;
        job.inputs = inputs;

        UnifiedBatchLane effective_lane = lane;
        if (effective_lane == UnifiedBatchLane::kAuto) {
#ifdef INFERFLUX_USE_COMMON_BACKEND_TYPES
          effective_lane = BatchAnalyzer::IsPrefillOnlyBatch(job.inputs)
#else
          effective_lane = LocalIsPrefillOnlyBatch(job.inputs)
#endif
                               ? UnifiedBatchLane::kPrefill
                               : UnifiedBatchLane::kDecode;
        }
        job.lane = effective_lane;
        AssignFenceTicketsLocked(&job, effective_lane);

        if (effective_lane == UnifiedBatchLane::kPrefill) {
          prefill_queue_.push_back(std::move(job));
          GlobalMetrics().RecordCudaLaneSubmission(/*decode_lane=*/false);
          GlobalMetrics().SetCudaLaneQueueDepth(
              /*decode_lane=*/false, static_cast<int>(prefill_queue_.size()));
          prefill_cv_.notify_one();
        } else {
          decode_queue_.push_back(std::move(job));
          GlobalMetrics().RecordCudaLaneSubmission(/*decode_lane=*/true);
          GlobalMetrics().SetCudaLaneQueueDepth(
              /*decode_lane=*/true, static_cast<int>(decode_queue_.size()));
          decode_cv_.notify_one();
        }
        return handle;
      }
    }

    const auto outputs = ExecuteUnifiedBatch(inputs);
    std::lock_guard<std::mutex> lock(async_mutex_);
    const UnifiedBatchHandle handle = next_async_handle_++;
    completed_[handle] = outputs;
    return handle;
  }

  bool TryCollectUnifiedBatchAsync(
      UnifiedBatchHandle handle,
      std::vector<UnifiedBatchOutput> *outputs) override {
    if (handle == 0 || !outputs) {
      return false;
    }
    std::lock_guard<std::mutex> lock(async_mutex_);
    auto it = completed_.find(handle);
    if (it == completed_.end()) {
      return false;
    }
    *outputs = std::move(it->second);
    completed_.erase(it);
    return true;
  }

  std::shared_ptr<LlamaCPUBackend> BackendHandle() const override {
    return decode_backend_;
  }

private:
  struct QueuedBatch {
    UnifiedBatchHandle handle{0};
    UnifiedBatchLane lane{UnifiedBatchLane::kAuto};
    std::vector<UnifiedBatchInput> inputs;
    uint64_t completion_fence_ticket{0};
    std::vector<uint64_t> required_fence_tickets;
  };

  void StartAsyncRuntimeLocked() {
    if (worker_running_) {
      return;
    }
    worker_stop_ = false;
    worker_running_ = true;
    decode_worker_ =
        std::thread([this]() { AsyncWorkerLoop(UnifiedBatchLane::kDecode); });
    prefill_worker_ =
        std::thread([this]() { AsyncWorkerLoop(UnifiedBatchLane::kPrefill); });
  }

  void StopAsyncRuntime() {
    std::thread decode_worker_to_join;
    std::thread prefill_worker_to_join;
    {
      std::lock_guard<std::mutex> lock(async_mutex_);
      if (!worker_running_) {
        return;
      }
      worker_stop_ = true;
      decode_cv_.notify_all();
      prefill_cv_.notify_all();
      fence_cv_.notify_all();
      decode_worker_to_join = std::move(decode_worker_);
      prefill_worker_to_join = std::move(prefill_worker_);
    }
    if (decode_worker_to_join.joinable()) {
      decode_worker_to_join.join();
    }
    if (prefill_worker_to_join.joinable()) {
      prefill_worker_to_join.join();
    }
    {
      std::lock_guard<std::mutex> lock(async_mutex_);
      worker_running_ = false;
      worker_stop_ = false;
      decode_queue_.clear();
      prefill_queue_.clear();
      completed_.clear();
      GlobalMetrics().SetCudaLaneQueueDepth(/*decode_lane=*/true, 0);
      GlobalMetrics().SetCudaLaneQueueDepth(/*decode_lane=*/false, 0);
    }
    {
      std::lock_guard<std::mutex> lock(fence_mutex_);
      sequence_pending_prefill_ticket_.clear();
      completed_fence_tickets_.clear();
      next_fence_ticket_ = 1;
    }
  }

  void AssignFenceTicketsLocked(QueuedBatch *job, UnifiedBatchLane lane) {
    if (!job) {
      return;
    }
    if (lane == UnifiedBatchLane::kPrefill) {
      std::lock_guard<std::mutex> lock(fence_mutex_);
      const uint64_t ticket = next_fence_ticket_++;
      job->completion_fence_ticket = ticket;
      for (const auto &input : job->inputs) {
        sequence_pending_prefill_ticket_[input.sequence_id] = ticket;
      }
      return;
    }

    std::lock_guard<std::mutex> lock(fence_mutex_);
    job->required_fence_tickets.clear();
    job->required_fence_tickets.reserve(job->inputs.size());
    for (const auto &input : job->inputs) {
      auto it = sequence_pending_prefill_ticket_.find(input.sequence_id);
      if (it != sequence_pending_prefill_ticket_.end() && it->second > 0) {
        job->required_fence_tickets.push_back(it->second);
      }
    }
    std::sort(job->required_fence_tickets.begin(),
              job->required_fence_tickets.end());
    job->required_fence_tickets.erase(
        std::unique(job->required_fence_tickets.begin(),
                    job->required_fence_tickets.end()),
        job->required_fence_tickets.end());
  }

  void WaitForRequiredFences(const QueuedBatch &job) {
    if (job.required_fence_tickets.empty()) {
      return;
    }
    std::unique_lock<std::mutex> lock(fence_mutex_);
    for (;;) {
      bool all_ready = true;
      for (auto ticket : job.required_fence_tickets) {
        if (completed_fence_tickets_.find(ticket) ==
            completed_fence_tickets_.end()) {
          all_ready = false;
          break;
        }
      }
      if (all_ready) {
        return;
      }
      if (worker_stop_.load(std::memory_order_relaxed)) {
        return;
      }
      fence_cv_.wait_for(lock, std::chrono::milliseconds(2));
    }
  }

  void MarkFenceComplete(uint64_t ticket) {
    if (ticket == 0) {
      return;
    }
    std::lock_guard<std::mutex> lock(fence_mutex_);
    completed_fence_tickets_.insert(ticket);
    fence_cv_.notify_all();
  }

  bool
  FinalizePrefillReplicaHandoffs(const QueuedBatch &job,
                                 std::vector<UnifiedBatchOutput> *outputs) {
    if (!prefill_backend_ || !decode_backend_ || !outputs ||
        outputs->size() != job.inputs.size()) {
      return false;
    }

    struct HandoffEntry {
      std::size_t output_index{0};
      int sequence_id{-1};
      std::vector<uint8_t> kv_blob;
      std::chrono::steady_clock::time_point start_time;
    };

    bool all_ok = true;
    std::vector<int> prefill_sequences_to_free;
    std::vector<HandoffEntry> handoffs;
    prefill_sequences_to_free.reserve(job.inputs.size());
    handoffs.reserve(job.inputs.size());

    const auto mark_handoff_failure = [&](std::size_t output_index, int seq_id,
                                          const char *reason) {
      auto &output = (*outputs)[output_index];
      output.ok = false;
      output.token = -1;
      output.piece.clear();
      all_ok = false;
      log::Warn("native_cuda_executor",
                "direct_llama_scaffold handoff failed for seq=" +
                    std::to_string(seq_id) + " (" + reason + ")");
    };

    for (std::size_t i = 0; i < job.inputs.size(); ++i) {
      const auto &input = job.inputs[i];
      auto &output = (*outputs)[i];
      const bool final_prefill_chunk =
          input.request_logits && input.tokens.size() > 1;
      if (!final_prefill_chunk) {
        continue;
      }

      const int seq_id = input.sequence_id;
      prefill_sequences_to_free.push_back(seq_id);

      if (!output.ok || output.token < 0) {
        continue;
      }

      HandoffEntry entry;
      entry.output_index = i;
      entry.sequence_id = seq_id;
      entry.start_time = std::chrono::steady_clock::now();
      entry.kv_blob = prefill_backend_->SerializeSequence(seq_id);
      if (entry.kv_blob.empty()) {
        mark_handoff_failure(i, seq_id, "serialize");
        continue;
      }
      handoffs.push_back(std::move(entry));
    }

    if (!handoffs.empty()) {
      std::lock_guard<std::mutex> decode_lock(decode_exec_mutex_);
      for (auto &handoff : handoffs) {
        decode_backend_->FreeSequence(handoff.sequence_id);
        const bool hydrated = decode_backend_->HydrateSequence(
            handoff.sequence_id, handoff.kv_blob);
        const auto transfer_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - handoff.start_time)
                .count();
        GlobalMetrics().RecordKVTransfer(transfer_ms);
        if (!hydrated) {
          mark_handoff_failure(handoff.output_index, handoff.sequence_id,
                               "hydrate");
        }
      }
    }

    for (int seq_id : prefill_sequences_to_free) {
      prefill_backend_->FreeSequence(seq_id);
    }
    return all_ok;
  }

  void AsyncWorkerLoop(UnifiedBatchLane lane) {
    for (;;) {
      QueuedBatch job;
      {
        std::unique_lock<std::mutex> lock(async_mutex_);
        auto &lane_cv =
            (lane == UnifiedBatchLane::kPrefill) ? prefill_cv_ : decode_cv_;
        lane_cv.wait(lock, [this, lane]() {
          if (worker_stop_) {
            return true;
          }
          if (lane == UnifiedBatchLane::kPrefill) {
            return !prefill_queue_.empty();
          }
          return !decode_queue_.empty();
        });
        const bool use_prefill_queue = (lane == UnifiedBatchLane::kPrefill);
        auto &queue = use_prefill_queue ? prefill_queue_ : decode_queue_;
        if (worker_stop_ && queue.empty()) {
          break;
        }
        if (queue.empty()) {
          continue;
        }
        job = std::move(queue.front());
        queue.pop_front();
        GlobalMetrics().SetCudaLaneQueueDepth(IsDecodeLane(lane),
                                              static_cast<int>(queue.size()));
      }

      WaitForRequiredFences(job);

      std::vector<UnifiedBatchOutput> outputs;
      struct LaneExecutionScope {
        explicit LaneExecutionScope(bool decode_lane)
            : decode_lane_(decode_lane) {
          GlobalMetrics().RecordCudaLaneExecutionStart(decode_lane_);
        }
        ~LaneExecutionScope() {
          GlobalMetrics().RecordCudaLaneExecutionStop(decode_lane_);
        }
        bool decode_lane_{false};
      } lane_execution_scope(IsDecodeLane(lane));

      if (lane == UnifiedBatchLane::kPrefill && prefill_replica_enabled_ &&
          prefill_backend_ && prefill_backend_ != decode_backend_) {
        outputs = prefill_backend_->ExecuteUnifiedBatch(job.inputs);
        if (outputs.empty()) {
          outputs.resize(job.inputs.size());
        }
        FinalizePrefillReplicaHandoffs(job, &outputs);
      } else {
        std::lock_guard<std::mutex> exec_lock(decode_exec_mutex_);
        if (job.lane == UnifiedBatchLane::kAuto) {
          outputs = ExecuteUnifiedBatch(job.inputs);
        } else {
          outputs = decode_backend_->ExecuteUnifiedBatch(job.inputs);
        }
      }

      {
        std::lock_guard<std::mutex> lock(async_mutex_);
        completed_[job.handle] = std::move(outputs);
      }
      GlobalMetrics().RecordCudaLaneCompletion(IsDecodeLane(lane));
      if (job.completion_fence_ticket > 0) {
        MarkFenceComplete(job.completion_fence_ticket);
      }
    }
  }

  std::filesystem::path model_path_;
  LlamaBackendConfig effective_config_{};
  std::shared_ptr<LlamaCPUBackend> decode_backend_;
  std::shared_ptr<LlamaCPUBackend> prefill_backend_;
  bool async_runtime_enabled_{false};
  bool prefill_replica_enabled_{false};
  bool worker_running_{false};
  std::atomic<bool> worker_stop_{false};
  UnifiedBatchHandle next_async_handle_{1};
  std::thread decode_worker_;
  std::thread prefill_worker_;
  mutable std::mutex async_mutex_;
  std::condition_variable decode_cv_;
  std::condition_variable prefill_cv_;
  std::deque<QueuedBatch> decode_queue_;
  std::deque<QueuedBatch> prefill_queue_;
  std::unordered_map<UnifiedBatchHandle, std::vector<UnifiedBatchOutput>>
      completed_;
  std::mutex decode_exec_mutex_;
  uint64_t next_fence_ticket_{1};
  std::mutex fence_mutex_;
  std::condition_variable fence_cv_;
  std::unordered_map<int, uint64_t> sequence_pending_prefill_ticket_;
  std::unordered_set<uint64_t> completed_fence_tickets_;
};

std::string NormalizeExecutorHint(const std::string &hint) {
  std::string lowered = hint;
  std::transform(
      lowered.begin(), lowered.end(), lowered.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return lowered;
}

} // namespace

std::unique_ptr<NativeCudaExecutor>
CreateNativeCudaExecutor(const std::string &executor_hint) {
  const std::string hint = NormalizeExecutorHint(executor_hint);
  if (hint.empty() || hint == "delegate") {
    return std::make_unique<DelegateCudaExecutor>();
  }
  if (hint == "direct_llama") {
    return std::make_unique<DirectLlamaCudaExecutor>();
  }
  if (hint == "native_kernel" || hint == "native") {
    return std::make_unique<NativeKernelExecutor>();
  }
  log::Warn("native_cuda_executor", "unknown native CUDA executor hint '" +
                                        hint +
                                        "'; falling back to delegate executor");
  return std::make_unique<DelegateCudaExecutor>();
}

} // namespace inferflux
