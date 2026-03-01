#include "scheduler/scheduler.h"

#include "scheduler/single_model_router.h"
#include "server/metrics/metrics.h"
#include "server/tracing/span.h"
#include "runtime/structured_output/structured_output_adapter.h"
#include "runtime/execution/batch_executor.h"

#include <algorithm>
#include <iostream>
#include <cstring>

namespace inferflux {

namespace {
constexpr std::size_t kMaxBatchSize = 4;
constexpr std::size_t kMaxBatchTokens = 8192;
constexpr double kFairnessAgingDivisorMs = 2000.0;  // every 2s of wait adds +1 to effective priority
// Number of distinct KV sequence slots available for phased prefill/decode.
// Must exceed kMaxBatchSize; mod-based assignment keeps concurrent requests collision-free.
constexpr int kMaxSequenceSlots = 16;

std::vector<uint8_t> SerializeTokens(const std::vector<int>& tokens) {
  std::vector<uint8_t> payload(tokens.size() * sizeof(int));
  if (!tokens.empty()) {
    std::memcpy(payload.data(), tokens.data(), payload.size());
  }
  return payload;
}
}

Scheduler::Scheduler(SimpleTokenizer tokenizer,
                     std::shared_ptr<CPUDeviceContext> device,
                     std::shared_ptr<PagedKVCache> cache,
                     std::shared_ptr<ModelRouter> router,
                     std::shared_ptr<SpeculativeDecoder> speculative_decoder,
                     std::shared_ptr<RadixPrefixCache> prefix_cache,
                     FairnessConfig fairness_config,
                     DisaggregatedConfig disagg_config)
    : tokenizer_(std::move(tokenizer)),
      device_(std::move(device)),
      cache_(std::move(cache)),
      router_(std::move(router)),
      speculative_decoder_(std::move(speculative_decoder)),
      prefix_cache_(std::move(prefix_cache)),
      fairness_controller_(fairness_config),
      fairness_config_(fairness_config),
      disagg_config_(disagg_config),
      seq_slots_free_(kMaxSequenceSlots, true) {
  executor_ = std::make_unique<BatchExecutor>(&tokenizer_, device_, cache_, router_, speculative_decoder_, prefix_cache_);
  worker_ = std::thread(&Scheduler::WorkerLoop, this);
  use_decode_workers_ = false;
  if (use_decode_workers_) {
    StartDecodeWorkers();
  }
}

Scheduler::~Scheduler() {
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    stop_ = true;
  }
  queue_cv_.notify_all();
  if (worker_.joinable()) {
    worker_.join();
  }
  if (use_decode_workers_) {
    StopDecodeWorkers();
  }
}

void Scheduler::StartDecodeWorkers() {
  // Decode worker pool is a forward-looking hook for disaggregated prefill/decode (§2.5).
  // Currently a no-op; decode happens on the single WorkerLoop thread.
  (void)disagg_config_.decode_pool_size;
}

void Scheduler::StopDecodeWorkers() {
  for (auto& t : decode_workers_) {
    if (t.joinable()) {
      t.join();
    }
  }
  decode_workers_.clear();
}

void Scheduler::DecodeWorkerLoop() {
  // Placeholder for future disaggregated decode worker (§2.5).
}

InferenceResult Scheduler::Generate(InferenceRequest request) {
  auto pending = std::make_shared<PendingRequest>();
  pending->inference = std::move(request);
  pending->priority = pending->inference.priority;
  pending->sequence = next_sequence_.fetch_add(1, std::memory_order_relaxed);
  pending->enqueue_time = std::chrono::steady_clock::now();
  pending->inference.id = pending->sequence;
  pending->inference.phase = RequestPhase::kPending;
  pending->inference.enqueue_time = pending->enqueue_time;
  if (pending->inference.max_tokens <= 0) {
    pending->inference.max_tokens = 1;
  }
  pending->inference.remaining_decode_tokens = pending->inference.max_tokens;
  pending->inference.accumulated_output.clear();
  if (pending->inference.prompt_tokens.empty()) {
    pending->inference.prompt_tokens = tokenizer_.Encode(pending->inference.prompt);
  }
  pending->inference.reported_prompt_tokens =
      static_cast<int>(pending->inference.prompt_tokens.size());
  pending->inference.output_tokens.clear();
  pending->inference.first_token_time = {};
  pending->inference.phase = RequestPhase::kPending;

  auto future = pending->promise.get_future();
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    pending_prefill_.push_back(pending);
    UpdateQueueDepthLocked();
  }
  queue_cv_.notify_one();
  return future.get();
}

void Scheduler::UpdateFairnessConfig(const FairnessConfig& config) {
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    fairness_config_ = config;
    fairness_controller_.UpdateConfig(config);
  }
}

void Scheduler::WorkerLoop() {
  while (true) {
    BatchSelection selection;
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      queue_cv_.wait(lock, [&] { return stop_ || !pending_prefill_.empty() || !pending_decode_.empty(); });
      if (stop_ && pending_prefill_.empty() && pending_decode_.empty()) {
        break;
      }
      selection = BuildBatchLocked();
    }
    if (!selection.pending.empty()) {
      ProcessBatch(std::move(selection));
    }
  }
}

Scheduler::BatchSelection Scheduler::BuildBatchLocked() {
  BatchSelection selection;
  if (pending_prefill_.empty() && pending_decode_.empty()) {
    return selection;
  }

  struct QueueItem {
    std::shared_ptr<PendingRequest> pending;
    bool from_decode{false};
    std::size_t index{0};
  };
  std::vector<QueueItem> queue_items;
  queue_items.reserve(pending_prefill_.size() + pending_decode_.size());
  for (std::size_t i = 0; i < pending_prefill_.size(); ++i) {
    queue_items.push_back(QueueItem{pending_prefill_[i], false, i});
  }
  for (std::size_t i = 0; i < pending_decode_.size(); ++i) {
    queue_items.push_back(QueueItem{pending_decode_[i], true, i});
  }

  auto now = std::chrono::steady_clock::now();
  std::stable_sort(queue_items.begin(),
                   queue_items.end(),
                   [&](const QueueItem& a, const QueueItem& b) {
                     double age_a = std::chrono::duration<double, std::milli>(now - a.pending->enqueue_time).count();
                     double age_b = std::chrono::duration<double, std::milli>(now - b.pending->enqueue_time).count();
                     double eff_a = static_cast<double>(a.pending->priority) + age_a / kFairnessAgingDivisorMs;
                     double eff_b = static_cast<double>(b.pending->priority) + age_b / kFairnessAgingDivisorMs;
                     if (eff_a != eff_b) {
                       return eff_a > eff_b;
                     }
                     return a.pending->sequence < b.pending->sequence;
                   });

  std::vector<std::size_t> to_remove_prefill;
  std::vector<std::size_t> to_remove_decode;
  std::size_t token_budget = 0;
  for (std::size_t i = 0; i < queue_items.size(); ++i) {
    auto& item = queue_items[i].pending;
    std::size_t tokens = item->inference.prompt_tokens.size();
    if (!selection.pending.empty() && token_budget + tokens > kMaxBatchTokens) {
      continue;
    }
    selection.pending.push_back(item);
    selection.batch.requests.push_back(&item->inference);
    item->inference.phase = queue_items[i].from_decode ? RequestPhase::kDecode : RequestPhase::kPrefill;
    token_budget += tokens;
    selection.total_tokens += tokens;
    if (queue_items[i].from_decode) {
      to_remove_decode.push_back(queue_items[i].index);
    } else {
      to_remove_prefill.push_back(queue_items[i].index);
    }
    if (selection.pending.size() >= kMaxBatchSize) {
      break;
    }
  }

  for (std::size_t idx = to_remove_prefill.size(); idx-- > 0;) {
    pending_prefill_.erase(pending_prefill_.begin() + static_cast<std::ptrdiff_t>(to_remove_prefill[idx]));
  }
  for (std::size_t idx = to_remove_decode.size(); idx-- > 0;) {
    pending_decode_.erase(pending_decode_.begin() + static_cast<std::ptrdiff_t>(to_remove_decode[idx]));
  }
  UpdateQueueDepthLocked();
  return selection;
}

void Scheduler::ProcessBatch(BatchSelection selection) {
  if (selection.pending.empty()) {
    return;
  }

  GlobalMetrics().SetDecodeQueueDepth(static_cast<int>(selection.pending.size()));
  ApplyFairness(&selection);
  ResolveBackends(selection.pending);
  auto batch_start = std::chrono::steady_clock::now();
  selection.batch.batch_id = next_batch_id_.fetch_add(1, std::memory_order_relaxed);
  std::vector<std::shared_ptr<PendingRequest>> staged_decode;
  staged_decode.reserve(selection.pending.size());
  std::vector<std::shared_ptr<PendingRequest>> decode_ready;
  decode_ready.reserve(selection.pending.size());
  for (auto& pending : selection.pending) {
    if (pending->inference.phase == RequestPhase::kPrefill) {
      // Option A phased prefill: run prompt evaluation on the local backend now so that
      // the decode slice (ExecuteRequest) can call Decode() from n_past instead of Generate().
      auto& inf = pending->inference;
      if (pending->resolved_backend && pending->resolved_backend->IsReady()) {
        // Allocate a dedicated sequence slot from the free-list.  Using a free-list
        // prevents two concurrent requests from sharing the same slot (which would
        // cause Prefill() to wipe the other request's KV state).
        int seq_id = AllocSeqSlot();
        if (seq_id >= 0) {
          auto pr = pending->resolved_backend->Prefill(inf.prompt, seq_id);
          if (pr.ok) {
            inf.n_past = pr.n_past;
            inf.sequence_id = seq_id;
          } else {
            FreeSeqSlot(seq_id);  // Prefill failed — return the slot immediately.
          }
        }
        // If seq_id < 0 (all slots busy), inf.n_past stays -1 and ExecuteRequest
        // falls back to the legacy Generate() path.
      }
      bool enqueued = false;
      if (use_decode_workers_ && disagg_config_.kv_channel) {
        // Only enqueue when decode workers are live and draining the channel.
        // Without active consumers the channel fills to capacity (default 64) and
        // Enqueue() returns false, causing requests to bounce back into pending_prefill_
        // indefinitely — a scheduler deadlock.
        disaggregated::KVPacket packet;
        packet.request_id = inf.id;
        packet.prompt_tokens = SerializeTokens(inf.prompt_tokens);
        packet.kv_blob = SerializeTokens(inf.prompt_tokens);
        packet.n_past = inf.n_past;
        packet.sequence_id = inf.sequence_id;
        packet.metadata = inf.model;
        enqueued = disagg_config_.kv_channel->Enqueue(std::move(packet));
      } else {
        enqueued = true;
      }
      if (enqueued) {
        staged_decode.push_back(pending);
        continue;
      } else {
        // Channel rejected the packet (full).  Undo any phased state so the request
        // retries cleanly and does not hold a stale slot or stale n_past.
        if (inf.sequence_id >= 0) {
          FreeSeqSlot(inf.sequence_id);
          inf.sequence_id = -1;
          inf.n_past = -1;
        }
        inf.phase = RequestPhase::kPending;
        std::lock_guard<std::mutex> lock(queue_mutex_);
        pending_prefill_.push_back(pending);
      }
    } else {
      decode_ready.push_back(pending);
    }
  }

  if (!staged_decode.empty()) {
    if (use_decode_workers_) {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      for (auto& pending : staged_decode) {
        pending->inference.phase = RequestPhase::kDecode;
        pending_decode_.push_back(pending);
      }
      UpdateQueueDepthLocked();
      queue_cv_.notify_all();
    } else {
      for (auto& pending : staged_decode) {
        pending->inference.phase = RequestPhase::kDecode;
        decode_ready.push_back(pending);
      }
    }
  }

  if (use_decode_workers_) {
    GlobalMetrics().SetDecodeQueueDepth(static_cast<int>(pending_decode_.size()));
    return;
  }

  if (decode_ready.empty()) {
    GlobalMetrics().SetDecodeQueueDepth(0);
    return;
  }

  selection.pending = decode_ready;
  for (auto& req : selection.pending) {
    double wait_ms = std::chrono::duration<double, std::milli>(batch_start - req->enqueue_time).count();
    GlobalMetrics().RecordQueueLatency(wait_ms);
  }

  if (selection.total_tokens == 0) {
    for (const auto* req : selection.batch.requests) {
      selection.total_tokens += req->prompt_tokens.size();
      selection.total_tokens += req->output_tokens.size();
    }
  }
  GlobalMetrics().RecordBatch(selection.batch.requests.size(), selection.total_tokens);

  RequestBatch exec_batch;
  exec_batch.batch_id = selection.batch.batch_id;
  std::vector<std::shared_ptr<LlamaCPUBackend>> overrides;
  std::vector<std::shared_ptr<PendingRequest>> exec_pending;
  overrides.reserve(selection.pending.size());
  exec_pending.reserve(selection.pending.size());
  for (auto& pending : selection.pending) {
    auto* inference = &pending->inference;
    if (inference->has_response_format && !inference->response_format_supported) {
      InferenceResult error;
      error.no_backend = true;
      error.completion = inference->response_format_error.empty()
                             ? "Selected model does not support response_format"
                             : inference->response_format_error;
      pending->promise.set_value(std::move(error));
      continue;
    }
    if (inference->has_response_format) {
      StructuredConstraint constraint;
      std::string adapter_error;
      if (!StructuredOutputAdapter::BuildConstraint(
              inference->response_format_type,
              inference->response_format_schema,
              inference->response_format_grammar,
              inference->response_format_root,
              &constraint,
              &adapter_error)) {
        InferenceResult error;
        error.no_backend = true;
        error.completion = adapter_error.empty()
                               ? "response_format could not be converted"
                               : adapter_error;
        pending->promise.set_value(std::move(error));
        continue;
      }
      inference->response_constraint = constraint;
      inference->response_format_ready = true;
    } else {
      inference->response_constraint = StructuredConstraint{};
      inference->response_format_ready = false;
    }
    exec_pending.push_back(pending);
    overrides.push_back(pending->resolved_backend);
    exec_batch.requests.push_back(&pending->inference);
  }

  if (exec_pending.empty()) {
    GlobalMetrics().SetDecodeQueueDepth(0);
    return;
  }

  auto responses = executor_->ExecuteBatch(exec_batch, overrides);
  std::vector<std::shared_ptr<PendingRequest>> requeue;
  requeue.reserve(exec_pending.size());
  for (std::size_t i = 0; i < exec_pending.size(); ++i) {
    auto& pending = exec_pending[i];
    auto* inference = &pending->inference;
    InferenceResult result;
    if (i < responses.size()) {
      result = std::move(responses[i]);
    }
    if (inference->fairness_yielded) {
      const std::string slice_text = result.completion;
      const int slice_tokens = result.completion_tokens;
      auto now = std::chrono::steady_clock::now();
      pending->enqueue_time = now;
      inference->enqueue_time = now;
      inference->phase = RequestPhase::kDecode;
      if (!slice_text.empty()) {
        inference->prompt.append(slice_text);
        inference->prompt_tokens = tokenizer_.Encode(inference->prompt);
      }
      SpanContext parent_ctx;
      parent_ctx.trace_id = inference->trace_id;
      Span fairness_span(
          "scheduler.fairness.yield",
          tracing::ChildContext(parent_ctx),
          [req_id = inference->id,
           remaining = inference->remaining_decode_tokens,
           slice = inference->last_timeslice_tokens,
           slice_tokens](const std::string& name,
                         const SpanContext& ctx,
                         double ms) {
            std::cout << "[fairness] " << name
                      << " request=" << req_id
                      << " trace=" << ctx.trace_id
                      << " limit=" << slice
                      << " generated=" << slice_tokens
                      << " remaining=" << remaining
                      << " duration_ms=" << ms << std::endl;
          });
      fairness_span.Finish();
      requeue.push_back(pending);
      continue;
    }
    if (!inference->accumulated_output.empty()) {
      result.completion = inference->accumulated_output;
      if (inference->total_completion_tokens > 0) {
        result.completion_tokens = inference->total_completion_tokens;
      }
    }
    if (inference->reported_prompt_tokens >= 0) {
      result.prompt_tokens = inference->reported_prompt_tokens;
    }
    // Return the sequence slot to the pool now that the request is fully done.
    // (batch_executor already called backend->FreeSequence() to release KV memory.)
    if (inference->sequence_id >= 0) {
      FreeSeqSlot(inference->sequence_id);
      inference->sequence_id = -1;
    }
    inference->phase = RequestPhase::kFinished;
    pending->promise.set_value(std::move(result));
  }

  if (!requeue.empty()) {
    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      for (auto& pending : requeue) {
        pending_decode_.push_back(pending);
      }
      UpdateQueueDepthLocked();
    }
    queue_cv_.notify_all();
  }
  auto batch_exec_end = std::chrono::steady_clock::now();
  double exec_ms = std::chrono::duration<double, std::milli>(batch_exec_end - batch_start).count();
  GlobalMetrics().RecordBatchExecution(exec_ms);
  GlobalMetrics().SetDecodeQueueDepth(0);
}

void Scheduler::UpdateQueueDepthLocked() const {
  int prefill_depth = static_cast<int>(pending_prefill_.size());
  int decode_depth = static_cast<int>(pending_decode_.size());
  GlobalMetrics().SetQueueDepth(prefill_depth + decode_depth);
  GlobalMetrics().SetPrefillQueueDepth(prefill_depth);
  GlobalMetrics().SetDecodeQueueDepth(decode_depth);
}
 
void Scheduler::ApplyFairness(BatchSelection* selection) {
  if (!selection || selection->pending.empty()) {
    return;
  }

  std::vector<FairnessEntry> batch_entries;
  batch_entries.reserve(selection->pending.size());
  for (std::size_t i = 0; i < selection->pending.size(); ++i) {
    auto& pending = selection->pending[i];
    pending->inference.priority_level = pending->inference.priority;
    if (pending->inference.total_completion_tokens > 0 &&
        pending->inference.remaining_decode_tokens > 0) {
      GlobalMetrics().RecordFairnessResume(pending->inference.priority_level);
      SpanContext parent_ctx;
      parent_ctx.trace_id = pending->inference.trace_id;
      Span resume_span("scheduler.fairness.resume",
                       tracing::ChildContext(parent_ctx),
                       [req_id = pending->inference.id,
                        remaining = pending->inference.remaining_decode_tokens](const std::string& name,
                                                                                const SpanContext& ctx,
                                                                                double ms) {
                         std::cout << "[fairness] " << name
                                   << " request=" << req_id
                                   << " trace=" << ctx.trace_id
                                   << " remaining=" << remaining
                                   << " duration_ms=" << ms << std::endl;
                       });
      resume_span.Finish();
    }
    batch_entries.push_back(
        FairnessEntry{&pending->inference, pending->inference.priority_level, i});
  }
  struct QueueItem {
    std::shared_ptr<PendingRequest> pending;
    bool from_decode{false};
    std::size_t index{0};
  };
  std::vector<QueueItem> queue_refs;
  queue_refs.reserve(pending_prefill_.size() + pending_decode_.size());
  for (std::size_t i = 0; i < pending_prefill_.size(); ++i) {
    auto& pending = pending_prefill_[i];
    pending->inference.priority_level = pending->inference.priority;
    queue_refs.push_back(QueueItem{pending, false, i});
  }
  for (std::size_t i = 0; i < pending_decode_.size(); ++i) {
    auto& pending = pending_decode_[i];
    pending->inference.priority_level = pending->inference.priority;
    queue_refs.push_back(QueueItem{pending, true, i});
  }
  std::vector<FairnessEntry> queue_entries;
  queue_entries.reserve(queue_refs.size());
  for (std::size_t i = 0; i < queue_refs.size(); ++i) {
    queue_entries.push_back(
        FairnessEntry{&queue_refs[i].pending->inference,
                      queue_refs[i].pending->inference.priority_level,
                      i});
  }

  auto decision = fairness_controller_.Evaluate(batch_entries, queue_entries);
  if (decision.swap &&
      decision.batch_index < selection->pending.size() &&
      decision.queue_index < queue_refs.size()) {
    auto queued_ref = queue_refs[decision.queue_index];
    auto queued = queued_ref.pending;
    auto displaced = selection->pending[decision.batch_index];
    selection->pending[decision.batch_index] = queued;
    if (queued_ref.from_decode) {
      pending_decode_.erase(pending_decode_.begin() + static_cast<std::ptrdiff_t>(queued_ref.index));
    } else {
      pending_prefill_.erase(pending_prefill_.begin() + static_cast<std::ptrdiff_t>(queued_ref.index));
    }
    displaced->inference.phase = RequestPhase::kPrefill;
    pending_prefill_.push_back(displaced);
    batch_entries[decision.batch_index].request = &queued->inference;
    GlobalMetrics().RecordFairnessPreemption(queued->inference.priority_level);
    UpdateQueueDepthLocked();
  }
  fairness_controller_.ApplyTimeslice(&batch_entries);
}

int Scheduler::AllocSeqSlot() {
  for (int i = 0; i < static_cast<int>(seq_slots_free_.size()); ++i) {
    if (seq_slots_free_[i]) {
      seq_slots_free_[i] = false;
      return i;
    }
  }
  return -1;  // All slots in use; caller falls back to the legacy Generate() path.
}

void Scheduler::FreeSeqSlot(int slot) {
  if (slot >= 0 && slot < static_cast<int>(seq_slots_free_.size())) {
    seq_slots_free_[slot] = true;
  }
}

void Scheduler::ResolveBackends(const std::vector<std::shared_ptr<PendingRequest>>& batch) {
  if (!router_) {
    for (auto& pending : batch) {
      pending->resolved_backend.reset();
      pending->inference.resolved_model.clear();
      pending->inference.response_format_supported = true;
      pending->inference.response_format_error.clear();
    }
    return;
  }
  for (auto& pending : batch) {
    pending->resolved_backend.reset();
    pending->inference.resolved_model.clear();
    pending->inference.response_format_supported = true;
    pending->inference.response_format_error.clear();
    auto* info = router_->Resolve(pending->inference.model);
    if (!info) {
      GlobalMetrics().RecordModelRoute(pending->inference.model, "", false);
      continue;
    }
    if (pending->inference.has_response_format && !info->supports_structured_output) {
      pending->inference.response_format_supported = false;
      pending->inference.response_format_error =
          "Selected model does not support response_format constraints";
      GlobalMetrics().RecordModelRoute(info->id, info->backend, false);
      continue;
    }
    auto backend = router_->GetBackend(info->id);
    if (backend && backend->IsReady()) {
      pending->resolved_backend = backend;
      pending->inference.resolved_model = info->id;
      GlobalMetrics().RecordModelRoute(info->id, info->backend, true);
    } else {
      pending->inference.resolved_model = info->id;
      GlobalMetrics().RecordModelRoute(info->id, info->backend, false);
    }
  }
}

}  // namespace inferflux
