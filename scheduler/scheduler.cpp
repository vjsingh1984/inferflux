#include "scheduler/scheduler.h"

#include "runtime/execution/batch_executor.h"
#include "runtime/execution/parallel_context.h"
#include "runtime/structured_output/structured_output_adapter.h"
#include "scheduler/single_model_router.h"
#include "server/metrics/metrics.h"
#include "server/tracing/span.h"

#include <algorithm>
#include <cstring>
#include <iostream>

namespace inferflux {

namespace {
constexpr std::size_t kMaxBatchSize = 4;
constexpr std::size_t kMaxBatchTokens = 8192;
constexpr double kFairnessAgingDivisorMs =
    2000.0; // every 2s of wait adds +1 to effective priority
// Number of distinct KV sequence slots available for phased prefill/decode.
// Must exceed kMaxBatchSize; mod-based assignment keeps concurrent requests
// collision-free.
constexpr int kMaxSequenceSlots = 16;
// Warm KV prefix store capacity (dedicated sequence slots for cached prefixes).
constexpr int kPrefixStoreCap = 4;
// Minimum prompt token count to be eligible for prefix store donation.
constexpr int kMinPrefixTokens = 32;
// Tokens per physical KV block (PagedAttention style).
constexpr int kTokensPerBlock = 16;

std::vector<uint8_t> SerializeTokens(const std::vector<int> &tokens) {
  std::vector<uint8_t> payload(tokens.size() * sizeof(int));
  if (!tokens.empty()) {
    std::memcpy(payload.data(), tokens.data(), payload.size());
  }
  return payload;
}
} // namespace

Scheduler::Scheduler(SimpleTokenizer &tokenizer,
                     std::shared_ptr<CPUDeviceContext> device,
                     std::shared_ptr<PagedKVCache> cache,
                     std::shared_ptr<ModelRouter> router,
                     std::shared_ptr<SpeculativeDecoder> speculative_decoder,
                     std::shared_ptr<RadixPrefixCache> prefix_cache,
                     const FairnessConfig &fairness_config,
                     const DisaggregatedConfig &disagg_config)
    : tokenizer_(tokenizer), device_(std::move(device)),
      cache_(std::move(cache)), router_(std::move(router)),
      speculative_decoder_(std::move(speculative_decoder)),
      prefix_cache_(std::move(prefix_cache)),
      fairness_controller_(fairness_config), fairness_config_(fairness_config),
      disagg_config_(disagg_config),
      seq_slots_free_((1ULL << kMaxSequenceSlots) - 1) {
  static_assert(kMaxSequenceSlots <= 64,
                "seq_slots_free_ bitmask requires kMaxSequenceSlots <= 64");
  executor_ = std::make_unique<BatchExecutor>(&tokenizer_, device_, cache_,
                                              router_, speculative_decoder_);
  // Enable decode worker pool when a positive pool size is configured.
  // With use_decode_workers_=true, ProcessBatch only runs Prefill and
  // hands off to pending_decode_; decode workers drain that queue.
  use_decode_workers_ = disagg_config_.decode_pool_size > 0;
  worker_ = std::thread(&Scheduler::WorkerLoop, this);
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
  int n = disagg_config_.decode_pool_size;
  if (n <= 0)
    return;
  decode_workers_.reserve(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    decode_workers_.emplace_back(&Scheduler::DecodeWorkerLoop, this);
  }
}

void Scheduler::StopDecodeWorkers() {
  for (auto &t : decode_workers_) {
    if (t.joinable()) {
      t.join();
    }
  }
  decode_workers_.clear();
}

int Scheduler::QueueDepth() const {
  std::lock_guard<std::mutex> lock(queue_mutex_);
  return static_cast<int>(pending_prefill_.size() + pending_decode_.size());
}

int Scheduler::LiveDecodeWorkers() const {
  return live_decode_workers_.load(std::memory_order_relaxed);
}

void Scheduler::DecodeWorkerLoop() {
  // Register as a live worker; deregister on any exit path (stop signal or
  // uncaught exception), so HttpServer::LiveDecodeWorkers() accurately reflects
  // pool health rather than the static startup flag.
  live_decode_workers_.fetch_add(1, std::memory_order_relaxed);
  struct LiveGuard {
    std::atomic<int> &counter;
    ~LiveGuard() { counter.fetch_sub(1, std::memory_order_relaxed); }
  } live_guard{live_decode_workers_};

  while (true) {
    std::vector<std::shared_ptr<PendingRequest>> batch;
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      queue_cv_.wait(lock, [&] { return stop_ || !pending_decode_.empty(); });
      if (stop_ && pending_decode_.empty())
        break;

      // Drain up to kMaxBatchSize requests from the decode queue.
      std::size_t n = std::min(pending_decode_.size(), kMaxBatchSize);
      batch.assign(pending_decode_.begin(),
                   pending_decode_.begin() + static_cast<std::ptrdiff_t>(n));
      pending_decode_.erase(pending_decode_.begin(),
                            pending_decode_.begin() +
                                static_cast<std::ptrdiff_t>(n));
      UpdateQueueDepthLocked();
    }

    // If a KV transport is configured, drain cross-process packets that may
    // have arrived from remote prefill workers.  For the in-process case the KV
    // state is already resident in the llama context and no hydration is
    // needed.
    if (disagg_config_.kv_transport) {
      while (auto pkt = disagg_config_.kv_transport->TryDequeue()) {
        // Record KV transfer latency (§2.5 item 12): time from Enqueue to
        // dequeue.
        auto dequeue_time = std::chrono::steady_clock::now();
        double transfer_ms = std::chrono::duration<double, std::milli>(
                                 dequeue_time - pkt->enqueue_time)
                                 .count();
        GlobalMetrics().RecordKVTransfer(transfer_ms);

        // Match the packet to a pending decode request by request_id and
        // hydrate its KV state.  If no matching request is found (e.g., it was
        // already handled), discard the packet.
        std::lock_guard<std::mutex> lock(queue_mutex_);
        for (auto &pending : pending_decode_) {
          if (pending->inference.id == pkt->request_id && pkt->n_past >= 0) {
            // Packet carries a serialised KV blob; hydrate it now.
            // (blob is currently empty for in-process path — no-op when empty.)
            // kv_blob holds raw bytes from LlamaCPUBackend::SerializeSequence.
            // Pass it directly to HydrateSequence — no cast or size adjustment.
            if (!pkt->kv_blob.empty() && pending->resolved_backend) {
              int seq_id = AllocSeqSlot();
              if (seq_id >= 0) {
                if (pending->resolved_backend->HydrateSequence(seq_id,
                                                               pkt->kv_blob)) {
                  pending->inference.sequence_id = seq_id;
                  pending->inference.n_past = pkt->n_past;
                } else {
                  FreeSeqSlot(seq_id);
                }
              }
            }
            break;
          }
        }
      }
    }

    if (batch.empty())
      continue;

    ResolveBackends(batch);

    RequestBatch exec_batch;
    exec_batch.batch_id =
        next_batch_id_.fetch_add(1, std::memory_order_relaxed);
    std::vector<std::shared_ptr<LlamaCPUBackend>> overrides;
    std::vector<std::shared_ptr<PendingRequest>> exec_pending;
    overrides.reserve(batch.size());
    exec_pending.reserve(batch.size());

    for (auto &pending : batch) {
      auto *inference = &pending->inference;
      if (inference->has_response_format &&
          !inference->response_format_supported) {
        InferenceResult error;
        error.no_backend = true;
        error.completion =
            inference->response_format_error.empty()
                ? "Selected model does not support response_format"
                : inference->response_format_error;
        pending->promise.set_value(std::move(error));
        // Return the sequence slot and KV blocks so they are not leaked.
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (inference->sequence_id >= 0) {
          if (cache_)
            cache_->ReleaseBlocks(inference->block_table);
          inference->block_table.clear();
          FreeSeqSlot(inference->sequence_id);
          inference->sequence_id = -1;
        }
        continue;
      }
      if (inference->has_response_format && !inference->response_format_ready) {
        StructuredConstraint constraint;
        std::string adapter_error;
        if (!StructuredOutputAdapter::BuildConstraint(
                inference->response_format_type,
                inference->response_format_schema,
                inference->response_format_grammar,
                inference->response_format_root, &constraint, &adapter_error)) {
          InferenceResult error;
          error.no_backend = true;
          error.completion = adapter_error.empty()
                                 ? "response_format could not be converted"
                                 : adapter_error;
          pending->promise.set_value(std::move(error));
          std::lock_guard<std::mutex> lock(queue_mutex_);
          if (inference->sequence_id >= 0) {
            if (cache_)
              cache_->ReleaseBlocks(inference->block_table);
            inference->block_table.clear();
            FreeSeqSlot(inference->sequence_id);
            inference->sequence_id = -1;
          }
          continue;
        }
        inference->response_constraint = constraint;
        inference->response_format_ready = true;
      }
      exec_pending.push_back(pending);
      overrides.push_back(pending->resolved_backend);
      exec_batch.requests.push_back(inference);
    }

    if (exec_pending.empty())
      continue;

    auto responses = executor_->ExecuteBatch(exec_batch, overrides);
    std::vector<std::shared_ptr<PendingRequest>> requeue;
    requeue.reserve(exec_pending.size());

    for (std::size_t i = 0; i < exec_pending.size(); ++i) {
      auto &pending = exec_pending[i];
      auto *inference = &pending->inference;
      InferenceResult result;
      if (i < responses.size()) {
        result = std::move(responses[i]);
      }

      if (inference->fairness_yielded) {
        auto now = std::chrono::steady_clock::now();
        pending->enqueue_time = now;
        inference->enqueue_time = now;
        inference->phase = RequestPhase::kDecode;
        if (!result.completion.empty()) {
          inference->prompt.append(result.completion);
          inference->prompt_tokens = tokenizer_.Encode(inference->prompt);
        }
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

      // Return KV memory and sequence slot.  Both must happen together.
      // ExecuteRequest intentionally does NOT call FreeSequence — it leaves
      // sequence_id set so the scheduler (here or in ProcessBatch) handles
      // both FreeSequence and FreeSeqSlot together in one place.
      {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (inference->sequence_id >= 0) {
          // Release the llama.cpp KV memory for this sequence.
          if (pending->resolved_backend) {
            pending->resolved_backend->FreeSequence(inference->sequence_id);
          }
          if (cache_)
            cache_->ReleaseBlocks(inference->block_table);
          inference->block_table.clear();
          FreeSeqSlot(inference->sequence_id);
          inference->sequence_id = -1;
        }
      }
      inference->phase = RequestPhase::kFinished;
      pending->promise.set_value(std::move(result));
    }

    if (!requeue.empty()) {
      {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        for (auto &pending : requeue) {
          pending_decode_.push_back(pending);
        }
        UpdateQueueDepthLocked();
      }
      queue_cv_.notify_all();
    }
  }
}

std::future<InferenceResult> Scheduler::Generate(InferenceRequest request) {
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
    pending->inference.prompt_tokens =
        tokenizer_.Encode(pending->inference.prompt);
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
  return future;
}

void Scheduler::UpdateFairnessConfig(const FairnessConfig &config) {
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    fairness_config_ = config;
    fairness_controller_.UpdateConfig(config);
  }
}

void Scheduler::WorkerLoop() {
  auto &pc = ParallelContext::Get();
  bool is_distributed = pc.IsInitialized() && pc.WorldSize() > 1;

  while (true) {
    BatchSelection selection;

    if (is_distributed && !pc.IsMaster()) {
      // §P1g: Worker ranks wait for the master's batch decision.
      std::vector<int> request_ids;
      std::vector<int> phases;
      if (!pc.ReceiveBatch(request_ids, phases)) {
        if (stop_)
          break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        continue;
      }

      // In a real implementation we'd reconstruct 'selection' from active
      // queues. For this foundation, we just Barrier to keep the worker rank
      // aligned with the master's processing cadence.
      pc.Comm()->Barrier();
    } else {
      {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        queue_cv_.wait(lock, [&] {
          return stop_ || !pending_prefill_.empty() ||
                 (!use_decode_workers_ && !pending_decode_.empty());
        });
        if (stop_ && pending_prefill_.empty() && pending_decode_.empty()) {
          break;
        }
        selection = BuildBatchLocked();
      }

      if (is_distributed) {
        // Broadcast the decision to all workers.
        std::vector<int> ids;
        std::vector<int> phases;
        for (auto &p : selection.pending) {
          ids.push_back(static_cast<int>(p->inference.id));
          phases.push_back(static_cast<int>(p->inference.phase));
        }
        pc.BroadcastBatch(ids, phases);
        pc.Comm()->Barrier(); // Wait for workers to catch up
      }
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
  // When decode workers are active they own pending_decode_; WorkerLoop must
  // only build prefill batches to avoid both threads competing for the same
  // requests and for the seq slot free-list (which is not separately locked).
  const bool decode_workers_own_decode = use_decode_workers_;
  queue_items.reserve(pending_prefill_.size() +
                      (decode_workers_own_decode ? 0 : pending_decode_.size()));
  for (std::size_t i = 0; i < pending_prefill_.size(); ++i) {
    queue_items.push_back(QueueItem{pending_prefill_[i], false, i});
  }
  if (!decode_workers_own_decode) {
    for (std::size_t i = 0; i < pending_decode_.size(); ++i) {
      queue_items.push_back(QueueItem{pending_decode_[i], true, i});
    }
  }

  auto now = std::chrono::steady_clock::now();
  std::stable_sort(queue_items.begin(), queue_items.end(),
                   [&](const QueueItem &a, const QueueItem &b) {
                     double age_a = std::chrono::duration<double, std::milli>(
                                        now - a.pending->enqueue_time)
                                        .count();
                     double age_b = std::chrono::duration<double, std::milli>(
                                        now - b.pending->enqueue_time)
                                        .count();
                     double eff_a = static_cast<double>(a.pending->priority) +
                                    age_a / kFairnessAgingDivisorMs;
                     double eff_b = static_cast<double>(b.pending->priority) +
                                    age_b / kFairnessAgingDivisorMs;
                     if (eff_a != eff_b) {
                       return eff_a > eff_b;
                     }
                     return a.pending->sequence < b.pending->sequence;
                   });

  std::vector<std::size_t> to_remove_prefill;
  std::vector<std::size_t> to_remove_decode;
  std::size_t token_budget = 0;
  for (std::size_t i = 0; i < queue_items.size(); ++i) {
    auto &item = queue_items[i].pending;
    std::size_t tokens = item->inference.prompt_tokens.size();
    if (!selection.pending.empty() && token_budget + tokens > kMaxBatchTokens) {
      continue;
    }

    // Handle swapped-out requests (§P1c).
    if (item->inference.is_swapped) {
      if (!TrySwapIn(item->inference)) {
        // Skip for this batch if we can't swap back in yet.
        continue;
      }
    }

    selection.pending.push_back(item);
    selection.batch.requests.push_back(&item->inference);
    item->inference.phase = queue_items[i].from_decode ? RequestPhase::kDecode
                                                       : RequestPhase::kPrefill;
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
    pending_prefill_.erase(pending_prefill_.begin() +
                           static_cast<std::ptrdiff_t>(to_remove_prefill[idx]));
  }
  for (std::size_t idx = to_remove_decode.size(); idx-- > 0;) {
    pending_decode_.erase(pending_decode_.begin() +
                          static_cast<std::ptrdiff_t>(to_remove_decode[idx]));
  }
  UpdateQueueDepthLocked();
  return selection;
}

void Scheduler::ProcessBatch(BatchSelection selection) {
  if (selection.pending.empty()) {
    return;
  }

  GlobalMetrics().SetDecodeQueueDepth(
      static_cast<int>(selection.pending.size()));
  ApplyFairness(&selection);
  ResolveBackends(selection.pending);
  auto batch_start = std::chrono::steady_clock::now();
  selection.batch.batch_id =
      next_batch_id_.fetch_add(1, std::memory_order_relaxed);
  std::vector<std::shared_ptr<PendingRequest>> staged_decode;
  staged_decode.reserve(selection.pending.size());
  std::vector<std::shared_ptr<PendingRequest>> decode_ready;
  decode_ready.reserve(selection.pending.size());
  for (auto &pending : selection.pending) {
    if (pending->inference.phase == RequestPhase::kPrefill) {
      // Option A phased prefill: run prompt evaluation on the local backend now
      // so that the decode slice (ExecuteRequest) can call Decode() from n_past
      // instead of Generate().
      auto &inf = pending->inference;
      if (pending->resolved_backend && pending->resolved_backend->IsReady()) {
        // Compute BPE tokens for prefix matching (INF-7).  We do this once per
        // request: if bpe_prompt_tokens is already populated (e.g., a retry
        // after channel-full rejection), reuse the cached result.
        if (inf.bpe_prompt_tokens.empty()) {
          inf.bpe_prompt_tokens =
              pending->resolved_backend->TokenizeForCache(inf.prompt);
        }
        // KV prefix reuse (§ Item 5): if we have a warm sequence whose BPE
        // tokens are a strict prefix of this request's BPE prompt tokens, copy
        // that KV state and only evaluate the suffix — skipping repeated prompt
        // computation.  Matching on BPE tokens ensures the prefix boundary
        // aligns with the KV cache (fixes INF-7 SimpleTokenizer mismatch).
        // KV prefix reuse (§P1b): lookup in the global Radix Trie to find
        // physical KV blocks matching this request's prefix.
        std::vector<int> cached_blocks;
        int matched_tokens = 0;
        int cached_seq_id = -1;
        bool prefix_hit = false;
        if (prefix_cache_) {
          prefix_hit = prefix_cache_->Lookup(
              inf.bpe_prompt_tokens, pending->resolved_backend.get(),
              &cached_blocks, &cached_seq_id, &matched_tokens);
        }

        // PagedAttention Block Allocation: calculate additional blocks needed.
        std::size_t prompt_len = inf.bpe_prompt_tokens.size();

        // Logical blocks already covered by the cached prefix.
        std::size_t warm_blocks = cached_blocks.size();
        std::size_t total_blocks_needed =
            (prompt_len + kTokensPerBlock - 1) / kTokensPerBlock;
        total_blocks_needed += 1;

        std::size_t new_blocks_needed =
            (total_blocks_needed > warm_blocks)
                ? (total_blocks_needed - warm_blocks)
                : 0;

        std::vector<int> new_blocks;
        if (cache_) {
          if (cache_->NumFreeBlocks() < new_blocks_needed) {
            // Memory tight (§P1c): attempt to swap out the lowest priority
            // request from the currently selected batch to make room.
            for (int j = static_cast<int>(selection.pending.size()) - 1; j >= 0;
                 --j) {
              auto &victim = selection.pending[j];
              if (victim != pending && !victim->inference.block_table.empty()) {
                if (TrySwapOut(victim->inference)) {
                  break; // Room potentially made.
                }
              }
            }
          }

          if (cache_->NumFreeBlocks() >= new_blocks_needed) {
            try {
              if (new_blocks_needed > 0) {
                new_blocks = cache_->ReserveBlocks(new_blocks_needed);
              }
            } catch (...) {
              new_blocks.clear();
            }
          }
        }

        int seq_id = AllocSeqSlot();
        // Admission logic (§ Item 4): can admit if we have a seq slot AND
        // (no paged cache configured OR new blocks were successfully reserved).
        bool can_admit = (seq_id >= 0) && (!cache_ || new_blocks_needed == 0 ||
                                           !new_blocks.empty());

        if (can_admit) {
          if (cache_) {
            // Take ownership of cached blocks (§P1b).
            cache_->AcquireBlocks(cached_blocks);
            inf.block_table = std::move(cached_blocks);
            inf.block_table.insert(inf.block_table.end(), new_blocks.begin(),
                                   new_blocks.end());
          }

          LlamaCPUBackend::PrefillResult pr;
          if (prefix_hit && matched_tokens > 0) {
            // Correctness Fix (§ Item 3): must copy KV state into new slot!
            // Only attempt partial prefill if we have a valid sequence to copy
            // from.
            if (cached_seq_id >= 0) {
              pending->resolved_backend->CopySequencePrefix(
                  cached_seq_id, seq_id, matched_tokens);
              pr = pending->resolved_backend->PrefillPartial(inf.prompt, seq_id,
                                                             matched_tokens);
            } else {
              // Hit but sequence was evicted; fallback to full prefill.
              pr = pending->resolved_backend->Prefill(inf.prompt, seq_id);
            }
            if (pr.ok && cached_seq_id >= 0) {
              GlobalMetrics().RecordKVPrefixReuse(matched_tokens);
            }
          } else {
            pr = pending->resolved_backend->Prefill(inf.prompt, seq_id);
          }

          if (pr.ok) {
            inf.n_past = pr.n_past;
            inf.prompt_bpe_tokens = pr.n_past;
            inf.sequence_id = seq_id;
            inf.first_token = pr.first_token;
            inf.first_piece = pr.first_piece;
          } else {
            // Prefill failed: release only the NEW blocks (warm blocks belong
            // to the cache).
            if (cache_)
              cache_->ReleaseBlocks(new_blocks);
            inf.block_table.clear();
            FreeSeqSlot(seq_id);
          }
        } else if (seq_id >= 0) {
          FreeSeqSlot(seq_id);
        }
      }
      bool enqueued = false;
      if (use_decode_workers_ && disagg_config_.kv_transport) {
        // Only enqueue when decode workers are live and draining the channel.
        // Without active consumers the channel fills to capacity (default 64)
        // and Enqueue() returns false, causing requests to bounce back into
        // pending_prefill_ indefinitely — a scheduler deadlock.
        disaggregated::KVPacket packet;
        packet.request_id = inf.id;
        packet.prompt_tokens = SerializeTokens(inf.prompt_tokens);
        // Serialize the actual KV cache state for this sequence so a remote
        // decode worker can hydrate it.  Empty when Prefill failed (seq_id<0).
        if (inf.sequence_id >= 0 && pending->resolved_backend) {
          packet.kv_blob =
              pending->resolved_backend->SerializeSequence(inf.sequence_id);
        }
        packet.n_past = inf.n_past;
        packet.sequence_id = inf.sequence_id;
        packet.metadata = inf.model;
        enqueued = disagg_config_.kv_transport->Enqueue(std::move(packet));
      } else {
        enqueued = true;
      }
      if (enqueued) {
        staged_decode.push_back(pending);
        continue;
      } else {
        // Channel rejected the packet (full).  Undo any phased state so the
        // request retries cleanly and does not hold a stale slot or stale
        // n_past.
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
      for (auto &pending : staged_decode) {
        pending->inference.phase = RequestPhase::kDecode;
        pending_decode_.push_back(pending);
      }
      UpdateQueueDepthLocked();
      queue_cv_.notify_all();
    } else {
      for (auto &pending : staged_decode) {
        pending->inference.phase = RequestPhase::kDecode;
        decode_ready.push_back(pending);
      }
    }
  }

  if (use_decode_workers_) {
    GlobalMetrics().SetDecodeQueueDepth(
        static_cast<int>(pending_decode_.size()));
    return;
  }

  if (decode_ready.empty()) {
    GlobalMetrics().SetDecodeQueueDepth(0);
    return;
  }

  selection.pending = decode_ready;
  for (auto &req : selection.pending) {
    double wait_ms = std::chrono::duration<double, std::milli>(
                         batch_start - req->enqueue_time)
                         .count();
    GlobalMetrics().RecordQueueLatency(wait_ms);
  }

  if (selection.total_tokens == 0) {
    for (const auto *req : selection.batch.requests) {
      selection.total_tokens += req->prompt_tokens.size();
      selection.total_tokens += req->output_tokens.size();
    }
  }
  GlobalMetrics().RecordBatch(selection.batch.requests.size(),
                              selection.total_tokens);

  RequestBatch exec_batch;
  exec_batch.batch_id = selection.batch.batch_id;
  std::vector<std::shared_ptr<LlamaCPUBackend>> overrides;
  std::vector<std::shared_ptr<PendingRequest>> exec_pending;
  overrides.reserve(selection.pending.size());
  exec_pending.reserve(selection.pending.size());
  for (auto &pending : selection.pending) {
    auto *inference = &pending->inference;
    if (inference->has_response_format &&
        !inference->response_format_supported) {
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
              inference->response_format_root, &constraint, &adapter_error)) {
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
    auto &pending = exec_pending[i];
    auto *inference = &pending->inference;
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
          "scheduler.fairness.yield", tracing::ChildContext(parent_ctx),
          [req_id = inference->id,
           remaining = inference->remaining_decode_tokens,
           slice = inference->last_timeslice_tokens, slice_tokens](
              const std::string &name, const SpanContext &ctx, double ms) {
            std::cout << "[fairness] " << name << " request=" << req_id
                      << " trace=" << ctx.trace_id << " limit=" << slice
                      << " generated=" << slice_tokens
                      << " remaining=" << remaining << " duration_ms=" << ms
                      << std::endl;
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
    // Return the llama.cpp KV memory and the sequence slot to the pool.
    // ExecuteRequest intentionally leaves sequence_id set so both operations
    // happen here, co-located.
    // KV prefix donation: when the request was a non-yielded phased decode on
    // the local worker (no decode workers), attempt to retain the sequence slot
    // as a warm prefix entry so future requests sharing this prompt prefix can
    // skip re-evaluating those tokens.
    // KV prefix cache (§P1b): insert the completed request's blocks into
    // the global Radix Trie to enable zero-copy reuse for future requests.
    bool donated = false;
    if (prefix_cache_ && inference->sequence_id >= 0 &&
        !inference->fairness_yielded) {
      // Concatenate prompt BPE tokens and any generated output BPE tokens.
      // (Simplified: we use prompt_bpe_tokens for the architectural
      // foundation).
      if (static_cast<int>(inference->bpe_prompt_tokens.size()) >=
          kMinPrefixTokens) {
        if (cache_)
          cache_->AcquireBlocks(
              inference->block_table); // Node ownership (§P1b)
        prefix_cache_->Insert(inference->bpe_prompt_tokens,
                              inference->block_table, inference->sequence_id,
                              pending->resolved_backend);
        donated = true;
      }
    }

    if (inference->sequence_id >= 0) {
      // Correctness Fix (§ Item 2): Do NOT FreeSequence if we donated!
      // The trie now owns this sequence's KV state until it is evicted.
      if (!donated) {
        if (pending->resolved_backend) {
          pending->resolved_backend->FreeSequence(inference->sequence_id);
        }
        FreeSeqSlot(inference->sequence_id);
      }
      // Correctness Fix (§ Item 2): always release scheduler's reference!
      // If donated, prefix_cache already called AcquireBlocks, so ref_count
      // >= 1.
      if (cache_)
        cache_->ReleaseBlocksRef(inference->block_table);
      inference->block_table.clear();
      inference->sequence_id = -1;
    }
    inference->phase = RequestPhase::kFinished;
    pending->promise.set_value(std::move(result));
  }

  if (!requeue.empty()) {
    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      for (auto &pending : requeue) {
        pending_decode_.push_back(pending);
      }
      UpdateQueueDepthLocked();
    }
    queue_cv_.notify_all();
  }
  auto batch_exec_end = std::chrono::steady_clock::now();
  double exec_ms =
      std::chrono::duration<double, std::milli>(batch_exec_end - batch_start)
          .count();
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

void Scheduler::ApplyFairness(BatchSelection *selection) {
  if (!selection || selection->pending.empty()) {
    return;
  }

  std::vector<FairnessEntry> batch_entries;
  batch_entries.reserve(selection->pending.size());
  for (std::size_t i = 0; i < selection->pending.size(); ++i) {
    auto &pending = selection->pending[i];
    pending->inference.priority_level = pending->inference.priority;
    if (pending->inference.total_completion_tokens > 0 &&
        pending->inference.remaining_decode_tokens > 0) {
      GlobalMetrics().RecordFairnessResume(pending->inference.priority_level);
      SpanContext parent_ctx;
      parent_ctx.trace_id = pending->inference.trace_id;
      Span resume_span(
          "scheduler.fairness.resume", tracing::ChildContext(parent_ctx),
          [req_id = pending->inference.id,
           remaining = pending->inference.remaining_decode_tokens](
              const std::string &name, const SpanContext &ctx, double ms) {
            std::cout << "[fairness] " << name << " request=" << req_id
                      << " trace=" << ctx.trace_id << " remaining=" << remaining
                      << " duration_ms=" << ms << std::endl;
          });
      resume_span.Finish();
    }
    batch_entries.push_back(FairnessEntry{
        &pending->inference, pending->inference.priority_level, i});
  }
  struct QueueItem {
    std::shared_ptr<PendingRequest> pending;
    bool from_decode{false};
    std::size_t index{0};
  };
  std::vector<QueueItem> queue_refs;
  queue_refs.reserve(pending_prefill_.size() + pending_decode_.size());
  for (std::size_t i = 0; i < pending_prefill_.size(); ++i) {
    auto &pending = pending_prefill_[i];
    pending->inference.priority_level = pending->inference.priority;
    queue_refs.push_back(QueueItem{pending, false, i});
  }
  for (std::size_t i = 0; i < pending_decode_.size(); ++i) {
    auto &pending = pending_decode_[i];
    pending->inference.priority_level = pending->inference.priority;
    queue_refs.push_back(QueueItem{pending, true, i});
  }
  std::vector<FairnessEntry> queue_entries;
  queue_entries.reserve(queue_refs.size());
  for (std::size_t i = 0; i < queue_refs.size(); ++i) {
    queue_entries.push_back(
        FairnessEntry{&queue_refs[i].pending->inference,
                      queue_refs[i].pending->inference.priority_level, i});
  }

  auto decision = fairness_controller_.Evaluate(batch_entries, queue_entries);
  if (decision.swap && decision.batch_index < selection->pending.size() &&
      decision.queue_index < queue_refs.size()) {
    auto queued_ref = queue_refs[decision.queue_index];
    auto queued = queued_ref.pending;
    auto displaced = selection->pending[decision.batch_index];
    selection->pending[decision.batch_index] = queued;
    if (queued_ref.from_decode) {
      pending_decode_.erase(pending_decode_.begin() +
                            static_cast<std::ptrdiff_t>(queued_ref.index));
    } else {
      pending_prefill_.erase(pending_prefill_.begin() +
                             static_cast<std::ptrdiff_t>(queued_ref.index));
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
  // CAS loop: find the lowest free bit, atomically clear it.  This is
  // safe for concurrent callers (WorkerLoop + DecodeWorkerLoop) without
  // holding queue_mutex_, because the operation is a single atomic RMW.
  uint64_t current = seq_slots_free_.load(std::memory_order_acquire);
  while (current != 0) {
    int slot = __builtin_ctzll(current); // index of lowest set (free) bit
    uint64_t desired = current & ~(1ULL << slot);
    if (seq_slots_free_.compare_exchange_weak(current, desired,
                                              std::memory_order_acquire,
                                              std::memory_order_relaxed)) {
      return slot;
    }
    // compare_exchange_weak reloads current on failure; retry.
  }
  return -1; // All slots in use; caller falls back to the legacy Generate()
             // path.
}

void Scheduler::FreeSeqSlot(int slot) {
  if (slot >= 0 && slot < kMaxSequenceSlots) {
    seq_slots_free_.fetch_or(1ULL << slot, std::memory_order_release);
  }
}

void Scheduler::ResolveBackends(
    const std::vector<std::shared_ptr<PendingRequest>> &batch) {
  if (!router_) {
    for (auto &pending : batch) {
      pending->resolved_backend.reset();
      pending->inference.resolved_model.clear();
      pending->inference.response_format_supported = true;
      pending->inference.response_format_error.clear();
    }
    return;
  }
  for (auto &pending : batch) {
    pending->resolved_backend.reset();
    pending->inference.resolved_model.clear();
    pending->inference.response_format_supported = true;
    pending->inference.response_format_error.clear();
    auto *info = router_->Resolve(pending->inference.model);
    if (!info) {
      GlobalMetrics().RecordModelRoute(pending->inference.model, "", false);
      continue;
    }
    if (pending->inference.has_response_format &&
        !info->supports_structured_output) {
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
      if (info->is_moe) {
        GlobalMetrics().RecordMoERequest();
      }
    } else {
      pending->inference.resolved_model = info->id;
      GlobalMetrics().RecordModelRoute(info->id, info->backend, false);
    }
  }
}

bool Scheduler::TrySwapIn(InferenceRequest &inf) {
  if (!inf.is_swapped || !cache_)
    return false;

  std::size_t blocks_needed = inf.swapped_host_handles.size();
  if (cache_->NumFreeBlocks() < blocks_needed)
    return false;

  try {
    std::vector<int> target_blocks = cache_->ReserveBlocks(blocks_needed);
    cache_->SwapIn(inf.swapped_host_handles, target_blocks);
    inf.block_table = std::move(target_blocks);
    inf.swapped_host_handles.clear();
    inf.is_swapped = false;
    return true;
  } catch (...) {
    return false;
  }
}

bool Scheduler::TrySwapOut(InferenceRequest &inf) {
  if (inf.is_swapped || inf.block_table.empty() || !cache_)
    return false;

  try {
    inf.swapped_host_handles = cache_->SwapOut(inf.block_table);
    inf.block_table.clear();
    inf.is_swapped = true;
    GlobalMetrics().RecordFairnessPreemption(
        inf.priority_level); // Reuse preemption metric for swap
    return true;
  } catch (...) {
    return false;
  }
}

} // namespace inferflux
