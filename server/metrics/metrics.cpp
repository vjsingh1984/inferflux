#include "server/metrics/metrics.h"

#include <cmath>
#include <iomanip>
#include <sstream>

namespace inferflux {

namespace {
MetricsRegistry g_metrics;
} // namespace

void LatencyHistogram::Record(double ms) {
  total.fetch_add(1, std::memory_order_relaxed);
  sum_ms.fetch_add(static_cast<uint64_t>(std::max(0.0, ms)),
                   std::memory_order_relaxed);
  // All buckets are cumulative: increment every bucket >= ms.
  for (std::size_t i = 0; i < kBuckets.size(); ++i) {
    if (ms <= kBuckets[i]) {
      counts[i].fetch_add(1, std::memory_order_relaxed);
    }
  }
  // +Inf bucket always increments.
  counts[kBuckets.size()].fetch_add(1, std::memory_order_relaxed);
}

void MetricsRegistry::SetBackend(const std::string &backend) {
  std::lock_guard<std::mutex> lock(backend_mutex_);
  backend_ = backend;
}

void MetricsRegistry::RecordSuccess(int prompt_tokens, int completion_tokens) {
  total_requests_.fetch_add(1, std::memory_order_relaxed);
  total_prompt_tokens_.fetch_add(prompt_tokens, std::memory_order_relaxed);
  total_completion_tokens_.fetch_add(completion_tokens,
                                     std::memory_order_relaxed);
}

void MetricsRegistry::RecordError() {
  total_errors_.fetch_add(1, std::memory_order_relaxed);
}

void MetricsRegistry::RecordSpeculative(std::size_t total_chunks,
                                        std::size_t accepted_chunks,
                                        std::size_t reused_tokens) {
  if (total_chunks == 0 && accepted_chunks == 0 && reused_tokens == 0) {
    return;
  }
  speculative_chunks_total_.fetch_add(total_chunks, std::memory_order_relaxed);
  speculative_chunks_accepted_.fetch_add(accepted_chunks,
                                         std::memory_order_relaxed);
  speculative_tokens_reused_.fetch_add(reused_tokens,
                                       std::memory_order_relaxed);
}

void MetricsRegistry::RecordBatch(std::size_t request_count,
                                  std::size_t token_count) {
  if (request_count == 0) {
    return;
  }
  total_batches_.fetch_add(1, std::memory_order_relaxed);
  total_batch_tokens_.fetch_add(token_count, std::memory_order_relaxed);
  uint64_t current_max = max_batch_size_.load(std::memory_order_relaxed);
  while (request_count > current_max &&
         !max_batch_size_.compare_exchange_weak(current_max, request_count,
                                                std::memory_order_relaxed)) {
  }
}

void MetricsRegistry::RecordSchedulerIteration(std::size_t prefill_requests,
                                               std::size_t decode_requests,
                                               std::size_t token_count) {
  if (prefill_requests == 0 && decode_requests == 0) {
    return;
  }
  scheduler_iteration_requests_total_.fetch_add(
      prefill_requests + decode_requests, std::memory_order_relaxed);
  if (token_count > 0) {
    scheduler_iteration_tokens_total_.fetch_add(token_count,
                                                std::memory_order_relaxed);
  }

  if (prefill_requests > 0 && decode_requests > 0) {
    scheduler_iterations_mixed_.fetch_add(1, std::memory_order_relaxed);
  } else if (prefill_requests > 0) {
    scheduler_iterations_prefill_.fetch_add(1, std::memory_order_relaxed);
  } else {
    scheduler_iterations_decode_.fetch_add(1, std::memory_order_relaxed);
  }
}

void MetricsRegistry::RecordBatchTokenBudgetSkip() {
  scheduler_batch_token_budget_skips_.fetch_add(1, std::memory_order_relaxed);
}

void MetricsRegistry::RecordPrefixLookup(bool hit) {
  if (hit) {
    prefix_hits_.fetch_add(1, std::memory_order_relaxed);
  } else {
    prefix_misses_.fetch_add(1, std::memory_order_relaxed);
  }
}

void MetricsRegistry::RecordPrefixMatchedTokens(int tokens) {
  if (tokens <= 0)
    return;
  prefix_matched_tokens_.fetch_add(static_cast<uint64_t>(tokens),
                                   std::memory_order_relaxed);
}

void MetricsRegistry::RecordPartialPrefixHit() {
  prefix_partial_hits_.fetch_add(1, std::memory_order_relaxed);
}

void MetricsRegistry::RecordKVPrefixReuse(int tokens_saved) {
  kv_prefix_reuse_count_.fetch_add(1, std::memory_order_relaxed);
  if (tokens_saved > 0) {
    kv_prefix_reuse_tokens_.fetch_add(static_cast<uint64_t>(tokens_saved),
                                      std::memory_order_relaxed);
  }
}

void MetricsRegistry::RecordStreamTokens(std::size_t tokens) {
  if (tokens == 0) {
    return;
  }
  stream_tokens_.fetch_add(tokens, std::memory_order_relaxed);
}

void MetricsRegistry::RecordStreamCacheHit() {
  stream_cache_hits_.fetch_add(1, std::memory_order_relaxed);
}

void MetricsRegistry::RecordFairnessTokens(int priority_level,
                                           std::size_t tokens) {
  if (tokens == 0) {
    return;
  }
  std::lock_guard<std::mutex> lock(fairness_metrics_mutex_);
  fairness_tokens_[priority_level] += tokens;
}

void MetricsRegistry::RecordFairnessPreemption(int priority_level) {
  fairness_preemptions_.fetch_add(1, std::memory_order_relaxed);
  std::lock_guard<std::mutex> lock(fairness_metrics_mutex_);
  (void)fairness_tokens_[priority_level];
}

void MetricsRegistry::RecordFairnessYield(int priority_level,
                                          std::size_t emitted_tokens,
                                          std::size_t remaining_tokens) {
  fairness_yields_.fetch_add(1, std::memory_order_relaxed);
  (void)priority_level;
  (void)emitted_tokens;
  (void)remaining_tokens;
}

void MetricsRegistry::RecordFairnessResume(int priority_level) {
  fairness_resumes_.fetch_add(1, std::memory_order_relaxed);
}

void MetricsRegistry::RecordQueueLatency(double wait_ms) {
  queue_latency_.Record(wait_ms);
}

void MetricsRegistry::RecordBatchExecution(double exec_ms) {
  batch_exec_latency_.Record(exec_ms);
}

void MetricsRegistry::RecordModelLoad(const std::string &model_id,
                                      const std::string &backend,
                                      double load_seconds) {
  if (model_id.empty()) {
    return;
  }
  std::lock_guard<std::mutex> lock(model_metrics_mutex_);
  auto &stats = model_stats_[model_id];
  if (!backend.empty()) {
    stats.backend = backend;
  }
  stats.load_seconds += std::max(0.0, load_seconds);
  stats.load_events += 1;
}

void MetricsRegistry::RecordModelReady(const std::string &model_id,
                                       const std::string &backend, bool ready) {
  if (model_id.empty()) {
    return;
  }
  std::lock_guard<std::mutex> lock(model_metrics_mutex_);
  auto &stats = model_stats_[model_id];
  if (!backend.empty()) {
    stats.backend = backend;
  }
  stats.ready = ready;
}

void MetricsRegistry::RecordModelRoute(const std::string &model_id,
                                       const std::string &backend, bool hit) {
  if (!hit || model_id.empty()) {
    model_route_misses_.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  std::lock_guard<std::mutex> lock(model_metrics_mutex_);
  auto &stats = model_stats_[model_id];
  if (!backend.empty()) {
    stats.backend = backend;
  }
  stats.routes += 1;
}

void MetricsRegistry::RecordModelTokens(const std::string &model_id,
                                        const std::string &backend,
                                        int prompt_tokens,
                                        int completion_tokens) {
  if (model_id.empty()) {
    return;
  }
  if (prompt_tokens <= 0 && completion_tokens <= 0) {
    return;
  }
  std::lock_guard<std::mutex> lock(model_metrics_mutex_);
  auto &stats = model_stats_[model_id];
  if (!backend.empty()) {
    stats.backend = backend;
  }
  if (prompt_tokens > 0) {
    stats.prompt_tokens += static_cast<uint64_t>(prompt_tokens);
  }
  if (completion_tokens > 0) {
    stats.completion_tokens += static_cast<uint64_t>(completion_tokens);
  }
}

void MetricsRegistry::RecordCapabilityRejection(const std::string &backend,
                                                const std::string &feature) {
  if (feature.empty()) {
    return;
  }
  const std::string backend_label = backend.empty() ? "unknown" : backend;
  const std::string key = backend_label + "|" + feature;
  std::lock_guard<std::mutex> lock(capability_metrics_mutex_);
  capability_rejections_[key] += 1;
}

void MetricsRegistry::RecordBackendExposure(
    const std::string &requested_backend, const std::string &exposed_backend,
    const std::string &provider, bool used_fallback) {
  if (exposed_backend.empty()) {
    return;
  }
  const std::string requested_label =
      requested_backend.empty() ? "auto" : requested_backend;
  const std::string exposed_label =
      exposed_backend.empty() ? "unknown" : exposed_backend;
  const std::string provider_label = provider.empty() ? "unknown" : provider;
  const std::string fallback_label = used_fallback ? "true" : "false";
  const std::string key = requested_label + "|" + exposed_label + "|" +
                          provider_label + "|" + fallback_label;
  std::lock_guard<std::mutex> lock(backend_exposure_mutex_);
  backend_exposure_counts_[key] += 1;
}

void MetricsRegistry::RecordCapabilityRouteFallback(
    const std::string &from_backend, const std::string &to_backend,
    const std::string &feature) {
  if (feature.empty()) {
    return;
  }
  const std::string from_label =
      from_backend.empty() ? "unknown" : from_backend;
  const std::string to_label = to_backend.empty() ? "unknown" : to_backend;
  const std::string key = from_label + "|" + to_label + "|" + feature;
  std::lock_guard<std::mutex> lock(capability_route_fallback_mutex_);
  capability_route_fallbacks_[key] += 1;
}

void MetricsRegistry::RecordImagePreprocess(int images, double /*decode_ms*/) {
  if (images > 0) {
    multimodal_images_.fetch_add(static_cast<uint64_t>(images),
                                 std::memory_order_relaxed);
    multimodal_requests_.fetch_add(1, std::memory_order_relaxed);
  }
}

void MetricsRegistry::RecordMoERequest() {
  moe_requests_.fetch_add(1, std::memory_order_relaxed);
}

void MetricsRegistry::SetFlashAttentionEnabled(bool enabled) {
  flash_attention_enabled_.store(enabled ? 1 : 0, std::memory_order_relaxed);
}

void MetricsRegistry::RecordFlashAttentionExecution(const std::string &kernel,
                                                    double duration_ms,
                                                    int prompt_tokens) {
  // Record execution time histogram
  flash_attention_exec_latency_.Record(duration_ms);

  // Record request counter for the specific kernel
  if (kernel == "fa2") {
    flash_attention_requests_fa2_.fetch_add(1, std::memory_order_relaxed);
  } else if (kernel == "fa3") {
    flash_attention_requests_fa3_.fetch_add(1, std::memory_order_relaxed);
  } else {
    flash_attention_requests_standard_.fetch_add(1, std::memory_order_relaxed);
  }
}

void MetricsRegistry::SetFlashAttentionMemoryMB(double memory_mb) {
  flash_attention_memory_mb_.store(memory_mb, std::memory_order_relaxed);
}

void MetricsRegistry::RecordFlashAttentionRequest(const std::string &kernel) {
  // This is called when a request starts processing with FlashAttention
  // Separate from RecordFlashAttentionExecution which is called on completion
  if (kernel == "fa2") {
    flash_attention_requests_fa2_.fetch_add(1, std::memory_order_relaxed);
  } else if (kernel == "fa3") {
    flash_attention_requests_fa3_.fetch_add(1, std::memory_order_relaxed);
  } else {
    flash_attention_requests_standard_.fetch_add(1, std::memory_order_relaxed);
  }
}

void MetricsRegistry::RecordRocmKernelSelection(const std::string &kernel) {
  // Track which kernel is selected for ROCm backend
  // This is called when the ROCm backend selects an attention kernel
  if (kernel == "fa2") {
    rocm_flash_attention_requests_.fetch_add(1, std::memory_order_relaxed);
  }
}

void MetricsRegistry::RecordRocmFlashAttentionExecution(double duration_ms,
                                                        int tokens) {
  // Record ROCm FlashAttention execution time
  // Reuse the FlashAttention histogram for ROCm
  flash_attention_exec_latency_.Record(duration_ms);
}

void MetricsRegistry::SetRocmMemoryUsageMB(double memory_mb) {
  rocm_memory_mb_.store(memory_mb, std::memory_order_relaxed);
}

void MetricsRegistry::RecordRocmDeviceProperties(int device_id,
                                                 const std::string &arch) {
  // Record ROCm device properties
  rocm_device_arch_ = arch;
}

void MetricsRegistry::RecordKVTransfer(double transfer_ms) {
  kv_transfer_latency_.Record(transfer_ms);
}

void MetricsRegistry::RecordLlamaPerf(double prefill_ms, double decode_ms,
                                      int32_t prompt_tokens,
                                      int32_t generated_tokens) {
  if (prefill_ms > 0)
    llama_prefill_latency_.Record(prefill_ms);
  if (decode_ms > 0)
    llama_decode_latency_.Record(decode_ms);
  if (prompt_tokens > 0)
    llama_prompt_tokens_.fetch_add(static_cast<uint64_t>(prompt_tokens),
                                   std::memory_order_relaxed);
  if (generated_tokens > 0)
    llama_gen_tokens_.fetch_add(static_cast<uint64_t>(generated_tokens),
                                std::memory_order_relaxed);
}

void MetricsRegistry::RecordLatency(double request_ms) {
  request_latency_.Record(request_ms);
}

void MetricsRegistry::RecordPrefillDuration(double prefill_ms) {
  prefill_latency_.Record(prefill_ms);
}

void MetricsRegistry::RecordDecodeDuration(double decode_ms) {
  decode_latency_.Record(decode_ms);
}

void MetricsRegistry::IncrementConnections() {
  active_connections_.fetch_add(1, std::memory_order_relaxed);
}

void MetricsRegistry::DecrementConnections() {
  active_connections_.fetch_sub(1, std::memory_order_relaxed);
}

void MetricsRegistry::SetQueueDepth(int depth) {
  queue_depth_.store(depth, std::memory_order_relaxed);
}

void MetricsRegistry::SetPrefillQueueDepth(int depth) {
  prefill_queue_depth_.store(depth, std::memory_order_relaxed);
}

void MetricsRegistry::SetDecodeQueueDepth(int depth) {
  decode_queue_depth_.store(depth, std::memory_order_relaxed);
}

void MetricsRegistry::SetSchedulerBatchLimits(int max_batch_size,
                                              int max_batch_tokens) {
  scheduler_batch_limit_size_.store(max_batch_size, std::memory_order_relaxed);
  scheduler_batch_limit_tokens_.store(max_batch_tokens,
                                      std::memory_order_relaxed);
}

void MetricsRegistry::RecordCudaLaneSubmission(bool decode_lane) {
  if (decode_lane) {
    cuda_decode_lane_submissions_.fetch_add(1, std::memory_order_relaxed);
  } else {
    cuda_prefill_lane_submissions_.fetch_add(1, std::memory_order_relaxed);
  }
}

void MetricsRegistry::RecordCudaLaneCompletion(bool decode_lane) {
  if (decode_lane) {
    cuda_decode_lane_completions_.fetch_add(1, std::memory_order_relaxed);
  } else {
    cuda_prefill_lane_completions_.fetch_add(1, std::memory_order_relaxed);
  }
}

void MetricsRegistry::RecordCudaLaneExecutionStart(bool decode_lane) {
  bool maybe_start_overlap = false;
  if (decode_lane) {
    cuda_decode_lane_inflight_.fetch_add(1, std::memory_order_relaxed);
    if (cuda_prefill_lane_inflight_.load(std::memory_order_relaxed) > 0) {
      cuda_lane_overlap_events_.fetch_add(1, std::memory_order_relaxed);
      maybe_start_overlap = true;
    }
  } else {
    cuda_prefill_lane_inflight_.fetch_add(1, std::memory_order_relaxed);
    if (cuda_decode_lane_inflight_.load(std::memory_order_relaxed) > 0) {
      cuda_lane_overlap_events_.fetch_add(1, std::memory_order_relaxed);
      maybe_start_overlap = true;
    }
  }

  if (maybe_start_overlap) {
    std::lock_guard<std::mutex> lock(cuda_overlap_timing_mutex_);
    if (!cuda_overlap_active_) {
      cuda_overlap_active_ = true;
      cuda_overlap_started_at_ = std::chrono::steady_clock::now();
    }
  }
}

void MetricsRegistry::RecordCudaLaneExecutionStop(bool decode_lane) {
  if (decode_lane) {
    const int previous =
        cuda_decode_lane_inflight_.fetch_sub(1, std::memory_order_relaxed);
    if (previous <= 0) {
      cuda_decode_lane_inflight_.store(0, std::memory_order_relaxed);
    }
  } else {
    const int previous =
        cuda_prefill_lane_inflight_.fetch_sub(1, std::memory_order_relaxed);
    if (previous <= 0) {
      cuda_prefill_lane_inflight_.store(0, std::memory_order_relaxed);
    }
  }

  if (cuda_decode_lane_inflight_.load(std::memory_order_relaxed) == 0 &&
      cuda_prefill_lane_inflight_.load(std::memory_order_relaxed) == 0) {
    std::lock_guard<std::mutex> lock(cuda_overlap_timing_mutex_);
    if (cuda_overlap_active_) {
      const auto elapsed_us =
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now() - cuda_overlap_started_at_)
              .count();
      if (elapsed_us > 0) {
        cuda_lane_overlap_duration_us_.fetch_add(
            static_cast<uint64_t>(elapsed_us), std::memory_order_relaxed);
      }
      cuda_overlap_active_ = false;
    }
  }
}

void MetricsRegistry::RecordCudaLaneOverlap(double duration_ms) {
  // Record an overlap event and its duration
  cuda_lane_overlap_events_.fetch_add(1, std::memory_order_relaxed);
  const auto duration_us = static_cast<uint64_t>(duration_ms * 1000.0);
  if (duration_us > 0) {
    cuda_lane_overlap_duration_us_.fetch_add(duration_us,
                                             std::memory_order_relaxed);
  }
}

void MetricsRegistry::SetCudaLaneQueueDepth(bool decode_lane, int depth) {
  if (decode_lane) {
    cuda_decode_lane_queue_depth_.store(depth, std::memory_order_relaxed);
  } else {
    cuda_prefill_lane_queue_depth_.store(depth, std::memory_order_relaxed);
  }
}

void MetricsRegistry::RecordNativeForwardPass(bool is_decode, int batch_size,
                                              double forward_ms) {
  if (is_decode) {
    native_forward_decode_total_.fetch_add(1, std::memory_order_relaxed);
  } else {
    native_forward_prefill_total_.fetch_add(1, std::memory_order_relaxed);
  }
  native_forward_batch_tokens_total_.fetch_add(batch_size,
                                               std::memory_order_relaxed);
  native_forward_latency_.Record(forward_ms);
}

void MetricsRegistry::RecordNativeSampling(int batch_size, double sampling_ms) {
  (void)batch_size;
  native_sampling_latency_.Record(sampling_ms);
}

void MetricsRegistry::RecordNativeBatchDecode(int batch_size, double total_ms) {
  (void)batch_size;
  (void)total_ms;
  // Tracked via forward + sampling histograms; this is for future use.
}

void MetricsRegistry::SetNativeKvCacheOccupancy(int active_sequences,
                                                int max_sequences) {
  native_kv_active_sequences_.store(active_sequences,
                                    std::memory_order_relaxed);
  native_kv_max_sequences_.store(max_sequences, std::memory_order_relaxed);
}

void MetricsRegistry::SetCudaAttentionKernel(const std::string &kernel) {
  std::string normalized = kernel;
  if (normalized != "fa3" && normalized != "fa2" && normalized != "standard") {
    normalized = "standard";
  }
  std::lock_guard<std::mutex> lock(cuda_attention_kernel_mutex_);
  cuda_attention_kernel_ = normalized;
}

void MetricsRegistry::RecordCudaAttentionKernelFallback(
    const std::string &requested_kernel, const std::string &selected_kernel,
    const std::string &reason) {
  const std::string requested =
      requested_kernel.empty() ? "auto" : requested_kernel;
  const std::string selected =
      selected_kernel.empty() ? "standard" : selected_kernel;
  const std::string fallback_reason = reason.empty() ? "unspecified" : reason;
  const std::string key = requested + "|" + selected + "|" + fallback_reason;
  std::lock_guard<std::mutex> lock(cuda_attention_fallback_mutex_);
  cuda_attention_fallback_counts_[key] += 1;
}

void MetricsRegistry::RecordCudaAttentionKernelSwitch(
    const std::string &from_kernel, const std::string &to_kernel) {
  const std::string from = from_kernel.empty() ? "unknown" : from_kernel;
  const std::string to = to_kernel.empty() ? "unknown" : to_kernel;
  if (from == to) {
    return;
  }
  const std::string key = from + "|" + to;
  std::lock_guard<std::mutex> lock(cuda_attention_switch_mutex_);
  cuda_attention_switch_counts_[key] += 1;
}

MetricsRegistry::CacheMetrics MetricsRegistry::GetCacheMetrics() const {
  CacheMetrics cm;
  cm.hits = prefix_hits_.load(std::memory_order_relaxed);
  cm.misses = prefix_misses_.load(std::memory_order_relaxed);
  cm.partial_hits = prefix_partial_hits_.load(std::memory_order_relaxed);
  cm.matched_tokens = prefix_matched_tokens_.load(std::memory_order_relaxed);
  cm.kv_reuse_count = kv_prefix_reuse_count_.load(std::memory_order_relaxed);
  cm.kv_reuse_tokens = kv_prefix_reuse_tokens_.load(std::memory_order_relaxed);
  return cm;
}

std::string MetricsRegistry::RenderPrometheus() const {
  std::string backend;
  {
    std::lock_guard<std::mutex> lock(backend_mutex_);
    backend = backend_;
  }
  std::ostringstream out;

  // --- Counters ---
  out << "# HELP inferflux_requests_total Total successful generation "
         "requests\n";
  out << "# TYPE inferflux_requests_total counter\n";
  out << "inferflux_requests_total{backend=\"" << backend << "\"} "
      << total_requests_.load() << "\n";

  out << "# HELP inferflux_errors_total Total generation errors\n";
  out << "# TYPE inferflux_errors_total counter\n";
  out << "inferflux_errors_total{backend=\"" << backend << "\"} "
      << total_errors_.load() << "\n";

  out << "# HELP inferflux_prompt_tokens_total Total prompt tokens processed\n";
  out << "# TYPE inferflux_prompt_tokens_total counter\n";
  out << "inferflux_prompt_tokens_total{backend=\"" << backend << "\"} "
      << total_prompt_tokens_.load() << "\n";

  out << "# HELP inferflux_completion_tokens_total Total completion tokens "
         "produced\n";
  out << "# TYPE inferflux_completion_tokens_total counter\n";
  out << "inferflux_completion_tokens_total{backend=\"" << backend << "\"} "
      << total_completion_tokens_.load() << "\n";

  out << "# HELP inferflux_spec_chunks_total Total speculative chunks "
         "proposed\n";
  out << "# TYPE inferflux_spec_chunks_total counter\n";
  out << "inferflux_spec_chunks_total{backend=\"" << backend << "\"} "
      << speculative_chunks_total_.load() << "\n";

  out << "# HELP inferflux_spec_chunks_accepted_total Speculative chunks "
         "accepted by validation\n";
  out << "# TYPE inferflux_spec_chunks_accepted_total counter\n";
  out << "inferflux_spec_chunks_accepted_total{backend=\"" << backend << "\"} "
      << speculative_chunks_accepted_.load() << "\n";

  out << "# HELP inferflux_spec_tokens_reused_total Draft tokens reused after "
         "validation\n";
  out << "# TYPE inferflux_spec_tokens_reused_total counter\n";
  out << "inferflux_spec_tokens_reused_total{backend=\"" << backend << "\"} "
      << speculative_tokens_reused_.load() << "\n";

  out << "# HELP inferflux_batches_total Scheduler batches processed\n";
  out << "# TYPE inferflux_batches_total counter\n";
  out << "inferflux_batches_total{backend=\"" << backend << "\"} "
      << total_batches_.load() << "\n";

  out << "# HELP inferflux_batch_tokens_total Total prompt tokens observed in "
         "batches\n";
  out << "# TYPE inferflux_batch_tokens_total counter\n";
  out << "inferflux_batch_tokens_total{backend=\"" << backend << "\"} "
      << total_batch_tokens_.load() << "\n";

  out << "# HELP inferflux_batch_size_max Largest batch size observed\n";
  out << "# TYPE inferflux_batch_size_max gauge\n";
  out << "inferflux_batch_size_max{backend=\"" << backend << "\"} "
      << max_batch_size_.load() << "\n";

  out << "# HELP inferflux_scheduler_iterations_total Scheduler iterations by "
         "phase composition\n";
  out << "# TYPE inferflux_scheduler_iterations_total counter\n";
  out << "inferflux_scheduler_iterations_total{backend=\"" << backend
      << "\",phase=\"prefill\"} " << scheduler_iterations_prefill_.load()
      << "\n";
  out << "inferflux_scheduler_iterations_total{backend=\"" << backend
      << "\",phase=\"decode\"} " << scheduler_iterations_decode_.load() << "\n";
  out << "inferflux_scheduler_iterations_total{backend=\"" << backend
      << "\",phase=\"mixed\"} " << scheduler_iterations_mixed_.load() << "\n";

  out << "# HELP inferflux_scheduler_iteration_requests_total Requests touched "
         "by scheduler iterations\n";
  out << "# TYPE inferflux_scheduler_iteration_requests_total counter\n";
  out << "inferflux_scheduler_iteration_requests_total{backend=\"" << backend
      << "\"} " << scheduler_iteration_requests_total_.load() << "\n";

  out << "# HELP inferflux_scheduler_iteration_tokens_total Estimated token "
         "budget touched by scheduler iterations\n";
  out << "# TYPE inferflux_scheduler_iteration_tokens_total counter\n";
  out << "inferflux_scheduler_iteration_tokens_total{backend=\"" << backend
      << "\"} " << scheduler_iteration_tokens_total_.load() << "\n";

  out << "# HELP inferflux_scheduler_batch_token_budget_skips_total Requests "
         "deferred because adding them would exceed the current batch token "
         "budget\n";
  out << "# TYPE inferflux_scheduler_batch_token_budget_skips_total counter\n";
  out << "inferflux_scheduler_batch_token_budget_skips_total{backend=\""
      << backend << "\"} " << scheduler_batch_token_budget_skips_.load()
      << "\n";

  out << "# HELP inferflux_fairness_preemptions_total Scheduler swaps "
         "triggered by fairness\n";
  out << "# TYPE inferflux_fairness_preemptions_total counter\n";
  out << "inferflux_fairness_preemptions_total{backend=\"" << backend << "\"} "
      << fairness_preemptions_.load() << "\n";

  out << "# HELP inferflux_fairness_yields_total Requests yielded "
         "mid-generation due to fairness timeslices\n";
  out << "# TYPE inferflux_fairness_yields_total counter\n";
  out << "inferflux_fairness_yields_total{backend=\"" << backend << "\"} "
      << fairness_yields_.load() << "\n";

  out << "# HELP inferflux_fairness_resumes_total Yielded requests rescheduled "
         "for completion\n";
  out << "# TYPE inferflux_fairness_resumes_total counter\n";
  out << "inferflux_fairness_resumes_total{backend=\"" << backend << "\"} "
      << fairness_resumes_.load() << "\n";

  out << "# HELP inferflux_fairness_tokens_total Tokens served per priority "
         "level\n";
  out << "# TYPE inferflux_fairness_tokens_total counter\n";
  {
    std::lock_guard<std::mutex> lock(fairness_metrics_mutex_);
    for (const auto &[priority, tokens] : fairness_tokens_) {
      out << "inferflux_fairness_tokens_total{priority=\"" << priority << "\"} "
          << tokens << "\n";
    }
  }

  // Queue wait histogram.
  out << "# HELP inferflux_queue_wait_duration_ms Time requests spend waiting "
         "before execution\n";
  out << "# TYPE inferflux_queue_wait_duration_ms histogram\n";
  for (std::size_t i = 0; i < LatencyHistogram::kBuckets.size(); ++i) {
    out << "inferflux_queue_wait_duration_ms_bucket{backend=\"" << backend
        << "\",le=\"" << std::fixed << std::setprecision(0)
        << LatencyHistogram::kBuckets[i] << "\"} "
        << queue_latency_.counts[i].load() << "\n";
  }
  out << "inferflux_queue_wait_duration_ms_bucket{backend=\"" << backend
      << "\",le=\"+Inf\"} "
      << queue_latency_.counts[LatencyHistogram::kBuckets.size()].load()
      << "\n";
  out << "inferflux_queue_wait_duration_ms_sum{backend=\"" << backend << "\"} "
      << queue_latency_.sum_ms.load() << "\n";
  out << "inferflux_queue_wait_duration_ms_count{backend=\"" << backend
      << "\"} " << queue_latency_.total.load() << "\n";

  out << "# HELP inferflux_batch_exec_duration_ms Time to execute a batch\n";
  out << "# TYPE inferflux_batch_exec_duration_ms histogram\n";
  for (std::size_t i = 0; i < LatencyHistogram::kBuckets.size(); ++i) {
    out << "inferflux_batch_exec_duration_ms_bucket{backend=\"" << backend
        << "\",le=\"" << std::fixed << std::setprecision(0)
        << LatencyHistogram::kBuckets[i] << "\"} "
        << batch_exec_latency_.counts[i].load() << "\n";
  }
  out << "inferflux_batch_exec_duration_ms_bucket{backend=\"" << backend
      << "\",le=\"+Inf\"} "
      << batch_exec_latency_.counts[LatencyHistogram::kBuckets.size()].load()
      << "\n";
  out << "inferflux_batch_exec_duration_ms_sum{backend=\"" << backend << "\"} "
      << batch_exec_latency_.sum_ms.load() << "\n";
  out << "inferflux_batch_exec_duration_ms_count{backend=\"" << backend
      << "\"} " << batch_exec_latency_.total.load() << "\n";

  out << "# HELP inferflux_prefix_hits_total Prefix cache hits\n";
  out << "# TYPE inferflux_prefix_hits_total counter\n";
  out << "inferflux_prefix_hits_total{backend=\"" << backend << "\"} "
      << prefix_hits_.load() << "\n";

  out << "# HELP inferflux_prefix_misses_total Prefix cache misses\n";
  out << "# TYPE inferflux_prefix_misses_total counter\n";
  out << "inferflux_prefix_misses_total{backend=\"" << backend << "\"} "
      << prefix_misses_.load() << "\n";

  out << "# HELP inferflux_prefix_matched_tokens_total Tokens matched in "
         "prefix (partial or exact)\n";
  out << "# TYPE inferflux_prefix_matched_tokens_total counter\n";
  out << "inferflux_prefix_matched_tokens_total{backend=\"" << backend << "\"} "
      << prefix_matched_tokens_.load() << "\n";

  out << "# HELP inferflux_prefix_partial_hits_total Lookups with a partial "
         "prefix match but no exact hit\n";
  out << "# TYPE inferflux_prefix_partial_hits_total counter\n";
  out << "inferflux_prefix_partial_hits_total{backend=\"" << backend << "\"} "
      << prefix_partial_hits_.load() << "\n";

  out << "# HELP inferflux_kv_prefix_reuse_total Requests that reused a warm "
         "KV prefix (skipped full Prefill)\n";
  out << "# TYPE inferflux_kv_prefix_reuse_total counter\n";
  out << "inferflux_kv_prefix_reuse_total " << kv_prefix_reuse_count_.load()
      << "\n";
  out << "# HELP inferflux_kv_prefix_reuse_tokens_total Prompt tokens skipped "
         "via KV prefix reuse\n";
  out << "# TYPE inferflux_kv_prefix_reuse_tokens_total counter\n";
  out << "inferflux_kv_prefix_reuse_tokens_total "
      << kv_prefix_reuse_tokens_.load() << "\n";

  out << "# HELP inferflux_stream_tokens_total Tokens streamed via SSE\n";
  out << "# TYPE inferflux_stream_tokens_total counter\n";
  out << "inferflux_stream_tokens_total{backend=\"" << backend << "\"} "
      << stream_tokens_.load() << "\n";

  out << "# HELP inferflux_stream_cache_hits_total SSE completions served "
         "entirely from cache\n";
  out << "# TYPE inferflux_stream_cache_hits_total counter\n";
  out << "inferflux_stream_cache_hits_total{backend=\"" << backend << "\"} "
      << stream_cache_hits_.load() << "\n";

  out << "# HELP inferflux_capability_rejections_total Requests rejected due "
         "to unsupported backend/model features\n";
  out << "# TYPE inferflux_capability_rejections_total counter\n";
  {
    std::lock_guard<std::mutex> lock(capability_metrics_mutex_);
    for (const auto &[key, count] : capability_rejections_) {
      auto split = key.find('|');
      std::string label_backend =
          split == std::string::npos ? backend : key.substr(0, split);
      std::string label_feature =
          split == std::string::npos ? key : key.substr(split + 1);
      out << "inferflux_capability_rejections_total{backend=\"" << label_backend
          << "\",feature=\"" << label_feature << "\"} " << count << "\n";
    }
  }

  out << "# HELP inferflux_backend_exposures_total Model backend exposure "
         "decisions by requested/selected provider path\n";
  out << "# TYPE inferflux_backend_exposures_total counter\n";
  {
    std::lock_guard<std::mutex> lock(backend_exposure_mutex_);
    for (const auto &[key, count] : backend_exposure_counts_) {
      auto p1 = key.find('|');
      auto p2 =
          (p1 == std::string::npos) ? std::string::npos : key.find('|', p1 + 1);
      auto p3 =
          (p2 == std::string::npos) ? std::string::npos : key.find('|', p2 + 1);
      if (p1 == std::string::npos || p2 == std::string::npos ||
          p3 == std::string::npos) {
        continue;
      }
      const std::string requested = key.substr(0, p1);
      const std::string exposed = key.substr(p1 + 1, p2 - p1 - 1);
      const std::string provider = key.substr(p2 + 1, p3 - p2 - 1);
      const std::string fallback = key.substr(p3 + 1);
      out << "inferflux_backend_exposures_total{requested_backend=\""
          << requested << "\",exposed_backend=\"" << exposed << "\",provider=\""
          << provider << "\",fallback=\"" << fallback << "\"} " << count
          << "\n";
    }
  }

  out << "# HELP inferflux_capability_route_fallbacks_total Requests rerouted "
         "to another backend due to unsupported features or backend "
         "unavailability\n";
  out << "# TYPE inferflux_capability_route_fallbacks_total counter\n";
  {
    std::lock_guard<std::mutex> lock(capability_route_fallback_mutex_);
    for (const auto &[key, count] : capability_route_fallbacks_) {
      auto p1 = key.find('|');
      auto p2 =
          (p1 == std::string::npos) ? std::string::npos : key.find('|', p1 + 1);
      if (p1 == std::string::npos || p2 == std::string::npos) {
        continue;
      }
      const std::string from = key.substr(0, p1);
      const std::string to = key.substr(p1 + 1, p2 - p1 - 1);
      const std::string feature = key.substr(p2 + 1);
      out << "inferflux_capability_route_fallbacks_total{from_backend=\""
          << from << "\",to_backend=\"" << to << "\",feature=\"" << feature
          << "\"} " << count << "\n";
    }
  }

  out << "# HELP inferflux_fairness_preemptions_total Number of fairness "
         "preemptions performed\n";
  out << "# TYPE inferflux_fairness_preemptions_total counter\n";
  out << "inferflux_fairness_preemptions_total{backend=\"" << backend << "\"} "
      << fairness_preemptions_.load() << "\n";

  out << "# HELP inferflux_fairness_tokens_total Tokens served per priority "
         "level\n";
  out << "# TYPE inferflux_fairness_tokens_total counter\n";
  {
    std::lock_guard<std::mutex> lock(fairness_metrics_mutex_);
    for (const auto &[priority, tokens] : fairness_tokens_) {
      out << "inferflux_fairness_tokens_total{backend=\"" << backend
          << "\",priority=\"" << priority << "\"} " << tokens << "\n";
    }
  }

  out << "# HELP inferflux_model_routes_total Total routing decisions per "
         "model\n";
  out << "# TYPE inferflux_model_routes_total counter\n";
  {
    std::lock_guard<std::mutex> lock(model_metrics_mutex_);
    for (const auto &[model, stats] : model_stats_) {
      const std::string &label =
          stats.backend.empty() ? backend : stats.backend;
      out << "inferflux_model_routes_total{model=\"" << model << "\",backend=\""
          << label << "\"} " << stats.routes << "\n";
    }
  }

  out << "# HELP inferflux_model_route_misses_total Requests that could not be "
         "routed to a model\n";
  out << "# TYPE inferflux_model_route_misses_total counter\n";
  out << "inferflux_model_route_misses_total " << model_route_misses_.load()
      << "\n";

  out << "# HELP inferflux_model_load_seconds_total Cumulative time spent "
         "loading models\n";
  out << "# TYPE inferflux_model_load_seconds_total counter\n";
  {
    std::lock_guard<std::mutex> lock(model_metrics_mutex_);
    for (const auto &[model, stats] : model_stats_) {
      const std::string &label =
          stats.backend.empty() ? backend : stats.backend;
      out << "inferflux_model_load_seconds_total{model=\"" << model
          << "\",backend=\"" << label << "\"} " << stats.load_seconds << "\n";
    }
  }

  out << "# HELP inferflux_model_load_events_total Number of times a model was "
         "loaded\n";
  out << "# TYPE inferflux_model_load_events_total counter\n";
  {
    std::lock_guard<std::mutex> lock(model_metrics_mutex_);
    for (const auto &[model, stats] : model_stats_) {
      const std::string &label =
          stats.backend.empty() ? backend : stats.backend;
      out << "inferflux_model_load_events_total{model=\"" << model
          << "\",backend=\"" << label << "\"} " << stats.load_events << "\n";
    }
  }

  out << "# HELP inferflux_model_ready Model readiness (1=ready)\n";
  out << "# TYPE inferflux_model_ready gauge\n";
  {
    std::lock_guard<std::mutex> lock(model_metrics_mutex_);
    for (const auto &[model, stats] : model_stats_) {
      const std::string &label =
          stats.backend.empty() ? backend : stats.backend;
      out << "inferflux_model_ready{model=\"" << model << "\",backend=\""
          << label << "\"} " << (stats.ready ? 1 : 0) << "\n";
    }
  }

  out << "# HELP inferflux_model_prompt_tokens_total Prompt tokens processed "
         "per model (use rate(...) for prompt throughput)\n";
  out << "# TYPE inferflux_model_prompt_tokens_total counter\n";
  {
    std::lock_guard<std::mutex> lock(model_metrics_mutex_);
    for (const auto &[model, stats] : model_stats_) {
      const std::string &label =
          stats.backend.empty() ? backend : stats.backend;
      out << "inferflux_model_prompt_tokens_total{model=\"" << model
          << "\",backend=\"" << label << "\"} " << stats.prompt_tokens << "\n";
    }
  }

  out << "# HELP inferflux_model_completion_tokens_total Completion tokens "
         "produced per model (use rate(...) for generation throughput)\n";
  out << "# TYPE inferflux_model_completion_tokens_total counter\n";
  {
    std::lock_guard<std::mutex> lock(model_metrics_mutex_);
    for (const auto &[model, stats] : model_stats_) {
      const std::string &label =
          stats.backend.empty() ? backend : stats.backend;
      out << "inferflux_model_completion_tokens_total{model=\"" << model
          << "\",backend=\"" << label << "\"} " << stats.completion_tokens
          << "\n";
    }
  }

  // --- Request latency histogram ---
  out << "# HELP inferflux_request_duration_ms Request end-to-end latency in "
         "milliseconds\n";
  out << "# TYPE inferflux_request_duration_ms histogram\n";
  for (std::size_t i = 0; i < LatencyHistogram::kBuckets.size(); ++i) {
    out << "inferflux_request_duration_ms_bucket{backend=\"" << backend
        << "\",le=\"" << std::fixed << std::setprecision(0)
        << LatencyHistogram::kBuckets[i] << "\"} "
        << request_latency_.counts[i].load() << "\n";
  }
  out << "inferflux_request_duration_ms_bucket{backend=\"" << backend
      << "\",le=\"+Inf\"} "
      << request_latency_.counts[LatencyHistogram::kBuckets.size()].load()
      << "\n";
  out << "inferflux_request_duration_ms_sum{backend=\"" << backend << "\"} "
      << request_latency_.sum_ms.load() << "\n";
  out << "inferflux_request_duration_ms_count{backend=\"" << backend << "\"} "
      << request_latency_.total.load() << "\n";

  // --- OBS-2: per-phase latency histograms ---
  out << "# HELP inferflux_prefill_duration_ms Prefill phase latency "
         "(tokenization + prompt eval)\n";
  out << "# TYPE inferflux_prefill_duration_ms histogram\n";
  for (std::size_t i = 0; i < LatencyHistogram::kBuckets.size(); ++i) {
    out << "inferflux_prefill_duration_ms_bucket{backend=\"" << backend
        << "\",le=\"" << std::fixed << std::setprecision(0)
        << LatencyHistogram::kBuckets[i] << "\"} "
        << prefill_latency_.counts[i].load() << "\n";
  }
  out << "inferflux_prefill_duration_ms_bucket{backend=\"" << backend
      << "\",le=\"+Inf\"} "
      << prefill_latency_.counts[LatencyHistogram::kBuckets.size()].load()
      << "\n";
  out << "inferflux_prefill_duration_ms_sum{backend=\"" << backend << "\"} "
      << prefill_latency_.sum_ms.load() << "\n";
  out << "inferflux_prefill_duration_ms_count{backend=\"" << backend << "\"} "
      << prefill_latency_.total.load() << "\n";

  out << "# HELP inferflux_decode_duration_ms Decode phase latency (token "
         "generation)\n";
  out << "# TYPE inferflux_decode_duration_ms histogram\n";
  for (std::size_t i = 0; i < LatencyHistogram::kBuckets.size(); ++i) {
    out << "inferflux_decode_duration_ms_bucket{backend=\"" << backend
        << "\",le=\"" << std::fixed << std::setprecision(0)
        << LatencyHistogram::kBuckets[i] << "\"} "
        << decode_latency_.counts[i].load() << "\n";
  }
  out << "inferflux_decode_duration_ms_bucket{backend=\"" << backend
      << "\",le=\"+Inf\"} "
      << decode_latency_.counts[LatencyHistogram::kBuckets.size()].load()
      << "\n";
  out << "inferflux_decode_duration_ms_sum{backend=\"" << backend << "\"} "
      << decode_latency_.sum_ms.load() << "\n";
  out << "inferflux_decode_duration_ms_count{backend=\"" << backend << "\"} "
      << decode_latency_.total.load() << "\n";

  // --- KV transfer latency (§2.5 item 12) ---
  out << "# HELP inferflux_kv_transfer_duration_ms Prefill-to-decode KV "
         "hand-off latency via KVChannel/ShmKVTransport\n";
  out << "# TYPE inferflux_kv_transfer_duration_ms histogram\n";
  for (std::size_t i = 0; i < LatencyHistogram::kBuckets.size(); ++i) {
    out << "inferflux_kv_transfer_duration_ms_bucket{backend=\"" << backend
        << "\",le=\"" << std::fixed << std::setprecision(0)
        << LatencyHistogram::kBuckets[i] << "\"} "
        << kv_transfer_latency_.counts[i].load() << "\n";
  }
  out << "inferflux_kv_transfer_duration_ms_bucket{backend=\"" << backend
      << "\",le=\"+Inf\"} "
      << kv_transfer_latency_.counts[LatencyHistogram::kBuckets.size()].load()
      << "\n";
  out << "inferflux_kv_transfer_duration_ms_sum{backend=\"" << backend << "\"} "
      << kv_transfer_latency_.sum_ms.load() << "\n";
  out << "inferflux_kv_transfer_duration_ms_count{backend=\"" << backend
      << "\"} " << kv_transfer_latency_.total.load() << "\n";

  // --- GGML-native perf (llama_perf_context) ---
  out << "# HELP inferflux_llama_prefill_ms GGML-native prompt eval latency "
         "(from llama_perf_context)\n";
  out << "# TYPE inferflux_llama_prefill_ms histogram\n";
  for (std::size_t i = 0; i < LatencyHistogram::kBuckets.size(); ++i) {
    out << "inferflux_llama_prefill_ms_bucket{backend=\"" << backend
        << "\",le=\"" << std::fixed << std::setprecision(0)
        << LatencyHistogram::kBuckets[i] << "\"} "
        << llama_prefill_latency_.counts[i].load() << "\n";
  }
  out << "inferflux_llama_prefill_ms_bucket{backend=\"" << backend
      << "\",le=\"+Inf\"} "
      << llama_prefill_latency_.counts[LatencyHistogram::kBuckets.size()].load()
      << "\n";
  out << "inferflux_llama_prefill_ms_sum{backend=\"" << backend << "\"} "
      << llama_prefill_latency_.sum_ms.load() << "\n";
  out << "inferflux_llama_prefill_ms_count{backend=\"" << backend << "\"} "
      << llama_prefill_latency_.total.load() << "\n";

  out << "# HELP inferflux_llama_decode_ms GGML-native token generation "
         "latency "
         "(from llama_perf_context)\n";
  out << "# TYPE inferflux_llama_decode_ms histogram\n";
  for (std::size_t i = 0; i < LatencyHistogram::kBuckets.size(); ++i) {
    out << "inferflux_llama_decode_ms_bucket{backend=\"" << backend
        << "\",le=\"" << std::fixed << std::setprecision(0)
        << LatencyHistogram::kBuckets[i] << "\"} "
        << llama_decode_latency_.counts[i].load() << "\n";
  }
  out << "inferflux_llama_decode_ms_bucket{backend=\"" << backend
      << "\",le=\"+Inf\"} "
      << llama_decode_latency_.counts[LatencyHistogram::kBuckets.size()].load()
      << "\n";
  out << "inferflux_llama_decode_ms_sum{backend=\"" << backend << "\"} "
      << llama_decode_latency_.sum_ms.load() << "\n";
  out << "inferflux_llama_decode_ms_count{backend=\"" << backend << "\"} "
      << llama_decode_latency_.total.load() << "\n";

  out << "# HELP inferflux_llama_prompt_tokens_total Prompt tokens processed "
         "per llama_perf_context\n";
  out << "# TYPE inferflux_llama_prompt_tokens_total counter\n";
  out << "inferflux_llama_prompt_tokens_total{backend=\"" << backend << "\"} "
      << llama_prompt_tokens_.load() << "\n";

  out << "# HELP inferflux_llama_generated_tokens_total Tokens generated per "
         "llama_perf_context\n";
  out << "# TYPE inferflux_llama_generated_tokens_total counter\n";
  out << "inferflux_llama_generated_tokens_total{backend=\"" << backend
      << "\"} " << llama_gen_tokens_.load() << "\n";

  // --- Multimodal (§2.2) ---
  out << "# HELP inferflux_multimodal_images_total Total images preprocessed "
         "from image_url parts\n";
  out << "# TYPE inferflux_multimodal_images_total counter\n";
  out << "inferflux_multimodal_images_total " << multimodal_images_.load()
      << "\n";

  out << "# HELP inferflux_multimodal_requests_total Total requests containing "
         "image_url parts\n";
  out << "# TYPE inferflux_multimodal_requests_total counter\n";
  out << "inferflux_multimodal_requests_total " << multimodal_requests_.load()
      << "\n";

  // --- MoE (§2.6) ---
  out << "# HELP inferflux_moe_requests_total Requests dispatched to MoE "
         "models\n";
  out << "# TYPE inferflux_moe_requests_total counter\n";
  out << "inferflux_moe_requests_total " << moe_requests_.load() << "\n";

  // --- Flash Attention (§2.7) ---
  out << "# HELP inferflux_flash_attention_enabled Flash Attention active for "
         "this instance (0=disabled, 1=enabled)\n";
  out << "# TYPE inferflux_flash_attention_enabled gauge\n";
  out << "inferflux_flash_attention_enabled " << flash_attention_enabled_.load()
      << "\n";

  out << "# HELP inferflux_flash_attention_requests_total Total requests "
         "processed "
         "by Flash Attention kernel type\n";
  out << "# TYPE inferflux_flash_attention_requests_total counter\n";
  out << "inferflux_flash_attention_requests_total{kernel=\"fa2\"} "
      << flash_attention_requests_fa2_.load() << "\n";
  out << "inferflux_flash_attention_requests_total{kernel=\"fa3\"} "
      << flash_attention_requests_fa3_.load() << "\n";
  out << "inferflux_flash_attention_requests_total{kernel=\"standard\"} "
      << flash_attention_requests_standard_.load() << "\n";

  out << "# HELP inferflux_flash_attention_memory_mb Flash Attention KV cache "
         "memory usage in megabytes\n";
  out << "# TYPE inferflux_flash_attention_memory_mb gauge\n";
  out << "inferflux_flash_attention_memory_mb "
      << flash_attention_memory_mb_.load() << "\n";

  out << "# HELP inferflux_flash_attention_execution_ms Flash Attention kernel "
         "execution time in milliseconds\n";
  out << "# TYPE inferflux_flash_attention_execution_ms histogram\n";
  for (std::size_t i = 0; i < LatencyHistogram::kBuckets.size(); ++i) {
    out << "inferflux_flash_attention_execution_ms_bucket{le=\"" << std::fixed
        << std::setprecision(0) << LatencyHistogram::kBuckets[i] << "\"} "
        << flash_attention_exec_latency_.counts[i].load() << "\n";
  }
  out << "inferflux_flash_attention_execution_ms_bucket{le=\"+Inf\"} "
      << flash_attention_exec_latency_.counts[LatencyHistogram::kBuckets.size()]
             .load()
      << "\n";
  out << "inferflux_flash_attention_execution_ms_sum "
      << flash_attention_exec_latency_.sum_ms.load() << "\n";
  out << "inferflux_flash_attention_execution_ms_count "
      << flash_attention_exec_latency_.total.load() << "\n";

  // --- Gauges ---
  out << "# HELP inferflux_active_connections Current number of active HTTP "
         "connections\n";
  out << "# TYPE inferflux_active_connections gauge\n";
  out << "inferflux_active_connections " << active_connections_.load() << "\n";

  out << "# HELP inferflux_scheduler_queue_depth Current number of requests in "
         "the scheduler queue\n";
  out << "# TYPE inferflux_scheduler_queue_depth gauge\n";
  out << "inferflux_scheduler_queue_depth " << queue_depth_.load() << "\n";

  out << "# HELP inferflux_prefill_queue_depth Current number of tickets "
         "waiting in the prefill pool\n";
  out << "# TYPE inferflux_prefill_queue_depth gauge\n";
  out << "inferflux_prefill_queue_depth " << prefill_queue_depth_.load()
      << "\n";

  out << "# HELP inferflux_decode_queue_depth Current number of tickets "
         "waiting in the decode pool\n";
  out << "# TYPE inferflux_decode_queue_depth gauge\n";
  out << "inferflux_decode_queue_depth " << decode_queue_depth_.load() << "\n";

  out << "# HELP inferflux_scheduler_batch_limit_size Configured scheduler "
         "max requests per batch\n";
  out << "# TYPE inferflux_scheduler_batch_limit_size gauge\n";
  out << "inferflux_scheduler_batch_limit_size "
      << scheduler_batch_limit_size_.load() << "\n";

  out << "# HELP inferflux_scheduler_batch_limit_tokens Configured scheduler "
         "max prompt tokens per batch\n";
  out << "# TYPE inferflux_scheduler_batch_limit_tokens gauge\n";
  out << "inferflux_scheduler_batch_limit_tokens "
      << scheduler_batch_limit_tokens_.load() << "\n";

  // --- Native CUDA backend metrics ---
  out << "# HELP inferflux_native_forward_passes_total Native CUDA forward "
         "passes by phase\n";
  out << "# TYPE inferflux_native_forward_passes_total counter\n";
  out << "inferflux_native_forward_passes_total{phase=\"prefill\"} "
      << native_forward_prefill_total_.load() << "\n";
  out << "inferflux_native_forward_passes_total{phase=\"decode\"} "
      << native_forward_decode_total_.load() << "\n";

  out << "# HELP inferflux_native_forward_batch_tokens_total Total tokens "
         "processed by native forward passes\n";
  out << "# TYPE inferflux_native_forward_batch_tokens_total counter\n";
  out << "inferflux_native_forward_batch_tokens_total "
      << native_forward_batch_tokens_total_.load() << "\n";

  out << "# HELP inferflux_native_forward_duration_ms Native forward pass "
         "latency in milliseconds\n";
  out << "# TYPE inferflux_native_forward_duration_ms histogram\n";
  for (std::size_t i = 0; i < LatencyHistogram::kBuckets.size(); ++i) {
    out << "inferflux_native_forward_duration_ms_bucket{le=\"" << std::fixed
        << std::setprecision(0) << LatencyHistogram::kBuckets[i] << "\"} "
        << native_forward_latency_.counts[i].load() << "\n";
  }
  out << "inferflux_native_forward_duration_ms_bucket{le=\"+Inf\"} "
      << native_forward_latency_.counts[LatencyHistogram::kBuckets.size()]
             .load()
      << "\n";
  out << "inferflux_native_forward_duration_ms_sum "
      << native_forward_latency_.sum_ms.load() << "\n";
  out << "inferflux_native_forward_duration_ms_count "
      << native_forward_latency_.total.load() << "\n";

  out << "# HELP inferflux_native_sampling_duration_ms Native sampling latency "
         "in milliseconds\n";
  out << "# TYPE inferflux_native_sampling_duration_ms histogram\n";
  for (std::size_t i = 0; i < LatencyHistogram::kBuckets.size(); ++i) {
    out << "inferflux_native_sampling_duration_ms_bucket{le=\"" << std::fixed
        << std::setprecision(0) << LatencyHistogram::kBuckets[i] << "\"} "
        << native_sampling_latency_.counts[i].load() << "\n";
  }
  out << "inferflux_native_sampling_duration_ms_bucket{le=\"+Inf\"} "
      << native_sampling_latency_.counts[LatencyHistogram::kBuckets.size()]
             .load()
      << "\n";
  out << "inferflux_native_sampling_duration_ms_sum "
      << native_sampling_latency_.sum_ms.load() << "\n";
  out << "inferflux_native_sampling_duration_ms_count "
      << native_sampling_latency_.total.load() << "\n";

  out << "# HELP inferflux_native_kv_active_sequences Active KV cache "
         "sequences in native backend\n";
  out << "# TYPE inferflux_native_kv_active_sequences gauge\n";
  out << "inferflux_native_kv_active_sequences "
      << native_kv_active_sequences_.load() << "\n";

  out << "# HELP inferflux_native_kv_max_sequences Maximum KV cache sequences "
         "in native backend\n";
  out << "# TYPE inferflux_native_kv_max_sequences gauge\n";
  out << "inferflux_native_kv_max_sequences " << native_kv_max_sequences_.load()
      << "\n";

  out << "# HELP inferflux_cuda_lane_submissions_total Unified-batch lane "
         "submissions in CUDA runtime\n";
  out << "# TYPE inferflux_cuda_lane_submissions_total counter\n";
  out << "inferflux_cuda_lane_submissions_total{lane=\"decode\"} "
      << cuda_decode_lane_submissions_.load() << "\n";
  out << "inferflux_cuda_lane_submissions_total{lane=\"prefill\"} "
      << cuda_prefill_lane_submissions_.load() << "\n";

  out << "# HELP inferflux_cuda_lane_completions_total Unified-batch lane "
         "completions in CUDA runtime\n";
  out << "# TYPE inferflux_cuda_lane_completions_total counter\n";
  out << "inferflux_cuda_lane_completions_total{lane=\"decode\"} "
      << cuda_decode_lane_completions_.load() << "\n";
  out << "inferflux_cuda_lane_completions_total{lane=\"prefill\"} "
      << cuda_prefill_lane_completions_.load() << "\n";

  out << "# HELP inferflux_cuda_lane_queue_depth Pending lane queue depth in "
         "CUDA runtime\n";
  out << "# TYPE inferflux_cuda_lane_queue_depth gauge\n";
  out << "inferflux_cuda_lane_queue_depth{lane=\"decode\"} "
      << cuda_decode_lane_queue_depth_.load() << "\n";
  out << "inferflux_cuda_lane_queue_depth{lane=\"prefill\"} "
      << cuda_prefill_lane_queue_depth_.load() << "\n";

  out << "# HELP inferflux_cuda_lane_overlap_events_total CUDA lane execution "
         "windows where decode/prefill overlapped\n";
  out << "# TYPE inferflux_cuda_lane_overlap_events_total counter\n";
  out << "inferflux_cuda_lane_overlap_events_total "
      << cuda_lane_overlap_events_.load() << "\n";

  out << "# HELP inferflux_cuda_lane_overlap_duration_ms_total Total "
         "wall-clock "
         "duration where decode/prefill lanes overlapped\n";
  out << "# TYPE inferflux_cuda_lane_overlap_duration_ms_total counter\n";
  const double overlap_duration_ms =
      static_cast<double>(
          cuda_lane_overlap_duration_us_.load(std::memory_order_relaxed)) /
      1000.0;
  out << "inferflux_cuda_lane_overlap_duration_ms_total " << overlap_duration_ms
      << "\n";

  int overlap_active = 0;
  {
    std::lock_guard<std::mutex> lock(cuda_overlap_timing_mutex_);
    overlap_active = cuda_overlap_active_ ? 1 : 0;
  }
  out << "# HELP inferflux_cuda_lane_overlap_active Whether decode/prefill "
         "lanes are currently overlapping (1=true)\n";
  out << "# TYPE inferflux_cuda_lane_overlap_active gauge\n";
  out << "inferflux_cuda_lane_overlap_active " << overlap_active << "\n";

  out << "# HELP inferflux_cuda_lane_inflight Active CUDA lane workers "
         "currently executing a batch\n";
  out << "# TYPE inferflux_cuda_lane_inflight gauge\n";
  out << "inferflux_cuda_lane_inflight{lane=\"decode\"} "
      << cuda_decode_lane_inflight_.load() << "\n";
  out << "inferflux_cuda_lane_inflight{lane=\"prefill\"} "
      << cuda_prefill_lane_inflight_.load() << "\n";

  out << "# HELP inferflux_cuda_attention_kernel_selected CUDA attention "
         "kernel currently selected (one-hot gauge)\n";
  out << "# TYPE inferflux_cuda_attention_kernel_selected gauge\n";
  std::string selected_kernel = "standard";
  {
    std::lock_guard<std::mutex> lock(cuda_attention_kernel_mutex_);
    selected_kernel = cuda_attention_kernel_;
  }
  for (const char *kernel : {"fa3", "fa2", "standard"}) {
    out << "inferflux_cuda_attention_kernel_selected{kernel=\"" << kernel
        << "\"} " << (selected_kernel == kernel ? 1 : 0) << "\n";
  }

  out << "# HELP inferflux_cuda_attention_kernel_fallbacks_total CUDA "
         "attention kernel fallback decisions by requested/selected kernel\n";
  out << "# TYPE inferflux_cuda_attention_kernel_fallbacks_total counter\n";
  {
    std::lock_guard<std::mutex> lock(cuda_attention_fallback_mutex_);
    for (const auto &[key, count] : cuda_attention_fallback_counts_) {
      auto p1 = key.find('|');
      auto p2 =
          (p1 == std::string::npos) ? std::string::npos : key.find('|', p1 + 1);
      if (p1 == std::string::npos || p2 == std::string::npos) {
        continue;
      }
      const std::string requested = key.substr(0, p1);
      const std::string selected = key.substr(p1 + 1, p2 - p1 - 1);
      const std::string reason = key.substr(p2 + 1);
      out << "inferflux_cuda_attention_kernel_fallbacks_total{requested=\""
          << requested << "\",selected=\"" << selected << "\",reason=\""
          << reason << "\"} " << count << "\n";
    }
  }

  out << "# HELP inferflux_cuda_attention_kernel_switches_total CUDA "
         "attention kernel selection switches across model reloads\n";
  out << "# TYPE inferflux_cuda_attention_kernel_switches_total counter\n";
  {
    std::lock_guard<std::mutex> lock(cuda_attention_switch_mutex_);
    for (const auto &[key, count] : cuda_attention_switch_counts_) {
      auto split = key.find('|');
      if (split == std::string::npos) {
        continue;
      }
      const std::string from = key.substr(0, split);
      const std::string to = key.substr(split + 1);
      out << "inferflux_cuda_attention_kernel_switches_total{from_kernel=\""
          << from << "\",to_kernel=\"" << to << "\"} " << count << "\n";
    }
  }

  return out.str();
}

MetricsRegistry &GlobalMetrics() { return g_metrics; }

} // namespace inferflux
