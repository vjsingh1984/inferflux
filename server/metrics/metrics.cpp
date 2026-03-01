#include "server/metrics/metrics.h"

#include <sstream>
#include <cmath>
#include <iomanip>

namespace inferflux {

namespace {
MetricsRegistry g_metrics;
}  // namespace

void LatencyHistogram::Record(double ms) {
  total.fetch_add(1, std::memory_order_relaxed);
  sum_ms.fetch_add(static_cast<uint64_t>(std::max(0.0, ms)), std::memory_order_relaxed);
  // All buckets are cumulative: increment every bucket >= ms.
  for (std::size_t i = 0; i < kBuckets.size(); ++i) {
    if (ms <= kBuckets[i]) {
      counts[i].fetch_add(1, std::memory_order_relaxed);
    }
  }
  // +Inf bucket always increments.
  counts[kBuckets.size()].fetch_add(1, std::memory_order_relaxed);
}

void MetricsRegistry::SetBackend(const std::string& backend) {
  std::lock_guard<std::mutex> lock(backend_mutex_);
  backend_ = backend;
}

void MetricsRegistry::RecordSuccess(int prompt_tokens, int completion_tokens) {
  total_requests_.fetch_add(1, std::memory_order_relaxed);
  total_prompt_tokens_.fetch_add(prompt_tokens, std::memory_order_relaxed);
  total_completion_tokens_.fetch_add(completion_tokens, std::memory_order_relaxed);
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
  speculative_chunks_accepted_.fetch_add(accepted_chunks, std::memory_order_relaxed);
  speculative_tokens_reused_.fetch_add(reused_tokens, std::memory_order_relaxed);
}

void MetricsRegistry::RecordBatch(std::size_t request_count, std::size_t token_count) {
  if (request_count == 0) {
    return;
  }
  total_batches_.fetch_add(1, std::memory_order_relaxed);
  total_batch_tokens_.fetch_add(token_count, std::memory_order_relaxed);
  uint64_t current_max = max_batch_size_.load(std::memory_order_relaxed);
  while (request_count > current_max &&
         !max_batch_size_.compare_exchange_weak(current_max, request_count, std::memory_order_relaxed)) {
  }
}

void MetricsRegistry::RecordPrefixLookup(bool hit) {
  if (hit) {
    prefix_hits_.fetch_add(1, std::memory_order_relaxed);
  } else {
    prefix_misses_.fetch_add(1, std::memory_order_relaxed);
  }
}

void MetricsRegistry::RecordPrefixMatchedTokens(int tokens) {
  if (tokens <= 0) return;
  prefix_matched_tokens_.fetch_add(static_cast<uint64_t>(tokens), std::memory_order_relaxed);
}

void MetricsRegistry::RecordPartialPrefixHit() {
  prefix_partial_hits_.fetch_add(1, std::memory_order_relaxed);
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

void MetricsRegistry::RecordFairnessTokens(int priority_level, std::size_t tokens) {
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

void MetricsRegistry::RecordModelLoad(const std::string& model_id,
                                      const std::string& backend,
                                      double load_seconds) {
  if (model_id.empty()) {
    return;
  }
  std::lock_guard<std::mutex> lock(model_metrics_mutex_);
  auto& stats = model_stats_[model_id];
  if (!backend.empty()) {
    stats.backend = backend;
  }
  stats.load_seconds += std::max(0.0, load_seconds);
  stats.load_events += 1;
}

void MetricsRegistry::RecordModelReady(const std::string& model_id,
                                       const std::string& backend,
                                       bool ready) {
  if (model_id.empty()) {
    return;
  }
  std::lock_guard<std::mutex> lock(model_metrics_mutex_);
  auto& stats = model_stats_[model_id];
  if (!backend.empty()) {
    stats.backend = backend;
  }
  stats.ready = ready;
}

void MetricsRegistry::RecordModelRoute(const std::string& model_id,
                                       const std::string& backend,
                                       bool hit) {
  if (!hit || model_id.empty()) {
    model_route_misses_.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  std::lock_guard<std::mutex> lock(model_metrics_mutex_);
  auto& stats = model_stats_[model_id];
  if (!backend.empty()) {
    stats.backend = backend;
  }
  stats.routes += 1;
}

void MetricsRegistry::RecordImagePreprocess(int images, double /*decode_ms*/) {
  if (images > 0) {
    multimodal_images_.fetch_add(static_cast<uint64_t>(images), std::memory_order_relaxed);
    multimodal_requests_.fetch_add(1, std::memory_order_relaxed);
  }
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

std::string MetricsRegistry::RenderPrometheus() const {
  std::string backend;
  {
    std::lock_guard<std::mutex> lock(backend_mutex_);
    backend = backend_;
  }
  std::ostringstream out;

  // --- Counters ---
  out << "# HELP inferflux_requests_total Total successful generation requests\n";
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

  out << "# HELP inferflux_completion_tokens_total Total completion tokens produced\n";
  out << "# TYPE inferflux_completion_tokens_total counter\n";
  out << "inferflux_completion_tokens_total{backend=\"" << backend << "\"} "
      << total_completion_tokens_.load() << "\n";

  out << "# HELP inferflux_spec_chunks_total Total speculative chunks proposed\n";
  out << "# TYPE inferflux_spec_chunks_total counter\n";
  out << "inferflux_spec_chunks_total{backend=\"" << backend << "\"} "
      << speculative_chunks_total_.load() << "\n";

  out << "# HELP inferflux_spec_chunks_accepted_total Speculative chunks accepted by validation\n";
  out << "# TYPE inferflux_spec_chunks_accepted_total counter\n";
  out << "inferflux_spec_chunks_accepted_total{backend=\"" << backend << "\"} "
      << speculative_chunks_accepted_.load() << "\n";

  out << "# HELP inferflux_spec_tokens_reused_total Draft tokens reused after validation\n";
  out << "# TYPE inferflux_spec_tokens_reused_total counter\n";
  out << "inferflux_spec_tokens_reused_total{backend=\"" << backend << "\"} "
      << speculative_tokens_reused_.load() << "\n";

  out << "# HELP inferflux_batches_total Scheduler batches processed\n";
  out << "# TYPE inferflux_batches_total counter\n";
  out << "inferflux_batches_total{backend=\"" << backend << "\"} "
      << total_batches_.load() << "\n";

  out << "# HELP inferflux_batch_tokens_total Total prompt tokens observed in batches\n";
  out << "# TYPE inferflux_batch_tokens_total counter\n";
  out << "inferflux_batch_tokens_total{backend=\"" << backend << "\"} "
      << total_batch_tokens_.load() << "\n";

  out << "# HELP inferflux_batch_size_max Largest batch size observed\n";
  out << "# TYPE inferflux_batch_size_max gauge\n";
  out << "inferflux_batch_size_max{backend=\"" << backend << "\"} "
      << max_batch_size_.load() << "\n";

  out << "# HELP inferflux_fairness_preemptions_total Scheduler swaps triggered by fairness\n";
  out << "# TYPE inferflux_fairness_preemptions_total counter\n";
  out << "inferflux_fairness_preemptions_total{backend=\"" << backend << "\"} "
      << fairness_preemptions_.load() << "\n";

  out << "# HELP inferflux_fairness_yields_total Requests yielded mid-generation due to fairness timeslices\n";
  out << "# TYPE inferflux_fairness_yields_total counter\n";
  out << "inferflux_fairness_yields_total{backend=\"" << backend << "\"} "
      << fairness_yields_.load() << "\n";

  out << "# HELP inferflux_fairness_resumes_total Yielded requests rescheduled for completion\n";
  out << "# TYPE inferflux_fairness_resumes_total counter\n";
  out << "inferflux_fairness_resumes_total{backend=\"" << backend << "\"} "
      << fairness_resumes_.load() << "\n";

  out << "# HELP inferflux_fairness_tokens_total Tokens served per priority level\n";
  out << "# TYPE inferflux_fairness_tokens_total counter\n";
  {
    std::lock_guard<std::mutex> lock(fairness_metrics_mutex_);
    for (const auto& [priority, tokens] : fairness_tokens_) {
      out << "inferflux_fairness_tokens_total{priority=\"" << priority << "\"} "
          << tokens << "\n";
    }
  }

  // Queue wait histogram.
  out << "# HELP inferflux_queue_wait_duration_ms Time requests spend waiting before execution\n";
  out << "# TYPE inferflux_queue_wait_duration_ms histogram\n";
  for (std::size_t i = 0; i < LatencyHistogram::kBuckets.size(); ++i) {
    out << "inferflux_queue_wait_duration_ms_bucket{backend=\"" << backend
        << "\",le=\"" << std::fixed << std::setprecision(0) << LatencyHistogram::kBuckets[i]
        << "\"} " << queue_latency_.counts[i].load() << "\n";
  }
  out << "inferflux_queue_wait_duration_ms_bucket{backend=\"" << backend
      << "\",le=\"+Inf\"} " << queue_latency_.counts[LatencyHistogram::kBuckets.size()].load() << "\n";
  out << "inferflux_queue_wait_duration_ms_sum{backend=\"" << backend << "\"} "
      << queue_latency_.sum_ms.load() << "\n";
  out << "inferflux_queue_wait_duration_ms_count{backend=\"" << backend << "\"} "
      << queue_latency_.total.load() << "\n";

  out << "# HELP inferflux_batch_exec_duration_ms Time to execute a batch\n";
  out << "# TYPE inferflux_batch_exec_duration_ms histogram\n";
  for (std::size_t i = 0; i < LatencyHistogram::kBuckets.size(); ++i) {
    out << "inferflux_batch_exec_duration_ms_bucket{backend=\"" << backend
        << "\",le=\"" << std::fixed << std::setprecision(0) << LatencyHistogram::kBuckets[i]
        << "\"} " << batch_exec_latency_.counts[i].load() << "\n";
  }
  out << "inferflux_batch_exec_duration_ms_bucket{backend=\"" << backend
      << "\",le=\"+Inf\"} " << batch_exec_latency_.counts[LatencyHistogram::kBuckets.size()].load() << "\n";
  out << "inferflux_batch_exec_duration_ms_sum{backend=\"" << backend << "\"} "
      << batch_exec_latency_.sum_ms.load() << "\n";
  out << "inferflux_batch_exec_duration_ms_count{backend=\"" << backend << "\"} "
      << batch_exec_latency_.total.load() << "\n";

  out << "# HELP inferflux_prefix_hits_total Prefix cache hits\n";
  out << "# TYPE inferflux_prefix_hits_total counter\n";
  out << "inferflux_prefix_hits_total{backend=\"" << backend << "\"} "
      << prefix_hits_.load() << "\n";

  out << "# HELP inferflux_prefix_misses_total Prefix cache misses\n";
  out << "# TYPE inferflux_prefix_misses_total counter\n";
  out << "inferflux_prefix_misses_total{backend=\"" << backend << "\"} "
      << prefix_misses_.load() << "\n";

  out << "# HELP inferflux_prefix_matched_tokens_total Tokens matched in prefix (partial or exact)\n";
  out << "# TYPE inferflux_prefix_matched_tokens_total counter\n";
  out << "inferflux_prefix_matched_tokens_total{backend=\"" << backend << "\"} "
      << prefix_matched_tokens_.load() << "\n";

  out << "# HELP inferflux_prefix_partial_hits_total Lookups with a partial prefix match but no exact hit\n";
  out << "# TYPE inferflux_prefix_partial_hits_total counter\n";
  out << "inferflux_prefix_partial_hits_total{backend=\"" << backend << "\"} "
      << prefix_partial_hits_.load() << "\n";

  out << "# HELP inferflux_stream_tokens_total Tokens streamed via SSE\n";
  out << "# TYPE inferflux_stream_tokens_total counter\n";
  out << "inferflux_stream_tokens_total{backend=\"" << backend << "\"} "
      << stream_tokens_.load() << "\n";

  out << "# HELP inferflux_stream_cache_hits_total SSE completions served entirely from cache\n";
  out << "# TYPE inferflux_stream_cache_hits_total counter\n";
  out << "inferflux_stream_cache_hits_total{backend=\"" << backend << "\"} "
      << stream_cache_hits_.load() << "\n";

  out << "# HELP inferflux_fairness_preemptions_total Number of fairness preemptions performed\n";
  out << "# TYPE inferflux_fairness_preemptions_total counter\n";
  out << "inferflux_fairness_preemptions_total{backend=\"" << backend << "\"} "
      << fairness_preemptions_.load() << "\n";

  out << "# HELP inferflux_fairness_tokens_total Tokens served per priority level\n";
  out << "# TYPE inferflux_fairness_tokens_total counter\n";
  {
    std::lock_guard<std::mutex> lock(fairness_metrics_mutex_);
    for (const auto& [priority, tokens] : fairness_tokens_) {
      out << "inferflux_fairness_tokens_total{backend=\"" << backend
          << "\",priority=\"" << priority << "\"} "
          << tokens << "\n";
    }
  }

  out << "# HELP inferflux_model_routes_total Total routing decisions per model\n";
  out << "# TYPE inferflux_model_routes_total counter\n";
  {
    std::lock_guard<std::mutex> lock(model_metrics_mutex_);
    for (const auto& [model, stats] : model_stats_) {
      const std::string& label = stats.backend.empty() ? backend : stats.backend;
      out << "inferflux_model_routes_total{model=\"" << model << "\",backend=\""
          << label << "\"} " << stats.routes << "\n";
    }
  }

  out << "# HELP inferflux_model_route_misses_total Requests that could not be routed to a model\n";
  out << "# TYPE inferflux_model_route_misses_total counter\n";
  out << "inferflux_model_route_misses_total "
      << model_route_misses_.load() << "\n";

  out << "# HELP inferflux_model_load_seconds_total Cumulative time spent loading models\n";
  out << "# TYPE inferflux_model_load_seconds_total counter\n";
  {
    std::lock_guard<std::mutex> lock(model_metrics_mutex_);
    for (const auto& [model, stats] : model_stats_) {
      const std::string& label = stats.backend.empty() ? backend : stats.backend;
      out << "inferflux_model_load_seconds_total{model=\"" << model << "\",backend=\""
          << label << "\"} " << stats.load_seconds << "\n";
    }
  }

  out << "# HELP inferflux_model_load_events_total Number of times a model was loaded\n";
  out << "# TYPE inferflux_model_load_events_total counter\n";
  {
    std::lock_guard<std::mutex> lock(model_metrics_mutex_);
    for (const auto& [model, stats] : model_stats_) {
      const std::string& label = stats.backend.empty() ? backend : stats.backend;
      out << "inferflux_model_load_events_total{model=\"" << model << "\",backend=\""
          << label << "\"} " << stats.load_events << "\n";
    }
  }

  out << "# HELP inferflux_model_ready Model readiness (1=ready)\n";
  out << "# TYPE inferflux_model_ready gauge\n";
  {
    std::lock_guard<std::mutex> lock(model_metrics_mutex_);
    for (const auto& [model, stats] : model_stats_) {
      const std::string& label = stats.backend.empty() ? backend : stats.backend;
      out << "inferflux_model_ready{model=\"" << model << "\",backend=\""
          << label << "\"} " << (stats.ready ? 1 : 0) << "\n";
    }
  }

  // --- Request latency histogram ---
  out << "# HELP inferflux_request_duration_ms Request end-to-end latency in milliseconds\n";
  out << "# TYPE inferflux_request_duration_ms histogram\n";
  for (std::size_t i = 0; i < LatencyHistogram::kBuckets.size(); ++i) {
    out << "inferflux_request_duration_ms_bucket{backend=\"" << backend
        << "\",le=\"" << std::fixed << std::setprecision(0)
        << LatencyHistogram::kBuckets[i] << "\"} "
        << request_latency_.counts[i].load() << "\n";
  }
  out << "inferflux_request_duration_ms_bucket{backend=\"" << backend
      << "\",le=\"+Inf\"} " << request_latency_.counts[LatencyHistogram::kBuckets.size()].load() << "\n";
  out << "inferflux_request_duration_ms_sum{backend=\"" << backend << "\"} "
      << request_latency_.sum_ms.load() << "\n";
  out << "inferflux_request_duration_ms_count{backend=\"" << backend << "\"} "
      << request_latency_.total.load() << "\n";

  // --- OBS-2: per-phase latency histograms ---
  out << "# HELP inferflux_prefill_duration_ms Prefill phase latency (tokenization + prompt eval)\n";
  out << "# TYPE inferflux_prefill_duration_ms histogram\n";
  for (std::size_t i = 0; i < LatencyHistogram::kBuckets.size(); ++i) {
    out << "inferflux_prefill_duration_ms_bucket{backend=\"" << backend
        << "\",le=\"" << std::fixed << std::setprecision(0)
        << LatencyHistogram::kBuckets[i] << "\"} "
        << prefill_latency_.counts[i].load() << "\n";
  }
  out << "inferflux_prefill_duration_ms_bucket{backend=\"" << backend
      << "\",le=\"+Inf\"} " << prefill_latency_.counts[LatencyHistogram::kBuckets.size()].load() << "\n";
  out << "inferflux_prefill_duration_ms_sum{backend=\"" << backend << "\"} "
      << prefill_latency_.sum_ms.load() << "\n";
  out << "inferflux_prefill_duration_ms_count{backend=\"" << backend << "\"} "
      << prefill_latency_.total.load() << "\n";

  out << "# HELP inferflux_decode_duration_ms Decode phase latency (token generation)\n";
  out << "# TYPE inferflux_decode_duration_ms histogram\n";
  for (std::size_t i = 0; i < LatencyHistogram::kBuckets.size(); ++i) {
    out << "inferflux_decode_duration_ms_bucket{backend=\"" << backend
        << "\",le=\"" << std::fixed << std::setprecision(0)
        << LatencyHistogram::kBuckets[i] << "\"} "
        << decode_latency_.counts[i].load() << "\n";
  }
  out << "inferflux_decode_duration_ms_bucket{backend=\"" << backend
      << "\",le=\"+Inf\"} " << decode_latency_.counts[LatencyHistogram::kBuckets.size()].load() << "\n";
  out << "inferflux_decode_duration_ms_sum{backend=\"" << backend << "\"} "
      << decode_latency_.sum_ms.load() << "\n";
  out << "inferflux_decode_duration_ms_count{backend=\"" << backend << "\"} "
      << decode_latency_.total.load() << "\n";

  // --- Multimodal (§2.2) ---
  out << "# HELP inferflux_multimodal_images_total Total images preprocessed from image_url parts\n";
  out << "# TYPE inferflux_multimodal_images_total counter\n";
  out << "inferflux_multimodal_images_total " << multimodal_images_.load() << "\n";

  out << "# HELP inferflux_multimodal_requests_total Total requests containing image_url parts\n";
  out << "# TYPE inferflux_multimodal_requests_total counter\n";
  out << "inferflux_multimodal_requests_total " << multimodal_requests_.load() << "\n";

  // --- Gauges ---
  out << "# HELP inferflux_active_connections Current number of active HTTP connections\n";
  out << "# TYPE inferflux_active_connections gauge\n";
  out << "inferflux_active_connections " << active_connections_.load() << "\n";

  out << "# HELP inferflux_scheduler_queue_depth Current number of requests in the scheduler queue\n";
  out << "# TYPE inferflux_scheduler_queue_depth gauge\n";
  out << "inferflux_scheduler_queue_depth " << queue_depth_.load() << "\n";

  out << "# HELP inferflux_prefill_queue_depth Current number of tickets waiting in the prefill pool\n";
  out << "# TYPE inferflux_prefill_queue_depth gauge\n";
  out << "inferflux_prefill_queue_depth " << prefill_queue_depth_.load() << "\n";

  out << "# HELP inferflux_decode_queue_depth Current number of tickets waiting in the decode pool\n";
  out << "# TYPE inferflux_decode_queue_depth gauge\n";
  out << "inferflux_decode_queue_depth " << decode_queue_depth_.load() << "\n";

  return out.str();
}

MetricsRegistry& GlobalMetrics() { return g_metrics; }

}  // namespace inferflux
