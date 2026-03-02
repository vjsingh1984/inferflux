#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

namespace inferflux {

// Latency histogram with fixed buckets (in milliseconds).
// Prometheus-compatible: cumulative counts per bucket + _sum + _count.
struct LatencyHistogram {
  // Upper bounds in milliseconds: 10, 50, 100, 250, 500, 1000, 2500, 5000, +Inf
  static constexpr std::array<double, 8> kBuckets{
      10.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0};
  // Cumulative bucket counts (bucket[i] = requests finishing within kBuckets[i]
  // ms).
  std::array<std::atomic<uint64_t>, 9> counts{}; // 8 finite + 1 +Inf
  std::atomic<uint64_t> sum_ms{0};
  std::atomic<uint64_t> total{0};

  void Record(double ms);
};

class MetricsRegistry {
public:
  void SetBackend(const std::string &backend);

  // Counters (existing).
  void RecordSuccess(int prompt_tokens, int completion_tokens);
  void RecordError();
  void RecordSpeculative(std::size_t total_chunks, std::size_t accepted_chunks,
                         std::size_t reused_tokens);
  void RecordBatch(std::size_t request_count, std::size_t token_count);
  void RecordPrefixLookup(bool hit);
  void RecordPrefixMatchedTokens(int tokens);
  void RecordPartialPrefixHit();
  // KV prefix reuse (§ Item 5): called when CopySequencePrefix+PrefillPartial
  // replaces a full Prefill.  tokens_saved = number of prefix tokens skipped.
  void RecordKVPrefixReuse(int tokens_saved);
  void RecordStreamTokens(std::size_t tokens);
  void RecordStreamCacheHit();
  void RecordFairnessTokens(int priority_level, std::size_t tokens);
  void RecordFairnessPreemption(int priority_level);
  void RecordFairnessYield(int priority_level, std::size_t emitted_tokens,
                           std::size_t remaining_tokens);
  void RecordFairnessResume(int priority_level);
  void RecordQueueLatency(double wait_ms);
  void RecordBatchExecution(double exec_ms);
  void RecordModelLoad(const std::string &model_id, const std::string &backend,
                       double load_seconds);
  void RecordModelReady(const std::string &model_id, const std::string &backend,
                        bool ready);
  void RecordModelRoute(const std::string &model_id, const std::string &backend,
                        bool hit);
  // Per-model token counters for throughput dashboards.
  // Exported as counters so Prometheus can derive tokens/sec via rate().
  void RecordModelTokens(const std::string &model_id,
                         const std::string &backend, int prompt_tokens,
                         int completion_tokens);
  void RecordCapabilityRejection(const std::string &backend,
                                 const std::string &feature);
  void RecordBackendExposure(const std::string &requested_backend,
                             const std::string &exposed_backend,
                             const std::string &provider, bool used_fallback);
  void RecordCapabilityRouteFallback(const std::string &from_backend,
                                     const std::string &to_backend,
                                     const std::string &feature);

  // Multimodal (§2.2): record image preprocessing events.
  // images: number of images decoded; decode_ms: preprocessing wall-clock time.
  void RecordImagePreprocess(int images, double decode_ms);

  // MoE (§2.6): record a request dispatched to a MoE model.
  void RecordMoERequest();

  // Flash Attention (§2.7): set the FA enabled gauge (0=disabled, 1=enabled).
  // Called once at server startup after LlamaBackendConfig is applied.
  void SetFlashAttentionEnabled(bool enabled);

  // KV transfer latency (§2.5 item 12): elapsed time from
  // KVPacket::enqueue_time to when a decode worker dequeues it from KVChannel /
  // ShmKVTransport.
  void RecordKVTransfer(double transfer_ms);

  // GGML-native perf counters (from llama_perf_context). Exposes ground-truth
  // kernel timings that subprocess wrappers cannot surface.
  void RecordLlamaPerf(double prefill_ms, double decode_ms,
                       int32_t prompt_tokens, int32_t generated_tokens);

  // Latency recording — call with full request duration in milliseconds.
  void RecordLatency(double request_ms);

  // Per-phase latency recording (OBS-2).
  // prefill_ms: time from tokenization start to first token produced.
  // decode_ms:  time spent generating completion tokens.
  void RecordPrefillDuration(double prefill_ms);
  void RecordDecodeDuration(double decode_ms);

  // Gauge helpers — active connections and scheduler queue depth.
  void IncrementConnections();
  void DecrementConnections();
  void SetQueueDepth(int depth);
  void SetPrefillQueueDepth(int depth);
  void SetDecodeQueueDepth(int depth);

  // Snapshot of prefix-cache metrics for the /v1/admin/cache endpoint.
  struct CacheMetrics {
    uint64_t hits{0};
    uint64_t misses{0};
    uint64_t partial_hits{0};
    uint64_t matched_tokens{0};
    uint64_t kv_reuse_count{0};
    uint64_t kv_reuse_tokens{0};
  };
  CacheMetrics GetCacheMetrics() const;

  std::string RenderPrometheus() const;

private:
  mutable std::mutex backend_mutex_;
  std::string backend_{"cpu"};

  // Counters.
  std::atomic<uint64_t> total_requests_{0};
  std::atomic<uint64_t> total_errors_{0};
  std::atomic<uint64_t> total_prompt_tokens_{0};
  std::atomic<uint64_t> total_completion_tokens_{0};
  std::atomic<uint64_t> speculative_chunks_total_{0};
  std::atomic<uint64_t> speculative_chunks_accepted_{0};
  std::atomic<uint64_t> speculative_tokens_reused_{0};
  std::atomic<uint64_t> total_batches_{0};
  std::atomic<uint64_t> total_batch_tokens_{0};
  std::atomic<uint64_t> max_batch_size_{0};
  std::atomic<uint64_t> prefix_hits_{0};
  std::atomic<uint64_t> prefix_misses_{0};
  std::atomic<uint64_t> prefix_matched_tokens_{0};
  std::atomic<uint64_t> prefix_partial_hits_{0};
  std::atomic<uint64_t> kv_prefix_reuse_count_{0};
  std::atomic<uint64_t> kv_prefix_reuse_tokens_{0};
  std::atomic<uint64_t> stream_tokens_{0};
  std::atomic<uint64_t> stream_cache_hits_{0};
  std::atomic<uint64_t> fairness_preemptions_{0};
  std::atomic<uint64_t> fairness_yields_{0};
  std::atomic<uint64_t> fairness_resumes_{0};
  mutable std::mutex fairness_metrics_mutex_;
  std::unordered_map<int, uint64_t> fairness_tokens_;
  std::atomic<uint64_t> model_route_misses_{0};
  std::atomic<uint64_t> multimodal_images_{
      0}; // §2.2: total images preprocessed.
  std::atomic<uint64_t> multimodal_requests_{
      0}; // §2.2: total requests with images.
  std::atomic<uint64_t> moe_requests_{
      0}; // §2.6: requests routed to MoE models.
  std::atomic<uint64_t> flash_attention_enabled_{
      0}; // §2.7: gauge — 0=disabled, 1=enabled.

  // Latency histograms.
  LatencyHistogram request_latency_;
  LatencyHistogram queue_latency_;
  LatencyHistogram batch_exec_latency_;
  LatencyHistogram prefill_latency_; // OBS-2: prefill phase
  LatencyHistogram decode_latency_;  // OBS-2: decode phase
  LatencyHistogram
      kv_transfer_latency_; // §2.5 item 12: prefill→decode KV hand-off
  LatencyHistogram llama_prefill_latency_; // GGML-native prompt eval latency
  LatencyHistogram llama_decode_latency_;  // GGML-native token gen latency
  std::atomic<uint64_t> llama_prompt_tokens_{0};
  std::atomic<uint64_t> llama_gen_tokens_{0};

  // Gauges.
  std::atomic<int> active_connections_{0};
  std::atomic<int> queue_depth_{0};
  std::atomic<int> prefill_queue_depth_{0};
  std::atomic<int> decode_queue_depth_{0};

  struct ModelStats {
    std::string backend;
    double load_seconds{0.0};
    uint64_t load_events{0};
    uint64_t routes{0};
    uint64_t prompt_tokens{0};
    uint64_t completion_tokens{0};
    bool ready{false};
  };

  mutable std::mutex model_metrics_mutex_;
  std::unordered_map<std::string, ModelStats> model_stats_;
  mutable std::mutex capability_metrics_mutex_;
  std::unordered_map<std::string, uint64_t> capability_rejections_;
  mutable std::mutex backend_exposure_mutex_;
  std::unordered_map<std::string, uint64_t> backend_exposure_counts_;
  mutable std::mutex capability_route_fallback_mutex_;
  std::unordered_map<std::string, uint64_t> capability_route_fallbacks_;
};

MetricsRegistry &GlobalMetrics();

} // namespace inferflux
