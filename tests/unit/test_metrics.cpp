#include <catch2/catch_amalgamated.hpp>

#include "server/metrics/metrics.h"

#include <string>

TEST_CASE("MetricsRegistry default backend is cpu", "[metrics]") {
  inferflux::MetricsRegistry registry;
  auto output = registry.RenderPrometheus();
  REQUIRE(output.find("backend=\"cpu\"") != std::string::npos);
}

TEST_CASE("MetricsRegistry records successes", "[metrics]") {
  inferflux::MetricsRegistry registry;
  registry.RecordSuccess(10, 20);
  registry.RecordSuccess(5, 15);

  auto output = registry.RenderPrometheus();
  REQUIRE(output.find("inferflux_requests_total{backend=\"cpu\"} 2") !=
          std::string::npos);
  REQUIRE(output.find("inferflux_prompt_tokens_total{backend=\"cpu\"} 15") !=
          std::string::npos);
  REQUIRE(
      output.find("inferflux_completion_tokens_total{backend=\"cpu\"} 35") !=
      std::string::npos);
}

TEST_CASE("MetricsRegistry records errors", "[metrics]") {
  inferflux::MetricsRegistry registry;
  registry.RecordError();
  registry.RecordError();
  registry.RecordError();

  auto output = registry.RenderPrometheus();
  REQUIRE(output.find("inferflux_errors_total{backend=\"cpu\"} 3") !=
          std::string::npos);
}

TEST_CASE("MetricsRegistry records speculative stats", "[metrics]") {
  inferflux::MetricsRegistry registry;
  registry.RecordSpeculative(4, 3, 6);

  auto output = registry.RenderPrometheus();
  REQUIRE(output.find("inferflux_spec_chunks_total{backend=\"cpu\"} 4") !=
          std::string::npos);
  REQUIRE(
      output.find("inferflux_spec_chunks_accepted_total{backend=\"cpu\"} 3") !=
      std::string::npos);
  REQUIRE(
      output.find("inferflux_spec_tokens_reused_total{backend=\"cpu\"} 6") !=
      std::string::npos);
}

TEST_CASE("MetricsRegistry skips zero speculative stats", "[metrics]") {
  inferflux::MetricsRegistry registry;
  registry.RecordSpeculative(0, 0, 0);

  auto output = registry.RenderPrometheus();
  REQUIRE(output.find("inferflux_spec_chunks_total{backend=\"cpu\"} 0") !=
          std::string::npos);
}

TEST_CASE("MetricsRegistry SetBackend changes label", "[metrics]") {
  inferflux::MetricsRegistry registry;
  registry.SetBackend("cuda");

  auto output = registry.RenderPrometheus();
  REQUIRE(output.find("backend=\"cuda\"") != std::string::npos);
  REQUIRE(output.find("backend=\"cpu\"") == std::string::npos);
}

TEST_CASE("MetricsRegistry latency histogram records buckets", "[metrics]") {
  inferflux::MetricsRegistry registry;
  // Record 80ms — should fall in the 100ms bucket (≤100) and all larger ones.
  registry.RecordLatency(80.0);

  auto output = registry.RenderPrometheus();
  REQUIRE(output.find("inferflux_request_duration_ms_bucket") !=
          std::string::npos);
  REQUIRE(
      output.find("inferflux_request_duration_ms_count{backend=\"cpu\"} 1") !=
      std::string::npos);
  // 80ms is above 50ms bucket but within 100ms bucket.
  REQUIRE(output.find("le=\"50\"} 0") != std::string::npos);
  REQUIRE(output.find("le=\"100\"} 1") != std::string::npos);
  REQUIRE(output.find("le=\"+Inf\"} 1") != std::string::npos);
}

TEST_CASE("MetricsRegistry active connections gauge", "[metrics]") {
  inferflux::MetricsRegistry registry;
  registry.IncrementConnections();
  registry.IncrementConnections();
  registry.DecrementConnections();

  auto output = registry.RenderPrometheus();
  REQUIRE(output.find("inferflux_active_connections 1") != std::string::npos);
}

TEST_CASE("MetricsRegistry queue depth gauge", "[metrics]") {
  inferflux::MetricsRegistry registry;
  registry.SetQueueDepth(7);

  auto output = registry.RenderPrometheus();
  REQUIRE(output.find("inferflux_scheduler_queue_depth 7") !=
          std::string::npos);
}

TEST_CASE("MetricsRegistry prefill/decode queue gauges", "[metrics]") {
  inferflux::MetricsRegistry registry;
  registry.SetPrefillQueueDepth(3);
  registry.SetDecodeQueueDepth(2);

  auto output = registry.RenderPrometheus();
  REQUIRE(output.find("inferflux_prefill_queue_depth 3") != std::string::npos);
  REQUIRE(output.find("inferflux_decode_queue_depth 2") != std::string::npos);
}

TEST_CASE("MetricsRegistry records per-model token counters", "[metrics]") {
  inferflux::MetricsRegistry registry;
  registry.RecordModelRoute("llama3-8b", "cuda", true);
  registry.RecordModelTokens("llama3-8b", "cuda", 12, 34);
  registry.RecordModelTokens("llama3-8b", "", 3, 6);

  auto output = registry.RenderPrometheus();
  REQUIRE(output.find("inferflux_model_prompt_tokens_total{model=\"llama3-8b\","
                      "backend=\"cuda\"} 15") != std::string::npos);
  REQUIRE(output.find("inferflux_model_completion_tokens_total{model=\"llama3-"
                      "8b\",backend=\"cuda\"} 40") != std::string::npos);
}

TEST_CASE("MetricsRegistry per-model counters fall back to instance backend",
          "[metrics]") {
  inferflux::MetricsRegistry registry;
  registry.SetBackend("mps");
  registry.RecordModelTokens("qwen2.5-7b", "", 5, 9);

  auto output = registry.RenderPrometheus();
  REQUIRE(output.find("inferflux_model_prompt_tokens_total{model=\"qwen2.5-"
                      "7b\",backend=\"mps\"} 5") != std::string::npos);
  REQUIRE(output.find("inferflux_model_completion_tokens_total{model=\"qwen2.5-"
                      "7b\",backend=\"mps\"} 9") != std::string::npos);
}

TEST_CASE("MetricsRegistry records capability rejection counters",
          "[metrics]") {
  inferflux::MetricsRegistry registry;
  registry.RecordCapabilityRejection("cuda", "vision");
  registry.RecordCapabilityRejection("cuda", "vision");
  registry.RecordCapabilityRejection("cpu", "structured_output");

  auto output = registry.RenderPrometheus();
  REQUIRE(output.find("inferflux_capability_rejections_total{backend=\"cuda\","
                      "feature=\"vision\"} 2") != std::string::npos);
  REQUIRE(output.find("inferflux_capability_rejections_total{backend=\"cpu\","
                      "feature=\"structured_output\"} 1") != std::string::npos);
}

TEST_CASE("MetricsRegistry records backend exposure counters", "[metrics]") {
  inferflux::MetricsRegistry registry;
  registry.RecordBackendExposure("cuda", "cuda", "universal", false);
  registry.RecordBackendExposure("cuda", "cpu", "universal", true);
  registry.RecordBackendExposure("cuda", "cpu", "universal", true);

  auto output = registry.RenderPrometheus();
  REQUIRE(output.find("inferflux_backend_exposures_total{requested_backend="
                      "\"cuda\",exposed_backend=\"cuda\",provider="
                      "\"universal\",fallback=\"false\"} 1") !=
          std::string::npos);
  REQUIRE(output.find("inferflux_backend_exposures_total{requested_backend="
                      "\"cuda\",exposed_backend=\"cpu\",provider=\"universal\","
                      "fallback=\"true\"} 2") != std::string::npos);
}

TEST_CASE("MetricsRegistry records capability route fallback counters",
          "[metrics]") {
  inferflux::MetricsRegistry registry;
  registry.RecordCapabilityRouteFallback("cuda", "cpu", "logprobs");
  registry.RecordCapabilityRouteFallback("cuda", "cpu", "logprobs");
  registry.RecordCapabilityRouteFallback("rocm", "cpu", "backend_unavailable");

  auto output = registry.RenderPrometheus();
  REQUIRE(output.find("inferflux_capability_route_fallbacks_total{"
                      "from_backend=\"cuda\",to_backend=\"cpu\","
                      "feature=\"logprobs\"} 2") != std::string::npos);
  REQUIRE(output.find("inferflux_capability_route_fallbacks_total{"
                      "from_backend=\"rocm\",to_backend=\"cpu\","
                      "feature=\"backend_unavailable\"} 1") !=
          std::string::npos);
}
