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
  REQUIRE(output.find("inferflux_requests_total{backend=\"cpu\"} 2") != std::string::npos);
  REQUIRE(output.find("inferflux_prompt_tokens_total{backend=\"cpu\"} 15") != std::string::npos);
  REQUIRE(output.find("inferflux_completion_tokens_total{backend=\"cpu\"} 35") != std::string::npos);
}

TEST_CASE("MetricsRegistry records errors", "[metrics]") {
  inferflux::MetricsRegistry registry;
  registry.RecordError();
  registry.RecordError();
  registry.RecordError();

  auto output = registry.RenderPrometheus();
  REQUIRE(output.find("inferflux_errors_total{backend=\"cpu\"} 3") != std::string::npos);
}

TEST_CASE("MetricsRegistry records speculative stats", "[metrics]") {
  inferflux::MetricsRegistry registry;
  registry.RecordSpeculative(4, 3, 6);

  auto output = registry.RenderPrometheus();
  REQUIRE(output.find("inferflux_spec_chunks_total{backend=\"cpu\"} 4") != std::string::npos);
  REQUIRE(output.find("inferflux_spec_chunks_accepted_total{backend=\"cpu\"} 3") != std::string::npos);
  REQUIRE(output.find("inferflux_spec_tokens_reused_total{backend=\"cpu\"} 6") != std::string::npos);
}

TEST_CASE("MetricsRegistry skips zero speculative stats", "[metrics]") {
  inferflux::MetricsRegistry registry;
  registry.RecordSpeculative(0, 0, 0);

  auto output = registry.RenderPrometheus();
  REQUIRE(output.find("inferflux_spec_chunks_total{backend=\"cpu\"} 0") != std::string::npos);
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
  REQUIRE(output.find("inferflux_request_duration_ms_bucket") != std::string::npos);
  REQUIRE(output.find("inferflux_request_duration_ms_count{backend=\"cpu\"} 1") != std::string::npos);
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
  REQUIRE(output.find("inferflux_scheduler_queue_depth 7") != std::string::npos);
}
