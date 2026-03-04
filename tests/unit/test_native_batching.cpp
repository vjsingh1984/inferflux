#include <catch2/catch_amalgamated.hpp>

#include "server/metrics/metrics.h"

// Test NVTX macros compile to no-ops when CUDA is disabled
#ifndef INFERFLUX_HAS_CUDA
#include "runtime/backends/cuda/native/nvtx_scoped.h"
#endif

TEST_CASE("Native metrics record forward passes", "[native_batch]") {
  inferflux::MetricsRegistry registry;

  registry.RecordNativeForwardPass(/*is_decode=*/true, /*batch_size=*/4,
                                   /*forward_ms=*/12.5);
  registry.RecordNativeForwardPass(/*is_decode=*/false, /*batch_size=*/128,
                                   /*forward_ms=*/45.0);

  auto output = registry.RenderPrometheus();
  REQUIRE(output.find("inferflux_native_forward_passes_total{phase=\"decode\"} "
                       "1") != std::string::npos);
  REQUIRE(
      output.find(
          "inferflux_native_forward_passes_total{phase=\"prefill\"} 1") !=
      std::string::npos);
  REQUIRE(output.find("inferflux_native_forward_batch_tokens_total 132") !=
          std::string::npos);
}

TEST_CASE("Native metrics record sampling", "[native_batch]") {
  inferflux::MetricsRegistry registry;

  registry.RecordNativeSampling(/*batch_size=*/4, /*sampling_ms=*/0.5);
  registry.RecordNativeSampling(/*batch_size=*/1, /*sampling_ms=*/0.3);

  auto output = registry.RenderPrometheus();
  REQUIRE(output.find("inferflux_native_sampling_duration_ms_count 2") !=
          std::string::npos);
}

TEST_CASE("Native KV cache occupancy gauges", "[native_batch]") {
  inferflux::MetricsRegistry registry;

  registry.SetNativeKvCacheOccupancy(/*active=*/5, /*max=*/32);

  auto output = registry.RenderPrometheus();
  REQUIRE(output.find("inferflux_native_kv_active_sequences 5") !=
          std::string::npos);
  REQUIRE(output.find("inferflux_native_kv_max_sequences 32") !=
          std::string::npos);
}

TEST_CASE("Native forward pass histogram buckets", "[native_batch]") {
  inferflux::MetricsRegistry registry;

  // Record a 5ms forward (should go in 10ms bucket)
  registry.RecordNativeForwardPass(true, 1, 5.0);

  auto output = registry.RenderPrometheus();
  // Should be in the 10ms bucket
  REQUIRE(output.find("inferflux_native_forward_duration_ms_bucket{le=\"10\"} "
                       "1") != std::string::npos);
  // And in +Inf
  REQUIRE(output.find(
              "inferflux_native_forward_duration_ms_bucket{le=\"+Inf\"} 1") !=
          std::string::npos);
}

#ifndef INFERFLUX_HAS_CUDA
TEST_CASE("NVTX macros are no-ops without CUDA", "[native_batch]") {
  // Just verify compilation succeeds
  NVTX_SCOPE("TestScope");
  REQUIRE(true);
}
#endif
