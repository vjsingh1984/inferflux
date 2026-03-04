#include <catch2/catch_amalgamated.hpp>

#ifdef INFERFLUX_NATIVE_KERNELS_READY
#include "runtime/backends/cuda/native/kv_cache_gpu.h"
#endif

namespace inferflux {

// GPU KV Cache tests require CUDA hardware.
// Tagged [gpu_kv_cache][cuda] so they can be skipped on CI without GPUs.

TEST_CASE("KvCacheGpu: interface compiles", "[gpu_kv_cache]") {
#ifdef INFERFLUX_NATIVE_KERNELS_READY
  // Verify the class interface compiles correctly
  KvCacheGpu cache;
  REQUIRE(cache.GetMemoryUsage() == 0);
  REQUIRE(cache.MaxSeqLen() == 0);
  REQUIRE(cache.MaxBatchSize() == 0);
#else
  REQUIRE(true);  // Placeholder
#endif
}

#ifdef INFERFLUX_NATIVE_KERNELS_READY
TEST_CASE("KvCacheGpu: Allocate and access", "[gpu_kv_cache][cuda]") {
  // This test requires a CUDA GPU
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    SKIP("No CUDA device available");
  }

  KvCacheGpu cache;
  int num_layers = 2;
  int num_kv_heads = 4;
  int head_dim = 64;
  int max_seq = 128;
  int max_batch = 4;

  REQUIRE(cache.Allocate(num_layers, num_kv_heads, head_dim, max_seq,
                         max_batch));
  REQUIRE(cache.GetMemoryUsage() > 0);
  REQUIRE(cache.MaxSeqLen() == max_seq);
  REQUIRE(cache.MaxBatchSize() == max_batch);

  // Verify K/V pointers are non-null and distinct
  half* k0 = cache.GetK(0, 0);
  half* v0 = cache.GetV(0, 0);
  REQUIRE(k0 != nullptr);
  REQUIRE(v0 != nullptr);
  REQUIRE(k0 != v0);

  // Different layers should have different pointers
  half* k1 = cache.GetK(1, 0);
  REQUIRE(k1 != k0);

  // Different batch slots should have different pointers
  half* k0_b1 = cache.GetK(0, 1);
  REQUIRE(k0_b1 != k0);

  // Clear should not crash
  cache.ClearSequence(0);
}
#endif

}  // namespace inferflux
