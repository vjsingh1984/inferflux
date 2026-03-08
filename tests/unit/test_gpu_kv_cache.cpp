#include <catch2/catch_amalgamated.hpp>
#include <vector>

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
  REQUIRE(true); // Placeholder
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

  REQUIRE(
      cache.Allocate(num_layers, num_kv_heads, head_dim, max_seq, max_batch));
  REQUIRE(cache.GetMemoryUsage() > 0);
  REQUIRE(cache.MaxSeqLen() == max_seq);
  REQUIRE(cache.MaxBatchSize() == max_batch);

  // Verify K/V pointers are non-null and distinct
  half *k0 = cache.GetK(0, 0);
  half *v0 = cache.GetV(0, 0);
  REQUIRE(k0 != nullptr);
  REQUIRE(v0 != nullptr);
  REQUIRE(k0 != v0);

  // Different layers should have different pointers
  half *k1 = cache.GetK(1, 0);
  REQUIRE(k1 != k0);

  // Different batch slots should have different pointers
  half *k0_b1 = cache.GetK(0, 1);
  REQUIRE(k0_b1 != k0);

  // Clear should not crash
  cache.ClearSequence(0);
}

TEST_CASE("KvCacheGpu: copy prefix + serialize hydrate",
          "[gpu_kv_cache][cuda]") {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    SKIP("No CUDA device available");
  }

  KvCacheGpu cache;
  REQUIRE(cache.Allocate(/*num_layers=*/1, /*num_kv_heads=*/1, /*head_dim=*/8,
                         /*max_seq=*/8, /*max_batch=*/2));

  cudaStream_t stream = nullptr;
  REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

  // Write deterministic source prefix values for K/V.
  constexpr int kPrefixTokens = 4;
  constexpr int kKvDim = 8;
  std::vector<half> k_src(static_cast<size_t>(kPrefixTokens * kKvDim));
  std::vector<half> v_src(static_cast<size_t>(kPrefixTokens * kKvDim));
  for (size_t i = 0; i < k_src.size(); ++i) {
    k_src[i] = __float2half(static_cast<float>(i + 1));
    v_src[i] = __float2half(static_cast<float>((i + 1) * 2));
  }

  REQUIRE(cudaMemcpyAsync(cache.GetK(0, 0), k_src.data(),
                          k_src.size() * sizeof(half), cudaMemcpyHostToDevice,
                          stream) == cudaSuccess);
  REQUIRE(cudaMemcpyAsync(cache.GetV(0, 0), v_src.data(),
                          v_src.size() * sizeof(half), cudaMemcpyHostToDevice,
                          stream) == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

  REQUIRE(cache.CopySequencePrefix(/*src_seq=*/0, /*dst_seq=*/1, kPrefixTokens,
                                   stream));

  std::vector<half> k_dst(k_src.size());
  std::vector<half> v_dst(v_src.size());
  REQUIRE(cudaMemcpy(k_dst.data(), cache.GetK(0, 1),
                     k_dst.size() * sizeof(half),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  REQUIRE(cudaMemcpy(v_dst.data(), cache.GetV(0, 1),
                     v_dst.size() * sizeof(half),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  for (size_t i = 0; i < k_src.size(); ++i) {
    REQUIRE(__half2float(k_dst[i]) == Catch::Approx(__half2float(k_src[i])));
    REQUIRE(__half2float(v_dst[i]) == Catch::Approx(__half2float(v_src[i])));
  }

  std::vector<uint8_t> blob;
  REQUIRE(cache.SerializeSequence(/*seq_id=*/0, &blob));
  REQUIRE_FALSE(blob.empty());

  cache.ClearSequence(/*seq_id=*/1);
  REQUIRE(cache.HydrateSequence(/*seq_id=*/1, blob, stream));

  std::vector<half> k_hydrated(k_src.size());
  std::vector<half> v_hydrated(v_src.size());
  REQUIRE(cudaMemcpy(k_hydrated.data(), cache.GetK(0, 1),
                     k_hydrated.size() * sizeof(half),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  REQUIRE(cudaMemcpy(v_hydrated.data(), cache.GetV(0, 1),
                     v_hydrated.size() * sizeof(half),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  for (size_t i = 0; i < k_src.size(); ++i) {
    REQUIRE(__half2float(k_hydrated[i]) ==
            Catch::Approx(__half2float(k_src[i])));
    REQUIRE(__half2float(v_hydrated[i]) ==
            Catch::Approx(__half2float(v_src[i])));
  }

  REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
}
#endif

} // namespace inferflux
