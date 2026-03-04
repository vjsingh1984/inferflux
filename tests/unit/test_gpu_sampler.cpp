#include <catch2/catch_amalgamated.hpp>

#ifdef INFERFLUX_NATIVE_KERNELS_READY
#include "runtime/backends/cuda/native/gpu_sampler.h"
#include <cuda_runtime.h>
#include <vector>
#endif

namespace inferflux {

TEST_CASE("GpuSampler: interface compiles", "[gpu_sampler]") {
#ifdef INFERFLUX_NATIVE_KERNELS_READY
  // Verify the class interface compiles
  GpuSampler sampler;
  // Without initialization, we can't call Sample
  REQUIRE(true);
#else
  REQUIRE(true);
#endif
}

#ifdef INFERFLUX_NATIVE_KERNELS_READY
TEST_CASE("GpuSampler: Greedy argmax", "[gpu_sampler][cuda]") {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    SKIP("No CUDA device available");
  }

  int vocab_size = 100;
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  GpuSampler sampler;
  REQUIRE(sampler.Initialize(vocab_size, stream));

  // Create logits with a known maximum at position 42
  std::vector<float> h_logits(vocab_size, 0.0f);
  h_logits[42] = 10.0f;

  float* d_logits;
  cudaMalloc(&d_logits, vocab_size * sizeof(float));
  cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float),
             cudaMemcpyHostToDevice);

  // Greedy (temperature=0) should return 42
  int token = sampler.Sample(d_logits, 0.0f, 0, 1.0f);
  REQUIRE(token == 42);

  cudaFree(d_logits);
  cudaStreamDestroy(stream);
}

TEST_CASE("GpuSampler: Stochastic sample returns valid token",
          "[gpu_sampler][cuda]") {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    SKIP("No CUDA device available");
  }

  int vocab_size = 50;
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  GpuSampler sampler;
  REQUIRE(sampler.Initialize(vocab_size, stream));

  // Uniform logits
  std::vector<float> h_logits(vocab_size, 1.0f);

  float* d_logits;
  cudaMalloc(&d_logits, vocab_size * sizeof(float));
  cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float),
             cudaMemcpyHostToDevice);

  // Stochastic sample should return a valid token
  int token = sampler.Sample(d_logits, 1.0f, 0, 1.0f, 42);
  REQUIRE(token >= 0);
  REQUIRE(token < vocab_size);

  cudaFree(d_logits);
  cudaStreamDestroy(stream);
}
#endif

}  // namespace inferflux
