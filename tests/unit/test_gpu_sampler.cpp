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

  float *d_logits;
  cudaMalloc(&d_logits, vocab_size * sizeof(float));
  cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float),
             cudaMemcpyHostToDevice);

  // Greedy (temperature=0) should return 42
  int token = sampler.Sample(d_logits, 0.0f, 0, 1.0f);
  REQUIRE(token == 42);

  cudaFree(d_logits);
  cudaStreamDestroy(stream);
}

TEST_CASE("GpuSampler: Batched greedy argmax", "[gpu_sampler][cuda]") {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    SKIP("No CUDA device available");
  }

  int vocab_size = 100;
  int batch_size = 4;
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  GpuSampler sampler;
  REQUIRE(sampler.Initialize(vocab_size, stream));

  // Create batched logits with known maxima at different positions
  std::vector<float> h_logits(batch_size * vocab_size, 0.0f);
  h_logits[0 * vocab_size + 10] = 10.0f; // seq 0 -> token 10
  h_logits[1 * vocab_size + 42] = 10.0f; // seq 1 -> token 42
  h_logits[2 * vocab_size + 0] = 10.0f;  // seq 2 -> token 0
  h_logits[3 * vocab_size + 99] = 10.0f; // seq 3 -> token 99

  float *d_logits;
  cudaMalloc(&d_logits, batch_size * vocab_size * sizeof(float));
  cudaMemcpy(d_logits, h_logits.data(), batch_size * vocab_size * sizeof(float),
             cudaMemcpyHostToDevice);

  // All greedy temperatures -> should use batched kernel
  std::vector<float> temps(batch_size, 0.0f);
  std::vector<int> top_ks(batch_size, 0);
  std::vector<float> top_ps(batch_size, 1.0f);
  std::vector<uint32_t> seeds(batch_size, UINT32_MAX);
  std::vector<int> out_tokens;

  sampler.SampleBatch(d_logits, batch_size, temps, top_ks, top_ps, seeds,
                      &out_tokens);

  REQUIRE(out_tokens.size() == 4);
  REQUIRE(out_tokens[0] == 10);
  REQUIRE(out_tokens[1] == 42);
  REQUIRE(out_tokens[2] == 0);
  REQUIRE(out_tokens[3] == 99);

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

  float *d_logits;
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

TEST_CASE("GpuSampler: Batched stochastic sampling preserves per-sequence seeds",
          "[gpu_sampler][cuda]") {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    SKIP("No CUDA device available");
  }

  constexpr int vocab_size = 64;
  constexpr int batch_size = 2;
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  GpuSampler batch_sampler;
  GpuSampler single_sampler;
  REQUIRE(batch_sampler.Initialize(vocab_size, stream));
  REQUIRE(single_sampler.Initialize(vocab_size, stream));

  std::vector<float> h_logits(batch_size * vocab_size, 0.0f);
  for (int b = 0; b < batch_size; ++b) {
    for (int i = 0; i < vocab_size; ++i) {
      h_logits[b * vocab_size + i] =
          0.01f * static_cast<float>((b + 1) * ((i % 7) - 3));
    }
  }

  float *d_logits = nullptr;
  cudaMalloc(&d_logits, batch_size * vocab_size * sizeof(float));
  cudaMemcpy(d_logits, h_logits.data(), batch_size * vocab_size * sizeof(float),
             cudaMemcpyHostToDevice);

  std::vector<float> temps(batch_size, 1.0f);
  std::vector<int> top_ks(batch_size, 0);
  std::vector<float> top_ps(batch_size, 1.0f);
  const std::vector<uint32_t> seeds = {123u, 98765u};
  std::vector<int> batch_tokens;

  batch_sampler.SampleBatch(d_logits, batch_size, temps, top_ks, top_ps, seeds,
                            &batch_tokens);

  REQUIRE(batch_tokens.size() == batch_size);
  for (int i = 0; i < batch_size; ++i) {
    const int single =
        single_sampler.Sample(d_logits + i * vocab_size, temps[i], top_ks[i],
                              top_ps[i], seeds[i]);
    REQUIRE(batch_tokens[i] == single);
  }

  cudaFree(d_logits);
  cudaStreamDestroy(stream);
}
#endif

} // namespace inferflux
