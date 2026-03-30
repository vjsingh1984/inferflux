/**
 * Parity tests for CUDA fused kernels.
 *
 * Test 1: FusedResidualAddRmsNorm vs unfused ResidualAdd + RmsNorm
 * Test 2: FusedBiasAddTriple vs three separate BiasAdd calls
 * Test 3: Fused softmax vs 4-kernel softmax pipeline
 *
 * All tests allocate small buffers (hidden_size=128) to keep GPU memory
 * footprint minimal and run time short.
 */

#ifdef INFERFLUX_HAS_CUDA

#include <catch2/catch_amalgamated.hpp>

#include "runtime/backends/cuda/native/cuda_kernels.cuh"

// NOTE: The following headers are included AFTER Catch2 to avoid nvcc CUDA
// 13.2 operator== resolution conflicts between cuda_fp16.hpp/cuda_bf16.hpp
// operator overloads and Catch2's operator!= templates. Headers that
// transitively pull in <string> (via native_execution_policy.h) or
// cuda_bf16.hpp (via dtype_traits.cuh) trigger this issue.
// Fused kernel parity tests that need these headers are validated via
// integration tests with the corresponding env var flags enabled.

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <random>
#include <vector>

// ============================================================================
// Helpers
// ============================================================================

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      FAIL("CUDA error at " << __FILE__ << ":" << __LINE__ << ": "            \
                             << cudaGetErrorString(err__));                     \
    }                                                                          \
  } while (0)

namespace {

/// Fill a host vector with random half values in [-1, 1].
void FillRandomHalf(std::vector<half> &buf, std::mt19937 &rng) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto &v : buf) {
    v = __float2half(dist(rng));
  }
}

/// Fill a host vector with random half values in [lo, hi].
void FillRandomHalfRange(std::vector<half> &buf, std::mt19937 &rng, float lo,
                         float hi) {
  std::uniform_real_distribution<float> dist(lo, hi);
  for (auto &v : buf) {
    v = __float2half(dist(rng));
  }
}

/// Fill a host vector with random floats in [lo, hi].
void FillRandomFloat(std::vector<float> &buf, std::mt19937 &rng, float lo,
                     float hi) {
  std::uniform_real_distribution<float> dist(lo, hi);
  for (auto &v : buf) {
    v = dist(rng);
  }
}

/// Compare two device half buffers, returning max absolute difference.
float MaxAbsDiffHalf(const half *d_a, const half *d_b, int count) {
  std::vector<half> ha(count), hb(count);
  cudaMemcpy(ha.data(), d_a, count * sizeof(half), cudaMemcpyDeviceToHost);
  cudaMemcpy(hb.data(), d_b, count * sizeof(half), cudaMemcpyDeviceToHost);
  float max_diff = 0.0f;
  for (int i = 0; i < count; ++i) {
    float diff =
        std::abs(__half2float(ha[i]) - __half2float(hb[i]));
    max_diff = std::max(max_diff, diff);
  }
  return max_diff;
}

/// Compare two device float buffers, returning max absolute difference.
float MaxAbsDiffFloat(const float *d_a, const float *d_b, int count) {
  std::vector<float> ha(count), hb(count);
  cudaMemcpy(ha.data(), d_a, count * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(hb.data(), d_b, count * sizeof(float), cudaMemcpyDeviceToHost);
  float max_diff = 0.0f;
  for (int i = 0; i < count; ++i) {
    float diff = std::abs(ha[i] - hb[i]);
    max_diff = std::max(max_diff, diff);
  }
  return max_diff;
}

/// Count CUDA graph nodes by capturing a lambda into a graph.
template <typename F>
int CountGraphNodes(cudaStream_t stream, F &&kernel_fn) {
  cudaGraph_t graph;
  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  kernel_fn();
  cudaStreamEndCapture(stream, &graph);
  size_t num_nodes = 0;
  cudaGraphGetNodes(graph, nullptr, &num_nodes);
  cudaGraphDestroy(graph);
  return static_cast<int>(num_nodes);
}

/// RAII wrapper for device memory.
template <typename T> struct DeviceBuf {
  T *ptr = nullptr;
  int count = 0;

  DeviceBuf() = default;
  explicit DeviceBuf(int n) : count(n) {
    cudaMalloc(&ptr, static_cast<size_t>(n) * sizeof(T));
  }
  ~DeviceBuf() {
    if (ptr)
      cudaFree(ptr);
  }

  DeviceBuf(const DeviceBuf &) = delete;
  DeviceBuf &operator=(const DeviceBuf &) = delete;

  void Upload(const std::vector<T> &src) {
    cudaMemcpy(ptr, src.data(), static_cast<size_t>(count) * sizeof(T),
               cudaMemcpyHostToDevice);
  }

  void Download(std::vector<T> &dst) const {
    dst.resize(count);
    cudaMemcpy(dst.data(), ptr, static_cast<size_t>(count) * sizeof(T),
               cudaMemcpyDeviceToHost);
  }

  void Zero() {
    cudaMemset(ptr, 0, static_cast<size_t>(count) * sizeof(T));
  }

  /// Copy contents from another device buffer of the same size.
  void CopyFrom(const DeviceBuf &other) {
    cudaMemcpy(ptr, other.ptr, static_cast<size_t>(count) * sizeof(T),
               cudaMemcpyDeviceToDevice);
  }
};

} // anonymous namespace

// ============================================================================
// Local softmax kernels for parity test (mirrors gpu_sampler.cu internals)
//
// These are declared in an anonymous namespace to avoid linker collisions
// with the production kernels in gpu_sampler.cu, which are file-static
// (internal linkage) and inaccessible from this TU.
// ============================================================================

namespace {

__global__ void TestTemperatureScaleKernel(float *__restrict__ logits,
                                           int vocab_size, float temperature) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= vocab_size)
    return;
  logits[idx] /= temperature;
}

__global__ void TestSoftmaxMaxKernel(const float *__restrict__ logits,
                                     float *__restrict__ max_val,
                                     int vocab_size) {
  extern __shared__ float smem[];
  int tid = threadIdx.x;
  float local_max = -FLT_MAX;
  for (int i = tid; i < vocab_size; i += blockDim.x) {
    local_max = fmaxf(local_max, logits[i]);
  }
  smem[tid] = local_max;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)
      smem[tid] = fmaxf(smem[tid], smem[tid + s]);
    __syncthreads();
  }
  if (tid == 0)
    *max_val = smem[0];
}

__global__ void TestSoftmaxExpSumKernel(const float *__restrict__ logits,
                                        float *__restrict__ probs,
                                        const float *__restrict__ max_val,
                                        float *__restrict__ sum_val,
                                        int vocab_size) {
  extern __shared__ float smem[];
  int tid = threadIdx.x;
  float m = *max_val;
  float local_sum = 0.0f;
  for (int i = tid; i < vocab_size; i += blockDim.x) {
    float e = expf(logits[i] - m);
    probs[i] = e;
    local_sum += e;
  }
  smem[tid] = local_sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)
      smem[tid] += smem[tid + s];
    __syncthreads();
  }
  if (tid == 0)
    *sum_val = smem[0];
}

__global__ void TestSoftmaxNormKernel(float *__restrict__ probs,
                                      const float *__restrict__ sum_val,
                                      int vocab_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= vocab_size)
    return;
  probs[idx] /= *sum_val;
}

/// Fused softmax: temperature + max + exp-sum + normalize in a single launch.
/// Single block, shared memory reduction. Suitable for vocab_size <= ~32K.
__global__ void TestFusedSoftmaxKernel(const float *__restrict__ logits,
                                       float *__restrict__ probs,
                                       int vocab_size, float temperature) {
  extern __shared__ float smem[];
  int tid = threadIdx.x;

  // Pass 1: temperature-scale and find max
  float local_max = -FLT_MAX;
  for (int i = tid; i < vocab_size; i += blockDim.x) {
    float v = logits[i] / temperature;
    probs[i] = v; // stash scaled logit
    local_max = fmaxf(local_max, v);
  }
  smem[tid] = local_max;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)
      smem[tid] = fmaxf(smem[tid], smem[tid + s]);
    __syncthreads();
  }
  float gmax = smem[0];
  __syncthreads();

  // Pass 2: exp and sum
  float local_sum = 0.0f;
  for (int i = tid; i < vocab_size; i += blockDim.x) {
    float e = expf(probs[i] - gmax);
    probs[i] = e;
    local_sum += e;
  }
  smem[tid] = local_sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)
      smem[tid] += smem[tid + s];
    __syncthreads();
  }
  float gsum = smem[0];
  __syncthreads();

  // Pass 3: normalize
  for (int i = tid; i < vocab_size; i += blockDim.x) {
    probs[i] /= gsum;
  }
}

} // anonymous namespace

namespace inferflux {

// ============================================================================
// Test 1: FusedResidualAddRmsNorm parity
// ============================================================================

TEST_CASE("FusedResidualAddRmsNorm matches unfused ResidualAdd + RmsNorm",
          "[cuda][fused_kernels]") {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    SKIP("No CUDA device available");
  }

  constexpr int kHiddenSize = 128;
  constexpr int kRows = 4;
  constexpr int kCount = kRows * kHiddenSize;
  constexpr float kEps = 1e-5f;
  constexpr float kTolerance = 0.002f; // FP16 epsilon

  std::mt19937 rng(42);
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Host data
  std::vector<half> h_residual(kCount);
  std::vector<half> h_input(kCount);
  std::vector<half> h_weight(kHiddenSize);

  FillRandomHalf(h_residual, rng);
  FillRandomHalf(h_input, rng);
  FillRandomHalfRange(h_weight, rng, 0.5f, 1.5f);

  // Device buffers — two copies of residual (unfused path modifies in-place)
  DeviceBuf<half> d_residual_unfused(kCount);
  DeviceBuf<half> d_residual_fused(kCount);
  DeviceBuf<half> d_input(kCount);
  DeviceBuf<half> d_weight(kHiddenSize);
  DeviceBuf<half> d_output_unfused(kCount);
  DeviceBuf<half> d_output_fused(kCount);

  d_residual_unfused.Upload(h_residual);
  d_residual_fused.Upload(h_residual);
  d_input.Upload(h_input);
  d_weight.Upload(h_weight);

  // -- Unfused path: ResidualAdd + RmsNorm separately --
  CUDA_CHECK(cuda_kernel::ResidualAdd<half>(d_residual_unfused.ptr, d_input.ptr,
                                            kCount, stream));
  CUDA_CHECK(cuda_kernel::RmsNorm<half>(d_residual_unfused.ptr, d_weight.ptr,
                                        d_output_unfused.ptr, kRows,
                                        kHiddenSize, kEps, stream));

  // -- Fused path: ResidualAddRmsNorm --
  CUDA_CHECK(cuda_kernel::ResidualAddRmsNorm<half>(
      d_residual_fused.ptr, d_input.ptr, d_weight.ptr, d_output_fused.ptr,
      kRows, kHiddenSize, kEps, stream));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Compare residual buffers (both paths should produce identical residual)
  float residual_diff =
      MaxAbsDiffHalf(d_residual_unfused.ptr, d_residual_fused.ptr, kCount);
  INFO("Max residual diff: " << residual_diff);
  REQUIRE(residual_diff < kTolerance);

  // Compare RmsNorm output buffers
  float output_diff =
      MaxAbsDiffHalf(d_output_unfused.ptr, d_output_fused.ptr, kCount);
  INFO("Max output diff: " << output_diff);
  REQUIRE(output_diff < kTolerance);

  CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST_CASE(
    "FusedResidualAddRmsNorm parity with single-row input",
    "[cuda][fused_kernels]") {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    SKIP("No CUDA device available");
  }

  constexpr int kHiddenSize = 256;
  constexpr int kRows = 1;
  constexpr int kCount = kRows * kHiddenSize;
  constexpr float kEps = 1e-5f;
  constexpr float kTolerance = 0.002f;

  std::mt19937 rng(123);
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  std::vector<half> h_residual(kCount);
  std::vector<half> h_input(kCount);
  std::vector<half> h_weight(kHiddenSize);

  FillRandomHalf(h_residual, rng);
  FillRandomHalf(h_input, rng);
  FillRandomHalfRange(h_weight, rng, 0.5f, 1.5f);

  DeviceBuf<half> d_residual_unfused(kCount);
  DeviceBuf<half> d_residual_fused(kCount);
  DeviceBuf<half> d_input(kCount);
  DeviceBuf<half> d_weight(kHiddenSize);
  DeviceBuf<half> d_output_unfused(kCount);
  DeviceBuf<half> d_output_fused(kCount);

  d_residual_unfused.Upload(h_residual);
  d_residual_fused.Upload(h_residual);
  d_input.Upload(h_input);
  d_weight.Upload(h_weight);

  CUDA_CHECK(cuda_kernel::ResidualAdd<half>(d_residual_unfused.ptr, d_input.ptr,
                                            kCount, stream));
  CUDA_CHECK(cuda_kernel::RmsNorm<half>(d_residual_unfused.ptr, d_weight.ptr,
                                        d_output_unfused.ptr, kRows,
                                        kHiddenSize, kEps, stream));

  CUDA_CHECK(cuda_kernel::ResidualAddRmsNorm<half>(
      d_residual_fused.ptr, d_input.ptr, d_weight.ptr, d_output_fused.ptr,
      kRows, kHiddenSize, kEps, stream));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  float residual_diff =
      MaxAbsDiffHalf(d_residual_unfused.ptr, d_residual_fused.ptr, kCount);
  INFO("Max residual diff (single-row): " << residual_diff);
  REQUIRE(residual_diff < kTolerance);

  float output_diff =
      MaxAbsDiffHalf(d_output_unfused.ptr, d_output_fused.ptr, kCount);
  INFO("Max output diff (single-row): " << output_diff);
  REQUIRE(output_diff < kTolerance);

  CUDA_CHECK(cudaStreamDestroy(stream));
}

// ============================================================================
// Test 2: FusedBiasAddTriple parity
// ============================================================================

TEST_CASE("FusedBiasAddTriple matches three separate BiasAdd calls",
          "[cuda][fused_kernels]") {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    SKIP("No CUDA device available");
  }

  constexpr int kRows = 4;
  constexpr int kQDim = 128; // e.g., num_heads * head_dim
  constexpr int kKDim = 32;  // e.g., num_kv_heads * head_dim
  constexpr int kVDim = 32;
  constexpr float kTolerance = 0.002f;

  std::mt19937 rng(99);
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Host data
  std::vector<half> h_q(kRows * kQDim);
  std::vector<half> h_k(kRows * kKDim);
  std::vector<half> h_v(kRows * kVDim);
  std::vector<half> h_q_bias(kQDim);
  std::vector<half> h_k_bias(kKDim);
  std::vector<half> h_v_bias(kVDim);

  FillRandomHalf(h_q, rng);
  FillRandomHalf(h_k, rng);
  FillRandomHalf(h_v, rng);
  FillRandomHalfRange(h_q_bias, rng, -0.1f, 0.1f);
  FillRandomHalfRange(h_k_bias, rng, -0.1f, 0.1f);
  FillRandomHalfRange(h_v_bias, rng, -0.1f, 0.1f);

  // Device buffers — two copies for unfused vs fused
  DeviceBuf<half> d_q_unfused(kRows * kQDim);
  DeviceBuf<half> d_k_unfused(kRows * kKDim);
  DeviceBuf<half> d_v_unfused(kRows * kVDim);
  DeviceBuf<half> d_q_fused(kRows * kQDim);
  DeviceBuf<half> d_k_fused(kRows * kKDim);
  DeviceBuf<half> d_v_fused(kRows * kVDim);

  DeviceBuf<half> d_q_bias(kQDim);
  DeviceBuf<half> d_k_bias(kKDim);
  DeviceBuf<half> d_v_bias(kVDim);

  d_q_unfused.Upload(h_q);
  d_k_unfused.Upload(h_k);
  d_v_unfused.Upload(h_v);
  d_q_fused.Upload(h_q);
  d_k_fused.Upload(h_k);
  d_v_fused.Upload(h_v);
  d_q_bias.Upload(h_q_bias);
  d_k_bias.Upload(h_k_bias);
  d_v_bias.Upload(h_v_bias);

  // -- Unfused path: three separate BiasAdd calls --
  CUDA_CHECK(cuda_kernel::BiasAdd<half>(d_q_unfused.ptr, d_q_bias.ptr, kRows,
                                        kQDim, stream));
  CUDA_CHECK(cuda_kernel::BiasAdd<half>(d_k_unfused.ptr, d_k_bias.ptr, kRows,
                                        kKDim, stream));
  CUDA_CHECK(cuda_kernel::BiasAdd<half>(d_v_unfused.ptr, d_v_bias.ptr, kRows,
                                        kVDim, stream));

  // -- Fused path: single BiasAddTriple call --
  CUDA_CHECK(cuda_kernel::BiasAddTriple<half>(
      d_q_fused.ptr, d_k_fused.ptr, d_v_fused.ptr, d_q_bias.ptr,
      d_k_bias.ptr, d_v_bias.ptr, kRows, kQDim, kKDim, kVDim, stream));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Compare Q outputs
  float q_diff =
      MaxAbsDiffHalf(d_q_unfused.ptr, d_q_fused.ptr, kRows * kQDim);
  INFO("Max Q diff: " << q_diff);
  REQUIRE(q_diff < kTolerance);

  // Compare K outputs
  float k_diff =
      MaxAbsDiffHalf(d_k_unfused.ptr, d_k_fused.ptr, kRows * kKDim);
  INFO("Max K diff: " << k_diff);
  REQUIRE(k_diff < kTolerance);

  // Compare V outputs
  float v_diff =
      MaxAbsDiffHalf(d_v_unfused.ptr, d_v_fused.ptr, kRows * kVDim);
  INFO("Max V diff: " << v_diff);
  REQUIRE(v_diff < kTolerance);

  CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST_CASE("FusedBiasAddTriple with uniform dimensions",
          "[cuda][fused_kernels]") {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    SKIP("No CUDA device available");
  }

  // All three projections have the same dimension (e.g., MHA where
  // num_kv_heads == num_heads).
  constexpr int kRows = 2;
  constexpr int kDim = 64;
  constexpr float kTolerance = 0.002f;

  std::mt19937 rng(7);
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  std::vector<half> h_q(kRows * kDim), h_k(kRows * kDim), h_v(kRows * kDim);
  std::vector<half> h_q_bias(kDim), h_k_bias(kDim), h_v_bias(kDim);

  FillRandomHalf(h_q, rng);
  FillRandomHalf(h_k, rng);
  FillRandomHalf(h_v, rng);
  FillRandomHalfRange(h_q_bias, rng, -0.5f, 0.5f);
  FillRandomHalfRange(h_k_bias, rng, -0.5f, 0.5f);
  FillRandomHalfRange(h_v_bias, rng, -0.5f, 0.5f);

  DeviceBuf<half> d_q_unfused(kRows * kDim), d_k_unfused(kRows * kDim),
      d_v_unfused(kRows * kDim);
  DeviceBuf<half> d_q_fused(kRows * kDim), d_k_fused(kRows * kDim),
      d_v_fused(kRows * kDim);
  DeviceBuf<half> d_q_bias(kDim), d_k_bias(kDim), d_v_bias(kDim);

  d_q_unfused.Upload(h_q);
  d_k_unfused.Upload(h_k);
  d_v_unfused.Upload(h_v);
  d_q_fused.Upload(h_q);
  d_k_fused.Upload(h_k);
  d_v_fused.Upload(h_v);
  d_q_bias.Upload(h_q_bias);
  d_k_bias.Upload(h_k_bias);
  d_v_bias.Upload(h_v_bias);

  CUDA_CHECK(
      cuda_kernel::BiasAdd<half>(d_q_unfused.ptr, d_q_bias.ptr, kRows, kDim, stream));
  CUDA_CHECK(
      cuda_kernel::BiasAdd<half>(d_k_unfused.ptr, d_k_bias.ptr, kRows, kDim, stream));
  CUDA_CHECK(
      cuda_kernel::BiasAdd<half>(d_v_unfused.ptr, d_v_bias.ptr, kRows, kDim, stream));

  CUDA_CHECK(cuda_kernel::BiasAddTriple<half>(
      d_q_fused.ptr, d_k_fused.ptr, d_v_fused.ptr, d_q_bias.ptr,
      d_k_bias.ptr, d_v_bias.ptr, kRows, kDim, kDim, kDim, stream));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  REQUIRE(MaxAbsDiffHalf(d_q_unfused.ptr, d_q_fused.ptr, kRows * kDim) <
          kTolerance);
  REQUIRE(MaxAbsDiffHalf(d_k_unfused.ptr, d_k_fused.ptr, kRows * kDim) <
          kTolerance);
  REQUIRE(MaxAbsDiffHalf(d_v_unfused.ptr, d_v_fused.ptr, kRows * kDim) <
          kTolerance);

  CUDA_CHECK(cudaStreamDestroy(stream));
}

// ============================================================================
// Test 3: Fused softmax vs 4-kernel softmax pipeline
// ============================================================================

TEST_CASE("FusedSoftmax matches 4-kernel softmax pipeline",
          "[cuda][fused_kernels]") {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    SKIP("No CUDA device available");
  }

  constexpr int kVocabSize = 512;
  constexpr float kTemperature = 0.8f;
  constexpr float kTolerance = 1e-5f; // FP32 tolerance

  std::mt19937 rng(314);
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Host logits — use a range that exercises numerical stability
  // (large positive/negative values).
  std::vector<float> h_logits(kVocabSize);
  FillRandomFloat(h_logits, rng, -10.0f, 10.0f);

  // Device buffers
  DeviceBuf<float> d_logits_unfused(kVocabSize);
  DeviceBuf<float> d_logits_fused(kVocabSize);
  DeviceBuf<float> d_probs_unfused(kVocabSize);
  DeviceBuf<float> d_probs_fused(kVocabSize);
  DeviceBuf<float> d_scratch(1); // max_val / sum_val

  d_logits_unfused.Upload(h_logits);
  d_logits_fused.Upload(h_logits);

  constexpr int kThreads = 256;
  int blocks = (kVocabSize + kThreads - 1) / kThreads;
  int smem = kThreads * sizeof(float);

  // -- Unfused 4-kernel path --
  // Step 1: Temperature scale (in-place on logits copy)
  TestTemperatureScaleKernel<<<blocks, kThreads, 0, stream>>>(
      d_logits_unfused.ptr, kVocabSize, kTemperature);
  // Step 2: Find max
  TestSoftmaxMaxKernel<<<1, kThreads, smem, stream>>>(d_logits_unfused.ptr,
                                                       d_scratch.ptr,
                                                       kVocabSize);
  // Step 3: Exp and sum (reuses d_scratch for sum_val output)
  TestSoftmaxExpSumKernel<<<1, kThreads, smem, stream>>>(
      d_logits_unfused.ptr, d_probs_unfused.ptr, d_scratch.ptr, d_scratch.ptr,
      kVocabSize);
  // Step 4: Normalize
  TestSoftmaxNormKernel<<<blocks, kThreads, 0, stream>>>(
      d_probs_unfused.ptr, d_scratch.ptr, kVocabSize);

  // -- Fused single-kernel path --
  TestFusedSoftmaxKernel<<<1, kThreads, smem, stream>>>(
      d_logits_fused.ptr, d_probs_fused.ptr, kVocabSize, kTemperature);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Compare probability outputs
  float prob_diff = MaxAbsDiffFloat(d_probs_unfused.ptr, d_probs_fused.ptr,
                                    kVocabSize);
  INFO("Max probability diff: " << prob_diff);
  REQUIRE(prob_diff < kTolerance);

  // Verify probabilities sum to ~1.0 (sanity check)
  std::vector<float> h_probs_fused;
  d_probs_fused.Download(h_probs_fused);
  float sum = 0.0f;
  for (float p : h_probs_fused)
    sum += p;
  INFO("Fused softmax sum: " << sum);
  REQUIRE(std::abs(sum - 1.0f) < 1e-4f);

  CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST_CASE("FusedSoftmax parity with temperature=1.0 (no scaling)",
          "[cuda][fused_kernels]") {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    SKIP("No CUDA device available");
  }

  constexpr int kVocabSize = 256;
  constexpr float kTemperature = 1.0f;
  constexpr float kTolerance = 1e-5f;

  std::mt19937 rng(2718);
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  std::vector<float> h_logits(kVocabSize);
  FillRandomFloat(h_logits, rng, -5.0f, 5.0f);

  DeviceBuf<float> d_logits_unfused(kVocabSize);
  DeviceBuf<float> d_logits_fused(kVocabSize);
  DeviceBuf<float> d_probs_unfused(kVocabSize);
  DeviceBuf<float> d_probs_fused(kVocabSize);
  DeviceBuf<float> d_scratch(1);

  d_logits_unfused.Upload(h_logits);
  d_logits_fused.Upload(h_logits);

  constexpr int kThreads = 256;
  int blocks = (kVocabSize + kThreads - 1) / kThreads;
  int smem = kThreads * sizeof(float);

  // Unfused: skip temperature scale when temp=1.0 (matches gpu_sampler logic)
  TestSoftmaxMaxKernel<<<1, kThreads, smem, stream>>>(d_logits_unfused.ptr,
                                                       d_scratch.ptr,
                                                       kVocabSize);
  TestSoftmaxExpSumKernel<<<1, kThreads, smem, stream>>>(
      d_logits_unfused.ptr, d_probs_unfused.ptr, d_scratch.ptr, d_scratch.ptr,
      kVocabSize);
  TestSoftmaxNormKernel<<<blocks, kThreads, 0, stream>>>(
      d_probs_unfused.ptr, d_scratch.ptr, kVocabSize);

  // Fused with temperature=1.0
  TestFusedSoftmaxKernel<<<1, kThreads, smem, stream>>>(
      d_logits_fused.ptr, d_probs_fused.ptr, kVocabSize, kTemperature);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  float prob_diff = MaxAbsDiffFloat(d_probs_unfused.ptr, d_probs_fused.ptr,
                                    kVocabSize);
  INFO("Max probability diff (T=1.0): " << prob_diff);
  REQUIRE(prob_diff < kTolerance);

  std::vector<float> h_probs;
  d_probs_fused.Download(h_probs);
  float sum = 0.0f;
  for (float p : h_probs)
    sum += p;
  INFO("Fused softmax sum (T=1.0): " << sum);
  REQUIRE(std::abs(sum - 1.0f) < 1e-4f);

  CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST_CASE("FusedSoftmax with peaked distribution",
          "[cuda][fused_kernels]") {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    SKIP("No CUDA device available");
  }

  // One dominant logit to test numerical stability with large differences.
  constexpr int kVocabSize = 128;
  constexpr float kTemperature = 0.5f;
  constexpr float kTolerance = 1e-5f;

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  std::vector<float> h_logits(kVocabSize, 0.0f);
  h_logits[42] = 50.0f; // very peaked

  DeviceBuf<float> d_logits_unfused(kVocabSize);
  DeviceBuf<float> d_logits_fused(kVocabSize);
  DeviceBuf<float> d_probs_unfused(kVocabSize);
  DeviceBuf<float> d_probs_fused(kVocabSize);
  DeviceBuf<float> d_scratch(1);

  d_logits_unfused.Upload(h_logits);
  d_logits_fused.Upload(h_logits);

  constexpr int kThreads = 256;
  int blocks = (kVocabSize + kThreads - 1) / kThreads;
  int smem = kThreads * sizeof(float);

  TestTemperatureScaleKernel<<<blocks, kThreads, 0, stream>>>(
      d_logits_unfused.ptr, kVocabSize, kTemperature);
  TestSoftmaxMaxKernel<<<1, kThreads, smem, stream>>>(d_logits_unfused.ptr,
                                                       d_scratch.ptr,
                                                       kVocabSize);
  TestSoftmaxExpSumKernel<<<1, kThreads, smem, stream>>>(
      d_logits_unfused.ptr, d_probs_unfused.ptr, d_scratch.ptr, d_scratch.ptr,
      kVocabSize);
  TestSoftmaxNormKernel<<<blocks, kThreads, 0, stream>>>(
      d_probs_unfused.ptr, d_scratch.ptr, kVocabSize);

  TestFusedSoftmaxKernel<<<1, kThreads, smem, stream>>>(
      d_logits_fused.ptr, d_probs_fused.ptr, kVocabSize, kTemperature);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  float prob_diff = MaxAbsDiffFloat(d_probs_unfused.ptr, d_probs_fused.ptr,
                                    kVocabSize);
  INFO("Max probability diff (peaked): " << prob_diff);
  REQUIRE(prob_diff < kTolerance);

  // The dominant token should have probability very close to 1.0
  std::vector<float> h_probs;
  d_probs_fused.Download(h_probs);
  INFO("Prob at token 42: " << h_probs[42]);
  REQUIRE(h_probs[42] > 0.99f);

  CUDA_CHECK(cudaStreamDestroy(stream));
}

// ============================================================================
// Test P3: BiasAdd parity (kernel-level, no fused_quant_gemm.h needed)
// ============================================================================

TEST_CASE("BiasAdd matches CPU reference",
          "[cuda][fused_kernels][mmvq_bias]") {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    SKIP("No CUDA device available");
  }

  constexpr int kB = 2;
  constexpr int kN = 64;
  constexpr float kTolerance = 0.002f;

  std::mt19937 rng(111);
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  std::vector<half> h_output(kB * kN), h_bias(kN);
  FillRandomHalf(h_output, rng);
  FillRandomHalfRange(h_bias, rng, -0.1f, 0.1f);

  DeviceBuf<half> d_out(kB * kN), d_bias(kN);
  d_out.Upload(h_output);
  d_bias.Upload(h_bias);
  CUDA_CHECK(
      cuda_kernel::BiasAdd<half>(d_out.ptr, d_bias.ptr, kB, kN, stream));

  // CPU reference
  std::vector<half> h_expected(kB * kN);
  for (int r = 0; r < kB; ++r)
    for (int c = 0; c < kN; ++c)
      h_expected[r * kN + c] = __float2half(
          __half2float(h_output[r * kN + c]) + __half2float(h_bias[c]));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<half> h_result(kB * kN);
  d_out.Download(h_result);
  float max_diff = 0.0f;
  for (int i = 0; i < kB * kN; ++i)
    max_diff = std::max(max_diff,
                        std::abs(__half2float(h_result[i]) -
                                 __half2float(h_expected[i])));
  INFO("Max bias diff: " << max_diff);
  REQUIRE(max_diff < kTolerance);

  CUDA_CHECK(cudaStreamDestroy(stream));
}

// ============================================================================
// Tests P1/P2: Unfused component validation
// Fused kernel parity tests require headers that conflict with Catch2
// under nvcc CUDA 13.2 (cuda_bf16.hpp operator== vs Catch2 operator!=).
// Parity is validated via integration tests with env var flags enabled.
// ============================================================================

TEST_CASE("BatchedRoPE + BatchedKvAppendStrided unfused path runs correctly",
          "[cuda][fused_kernels][rope]") {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    SKIP("No CUDA device available");
  }

  constexpr int kB = 2;
  constexpr int kNumHeads = 2;
  constexpr int kNumKvHeads = 1;
  constexpr int kHeadDim = 32;
  constexpr int kKvDim = kNumKvHeads * kHeadDim;
  constexpr int kQSize = kB * kNumHeads * kHeadDim;
  constexpr int kKSize = kB * kKvDim;
  constexpr float kFreqBase = 10000.0f;

  constexpr int kMaxSeq = 32;
  constexpr size_t kKvStride = static_cast<size_t>(kMaxSeq) * kKvDim;
  constexpr size_t kLayerStride = 2 * kKvStride;
  constexpr size_t kSlotStride = 1 * kLayerStride;
  constexpr int kCacheSize = 2 * 1 * 2 * kMaxSeq * kKvDim;

  std::mt19937 rng(777);
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  std::vector<half> h_q(kQSize), h_k(kKSize), h_v(kKSize);
  FillRandomHalf(h_q, rng);
  FillRandomHalf(h_k, rng);
  FillRandomHalf(h_v, rng);
  std::vector<int> h_n_past = {5, 10};
  std::vector<int> h_seq_ids = {0, 1};

  DeviceBuf<half> d_q(kQSize), d_k(kKSize), d_v(kKSize);
  DeviceBuf<half> d_cache(kCacheSize);
  DeviceBuf<int> d_n_past(kB), d_seq_ids(kB);
  d_q.Upload(h_q);
  d_k.Upload(h_k);
  d_v.Upload(h_v);
  d_cache.Zero();
  d_n_past.Upload(h_n_past);
  d_seq_ids.Upload(h_seq_ids);

  CUDA_CHECK(cuda_kernel::BatchedRoPE<half>(
      d_q.ptr, d_k.ptr, kB, kNumHeads, kNumKvHeads, kHeadDim, d_n_past.ptr,
      kFreqBase, stream));
  CUDA_CHECK(cuda_kernel::BatchedKvAppendStrided<half>(
      d_k.ptr, d_v.ptr, d_cache.ptr, d_seq_ids.ptr, d_n_past.ptr,
      /*layer=*/0, kB, kKvDim, kSlotStride, kLayerStride, kKvStride, stream));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Verify Q was modified by RoPE
  std::vector<half> h_q_out(kQSize);
  d_q.Download(h_q_out);
  bool q_changed = false;
  for (int i = 0; i < kQSize; ++i) {
    if (__half2float(h_q[i]) != __half2float(h_q_out[i])) {
      q_changed = true;
      break;
    }
  }
  REQUIRE(q_changed);

  CUDA_CHECK(cudaStreamDestroy(stream));
}

} // namespace inferflux

#endif // INFERFLUX_HAS_CUDA
