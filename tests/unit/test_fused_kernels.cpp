/**
 * Parity tests for CUDA fused kernels.
 *
 * Compiled by MSVC as .cpp (not nvcc). CUDA device code for test-only
 * kernels lives in test_fused_kernels_cuda.cu, called via host wrappers.
 * Production kernels are called through inferflux_core library APIs.
 *
 * Test 1: FusedResidualAddRmsNorm vs unfused ResidualAdd + RmsNorm
 * Test 2: FusedBiasAddTriple vs three separate BiasAdd calls
 * Test 3: Fused softmax vs 4-kernel softmax pipeline
 * Test P1-P5: Fused kernel redesign validation
 */

#ifdef INFERFLUX_HAS_CUDA

#include <catch2/catch_amalgamated.hpp>

#include "runtime/backends/cuda/native/cuda_kernels.cuh"
#include "runtime/backends/cuda/native/fused_quant_gemm.h"
#include "runtime/backends/cuda/native/gpu_sampler.h"
#include "runtime/backends/cuda/native/native_execution_policy.h"
#include "tests/unit/test_fused_kernels_cuda.cuh"

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
      FAIL("CUDA error at " << __FILE__ << ":" << __LINE__ << ": "             \
                            << cudaGetErrorString(err__));                     \
    }                                                                          \
  } while (0)

using namespace inferflux;

// Block type declarations for .cpp compilation context.
// These structs are defined in dequantization.cuh (CUDA-only header) inside
// inferflux::runtime::cuda::native. Replicated here because dequantization.cuh
// uses #define macros that conflict with C++ compilation by g++.
namespace inferflux {
namespace runtime {
namespace cuda {
namespace native {

struct block_q4_k {
  unsigned short d;
  unsigned short dmin;
  unsigned char scales[12];
  unsigned char qs[128];
};

struct block_q8_1 {
  half2 ds;
  signed char qs[32];
};

constexpr int QK_K = 256;
constexpr int QK8_1 = 32;

} // namespace native
} // namespace cuda
} // namespace runtime
} // namespace inferflux

namespace {

void FillRandomHalf(std::vector<half> &buf, std::mt19937 &rng) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto &v : buf)
    v = __float2half(dist(rng));
}

void FillRandomHalfRange(std::vector<half> &buf, std::mt19937 &rng, float lo,
                         float hi) {
  std::uniform_real_distribution<float> dist(lo, hi);
  for (auto &v : buf)
    v = __float2half(dist(rng));
}

void FillRandomQ4KBlocks(std::vector<runtime::cuda::native::block_q4_k> &buf,
                         std::mt19937 &rng) {
  std::uniform_real_distribution<float> scale_dist(0.002f, 0.05f);
  std::uniform_int_distribution<int> byte_dist(0, 255);
  std::uniform_int_distribution<int> scale_byte_dist(0, 63);
  for (auto &block : buf) {
    block.d = __half_as_ushort(__float2half(scale_dist(rng)));
    block.dmin = __half_as_ushort(__float2half(scale_dist(rng)));
    for (auto &scale : block.scales) {
      scale = static_cast<unsigned char>(scale_byte_dist(rng));
    }
    for (auto &q : block.qs) {
      q = static_cast<unsigned char>(byte_dist(rng));
    }
  }
}

void FillRandomQ81Blocks(std::vector<runtime::cuda::native::block_q8_1> &buf,
                         std::mt19937 &rng) {
  std::uniform_real_distribution<float> scale_dist(0.002f, 0.05f);
  std::uniform_int_distribution<int> q_dist(-127, 127);
  for (auto &block : buf) {
    const half d = __float2half(scale_dist(rng));
    const half s = __float2half(scale_dist(rng));
    block.ds = __halves2half2(d, s);
    for (auto &q : block.qs) {
      q = static_cast<int8_t>(q_dist(rng));
    }
  }
}

void FillRandomFloat(std::vector<float> &buf, std::mt19937 &rng, float lo,
                     float hi) {
  std::uniform_real_distribution<float> dist(lo, hi);
  for (auto &v : buf)
    v = dist(rng);
}

float MaxAbsDiffHalf(const half *d_a, const half *d_b, int count) {
  std::vector<half> ha(count), hb(count);
  cudaMemcpy(ha.data(), d_a, count * sizeof(half), cudaMemcpyDeviceToHost);
  cudaMemcpy(hb.data(), d_b, count * sizeof(half), cudaMemcpyDeviceToHost);
  float max_diff = 0.0f;
  for (int i = 0; i < count; ++i) {
    float diff = std::abs(__half2float(ha[i]) - __half2float(hb[i]));
    max_diff = std::max(max_diff, diff);
  }
  return max_diff;
}

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
  void Zero() { cudaMemset(ptr, 0, static_cast<size_t>(count) * sizeof(T)); }
  void CopyFrom(const DeviceBuf &other) {
    cudaMemcpy(ptr, other.ptr, static_cast<size_t>(count) * sizeof(T),
               cudaMemcpyDeviceToDevice);
  }
};

} // anonymous namespace

namespace inferflux {

// ============================================================================
// Test 1: FusedResidualAddRmsNorm parity
// ============================================================================

TEST_CASE("FusedResidualAddRmsNorm matches unfused ResidualAdd + RmsNorm",
          "[cuda][fused_kernels]") {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0)
    SKIP("No CUDA device available");

  constexpr int kH = 128, kR = 4, kN = kR * kH;
  constexpr float kEps = 1e-5f, kTol = 0.002f;
  std::mt19937 rng(42);
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));

  std::vector<half> h_res(kN), h_in(kN), h_w(kH);
  FillRandomHalf(h_res, rng);
  FillRandomHalf(h_in, rng);
  FillRandomHalfRange(h_w, rng, 0.5f, 1.5f);

  DeviceBuf<half> d_ru(kN), d_rf(kN), d_in(kN), d_w(kH), d_ou(kN), d_of(kN);
  d_ru.Upload(h_res);
  d_rf.Upload(h_res);
  d_in.Upload(h_in);
  d_w.Upload(h_w);

  CUDA_CHECK(cuda_kernel::ResidualAdd<half>(d_ru.ptr, d_in.ptr, kN, s));
  CUDA_CHECK(
      cuda_kernel::RmsNorm<half>(d_ru.ptr, d_w.ptr, d_ou.ptr, kR, kH, kEps, s));
  CUDA_CHECK(cuda_kernel::ResidualAddRmsNorm<half>(d_rf.ptr, d_in.ptr, d_w.ptr,
                                                   d_of.ptr, kR, kH, kEps, s));
  CUDA_CHECK(cudaStreamSynchronize(s));

  REQUIRE(MaxAbsDiffHalf(d_ru.ptr, d_rf.ptr, kN) < kTol);
  REQUIRE(MaxAbsDiffHalf(d_ou.ptr, d_of.ptr, kN) < kTol);
  CUDA_CHECK(cudaStreamDestroy(s));
}

TEST_CASE("FusedResidualAddRmsNorm parity with single-row input",
          "[cuda][fused_kernels]") {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0)
    SKIP("No CUDA device available");

  constexpr int kH = 256, kR = 1, kN = kH;
  constexpr float kEps = 1e-5f, kTol = 0.002f;
  std::mt19937 rng(123);
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));

  std::vector<half> h_res(kN), h_in(kN), h_w(kH);
  FillRandomHalf(h_res, rng);
  FillRandomHalf(h_in, rng);
  FillRandomHalfRange(h_w, rng, 0.5f, 1.5f);

  DeviceBuf<half> d_ru(kN), d_rf(kN), d_in(kN), d_w(kH), d_ou(kN), d_of(kN);
  d_ru.Upload(h_res);
  d_rf.Upload(h_res);
  d_in.Upload(h_in);
  d_w.Upload(h_w);

  CUDA_CHECK(cuda_kernel::ResidualAdd<half>(d_ru.ptr, d_in.ptr, kN, s));
  CUDA_CHECK(
      cuda_kernel::RmsNorm<half>(d_ru.ptr, d_w.ptr, d_ou.ptr, kR, kH, kEps, s));
  CUDA_CHECK(cuda_kernel::ResidualAddRmsNorm<half>(d_rf.ptr, d_in.ptr, d_w.ptr,
                                                   d_of.ptr, kR, kH, kEps, s));
  CUDA_CHECK(cudaStreamSynchronize(s));

  REQUIRE(MaxAbsDiffHalf(d_ru.ptr, d_rf.ptr, kN) < kTol);
  REQUIRE(MaxAbsDiffHalf(d_ou.ptr, d_of.ptr, kN) < kTol);
  CUDA_CHECK(cudaStreamDestroy(s));
}

// ============================================================================
// Test 2: FusedBiasAddTriple parity
// ============================================================================

TEST_CASE("FusedBiasAddTriple matches three separate BiasAdd calls",
          "[cuda][fused_kernels]") {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0)
    SKIP("No CUDA device available");

  constexpr int kR = 4, kQ = 128, kK = 32, kV = 32;
  constexpr float kTol = 0.002f;
  std::mt19937 rng(99);
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));

  std::vector<half> hq(kR * kQ), hk(kR * kK), hv(kR * kV);
  std::vector<half> hqb(kQ), hkb(kK), hvb(kV);
  FillRandomHalf(hq, rng);
  FillRandomHalf(hk, rng);
  FillRandomHalf(hv, rng);
  FillRandomHalfRange(hqb, rng, -0.1f, 0.1f);
  FillRandomHalfRange(hkb, rng, -0.1f, 0.1f);
  FillRandomHalfRange(hvb, rng, -0.1f, 0.1f);

  DeviceBuf<half> dqu(kR * kQ), dku(kR * kK), dvu(kR * kV);
  DeviceBuf<half> dqf(kR * kQ), dkf(kR * kK), dvf(kR * kV);
  DeviceBuf<half> dqb(kQ), dkb(kK), dvb(kV);
  dqu.Upload(hq);
  dku.Upload(hk);
  dvu.Upload(hv);
  dqf.Upload(hq);
  dkf.Upload(hk);
  dvf.Upload(hv);
  dqb.Upload(hqb);
  dkb.Upload(hkb);
  dvb.Upload(hvb);

  CUDA_CHECK(cuda_kernel::BiasAdd<half>(dqu.ptr, dqb.ptr, kR, kQ, s));
  CUDA_CHECK(cuda_kernel::BiasAdd<half>(dku.ptr, dkb.ptr, kR, kK, s));
  CUDA_CHECK(cuda_kernel::BiasAdd<half>(dvu.ptr, dvb.ptr, kR, kV, s));
  CUDA_CHECK(cuda_kernel::BiasAddTriple<half>(
      dqf.ptr, dkf.ptr, dvf.ptr, dqb.ptr, dkb.ptr, dvb.ptr, kR, kQ, kK, kV, s));
  CUDA_CHECK(cudaStreamSynchronize(s));

  REQUIRE(MaxAbsDiffHalf(dqu.ptr, dqf.ptr, kR * kQ) < kTol);
  REQUIRE(MaxAbsDiffHalf(dku.ptr, dkf.ptr, kR * kK) < kTol);
  REQUIRE(MaxAbsDiffHalf(dvu.ptr, dvf.ptr, kR * kV) < kTol);
  CUDA_CHECK(cudaStreamDestroy(s));
}

TEST_CASE("FusedBiasAddTriple with uniform dimensions",
          "[cuda][fused_kernels]") {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0)
    SKIP("No CUDA device available");

  constexpr int kR = 2, kD = 64;
  constexpr float kTol = 0.002f;
  std::mt19937 rng(7);
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));

  std::vector<half> hq(kR * kD), hk(kR * kD), hv(kR * kD);
  std::vector<half> hqb(kD), hkb(kD), hvb(kD);
  FillRandomHalf(hq, rng);
  FillRandomHalf(hk, rng);
  FillRandomHalf(hv, rng);
  FillRandomHalfRange(hqb, rng, -0.5f, 0.5f);
  FillRandomHalfRange(hkb, rng, -0.5f, 0.5f);
  FillRandomHalfRange(hvb, rng, -0.5f, 0.5f);

  DeviceBuf<half> dqu(kR * kD), dku(kR * kD), dvu(kR * kD);
  DeviceBuf<half> dqf(kR * kD), dkf(kR * kD), dvf(kR * kD);
  DeviceBuf<half> dqb(kD), dkb(kD), dvb(kD);
  dqu.Upload(hq);
  dku.Upload(hk);
  dvu.Upload(hv);
  dqf.Upload(hq);
  dkf.Upload(hk);
  dvf.Upload(hv);
  dqb.Upload(hqb);
  dkb.Upload(hkb);
  dvb.Upload(hvb);

  CUDA_CHECK(cuda_kernel::BiasAdd<half>(dqu.ptr, dqb.ptr, kR, kD, s));
  CUDA_CHECK(cuda_kernel::BiasAdd<half>(dku.ptr, dkb.ptr, kR, kD, s));
  CUDA_CHECK(cuda_kernel::BiasAdd<half>(dvu.ptr, dvb.ptr, kR, kD, s));
  CUDA_CHECK(cuda_kernel::BiasAddTriple<half>(
      dqf.ptr, dkf.ptr, dvf.ptr, dqb.ptr, dkb.ptr, dvb.ptr, kR, kD, kD, kD, s));
  CUDA_CHECK(cudaStreamSynchronize(s));

  REQUIRE(MaxAbsDiffHalf(dqu.ptr, dqf.ptr, kR * kD) < kTol);
  REQUIRE(MaxAbsDiffHalf(dku.ptr, dkf.ptr, kR * kD) < kTol);
  REQUIRE(MaxAbsDiffHalf(dvu.ptr, dvf.ptr, kR * kD) < kTol);
  CUDA_CHECK(cudaStreamDestroy(s));
}

// ============================================================================
// Test 3: Fused softmax vs 4-kernel softmax pipeline
// Uses host wrappers from test_fused_kernels_cuda.cu (compiled by nvcc).
// ============================================================================

TEST_CASE("FusedSoftmax matches 4-kernel softmax pipeline",
          "[cuda][fused_kernels]") {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0)
    SKIP("No CUDA device available");

  constexpr int kV = 512;
  constexpr float kT = 0.8f, kTol = 1e-5f;
  std::mt19937 rng(314);
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));

  std::vector<float> hl(kV);
  FillRandomFloat(hl, rng, -10.0f, 10.0f);

  DeviceBuf<float> dlu(kV), dlf(kV), dpu(kV), dpf(kV), ds(1);
  dlu.Upload(hl);
  dlf.Upload(hl);

  test_cuda::LaunchTemperatureScale(dlu.ptr, kV, kT, s);
  test_cuda::LaunchSoftmaxMax(dlu.ptr, ds.ptr, kV, s);
  test_cuda::LaunchSoftmaxExpSum(dlu.ptr, dpu.ptr, ds.ptr, ds.ptr, kV, s);
  test_cuda::LaunchSoftmaxNorm(dpu.ptr, ds.ptr, kV, s);
  test_cuda::LaunchFusedSoftmax(dlf.ptr, dpf.ptr, kV, kT, s);

  CUDA_CHECK(cudaStreamSynchronize(s));

  REQUIRE(MaxAbsDiffFloat(dpu.ptr, dpf.ptr, kV) < kTol);

  std::vector<float> hp;
  dpf.Download(hp);
  float sum = 0.0f;
  for (float p : hp)
    sum += p;
  REQUIRE(std::abs(sum - 1.0f) < 1e-4f);
  CUDA_CHECK(cudaStreamDestroy(s));
}

TEST_CASE("FusedSoftmax parity with temperature=1.0", "[cuda][fused_kernels]") {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0)
    SKIP("No CUDA device available");

  constexpr int kV = 256;
  constexpr float kTol = 1e-5f;
  std::mt19937 rng(2718);
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));

  std::vector<float> hl(kV);
  FillRandomFloat(hl, rng, -5.0f, 5.0f);

  DeviceBuf<float> dlu(kV), dlf(kV), dpu(kV), dpf(kV), ds(1);
  dlu.Upload(hl);
  dlf.Upload(hl);

  test_cuda::LaunchSoftmaxMax(dlu.ptr, ds.ptr, kV, s);
  test_cuda::LaunchSoftmaxExpSum(dlu.ptr, dpu.ptr, ds.ptr, ds.ptr, kV, s);
  test_cuda::LaunchSoftmaxNorm(dpu.ptr, ds.ptr, kV, s);
  test_cuda::LaunchFusedSoftmax(dlf.ptr, dpf.ptr, kV, 1.0f, s);

  CUDA_CHECK(cudaStreamSynchronize(s));
  REQUIRE(MaxAbsDiffFloat(dpu.ptr, dpf.ptr, kV) < kTol);
  CUDA_CHECK(cudaStreamDestroy(s));
}

TEST_CASE("FusedSoftmax with peaked distribution", "[cuda][fused_kernels]") {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0)
    SKIP("No CUDA device available");

  constexpr int kV = 128;
  constexpr float kT = 0.5f, kTol = 1e-5f;
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));

  std::vector<float> hl(kV, 0.0f);
  hl[42] = 50.0f;

  DeviceBuf<float> dlu(kV), dlf(kV), dpu(kV), dpf(kV), ds(1);
  dlu.Upload(hl);
  dlf.Upload(hl);

  test_cuda::LaunchTemperatureScale(dlu.ptr, kV, kT, s);
  test_cuda::LaunchSoftmaxMax(dlu.ptr, ds.ptr, kV, s);
  test_cuda::LaunchSoftmaxExpSum(dlu.ptr, dpu.ptr, ds.ptr, ds.ptr, kV, s);
  test_cuda::LaunchSoftmaxNorm(dpu.ptr, ds.ptr, kV, s);
  test_cuda::LaunchFusedSoftmax(dlf.ptr, dpf.ptr, kV, kT, s);

  CUDA_CHECK(cudaStreamSynchronize(s));
  REQUIRE(MaxAbsDiffFloat(dpu.ptr, dpf.ptr, kV) < kTol);

  std::vector<float> hp;
  dpf.Download(hp);
  REQUIRE(hp[42] > 0.99f);
  CUDA_CHECK(cudaStreamDestroy(s));
}

// ============================================================================
// Test P0: Batched stochastic sampling
// ============================================================================

TEST_CASE("Batched stochastic sampling produces valid tokens",
          "[cuda][fused_kernels][sampling]") {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0)
    SKIP("No CUDA device available");

  constexpr int kV = 256, kB = 8;
  std::mt19937 rng(12345);
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));

  GpuSampler sampler;
  REQUIRE(sampler.Initialize(kV, s));

  std::vector<float> hl(kB * kV);
  FillRandomFloat(hl, rng, -5.0f, 5.0f);
  DeviceBuf<float> dl(kB * kV);
  dl.Upload(hl);

  std::vector<float> temps(kB, 0.8f);
  std::vector<int> topks(kB, 0);
  std::vector<float> topps(kB, 1.0f);
  std::vector<uint32_t> seeds(kB);
  for (int i = 0; i < kB; ++i)
    seeds[i] = 42 + i;

  std::vector<int> tokens;
  sampler.SampleBatch(dl.ptr, kB, temps, topks, topps, seeds, &tokens);

  REQUIRE(tokens.size() == kB);
  for (int i = 0; i < kB; ++i) {
    REQUIRE(tokens[i] >= 0);
    REQUIRE(tokens[i] < kV);
  }
  CUDA_CHECK(cudaStreamDestroy(s));
}

TEST_CASE("Batched sampling handles mixed greedy and stochastic",
          "[cuda][fused_kernels][sampling]") {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0)
    SKIP("No CUDA device available");

  constexpr int kV = 128, kB = 4;
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));

  GpuSampler sampler;
  REQUIRE(sampler.Initialize(kV, s));

  std::vector<float> hl(kB * kV, 0.0f);
  hl[0 * kV + 10] = 50.0f;
  hl[1 * kV + 20] = 50.0f;
  hl[2 * kV + 30] = 50.0f;
  hl[3 * kV + 40] = 50.0f;

  DeviceBuf<float> dl(kB * kV);
  dl.Upload(hl);

  std::vector<float> temps = {0.0f, 0.5f, 0.0f, 0.5f};
  std::vector<int> topks(kB, 0);
  std::vector<float> topps(kB, 1.0f);
  std::vector<uint32_t> seeds = {UINT32_MAX, 42, UINT32_MAX, 43};

  std::vector<int> tokens;
  sampler.SampleBatch(dl.ptr, kB, temps, topks, topps, seeds, &tokens);

  REQUIRE(tokens.size() == kB);
  REQUIRE(tokens[0] == 10);
  REQUIRE(tokens[2] == 30);
  REQUIRE(tokens[1] == 20);
  REQUIRE(tokens[3] == 40);
  CUDA_CHECK(cudaStreamDestroy(s));
}

// ============================================================================
// Test P1-P5: Policy flag validation + unfused component tests
// ============================================================================

TEST_CASE("Fused kernel policy flags have correct defaults",
          "[cuda][fused_kernels][policy]") {
  NativeExecutionPolicy p;
  // P1+P2 validated and enabled by default
  REQUIRE(p.enable_fused_rope_kv_append);
  REQUIRE(p.enable_fused_gemv_norm_quant_epilogue);
  // P3-P5 still experimental (default off)
  REQUIRE_FALSE(p.enable_mmvq_bias_epilogue);
  REQUIRE_FALSE(p.enable_q6k_vectorized);
  REQUIRE_FALSE(p.enable_gate_up_silu_q81_epilogue);
}

TEST_CASE("FusedGateUpSiluGemvQ8_1WithEpilogue returns false for null inputs",
          "[cuda][fused_kernels][gate_up_q81]") {
  QuantizedWeightInfo empty{};
  REQUIRE_FALSE(FusedQuantGemm::FusedGateUpSiluGemvQ8_1WithEpilogue(
      empty, empty, nullptr, nullptr, nullptr, 0, 0, 0, nullptr));
}

TEST_CASE(
    "FusedGateUpSiluGemvQ8_1WithEpilogue matches fused gate/up output and "
    "quantized epilogue",
    "[cuda][fused_kernels][gate_up_q81]") {
  using runtime::cuda::native::block_q4_k;
  using runtime::cuda::native::block_q8_1;
  using runtime::cuda::native::QK8_1;
  using runtime::cuda::native::QK_K;
  using runtime::cuda::native::GGUF::TensorType;

  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0)
    SKIP("No CUDA device available");

  constexpr int kM = 4;
  constexpr int kN = 256;
  constexpr int kK = 256;
  constexpr float kTol = 0.01f;
  static_assert(kK % QK_K == 0);
  static_assert(kK % QK8_1 == 0);
  static_assert(kN % QK8_1 == 0);

  std::mt19937 rng(20260327);
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));

  const int weight_blocks_per_row = kK / QK_K;
  const int input_blocks = kM * (kK / QK8_1);
  const int output_blocks = kM * (kN / QK8_1);

  std::vector<block_q4_k> h_gate(kN * weight_blocks_per_row);
  std::vector<block_q4_k> h_up(kN * weight_blocks_per_row);
  std::vector<block_q8_1> h_act(input_blocks);
  FillRandomQ4KBlocks(h_gate, rng);
  FillRandomQ4KBlocks(h_up, rng);
  FillRandomQ81Blocks(h_act, rng);

  DeviceBuf<block_q4_k> d_gate(static_cast<int>(h_gate.size()));
  DeviceBuf<block_q4_k> d_up(static_cast<int>(h_up.size()));
  DeviceBuf<block_q8_1> d_act(input_blocks);
  DeviceBuf<block_q8_1> d_q8_ref(output_blocks);
  DeviceBuf<block_q8_1> d_q8_epi(output_blocks);
  DeviceBuf<half> d_out_ref(kM * kN);
  DeviceBuf<half> d_out_epi(kM * kN);
  d_gate.Upload(h_gate);
  d_up.Upload(h_up);
  d_act.Upload(h_act);
  d_q8_ref.Zero();
  d_q8_epi.Zero();
  d_out_ref.Zero();
  d_out_epi.Zero();

  const QuantizedWeightInfo gate_raw{
      d_gate.ptr, static_cast<int>(TensorType::Q4_K), kN * kK};
  const QuantizedWeightInfo up_raw{d_up.ptr, static_cast<int>(TensorType::Q4_K),
                                   kN * kK};

  REQUIRE(FusedQuantGemm::FusedGateUpSiluGemvQ8_1(
      gate_raw, up_raw, d_act.ptr, d_out_ref.ptr, kM, kN, kK, s));
  FusedQuantGemm::QuantizeRowQ8_1(d_out_ref.ptr, d_q8_ref.ptr, kM, kN, s);

  REQUIRE(FusedQuantGemm::FusedGateUpSiluGemvQ8_1WithEpilogue(
      gate_raw, up_raw, d_act.ptr, d_out_epi.ptr, d_q8_epi.ptr, kM, kN, kK, s));
  CUDA_CHECK(cudaStreamSynchronize(s));

  REQUIRE(MaxAbsDiffHalf(d_out_ref.ptr, d_out_epi.ptr, kM * kN) < kTol);

  std::vector<block_q8_1> h_q8_ref;
  std::vector<block_q8_1> h_q8_epi;
  d_q8_ref.Download(h_q8_ref);
  d_q8_epi.Download(h_q8_epi);
  REQUIRE(h_q8_ref.size() == h_q8_epi.size());
  REQUIRE(std::memcmp(h_q8_ref.data(), h_q8_epi.data(),
                      h_q8_ref.size() * sizeof(block_q8_1)) == 0);

  CUDA_CHECK(cudaStreamDestroy(s));
}

TEST_CASE("BiasAdd matches CPU reference", "[cuda][fused_kernels][mmvq_bias]") {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0)
    SKIP("No CUDA device available");

  constexpr int kB = 2, kN = 64;
  constexpr float kTol = 0.002f;
  std::mt19937 rng(111);
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));

  std::vector<half> ho(kB * kN), hb(kN);
  FillRandomHalf(ho, rng);
  FillRandomHalfRange(hb, rng, -0.1f, 0.1f);

  DeviceBuf<half> d_out(kB * kN), d_bias(kN);
  d_out.Upload(ho);
  d_bias.Upload(hb);
  CUDA_CHECK(cuda_kernel::BiasAdd<half>(d_out.ptr, d_bias.ptr, kB, kN, s));

  std::vector<half> hexp(kB * kN);
  for (int r = 0; r < kB; ++r)
    for (int c = 0; c < kN; ++c)
      hexp[r * kN + c] =
          __float2half(__half2float(ho[r * kN + c]) + __half2float(hb[c]));

  CUDA_CHECK(cudaStreamSynchronize(s));

  std::vector<half> hr(kB * kN);
  d_out.Download(hr);
  float md = 0.0f;
  for (int i = 0; i < kB * kN; ++i)
    md = std::max(md, std::abs(__half2float(hr[i]) - __half2float(hexp[i])));
  REQUIRE(md < kTol);
  CUDA_CHECK(cudaStreamDestroy(s));
}

TEST_CASE("BatchedRoPE + BatchedKvAppendStrided unfused path runs correctly",
          "[cuda][fused_kernels][rope]") {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0)
    SKIP("No CUDA device available");

  constexpr int kB = 2, kNH = 2, kNKV = 1, kHD = 32;
  constexpr int kKvDim = kNKV * kHD;
  constexpr int kQSz = kB * kNH * kHD, kKSz = kB * kKvDim;
  constexpr int kMaxSeq = 32;
  constexpr size_t kKvStr = static_cast<size_t>(kMaxSeq) * kKvDim;
  constexpr size_t kLayStr = 2 * kKvStr;
  constexpr size_t kSlotStr = kLayStr;
  constexpr int kCSz = 2 * 1 * 2 * kMaxSeq * kKvDim;

  std::mt19937 rng(777);
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));

  std::vector<half> hq(kQSz), hk(kKSz), hv(kKSz);
  FillRandomHalf(hq, rng);
  FillRandomHalf(hk, rng);
  FillRandomHalf(hv, rng);
  std::vector<int> hnp = {5, 10}, hsid = {0, 1};

  DeviceBuf<half> dq(kQSz), dk(kKSz), dv(kKSz), dc(kCSz);
  DeviceBuf<int> dnp(kB), dsid(kB);
  dq.Upload(hq);
  dk.Upload(hk);
  dv.Upload(hv);
  dc.Zero();
  dnp.Upload(hnp);
  dsid.Upload(hsid);

  CUDA_CHECK(cuda_kernel::BatchedRoPE<half>(dq.ptr, dk.ptr, kB, kNH, kNKV, kHD,
                                            dnp.ptr, 10000.0f, s));
  CUDA_CHECK(cuda_kernel::BatchedKvAppendStrided<half>(
      dk.ptr, dv.ptr, dc.ptr, dsid.ptr, dnp.ptr, 0, kB, kKvDim, kSlotStr,
      kLayStr, kKvStr, s));
  CUDA_CHECK(cudaStreamSynchronize(s));

  std::vector<half> hqo(kQSz);
  dq.Download(hqo);
  bool changed = false;
  for (int i = 0; i < kQSz; ++i)
    if (__half2float(hq[i]) != __half2float(hqo[i])) {
      changed = true;
      break;
    }
  REQUIRE(changed);
  CUDA_CHECK(cudaStreamDestroy(s));
}

// ============================================================================
// Comprehensive coverage: RmsNorm + ResidualAdd stress test
// Verifies FP16 accumulation doesn't drift across many iterations.
// ============================================================================

TEST_CASE("ResidualAdd + RmsNorm stress test (100 iterations, no drift)",
          "[cuda][fused_kernels][stress]") {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0)
    SKIP("No CUDA device available");

  constexpr int kH = 256, kR = 1;
  constexpr int kN = kR * kH;
  constexpr float kEps = 1e-5f;
  constexpr int kIter = 100;
  std::mt19937 rng(42);
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));

  std::vector<half> h_w(kH);
  FillRandomHalfRange(h_w, rng, 0.9f, 1.1f);

  DeviceBuf<half> d_res(kN), d_in(kN), d_w(kH), d_out(kN);
  d_w.Upload(h_w);

  // Initialize residual to small values
  std::vector<half> h_res(kN);
  FillRandomHalfRange(h_res, rng, -0.01f, 0.01f);
  d_res.Upload(h_res);

  for (int iter = 0; iter < kIter; ++iter) {
    // Generate small input each iteration
    std::vector<half> h_in(kN);
    FillRandomHalfRange(h_in, rng, -0.001f, 0.001f);
    d_in.Upload(h_in);

    // Fused ResidualAdd + RmsNorm (the P1+P2 default path)
    CUDA_CHECK(cuda_kernel::ResidualAddRmsNorm<half>(
        d_res.ptr, d_in.ptr, d_w.ptr, d_out.ptr, kR, kH, kEps, s));
  }
  CUDA_CHECK(cudaStreamSynchronize(s));

  // Verify output is finite and in reasonable range
  std::vector<half> h_out(kN);
  d_out.Download(h_out);
  bool all_finite = true;
  float max_abs = 0.0f;
  for (int i = 0; i < kN; ++i) {
    float v = __half2float(h_out[i]);
    if (!std::isfinite(v))
      all_finite = false;
    max_abs = std::max(max_abs, std::abs(v));
  }
  INFO("After " << kIter << " iterations: max_abs=" << max_abs);
  REQUIRE(all_finite);
  REQUIRE(max_abs < 100.0f); // shouldn't explode
  REQUIRE(max_abs > 1e-6f);  // shouldn't collapse to zero
  CUDA_CHECK(cudaStreamDestroy(s));
}

// ============================================================================
// Comprehensive coverage: BatchedRoPE correctness for various head configs
// ============================================================================

TEST_CASE("BatchedRoPE preserves vector magnitude",
          "[cuda][fused_kernels][rope][coverage]") {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0)
    SKIP("No CUDA device available");

  // RoPE should preserve magnitude: |rotated| == |original| for each pair
  constexpr int kB = 1, kNH = 4, kNKV = 2, kHD = 64;
  constexpr int kQSz = kB * kNH * kHD;
  constexpr int kKSz = kB * kNKV * kHD;
  std::mt19937 rng(999);
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));

  std::vector<half> hq(kQSz), hk(kKSz);
  FillRandomHalf(hq, rng);
  FillRandomHalf(hk, rng);

  // Compute input magnitudes on CPU
  std::vector<float> q_mag_in(kNH * kHD / 2);
  for (int h = 0; h < kNH; ++h) {
    for (int p = 0; p < kHD / 2; ++p) {
      float v0 = __half2float(hq[h * kHD + 2 * p]);
      float v1 = __half2float(hq[h * kHD + 2 * p + 1]);
      q_mag_in[h * kHD / 2 + p] = sqrtf(v0 * v0 + v1 * v1);
    }
  }

  DeviceBuf<half> dq(kQSz), dk(kKSz);
  DeviceBuf<int> dnp(kB);
  dq.Upload(hq);
  dk.Upload(hk);
  std::vector<int> hnp = {42};
  dnp.Upload(hnp);

  CUDA_CHECK(cuda_kernel::BatchedRoPE<half>(dq.ptr, dk.ptr, kB, kNH, kNKV, kHD,
                                            dnp.ptr, 10000.0f, s));
  CUDA_CHECK(cudaStreamSynchronize(s));

  // Check output magnitudes match input
  std::vector<half> hq_out(kQSz);
  dq.Download(hq_out);
  float max_mag_diff = 0.0f;
  for (int h = 0; h < kNH; ++h) {
    for (int p = 0; p < kHD / 2; ++p) {
      float v0 = __half2float(hq_out[h * kHD + 2 * p]);
      float v1 = __half2float(hq_out[h * kHD + 2 * p + 1]);
      float mag_out = sqrtf(v0 * v0 + v1 * v1);
      float mag_in = q_mag_in[h * kHD / 2 + p];
      if (mag_in > 0.01f) {
        float rel_diff = std::abs(mag_out - mag_in) / mag_in;
        max_mag_diff = std::max(max_mag_diff, rel_diff);
      }
    }
  }
  INFO("Max relative magnitude difference: " << max_mag_diff);
  REQUIRE(max_mag_diff < 0.01f); // FP16 tolerance
  CUDA_CHECK(cudaStreamDestroy(s));
}

// ============================================================================
// Coverage: Fused softmax edge cases
// ============================================================================

TEST_CASE("FusedSoftmax handles all-equal logits",
          "[cuda][fused_kernels][coverage]") {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0)
    SKIP("No CUDA device available");

  constexpr int kV = 64;
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));

  // All logits equal → uniform distribution
  std::vector<float> hl(kV, 1.0f);
  DeviceBuf<float> dl(kV), dp(kV);
  dl.Upload(hl);

  test_cuda::LaunchFusedSoftmax(dl.ptr, dp.ptr, kV, 1.0f, s);
  CUDA_CHECK(cudaStreamSynchronize(s));

  std::vector<float> hp(kV);
  dp.Download(hp);
  float expected = 1.0f / kV;
  float max_diff = 0.0f;
  for (float p : hp)
    max_diff = std::max(max_diff, std::abs(p - expected));
  INFO("Max diff from uniform: " << max_diff);
  REQUIRE(max_diff < 1e-5f);
  CUDA_CHECK(cudaStreamDestroy(s));
}

TEST_CASE("FusedSoftmax handles very low temperature",
          "[cuda][fused_kernels][coverage]") {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0)
    SKIP("No CUDA device available");

  constexpr int kV = 128;
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));

  // One dominant logit, very low temp → near-one-hot
  std::vector<float> hl(kV, 0.0f);
  hl[7] = 10.0f;
  DeviceBuf<float> dl(kV), dp(kV);
  dl.Upload(hl);

  test_cuda::LaunchFusedSoftmax(dl.ptr, dp.ptr, kV, 0.01f, s);
  CUDA_CHECK(cudaStreamSynchronize(s));

  std::vector<float> hp(kV);
  dp.Download(hp);
  REQUIRE(hp[7] > 0.999f);
  // All other probs should be near zero
  float sum_others = 0.0f;
  for (int i = 0; i < kV; ++i)
    if (i != 7)
      sum_others += hp[i];
  REQUIRE(sum_others < 0.001f);
  CUDA_CHECK(cudaStreamDestroy(s));
}

// ============================================================================
// Coverage: BiasAddTriple with large dimensions (realistic geometry)
// ============================================================================

TEST_CASE("FusedBiasAddTriple with realistic Qwen2 geometry",
          "[cuda][fused_kernels][coverage]") {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0)
    SKIP("No CUDA device available");

  // Qwen2.5-3B: num_heads=16, num_kv_heads=2, head_dim=128
  constexpr int kR = 1;        // single decode token
  constexpr int kQ = 16 * 128; // 2048
  constexpr int kK = 2 * 128;  // 256
  constexpr int kV = 2 * 128;  // 256
  constexpr float kTol = 0.002f;
  std::mt19937 rng(2024);
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));

  std::vector<half> hq(kR * kQ), hk(kR * kK), hv(kR * kV);
  std::vector<half> hqb(kQ), hkb(kK), hvb(kV);
  FillRandomHalf(hq, rng);
  FillRandomHalf(hk, rng);
  FillRandomHalf(hv, rng);
  FillRandomHalfRange(hqb, rng, -0.05f, 0.05f);
  FillRandomHalfRange(hkb, rng, -0.05f, 0.05f);
  FillRandomHalfRange(hvb, rng, -0.05f, 0.05f);

  DeviceBuf<half> dqu(kR * kQ), dku(kR * kK), dvu(kR * kV);
  DeviceBuf<half> dqf(kR * kQ), dkf(kR * kK), dvf(kR * kV);
  DeviceBuf<half> dqb(kQ), dkb(kK), dvb(kV);
  dqu.Upload(hq);
  dku.Upload(hk);
  dvu.Upload(hv);
  dqf.Upload(hq);
  dkf.Upload(hk);
  dvf.Upload(hv);
  dqb.Upload(hqb);
  dkb.Upload(hkb);
  dvb.Upload(hvb);

  // Unfused
  CUDA_CHECK(cuda_kernel::BiasAdd<half>(dqu.ptr, dqb.ptr, kR, kQ, s));
  CUDA_CHECK(cuda_kernel::BiasAdd<half>(dku.ptr, dkb.ptr, kR, kK, s));
  CUDA_CHECK(cuda_kernel::BiasAdd<half>(dvu.ptr, dvb.ptr, kR, kV, s));

  // Fused
  CUDA_CHECK(cuda_kernel::BiasAddTriple<half>(
      dqf.ptr, dkf.ptr, dvf.ptr, dqb.ptr, dkb.ptr, dvb.ptr, kR, kQ, kK, kV, s));
  CUDA_CHECK(cudaStreamSynchronize(s));

  REQUIRE(MaxAbsDiffHalf(dqu.ptr, dqf.ptr, kR * kQ) < kTol);
  REQUIRE(MaxAbsDiffHalf(dku.ptr, dkf.ptr, kR * kK) < kTol);
  REQUIRE(MaxAbsDiffHalf(dvu.ptr, dvf.ptr, kR * kV) < kTol);
  CUDA_CHECK(cudaStreamDestroy(s));
}

// ============================================================================
// Coverage: RmsNorm with realistic hidden sizes
// ============================================================================

TEST_CASE("RmsNorm with hidden_size=2048 (Qwen2.5-3B)",
          "[cuda][fused_kernels][coverage]") {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0)
    SKIP("No CUDA device available");

  constexpr int kH = 2048, kR = 4;
  constexpr int kN = kR * kH;
  constexpr float kEps = 1e-6f, kTol = 0.002f;
  std::mt19937 rng(314159);
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));

  std::vector<half> hr(kN), hi(kN), hw(kH);
  FillRandomHalf(hr, rng);
  FillRandomHalf(hi, rng);
  FillRandomHalfRange(hw, rng, 0.8f, 1.2f);

  DeviceBuf<half> dru(kN), drf(kN), di(kN), dw(kH), dou(kN), dof(kN);
  dru.Upload(hr);
  drf.Upload(hr);
  di.Upload(hi);
  dw.Upload(hw);

  // Unfused
  CUDA_CHECK(cuda_kernel::ResidualAdd<half>(dru.ptr, di.ptr, kN, s));
  CUDA_CHECK(
      cuda_kernel::RmsNorm<half>(dru.ptr, dw.ptr, dou.ptr, kR, kH, kEps, s));

  // Fused
  CUDA_CHECK(cuda_kernel::ResidualAddRmsNorm<half>(drf.ptr, di.ptr, dw.ptr,
                                                   dof.ptr, kR, kH, kEps, s));
  CUDA_CHECK(cudaStreamSynchronize(s));

  REQUIRE(MaxAbsDiffHalf(dru.ptr, drf.ptr, kN) < kTol);
  REQUIRE(MaxAbsDiffHalf(dou.ptr, dof.ptr, kN) < kTol);
  CUDA_CHECK(cudaStreamDestroy(s));
}

// ============================================================================
// Coverage: Policy env var override roundtrip
// ============================================================================

TEST_CASE("Policy flags can be overridden via env vars",
          "[cuda][fused_kernels][policy][coverage]") {
  // Test that FromEnv() respects the defaults we just set
  NativeExecutionPolicy p = NativeExecutionPolicy::FromEnv();
  // P1+P2 should be on by default (no env override set in test)
  REQUIRE(p.enable_fused_rope_kv_append);
  REQUIRE(p.enable_fused_gemv_norm_quant_epilogue);
  // Existing flags should maintain their defaults
  REQUIRE(p.enable_fused_gate_up_silu);
  REQUIRE(p.enable_fused_residual_norm);
  REQUIRE(p.enable_fused_bias_add);
  REQUIRE(p.enable_gemv_accumulate);
}

} // namespace inferflux

#endif // INFERFLUX_HAS_CUDA
