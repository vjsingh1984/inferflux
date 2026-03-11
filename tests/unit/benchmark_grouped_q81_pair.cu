#include "runtime/backends/cuda/native/fused_quant_gemm.h"
#include "runtime/backends/cuda/native/kernels/dequantization.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using inferflux::FusedDispatchGeometry;
using inferflux::FusedQuantGemm;
using inferflux::NativeExecutionPolicy;
using inferflux::PackedProjectionSpec;
namespace native = inferflux::runtime::cuda::native;
namespace GGUF = native::GGUF;

namespace {

constexpr int kWarmupIters = 30;
constexpr int kBenchmarkIters = 200;
constexpr int kK = 2048;
constexpr int kN0 = 11008;
constexpr int kN1 = 11008;
constexpr int kBlocksPerRow = kK / QK_K;

struct BenchmarkResult {
  float mean_ms{0.0f};
  std::vector<half> output;
};

unsigned short EncodeHalfBits(float value) {
  const half h = __float2half(value);
  unsigned short bits = 0;
  std::memcpy(&bits, &h, sizeof(bits));
  return bits;
}

std::vector<half> MakeWaveTensor(size_t count, float scale, float bias = 0.0f) {
  std::vector<half> out(count);
  for (size_t i = 0; i < count; ++i) {
    const float value =
        bias + scale * std::sin(0.173f * static_cast<float>(i) + 0.031f);
    out[i] = __float2half(value);
  }
  return out;
}

std::vector<native::block_q4_k> MakeQ4Rows(int rows, int seed) {
  std::vector<native::block_q4_k> blocks(static_cast<size_t>(rows) *
                                         kBlocksPerRow);
  for (int row = 0; row < rows; ++row) {
    for (int blk = 0; blk < kBlocksPerRow; ++blk) {
      auto &block = blocks[static_cast<size_t>(row) * kBlocksPerRow + blk];
      block.d =
          EncodeHalfBits(0.02f * static_cast<float>(((row + blk + seed) % 4) + 1));
      block.dmin =
          EncodeHalfBits(0.01f * static_cast<float>(((row + blk + seed) % 3) + 1));
      for (int i = 0; i < 12; ++i) {
        block.scales[i] = static_cast<unsigned char>(
            (seed * 17 + row * 7 + blk * 11 + i * 5) & 0xFF);
      }
      for (int i = 0; i < QK_K / 2; ++i) {
        block.qs[i] = static_cast<unsigned char>(
            (seed * 11 + row * 13 + blk * 19 + i * 3) & 0xFF);
      }
    }
  }
  return blocks;
}

BenchmarkResult RunCase(int m, const NativeExecutionPolicy &policy) {
  native::block_q4_k *d_w0 = nullptr;
  native::block_q4_k *d_w1 = nullptr;
  half *d_input = nullptr;
  void *d_act_q8_1 = nullptr;
  half *d_out = nullptr;
  cudaStream_t stream = nullptr;
  cudaStreamCreate(&stream);

  const auto w0 = MakeQ4Rows(kN0, 3);
  const auto w1 = MakeQ4Rows(kN1, 7);
  const auto input = MakeWaveTensor(static_cast<size_t>(m) * kK, 0.015f,
                                    -0.002f);

  cudaMalloc(reinterpret_cast<void **>(&d_w0), w0.size() * sizeof(native::block_q4_k));
  cudaMalloc(reinterpret_cast<void **>(&d_w1), w1.size() * sizeof(native::block_q4_k));
  cudaMalloc(reinterpret_cast<void **>(&d_input), input.size() * sizeof(half));
  cudaMalloc(&d_act_q8_1,
             static_cast<size_t>(m) * (kK / QK8_1) * sizeof(native::block_q8_1));
  cudaMalloc(reinterpret_cast<void **>(&d_out),
             static_cast<size_t>(m) * (kN0 + kN1) * sizeof(half));

  cudaMemcpy(d_w0, w0.data(), w0.size() * sizeof(native::block_q4_k),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_w1, w1.data(), w1.size() * sizeof(native::block_q4_k),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_input, input.data(), input.size() * sizeof(half),
             cudaMemcpyHostToDevice);
  FusedQuantGemm::QuantizeRowQ8_1(d_input, d_act_q8_1, m, kK, stream);
  cudaStreamSynchronize(stream);

  const std::array<PackedProjectionSpec, 2> projections = {{
      {{d_w0, static_cast<int>(GGUF::TensorType::Q4_K),
        static_cast<int64_t>(kN0) * kK},
       d_out,
       kN0},
      {{d_w1, static_cast<int>(GGUF::TensorType::Q4_K),
        static_cast<int64_t>(kN1) * kK},
       d_out + static_cast<size_t>(m) * kN0,
       kN1},
  }};

  const auto bench_ms = [&]() {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < kWarmupIters; ++i) {
      FusedQuantGemm::GemvQ8_1Pair(projections, d_act_q8_1, m, kK, stream,
                                   &policy);
    }
    cudaStreamSynchronize(stream);
    cudaEventRecord(start, stream);
    for (int i = 0; i < kBenchmarkIters; ++i) {
      FusedQuantGemm::GemvQ8_1Pair(projections, d_act_q8_1, m, kK, stream,
                                   &policy);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    return ms / static_cast<float>(kBenchmarkIters);
  };

  const float ms = bench_ms();
  std::vector<half> output(static_cast<size_t>(m) * (kN0 + kN1));
  cudaMemcpy(output.data(), d_out, output.size() * sizeof(half),
             cudaMemcpyDeviceToHost);

  cudaFree(d_out);
  cudaFree(d_act_q8_1);
  cudaFree(d_input);
  cudaFree(d_w1);
  cudaFree(d_w0);
  cudaStreamDestroy(stream);
  return BenchmarkResult{ms, std::move(output)};
}

float MaxAbsDiff(const std::vector<half> &lhs, const std::vector<half> &rhs) {
  float max_diff = 0.0f;
  for (size_t i = 0; i < lhs.size(); ++i) {
    const float diff = std::fabs(__half2float(lhs[i]) - __half2float(rhs[i]));
    max_diff = std::max(max_diff, diff);
  }
  return max_diff;
}

void PrintCaseSummary(const char *label, int m, const BenchmarkResult &generic,
                      const BenchmarkResult &candidate,
                      const char *candidate_name) {
  std::printf("Case: %s Q4_K M=%d N=%d/%d K=%d\n", label, m, kN0, kN1, kK);
  std::printf("  generic: %.3f ms\n", generic.mean_ms);
  std::printf("  %s: %8.3f ms\n", candidate_name, candidate.mean_ms);
  std::printf("  speedup: %.3fx\n", generic.mean_ms / candidate.mean_ms);
  std::printf("  max_abs_diff_vs_generic: %.6f\n",
              MaxAbsDiff(generic.output, candidate.output));
}

void PrintRawCaseSummary(const char *label, int m, const BenchmarkResult &result,
                         const char *mode_name) {
  std::printf("Case: %s Q4_K M=%d N=%d/%d K=%d\n", label, m, kN0, kN1, kK);
  std::printf("  %s: %.3f ms\n", mode_name, result.mean_ms);
}

} // namespace

int main(int argc, char **argv) {
#ifndef INFERFLUX_HAS_CUDA
  std::puts("benchmark_grouped_q81_pair requires native CUDA kernels");
  return 0;
#else
  bool force_v2 = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--v2") == 0) {
      force_v2 = true;
    } else {
      std::fprintf(stderr, "Unknown argument: %s\n", argv[i]);
      return 1;
    }
  }
  if (force_v2) {
    setenv("INFERFLUX_GEMV_V2", "1", 1);
  }

  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    std::puts("No CUDA device available");
    return 0;
  }

  std::puts("========================================");
  std::puts("Grouped Q8_1 Pair Benchmark");
  std::puts("========================================");
  std::printf("Mode: %s\n", force_v2 ? "v2" : "v1");

  NativeExecutionPolicy generic_policy;
  generic_policy.enable_experimental_q81_grouped_hot_q4k = false;
  generic_policy.enable_experimental_q81_grouped_rowpair_w4 = false;

  NativeExecutionPolicy m1_hot_policy = generic_policy;
  m1_hot_policy.enable_experimental_q81_grouped_hot_q4k = true;

  NativeExecutionPolicy m2_rowpair_policy = generic_policy;
  m2_rowpair_policy.enable_experimental_q81_grouped_rowpair_w4 = true;

  const BenchmarkResult m1_generic = RunCase(1, generic_policy);
  const BenchmarkResult m2_generic = RunCase(2, generic_policy);
  const BenchmarkResult m4_generic = RunCase(4, generic_policy);

  if (force_v2) {
    PrintRawCaseSummary("generic", 1, m1_generic, "v2");
    PrintRawCaseSummary("generic", 2, m2_generic, "v2");
    PrintRawCaseSummary("generic", 4, m4_generic, "v2");
    return 0;
  }

  const BenchmarkResult m1_hot = RunCase(1, m1_hot_policy);
  const BenchmarkResult m2_rowpair = RunCase(2, m2_rowpair_policy);
  PrintCaseSummary("single-row hot", 1, m1_generic, m1_hot, "hot");
  std::puts("");
  PrintCaseSummary("row-pair", 2, m2_generic, m2_rowpair, "rowpair");
  std::puts("");
  PrintCaseSummary("generic row-quad baseline", 4, m4_generic, m4_generic,
                   "generic");
  return 0;
#endif
}
