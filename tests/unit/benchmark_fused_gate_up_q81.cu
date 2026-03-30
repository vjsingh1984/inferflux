#include "runtime/backends/cuda/native/cuda_kernels.cuh"
#include "runtime/backends/cuda/native/fused_quant_gemm.h"
#include "runtime/backends/cuda/native/kernels/dequantization.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using inferflux::FusedDispatchGeometry;
using inferflux::FusedQuantGemm;
using inferflux::NativeExecutionPolicy;
using inferflux::PackedProjectionSpec;
using inferflux::QuantizedWeightInfo;
namespace native = inferflux::runtime::cuda::native;
namespace GGUF = native::GGUF;

namespace {

constexpr int kWarmupIters = 30;
constexpr int kBenchmarkIters = 200;
constexpr int kHiddenSize = 2048;
constexpr int kIntermediateSize = 11008;
constexpr int kBlocksPerRow = kHiddenSize / QK_K;
constexpr int kQ8BlocksPerRow = kHiddenSize / QK8_1;

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

std::vector<native::block_q4_k> MakeQ4Rows(int rows, int blocks_per_row,
                                           int seed) {
  std::vector<native::block_q4_k> blocks(static_cast<size_t>(rows) *
                                         blocks_per_row);
  for (int row = 0; row < rows; ++row) {
    for (int blk = 0; blk < blocks_per_row; ++blk) {
      auto &block = blocks[static_cast<size_t>(row) * blocks_per_row + blk];
      block.d = EncodeHalfBits(
          0.012f * static_cast<float>(((row + blk + seed) % 5) + 1));
      block.dmin = EncodeHalfBits(
          0.006f * static_cast<float>(((row + blk + seed) % 3) + 1));
      for (int i = 0; i < 12; ++i) {
        block.scales[i] = static_cast<unsigned char>(
            (seed * 17 + row * 7 + blk * 11 + i * 5) & 0xFF);
      }
      for (int i = 0; i < QK_K / 2; ++i) {
        block.qs[i] = static_cast<unsigned char>(
            (seed * 13 + row * 19 + blk * 23 + i * 3) & 0xFF);
      }
    }
  }
  return blocks;
}

std::vector<half> CopyDeviceHalfs(const half *device, size_t count) {
  std::vector<half> host(count);
  cudaMemcpy(host.data(), device, count * sizeof(half), cudaMemcpyDeviceToHost);
  return host;
}

float MaxAbsDiff(const std::vector<half> &lhs, const std::vector<half> &rhs) {
  float max_abs_diff = 0.0f;
  for (size_t i = 0; i < lhs.size(); ++i) {
    max_abs_diff = std::max(
        max_abs_diff, std::fabs(__half2float(lhs[i]) - __half2float(rhs[i])));
  }
  return max_abs_diff;
}

BenchmarkResult RunSeparatePathBenchmark(const QuantizedWeightInfo &gate_info,
                                         const QuantizedWeightInfo &up_info,
                                         const void *d_act_q8_1, int m,
                                         const NativeExecutionPolicy &policy,
                                         cudaStream_t stream) {
  half *d_gate = nullptr;
  half *d_up = nullptr;
  half *d_out = nullptr;
  cudaMalloc(reinterpret_cast<void **>(&d_gate),
             static_cast<size_t>(m) * kIntermediateSize * sizeof(half));
  cudaMalloc(reinterpret_cast<void **>(&d_up),
             static_cast<size_t>(m) * kIntermediateSize * sizeof(half));
  cudaMalloc(reinterpret_cast<void **>(&d_out),
             static_cast<size_t>(m) * kIntermediateSize * sizeof(half));

  const std::array<PackedProjectionSpec, 2> projections = {{
      {gate_info, d_gate, kIntermediateSize},
      {up_info, d_up, kIntermediateSize},
  }};

  auto bench_ms = [&]() {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < kWarmupIters; ++i) {
      FusedQuantGemm::GemvQ8_1Pair(projections, d_act_q8_1, m, kHiddenSize,
                                   stream, &policy);
      inferflux::cuda_kernel::SiluMul(d_gate, d_up, d_out,
                                      m * kIntermediateSize, stream);
    }
    cudaStreamSynchronize(stream);
    cudaEventRecord(start, stream);
    for (int i = 0; i < kBenchmarkIters; ++i) {
      FusedQuantGemm::GemvQ8_1Pair(projections, d_act_q8_1, m, kHiddenSize,
                                   stream, &policy);
      inferflux::cuda_kernel::SiluMul(d_gate, d_up, d_out,
                                      m * kIntermediateSize, stream);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    return ms / static_cast<float>(kBenchmarkIters);
  };

  const float mean_ms = bench_ms();
  auto output = CopyDeviceHalfs(d_out, static_cast<size_t>(m) * kIntermediateSize);

  cudaFree(d_out);
  cudaFree(d_up);
  cudaFree(d_gate);
  return BenchmarkResult{mean_ms, std::move(output)};
}

BenchmarkResult RunFusedPathBenchmark(const QuantizedWeightInfo &gate_info,
                                      const QuantizedWeightInfo &up_info,
                                      const void *d_act_q8_1, int m,
                                      cudaStream_t stream) {
  half *d_out = nullptr;
  cudaMalloc(reinterpret_cast<void **>(&d_out),
             static_cast<size_t>(m) * kIntermediateSize * sizeof(half));

  auto bench_ms = [&]() {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < kWarmupIters; ++i) {
      FusedQuantGemm::FusedGateUpSiluGemvQ8_1(
          gate_info, up_info, d_act_q8_1, d_out, m, kIntermediateSize,
          kHiddenSize, stream);
    }
    cudaStreamSynchronize(stream);
    cudaEventRecord(start, stream);
    for (int i = 0; i < kBenchmarkIters; ++i) {
      FusedQuantGemm::FusedGateUpSiluGemvQ8_1(
          gate_info, up_info, d_act_q8_1, d_out, m, kIntermediateSize,
          kHiddenSize, stream);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    return ms / static_cast<float>(kBenchmarkIters);
  };

  const float mean_ms = bench_ms();
  auto output = CopyDeviceHalfs(d_out, static_cast<size_t>(m) * kIntermediateSize);

  cudaFree(d_out);
  return BenchmarkResult{mean_ms, std::move(output)};
}

BenchmarkResult RunFusedEpilogueBenchmark(const QuantizedWeightInfo &gate_info,
                                          const QuantizedWeightInfo &up_info,
                                          const void *d_act_q8_1, int m,
                                          cudaStream_t stream) {
  half *d_out = nullptr;
  void *d_act_q8_1_out = nullptr;
  cudaMalloc(reinterpret_cast<void **>(&d_out),
             static_cast<size_t>(m) * kIntermediateSize * sizeof(half));
  cudaMalloc(&d_act_q8_1_out,
             static_cast<size_t>(m) * (kIntermediateSize / QK8_1) *
                 sizeof(native::block_q8_1));

  auto bench_ms = [&]() {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < kWarmupIters; ++i) {
      FusedQuantGemm::FusedGateUpSiluGemvQ8_1WithEpilogue(
          gate_info, up_info, d_act_q8_1, d_out, d_act_q8_1_out, m,
          kIntermediateSize, kHiddenSize, stream);
    }
    cudaStreamSynchronize(stream);
    cudaEventRecord(start, stream);
    for (int i = 0; i < kBenchmarkIters; ++i) {
      FusedQuantGemm::FusedGateUpSiluGemvQ8_1WithEpilogue(
          gate_info, up_info, d_act_q8_1, d_out, d_act_q8_1_out, m,
          kIntermediateSize, kHiddenSize, stream);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    return ms / static_cast<float>(kBenchmarkIters);
  };

  const float mean_ms = bench_ms();
  auto output = CopyDeviceHalfs(d_out, static_cast<size_t>(m) * kIntermediateSize);

  cudaFree(d_act_q8_1_out);
  cudaFree(d_out);
  return BenchmarkResult{mean_ms, std::move(output)};
}

void RunCase(int m) {
  std::printf("\nCase: M=%d N=%d K=%d\n", m, kIntermediateSize, kHiddenSize);

  native::block_q4_k *d_gate = nullptr;
  native::block_q4_k *d_up = nullptr;
  half *d_input = nullptr;
  void *d_act_q8_1 = nullptr;
  cudaStream_t stream = nullptr;
  cudaStreamCreate(&stream);

  const auto gate = MakeQ4Rows(kIntermediateSize, kBlocksPerRow, 3);
  const auto up = MakeQ4Rows(kIntermediateSize, kBlocksPerRow, 7);
  const auto input = MakeWaveTensor(static_cast<size_t>(m) * kHiddenSize,
                                    0.015f, -0.002f);

  cudaMalloc(reinterpret_cast<void **>(&d_gate),
             gate.size() * sizeof(native::block_q4_k));
  cudaMalloc(reinterpret_cast<void **>(&d_up),
             up.size() * sizeof(native::block_q4_k));
  cudaMalloc(reinterpret_cast<void **>(&d_input), input.size() * sizeof(half));
  cudaMalloc(&d_act_q8_1, static_cast<size_t>(m) * kQ8BlocksPerRow *
                              sizeof(native::block_q8_1));

  cudaMemcpy(d_gate, gate.data(), gate.size() * sizeof(native::block_q4_k),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_up, up.data(), up.size() * sizeof(native::block_q4_k),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_input, input.data(), input.size() * sizeof(half),
             cudaMemcpyHostToDevice);
  FusedQuantGemm::QuantizeRowQ8_1(d_input, d_act_q8_1, m, kHiddenSize, stream);
  cudaStreamSynchronize(stream);

  const QuantizedWeightInfo gate_info{
      d_gate, static_cast<int>(GGUF::TensorType::Q4_K),
      static_cast<int64_t>(kIntermediateSize) * kHiddenSize};
  const QuantizedWeightInfo up_info{
      d_up, static_cast<int>(GGUF::TensorType::Q4_K),
      static_cast<int64_t>(kIntermediateSize) * kHiddenSize};

  NativeExecutionPolicy default_policy;
  const auto separate_op = FusedQuantGemm::SelectFfnProjOperator(
      static_cast<int>(GGUF::TensorType::Q4_K),
      static_cast<int>(GGUF::TensorType::Q4_K),
      FusedDispatchGeometry{m, kIntermediateSize, kHiddenSize, 2, true, false},
      true, false, &default_policy);
  std::printf("  separate grouped op: %s\n",
              FusedQuantGemm::FfnProjOperatorName(separate_op));

  const BenchmarkResult separate =
      RunSeparatePathBenchmark(gate_info, up_info, d_act_q8_1, m,
                               default_policy, stream);
  const BenchmarkResult fused =
      RunFusedPathBenchmark(gate_info, up_info, d_act_q8_1, m, stream);
  const BenchmarkResult epilogue =
      RunFusedEpilogueBenchmark(gate_info, up_info, d_act_q8_1, m, stream);

  const float fused_diff = MaxAbsDiff(separate.output, fused.output);
  const float epilogue_diff = MaxAbsDiff(separate.output, epilogue.output);

  std::printf("  separate pair+silu: %.3f ms\n", separate.mean_ms);
  std::printf("  fused gate/up:      %.3f ms (%.3fx, max diff %.6f)\n",
              fused.mean_ms, separate.mean_ms / fused.mean_ms, fused_diff);
  std::printf("  fused + q8 epilogue: %.3f ms (%.3fx, max diff %.6f)\n",
              epilogue.mean_ms, separate.mean_ms / epilogue.mean_ms,
              epilogue_diff);

  cudaFree(d_act_q8_1);
  cudaFree(d_input);
  cudaFree(d_up);
  cudaFree(d_gate);
  cudaStreamDestroy(stream);
}

} // namespace

int main(int argc, char **argv) {
  std::puts("========================================");
  std::puts("Fused Gate+Up+SiLU Q8_1 Benchmark");
  std::puts("========================================");

#ifndef INFERFLUX_HAS_CUDA
  std::puts("benchmark_fused_gate_up_q81 requires InferFlux CUDA kernels");
  return 0;
#else
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    std::puts("No CUDA device available");
    return 0;
  }

  std::vector<int> cases = {1, 2, 4, 8};
  if (argc > 1) {
    cases.clear();
    for (int i = 1; i < argc; ++i) {
      const int m = std::atoi(argv[i]);
      if (m > 0) {
        cases.push_back(m);
      }
    }
    if (cases.empty()) {
      std::fprintf(stderr, "usage: %s [M...]\n", argv[0]);
      return 1;
    }
  }

  for (int m : cases) {
    RunCase(m);
  }
  return 0;
#endif
}
