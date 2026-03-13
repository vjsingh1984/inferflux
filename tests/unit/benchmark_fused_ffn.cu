#include "runtime/backends/cuda/native/fused_quant_gemm.h"
#include "runtime/backends/cuda/native/kernels/dequantization.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using inferflux::FusedQuantGemm;
using inferflux::PackedProjectionSpec;
using inferflux::QuantizedWeightInfo;
using inferflux::runtime::cuda::native::block_q4_k;
namespace GGUF = inferflux::runtime::cuda::native::GGUF;

namespace {

constexpr int kWarmupIters = 20;
constexpr int kBenchmarkIters = 100;

unsigned short EncodeHalfBits(float value) {
  const half h = __float2half(value);
  unsigned short bits = 0;
  std::memcpy(&bits, &h, sizeof(bits));
  return bits;
}

std::vector<block_q4_k> MakeQ4Rows(int rows, int blocks_per_row, int seed) {
  std::vector<block_q4_k> blocks(static_cast<size_t>(rows) * blocks_per_row);
  for (int row = 0; row < rows; ++row) {
    for (int blk = 0; blk < blocks_per_row; ++blk) {
      auto &block = blocks[static_cast<size_t>(row) * blocks_per_row + blk];
      block.d =
          EncodeHalfBits(0.02f * static_cast<float>(((row + blk + seed) % 5) + 1));
      block.dmin =
          EncodeHalfBits(0.01f * static_cast<float>(((row + blk + seed) % 3) + 1));
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

std::vector<half> MakeWaveTensor(size_t count, float scale, float bias = 0.0f) {
  std::vector<half> out(count);
  for (size_t i = 0; i < count; ++i) {
    const float value =
        bias + scale * std::sin(0.173f * static_cast<float>(i) + 0.031f);
    out[i] = __float2half(value);
  }
  return out;
}

std::vector<half> CopyDeviceHalfs(const half *device, size_t count) {
  std::vector<half> host(count);
  cudaMemcpy(host.data(), device, count * sizeof(half), cudaMemcpyDeviceToHost);
  return host;
}

struct DeviceBuffers {
  block_q4_k *d_gate{nullptr};
  block_q4_k *d_up{nullptr};
  block_q4_k *d_down{nullptr};
  half *d_input{nullptr};
  half *d_gate_out{nullptr};
  half *d_up_out{nullptr};
  half *d_output_baseline{nullptr};
  half *d_output_fused{nullptr};
  void *d_q8_input{nullptr};
  void *d_q8_ffn{nullptr};
};

void FreeBuffers(DeviceBuffers *buf) {
  cudaFree(buf->d_q8_ffn);
  cudaFree(buf->d_q8_input);
  cudaFree(buf->d_output_fused);
  cudaFree(buf->d_output_baseline);
  cudaFree(buf->d_up_out);
  cudaFree(buf->d_gate_out);
  cudaFree(buf->d_input);
  cudaFree(buf->d_down);
  cudaFree(buf->d_up);
  cudaFree(buf->d_gate);
}

bool RunCurrentPath(const QuantizedWeightInfo &gate_info,
                    const QuantizedWeightInfo &up_info,
                    const QuantizedWeightInfo &down_info, const half *d_input,
                    half *d_gate_out, half *d_up_out, void *d_q8_input,
                    void *d_q8_ffn, half *d_output, int M, int N_inter,
                    int N_hidden, int K, cudaStream_t stream) {
  FusedQuantGemm::QuantizeRowQ8_1(d_input, d_q8_input, M, K, stream);

  const std::array<PackedProjectionSpec, 2> grouped = {{
      {gate_info, d_gate_out, N_inter},
      {up_info, d_up_out, N_inter},
  }};
  if (!FusedQuantGemm::GemvQ8_1Pair(grouped, d_q8_input, M, K, stream)) {
    return false;
  }

  FusedQuantGemm::SiluMulQuantizeQ8_1(d_gate_out, d_up_out, d_q8_ffn, M,
                                      N_inter, stream);
  return FusedQuantGemm::GemvQ8_1(down_info, d_q8_ffn, d_output, M, N_hidden,
                                  N_inter, stream);
}

float BenchmarkMs(void (*fn)(void *), void *ctx, cudaStream_t stream) {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (int i = 0; i < kWarmupIters; ++i) {
    fn(ctx);
  }
  cudaStreamSynchronize(stream);

  cudaEventRecord(start, stream);
  for (int i = 0; i < kBenchmarkIters; ++i) {
    fn(ctx);
  }
  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  cudaEventDestroy(stop);
  cudaEventDestroy(start);
  return ms / static_cast<float>(kBenchmarkIters);
}

struct BaselineContext {
  QuantizedWeightInfo gate_info;
  QuantizedWeightInfo up_info;
  QuantizedWeightInfo down_info;
  const half *d_input;
  half *d_gate_out;
  half *d_up_out;
  void *d_q8_input;
  void *d_q8_ffn;
  half *d_output;
  int M;
  int N_inter;
  int N_hidden;
  int K;
  cudaStream_t stream;
};

struct FusedContext {
  QuantizedWeightInfo gate_info;
  QuantizedWeightInfo up_info;
  QuantizedWeightInfo down_info;
  const half *d_input;
  half *d_output;
  int M;
  int N_inter;
  int N_hidden;
  int K;
  cudaStream_t stream;
};

void RunBaselineThunk(void *opaque) {
  auto *ctx = static_cast<BaselineContext *>(opaque);
  RunCurrentPath(ctx->gate_info, ctx->up_info, ctx->down_info, ctx->d_input,
                 ctx->d_gate_out, ctx->d_up_out, ctx->d_q8_input, ctx->d_q8_ffn,
                 ctx->d_output, ctx->M, ctx->N_inter, ctx->N_hidden, ctx->K,
                 ctx->stream);
}

void RunFusedThunk(void *opaque) {
  auto *ctx = static_cast<FusedContext *>(opaque);
  FusedQuantGemm::FusedFfnQ4K(ctx->gate_info, ctx->up_info, ctx->down_info,
                              ctx->d_input, ctx->d_output, ctx->M,
                              ctx->N_inter, ctx->N_hidden, ctx->K, ctx->stream);
}

bool RunCase(int M, int K, int N_inter, int N_hidden) {
  printf("\nCase: M=%d K=%d N_inter=%d N_hidden=%d\n", M, K, N_inter, N_hidden);

  if ((K % QK_K) != 0 || (N_inter % QK_K) != 0) {
    printf("  invalid geometry for Q4_K\n");
    return false;
  }

  const auto gate = MakeQ4Rows(N_inter, K / QK_K, 3);
  const auto up = MakeQ4Rows(N_inter, K / QK_K, 7);
  const auto down = MakeQ4Rows(N_hidden, N_inter / QK_K, 11);
  const auto input = MakeWaveTensor(static_cast<size_t>(M) * K, 0.015f, -0.0025f);

  DeviceBuffers buf;
  cudaStream_t stream = nullptr;
  cudaStreamCreate(&stream);

  const size_t q8_input_bytes =
      static_cast<size_t>(M) * (K / 32) * sizeof(inferflux::runtime::cuda::native::block_q8_1);
  const size_t q8_ffn_bytes =
      static_cast<size_t>(M) * (N_inter / 32) *
      sizeof(inferflux::runtime::cuda::native::block_q8_1);

  cudaMalloc(reinterpret_cast<void **>(&buf.d_gate),
             gate.size() * sizeof(block_q4_k));
  cudaMalloc(reinterpret_cast<void **>(&buf.d_up), up.size() * sizeof(block_q4_k));
  cudaMalloc(reinterpret_cast<void **>(&buf.d_down),
             down.size() * sizeof(block_q4_k));
  cudaMalloc(reinterpret_cast<void **>(&buf.d_input), input.size() * sizeof(half));
  cudaMalloc(reinterpret_cast<void **>(&buf.d_gate_out),
             static_cast<size_t>(M) * N_inter * sizeof(half));
  cudaMalloc(reinterpret_cast<void **>(&buf.d_up_out),
             static_cast<size_t>(M) * N_inter * sizeof(half));
  cudaMalloc(reinterpret_cast<void **>(&buf.d_output_baseline),
             static_cast<size_t>(M) * N_hidden * sizeof(half));
  cudaMalloc(reinterpret_cast<void **>(&buf.d_output_fused),
             static_cast<size_t>(M) * N_hidden * sizeof(half));
  cudaMalloc(&buf.d_q8_input, q8_input_bytes);
  cudaMalloc(&buf.d_q8_ffn, q8_ffn_bytes);

  cudaMemcpy(buf.d_gate, gate.data(), gate.size() * sizeof(block_q4_k),
             cudaMemcpyHostToDevice);
  cudaMemcpy(buf.d_up, up.data(), up.size() * sizeof(block_q4_k),
             cudaMemcpyHostToDevice);
  cudaMemcpy(buf.d_down, down.data(), down.size() * sizeof(block_q4_k),
             cudaMemcpyHostToDevice);
  cudaMemcpy(buf.d_input, input.data(), input.size() * sizeof(half),
             cudaMemcpyHostToDevice);

  QuantizedWeightInfo gate_info{
      buf.d_gate, static_cast<int>(GGUF::TensorType::Q4_K),
      static_cast<int64_t>(N_inter) * K};
  QuantizedWeightInfo up_info{buf.d_up, static_cast<int>(GGUF::TensorType::Q4_K),
                              static_cast<int64_t>(N_inter) * K};
  QuantizedWeightInfo down_info{
      buf.d_down, static_cast<int>(GGUF::TensorType::Q4_K),
      static_cast<int64_t>(N_hidden) * N_inter};

  BaselineContext baseline{gate_info,
                           up_info,
                           down_info,
                           buf.d_input,
                           buf.d_gate_out,
                           buf.d_up_out,
                           buf.d_q8_input,
                           buf.d_q8_ffn,
                           buf.d_output_baseline,
                           M,
                           N_inter,
                           N_hidden,
                           K,
                           stream};
  FusedContext fused{
      gate_info, up_info, down_info, buf.d_input, buf.d_output_fused,
      M,         N_inter, N_hidden,  K,           stream};

  if (!RunCurrentPath(gate_info, up_info, down_info, buf.d_input, buf.d_gate_out,
                      buf.d_up_out, buf.d_q8_input, buf.d_q8_ffn,
                      buf.d_output_baseline, M, N_inter, N_hidden, K, stream)) {
    printf("  baseline path unavailable\n");
    FreeBuffers(&buf);
    cudaStreamDestroy(stream);
    return false;
  }
  if (!FusedQuantGemm::FusedFfnQ4K(gate_info, up_info, down_info, buf.d_input,
                                   buf.d_output_fused, M, N_inter, N_hidden, K,
                                   stream)) {
    printf("  fused path unavailable\n");
    FreeBuffers(&buf);
    cudaStreamDestroy(stream);
    return false;
  }
  cudaStreamSynchronize(stream);

  const float baseline_ms = BenchmarkMs(RunBaselineThunk, &baseline, stream);
  const float fused_ms = BenchmarkMs(RunFusedThunk, &fused, stream);

  const auto baseline_out =
      CopyDeviceHalfs(buf.d_output_baseline, static_cast<size_t>(M) * N_hidden);
  const auto fused_out =
      CopyDeviceHalfs(buf.d_output_fused, static_cast<size_t>(M) * N_hidden);
  float max_abs_diff = 0.0f;
  for (size_t i = 0; i < baseline_out.size(); ++i) {
    max_abs_diff = std::max(
        max_abs_diff,
        std::fabs(__half2float(baseline_out[i]) - __half2float(fused_out[i])));
  }

  printf("  baseline current FFN path: %.3f ms\n", baseline_ms);
  printf("  fused tiled FFN path:      %.3f ms\n", fused_ms);
  printf("  speedup (baseline/fused):  %.3fx\n", baseline_ms / fused_ms);
  printf("  max abs diff:              %.6f\n", max_abs_diff);

  FreeBuffers(&buf);
  cudaStreamDestroy(stream);
  return true;
}

} // namespace

int main() {
#ifndef INFERFLUX_NATIVE_KERNELS_READY
  std::puts("benchmark_fused_ffn requires InferFlux CUDA kernels");
  return 0;
#else
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    std::puts("No CUDA device available");
    return 0;
  }

  std::puts("========================================");
  std::puts("Fused FFN Microbenchmark");
  std::puts("========================================");
  std::puts("Comparing:");
  std::puts("  1. Current path: Quantize + GemvQ8_1Pair + SiluMulQuantizeQ8_1 + GemvQ8_1");
  std::puts("  2. Bring-up fused path: FusedFfnQ4K");

  bool ok = true;
  ok &= RunCase(/*M=*/1, /*K=*/2048, /*N_inter=*/11008, /*N_hidden=*/2048);
  ok &= RunCase(/*M=*/2, /*K=*/2048, /*N_inter=*/11008, /*N_hidden=*/2048);
  return ok ? 0 : 1;
#endif
}
