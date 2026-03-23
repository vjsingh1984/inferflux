#pragma once

// Test-only CUDA kernel declarations for fused kernel parity tests.
// Kernels are defined in test_fused_kernels_cuda.cu and called from
// test_fused_kernels.cpp (Catch2, compiled by MSVC not nvcc).

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace inferflux {
namespace test_cuda {

// Unfused softmax pipeline kernels (mirrors gpu_sampler.cu internals)
void LaunchTemperatureScale(float *logits, int vocab_size, float temperature,
                            cudaStream_t stream);
void LaunchSoftmaxMax(const float *logits, float *max_val, int vocab_size,
                      cudaStream_t stream);
void LaunchSoftmaxExpSum(const float *logits, float *probs,
                         const float *max_val, float *sum_val, int vocab_size,
                         cudaStream_t stream);
void LaunchSoftmaxNorm(float *probs, const float *sum_val, int vocab_size,
                       cudaStream_t stream);
void LaunchFusedSoftmax(const float *logits, float *probs, int vocab_size,
                        float temperature, cudaStream_t stream);

} // namespace test_cuda
} // namespace inferflux
