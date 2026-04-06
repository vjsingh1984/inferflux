/**
 * Test-only CUDA kernels for fused kernel parity tests.
 * Compiled by nvcc as a .cu file (no Catch2 dependency).
 * Host wrappers are called from test_fused_kernels.cpp (Catch2, MSVC).
 */

#include "tests/unit/test_fused_kernels_cuda.cuh"

#include <cfloat>

namespace inferflux {
namespace test_cuda {

// ============================================================================
// Unfused softmax pipeline kernels
// ============================================================================

static __global__ void TemperatureScaleKernel(float *__restrict__ logits,
                                              int vocab_size,
                                              float temperature) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= vocab_size)
    return;
  logits[idx] /= temperature;
}

static __global__ void SoftmaxMaxKernel(const float *__restrict__ logits,
                                        float *__restrict__ max_val,
                                        int vocab_size) {
  extern __shared__ float smem[];
  int tid = threadIdx.x;
  float local_max = -FLT_MAX;
  for (int i = tid; i < vocab_size; i += blockDim.x)
    local_max = fmaxf(local_max, logits[i]);
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

static __global__ void SoftmaxExpSumKernel(const float *__restrict__ logits,
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

static __global__ void SoftmaxNormKernel(float *__restrict__ probs,
                                         const float *__restrict__ sum_val,
                                         int vocab_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= vocab_size)
    return;
  probs[idx] /= *sum_val;
}

static __global__ void FusedSoftmaxKernel(const float *__restrict__ logits,
                                          float *__restrict__ probs,
                                          int vocab_size, float temperature) {
  extern __shared__ float smem[];
  int tid = threadIdx.x;

  float local_max = -FLT_MAX;
  for (int i = tid; i < vocab_size; i += blockDim.x) {
    float v = logits[i] / temperature;
    probs[i] = v;
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

  for (int i = tid; i < vocab_size; i += blockDim.x)
    probs[i] /= gsum;
}

// ============================================================================
// Host wrappers
// ============================================================================

void LaunchTemperatureScale(float *logits, int vocab_size, float temperature,
                            cudaStream_t stream) {
  int threads = 256;
  int blocks = (vocab_size + threads - 1) / threads;
  TemperatureScaleKernel<<<blocks, threads, 0, stream>>>(logits, vocab_size,
                                                          temperature);
}

void LaunchSoftmaxMax(const float *logits, float *max_val, int vocab_size,
                      cudaStream_t stream) {
  int threads = 256;
  int smem = threads * sizeof(float);
  SoftmaxMaxKernel<<<1, threads, smem, stream>>>(logits, max_val, vocab_size);
}

void LaunchSoftmaxExpSum(const float *logits, float *probs,
                         const float *max_val, float *sum_val, int vocab_size,
                         cudaStream_t stream) {
  int threads = 256;
  int smem = threads * sizeof(float);
  SoftmaxExpSumKernel<<<1, threads, smem, stream>>>(logits, probs, max_val,
                                                     sum_val, vocab_size);
}

void LaunchSoftmaxNorm(float *probs, const float *sum_val, int vocab_size,
                       cudaStream_t stream) {
  int threads = 256;
  int blocks = (vocab_size + threads - 1) / threads;
  SoftmaxNormKernel<<<blocks, threads, 0, stream>>>(probs, sum_val,
                                                     vocab_size);
}

void LaunchFusedSoftmax(const float *logits, float *probs, int vocab_size,
                        float temperature, cudaStream_t stream) {
  int threads = 256;
  int smem = threads * sizeof(float);
  FusedSoftmaxKernel<<<1, threads, smem, stream>>>(logits, probs, vocab_size,
                                                    temperature);
}

} // namespace test_cuda
} // namespace inferflux
