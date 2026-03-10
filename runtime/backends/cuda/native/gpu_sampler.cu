#include "runtime/backends/cuda/native/gpu_sampler.h"
#include "runtime/backends/cuda/native/nvtx_scoped.h"

#include "server/logging/logger.h"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>

namespace inferflux {

// ============================================================================
// Kernels
// ============================================================================

// Temperature scaling: logits /= temperature
__global__ void TemperatureScaleKernel(float *__restrict__ logits,
                                       int vocab_size, float temperature) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= vocab_size)
    return;
  logits[idx] /= temperature;
}

// Softmax: two-pass (find max, then exp and sum)
__global__ void SoftmaxMaxKernel(const float *__restrict__ logits,
                                 float *__restrict__ max_val, int vocab_size) {
  extern __shared__ float smem[];
  int tid = threadIdx.x;

  float local_max = -FLT_MAX;
  for (int i = tid; i < vocab_size; i += blockDim.x) {
    local_max = fmaxf(local_max, logits[i]);
  }
  smem[tid] = local_max;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      smem[tid] = fmaxf(smem[tid], smem[tid + s]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    *max_val = smem[0];
  }
}

__global__ void SoftmaxExpSumKernel(const float *__restrict__ logits,
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
    if (tid < s) {
      smem[tid] += smem[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    *sum_val = smem[0];
  }
}

__global__ void SoftmaxNormKernel(float *__restrict__ probs,
                                  const float *__restrict__ sum_val,
                                  int vocab_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= vocab_size)
    return;
  probs[idx] /= *sum_val;
}

// Argmax: find token with highest probability
__global__ void ArgmaxKernel(const float *__restrict__ logits,
                             int *__restrict__ result, int vocab_size) {
  extern __shared__ char smem_raw[];
  float *s_vals = reinterpret_cast<float *>(smem_raw);
  int *s_idxs = reinterpret_cast<int *>(s_vals + blockDim.x);
  int tid = threadIdx.x;

  float best_val = -FLT_MAX;
  int best_idx = 0;
  for (int i = tid; i < vocab_size; i += blockDim.x) {
    if (logits[i] > best_val) {
      best_val = logits[i];
      best_idx = i;
    }
  }
  s_vals[tid] = best_val;
  s_idxs[tid] = best_idx;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      if (s_vals[tid + s] > s_vals[tid]) {
        s_vals[tid] = s_vals[tid + s];
        s_idxs[tid] = s_idxs[tid + s];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    *result = s_idxs[0];
  }
}

// Top-k: zero out all entries below the k-th largest
// Simple approach: find k-th threshold then zero below it
__global__ void TopKMaskKernel(float *__restrict__ probs, int vocab_size,
                               int top_k, float threshold) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= vocab_size)
    return;
  if (probs[idx] < threshold) {
    probs[idx] = 0.0f;
  }
}

// Top-p: zero out entries where cumulative sum exceeds p
// Requires sorted probs. Simple single-thread approach for correctness.
__global__ void TopPMaskKernel(float *__restrict__ probs, int vocab_size,
                               float top_p) {
  // Single thread scans sorted probs
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;

  // Simple bubble-sort approach for small vocab isn't practical.
  // Instead, scan from max prob downward using the unsorted array.
  // This is approximate but functional.
  float cumsum = 0.0f;
  // We need to iterate in sorted order. For a production kernel,
  // use CUB radix sort. Here we use a simple iterative max approach.
  // Mark visited entries by negating them temporarily.

  // Copy probs to find top-p threshold
  for (int iter = 0; iter < vocab_size && cumsum < top_p; iter++) {
    float max_val = 0.0f;
    int max_idx = -1;
    for (int i = 0; i < vocab_size; i++) {
      if (probs[i] > max_val) {
        max_val = probs[i];
        max_idx = i;
      }
    }
    if (max_idx < 0)
      break;
    cumsum += max_val;
    probs[max_idx] = -probs[max_idx]; // Mark as visited
  }

  // Restore marked entries, zero those below threshold
  for (int i = 0; i < vocab_size; i++) {
    if (probs[i] < 0.0f) {
      probs[i] = -probs[i]; // Restore
    } else {
      probs[i] = 0.0f; // Zero out
    }
  }
}

// Batched argmax: one block per sequence, parallel reduction within each block
__global__ void BatchedArgmaxKernel(const float *__restrict__ logits,
                                    int *__restrict__ results, int vocab_size,
                                    int batch_size) {
  int seq = blockIdx.x;
  if (seq >= batch_size)
    return;
  extern __shared__ char smem_raw[];
  float *s_vals = reinterpret_cast<float *>(smem_raw);
  int *s_idxs = reinterpret_cast<int *>(s_vals + blockDim.x);
  int tid = threadIdx.x;

  const float *row = logits + seq * vocab_size;

  float best_val = -FLT_MAX;
  int best_idx = 0;
  for (int i = tid; i < vocab_size; i += blockDim.x) {
    if (row[i] > best_val) {
      best_val = row[i];
      best_idx = i;
    }
  }
  s_vals[tid] = best_val;
  s_idxs[tid] = best_idx;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      if (s_vals[tid + s] > s_vals[tid]) {
        s_vals[tid] = s_vals[tid + s];
        s_idxs[tid] = s_idxs[tid + s];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    results[seq] = s_idxs[0];
  }
}

// Multinomial sample: walk CDF until uniform < cumsum
__global__ void MultinomialSampleKernel(const float *__restrict__ probs,
                                        const float *__restrict__ uniform,
                                        int *__restrict__ result,
                                        int vocab_size) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;

  float u = *uniform;
  float cumsum = 0.0f;
  // First renormalize
  float total = 0.0f;
  for (int i = 0; i < vocab_size; i++) {
    total += probs[i];
  }
  if (total <= 0.0f) {
    *result = 0;
    return;
  }

  for (int i = 0; i < vocab_size; i++) {
    cumsum += probs[i] / total;
    if (u < cumsum) {
      *result = i;
      return;
    }
  }
  *result = vocab_size - 1; // Fallback to last token
}

// ============================================================================
// GpuSampler implementation
// ============================================================================

GpuSampler::~GpuSampler() {
  if (d_probs_)
    cudaFree(d_probs_);
  if (d_indices_)
    cudaFree(d_indices_);
  if (d_temp_)
    cudaFree(d_temp_);
  if (d_result_)
    cudaFree(d_result_);
  if (d_max_val_)
    cudaFree(d_max_val_);
  if (d_max_idx_)
    cudaFree(d_max_idx_);
  if (d_result_batch_)
    cudaFree(d_result_batch_);
  if (d_uniform_)
    cudaFree(d_uniform_);
  if (rng_initialized_)
    curandDestroyGenerator(rng_);
}

bool GpuSampler::Initialize(int vocab_size, cudaStream_t stream) {
  vocab_size_ = vocab_size;
  stream_ = stream;

  cudaError_t err;
  err = cudaMalloc(&d_probs_, vocab_size * sizeof(float));
  if (err != cudaSuccess)
    return false;
  err = cudaMalloc(&d_temp_, vocab_size * sizeof(float));
  if (err != cudaSuccess)
    return false;
  err = cudaMalloc(&d_result_, sizeof(int));
  if (err != cudaSuccess)
    return false;
  err = cudaMalloc(&d_max_val_, sizeof(float));
  if (err != cudaSuccess)
    return false;
  err = cudaMalloc(&d_max_idx_, sizeof(int));
  if (err != cudaSuccess)
    return false;
  err = cudaMalloc(&d_uniform_, sizeof(float));
  if (err != cudaSuccess)
    return false;
  err = cudaMalloc(&d_result_batch_, kMaxBatchSize * sizeof(int));
  if (err != cudaSuccess)
    return false;

  // Initialize cuRAND
  curandStatus_t rng_st =
      curandCreateGenerator(&rng_, CURAND_RNG_PSEUDO_DEFAULT);
  if (rng_st != CURAND_STATUS_SUCCESS) {
    log::Error("gpu_sampler", "curandCreateGenerator failed");
    return false;
  }
  curandSetStream(rng_, stream);
  rng_initialized_ = true;

  log::Info("gpu_sampler",
            "Initialized: vocab_size=" + std::to_string(vocab_size));
  return true;
}

int GpuSampler::GreedyArgmax(const float *d_logits) {
  NVTX_SCOPE("Sampler_Argmax");
  int threads = 256;
  int smem = threads * (sizeof(float) + sizeof(int));

  ArgmaxKernel<<<1, threads, smem, stream_>>>(d_logits, d_result_, vocab_size_);

  cudaMemcpyAsync(&h_result_, d_result_, sizeof(int), cudaMemcpyDeviceToHost,
                  stream_);
  cudaStreamSynchronize(stream_);
  return h_result_;
}

int GpuSampler::StochasticSample(const float *d_logits, float temperature,
                                 int top_k, float top_p) {
  NVTX_SCOPE("Sampler_Stochastic");
  int threads = 256;
  int blocks = (vocab_size_ + threads - 1) / threads;

  // Step 1: Copy logits to probs buffer
  cudaMemcpyAsync(d_probs_, d_logits, vocab_size_ * sizeof(float),
                  cudaMemcpyDeviceToDevice, stream_);

  // Step 2: Temperature scaling
  if (temperature != 1.0f) {
    TemperatureScaleKernel<<<blocks, threads, 0, stream_>>>(
        d_probs_, vocab_size_, temperature);
  }

  // Step 3: Softmax (no host sync — sum is always positive for valid logits)
  int smem = threads * sizeof(float);
  SoftmaxMaxKernel<<<1, threads, smem, stream_>>>(d_probs_, d_max_val_,
                                                  vocab_size_);
  SoftmaxExpSumKernel<<<1, threads, smem, stream_>>>(
      d_probs_, d_temp_, d_max_val_, d_max_val_, vocab_size_);
  // d_max_val_ reused as sum_val — SoftmaxNormKernel reads it on device
  SoftmaxNormKernel<<<blocks, threads, 0, stream_>>>(d_temp_, d_max_val_,
                                                     vocab_size_);

  // Copy normalized probs back
  cudaMemcpyAsync(d_probs_, d_temp_, vocab_size_ * sizeof(float),
                  cudaMemcpyDeviceToDevice, stream_);

  // Step 4: Top-k filtering (if enabled)
  if (top_k > 0 && top_k < vocab_size_) {
    // Find k-th largest value on CPU (for simplicity)
    // In production, use CUB radix sort
    // For now, skip top-k and rely on top-p
  }

  // Step 5: Top-p / nucleus filtering (if enabled)
  if (top_p < 1.0f && top_p > 0.0f) {
    TopPMaskKernel<<<1, 1, 0, stream_>>>(d_probs_, vocab_size_, top_p);
  }

  // Step 6: Generate random uniform
  curandGenerateUniform(rng_, d_uniform_, 1);

  // Step 7: Multinomial sample
  MultinomialSampleKernel<<<1, 1, 0, stream_>>>(d_probs_, d_uniform_, d_result_,
                                                vocab_size_);

  cudaMemcpyAsync(&h_result_, d_result_, sizeof(int), cudaMemcpyDeviceToHost,
                  stream_);
  cudaStreamSynchronize(stream_);
  return h_result_;
}

int GpuSampler::Sample(const float *d_logits, float temperature, int top_k,
                       float top_p, uint32_t seed) {
  if (seed != UINT32_MAX && rng_initialized_) {
    curandSetPseudoRandomGeneratorSeed(rng_, seed);
  }

  if (temperature <= 0.0f) {
    return GreedyArgmax(d_logits);
  }

  return StochasticSample(d_logits, temperature, top_k, top_p);
}

void GpuSampler::GreedyArgmaxBatch(const float *d_logits, int batch_size,
                                   std::vector<int> *out_tokens) {
  NVTX_SCOPE("Sampler_BatchedArgmax");
  int B = std::min(batch_size, kMaxBatchSize);
  int threads = 256;
  int smem = threads * (sizeof(float) + sizeof(int));

  BatchedArgmaxKernel<<<B, threads, smem, stream_>>>(d_logits, d_result_batch_,
                                                     vocab_size_, B);

  cudaMemcpyAsync(h_result_batch_, d_result_batch_, B * sizeof(int),
                  cudaMemcpyDeviceToHost, stream_);
  cudaStreamSynchronize(stream_);

  out_tokens->resize(B);
  for (int i = 0; i < B; ++i) {
    (*out_tokens)[i] = h_result_batch_[i];
  }
}

void GpuSampler::SampleBatch(const float *d_logits, int batch_size,
                             const std::vector<float> &temperatures,
                             const std::vector<int> &top_ks,
                             const std::vector<float> &top_ps,
                             const std::vector<uint32_t> &seeds,
                             std::vector<int> *out_tokens) {
  NVTX_SCOPE("SampleBatch");
  out_tokens->resize(batch_size);

  // Fast path: if ALL sequences are greedy, use batched kernel (1 sync total)
  bool all_greedy = std::all_of(temperatures.begin(), temperatures.end(),
                                [](float t) { return t <= 0.0f; });
  if (all_greedy) {
    GreedyArgmaxBatch(d_logits, batch_size, out_tokens);
    return;
  }

  // Fallback: per-sequence sampling (stochastic needs per-seq state)
  for (int i = 0; i < batch_size; ++i) {
    const float *logits_i = d_logits + i * vocab_size_;
    const uint32_t seed =
        i < static_cast<int>(seeds.size()) ? seeds[static_cast<size_t>(i)]
                                           : UINT32_MAX;
    (*out_tokens)[i] =
        Sample(logits_i, temperatures[i], top_ks[i], top_ps[i], seed);
  }
}

void GpuSampler::CopyLogitsToHost(const float *d_logits, float *host_buf) {
  NVTX_SCOPE("Sampler_CopyLogits");
  cudaMemcpyAsync(host_buf, d_logits, vocab_size_ * sizeof(float),
                  cudaMemcpyDeviceToHost, stream_);
  cudaStreamSynchronize(stream_);
}

} // namespace inferflux
