#include "runtime/backends/cuda/common/dtype_traits.cuh"
#include "runtime/backends/cuda/native/cuda_kernels.cuh"
#include <cmath>

namespace inferflux {
namespace cuda_kernel {

// ============================================================================
// RMS Normalization (templated)
// ============================================================================

template <typename T>
__global__ void
RmsNormKernel(const T *__restrict__ input, const T *__restrict__ weight,
              T *__restrict__ output, int hidden_size, float eps) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const T *x = input + row * hidden_size;
  T *y = output + row * hidden_size;

  extern __shared__ float shared[];

  float local_sum = 0.0f;
  for (int i = tid; i < hidden_size; i += blockDim.x) {
    float val = DtypeTraits<T>::to_float(x[i]);
    local_sum += val * val;
  }
  shared[tid] = local_sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared[tid] += shared[tid + s];
    }
    __syncthreads();
  }

  float rms = rsqrtf(shared[0] / static_cast<float>(hidden_size) + eps);

  for (int i = tid; i < hidden_size; i += blockDim.x) {
    float val = DtypeTraits<T>::to_float(x[i]) * rms *
                DtypeTraits<T>::to_float(weight[i]);
    y[i] = DtypeTraits<T>::from_float(val);
  }
}

template <typename T>
cudaError_t RmsNorm(const T *input, const T *weight, T *output, int count,
                    int hidden_size, float eps, cudaStream_t stream) {
  int threads = min(1024, hidden_size);
  int t = 1;
  while (t < threads)
    t <<= 1;
  threads = t;
  int smem = threads * sizeof(float);

  RmsNormKernel<T><<<count, threads, smem, stream>>>(input, weight, output,
                                                     hidden_size, eps);
  return cudaGetLastError();
}

// ============================================================================
// Rotary Position Embedding (RoPE) (templated)
// ============================================================================

// RoPE kernel supporting both pairing strategies:
//   rope_type=0 (NORM):  consecutive pairs (0,1),(2,3),... — LLaMA family
//   rope_type=2 (NEOX):  split-half pairs (0,d/2),(1,d/2+1),... —
//   GPT-NeoX/Falcon/Qwen
template <typename T>
__global__ void RoPEKernel(T *__restrict__ q, T *__restrict__ k, int seq_len,
                           int num_heads, int num_kv_heads, int head_dim,
                           int n_past, float freq_base, int rope_type,
                           float freq_scale) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int half_dim = head_dim / 2;
  const int total_q_pairs = seq_len * num_heads * half_dim;
  const int total_k_pairs = seq_len * num_kv_heads * half_dim;
  const int total_pairs = total_q_pairs + total_k_pairs;

  if (idx >= total_pairs)
    return;

  bool is_q = (idx < total_q_pairs);
  int local_idx = is_q ? idx : (idx - total_q_pairs);
  int n_heads = is_q ? num_heads : num_kv_heads;
  T *tensor = is_q ? q : k;

  int pair_idx = local_idx % half_dim;
  int head_idx = (local_idx / half_dim) % n_heads;
  int pos_idx = local_idx / (half_dim * n_heads);

  int position = n_past + pos_idx;
  float freq =
      (1.0f / powf(freq_base, 2.0f * pair_idx / static_cast<float>(head_dim))) *
      freq_scale;
  float angle = position * freq;
  float cos_val = cosf(angle);
  float sin_val = sinf(angle);

  int offset = pos_idx * n_heads * head_dim + head_idx * head_dim;
  int i0, i1;
  if (rope_type == 2) {
    // NEOX: split-half — pair (pair_idx) with (pair_idx + half_dim)
    i0 = offset + pair_idx;
    i1 = offset + pair_idx + half_dim;
  } else {
    // NORM: consecutive — pair (2*pair_idx) with (2*pair_idx + 1)
    i0 = offset + 2 * pair_idx;
    i1 = offset + 2 * pair_idx + 1;
  }

  float v0 = DtypeTraits<T>::to_float(tensor[i0]);
  float v1 = DtypeTraits<T>::to_float(tensor[i1]);

  tensor[i0] = DtypeTraits<T>::from_float(v0 * cos_val - v1 * sin_val);
  tensor[i1] = DtypeTraits<T>::from_float(v0 * sin_val + v1 * cos_val);
}

template <typename T>
cudaError_t RoPE(T *q, T *k, int seq_len, int num_heads, int num_kv_heads,
                 int head_dim, int n_past, float freq_base, cudaStream_t stream,
                 int rope_type, float freq_scale) {
  int half_dim = head_dim / 2;
  int total_pairs =
      seq_len * num_heads * half_dim + seq_len * num_kv_heads * half_dim;
  int threads = 256;
  int blocks = (total_pairs + threads - 1) / threads;

  RoPEKernel<T><<<blocks, threads, 0, stream>>>(
      q, k, seq_len, num_heads, num_kv_heads, head_dim, n_past, freq_base,
      rope_type, freq_scale);
  return cudaGetLastError();
}

// ============================================================================
// Batched RoPE: one kernel launch for B sequences with per-sequence n_past
// ============================================================================

// Each thread handles one (batch, head_type, head, pair) combination.
// Q layout: [B, num_heads * head_dim], K layout: [B, num_kv_heads * head_dim]
// Each sequence has seq_len=1 (decode batch).
template <typename T>
__global__ void BatchRoPEKernel(T *__restrict__ q, T *__restrict__ k,
                                int batch_size, int num_heads, int num_kv_heads,
                                int head_dim, const int *__restrict__ d_n_past,
                                float freq_base, int rope_type,
                                float freq_scale) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int half_dim = head_dim / 2;
  const int pairs_per_seq_q = num_heads * half_dim;
  const int pairs_per_seq_k = num_kv_heads * half_dim;
  const int pairs_per_seq = pairs_per_seq_q + pairs_per_seq_k;
  const int total_pairs = batch_size * pairs_per_seq;

  if (idx >= total_pairs)
    return;

  int batch_idx = idx / pairs_per_seq;
  int local_idx = idx % pairs_per_seq;

  bool is_q = (local_idx < pairs_per_seq_q);
  int within = is_q ? local_idx : (local_idx - pairs_per_seq_q);
  int n_heads = is_q ? num_heads : num_kv_heads;

  int pair_idx = within % half_dim;
  int head_idx = within / half_dim;

  int position = d_n_past[batch_idx]; // seq_len=1, pos_idx=0
  float freq =
      (1.0f / powf(freq_base, 2.0f * pair_idx / static_cast<float>(head_dim))) *
      freq_scale;
  float angle = position * freq;
  float cos_val = cosf(angle);
  float sin_val = sinf(angle);

  // Tensor pointer: row = batch_idx, offset within row = head_idx * head_dim
  int stride = is_q ? (num_heads * head_dim) : (num_kv_heads * head_dim);
  T *tensor = is_q ? (q + batch_idx * stride) : (k + batch_idx * stride);
  int base = head_idx * head_dim;

  int i0, i1;
  if (rope_type == 2) {
    i0 = base + pair_idx;
    i1 = base + pair_idx + half_dim;
  } else {
    i0 = base + 2 * pair_idx;
    i1 = base + 2 * pair_idx + 1;
  }

  float v0 = DtypeTraits<T>::to_float(tensor[i0]);
  float v1 = DtypeTraits<T>::to_float(tensor[i1]);

  tensor[i0] = DtypeTraits<T>::from_float(v0 * cos_val - v1 * sin_val);
  tensor[i1] = DtypeTraits<T>::from_float(v0 * sin_val + v1 * cos_val);
}

template <typename T>
cudaError_t BatchRoPE(T *q, T *k, int batch_size, int num_heads,
                      int num_kv_heads, int head_dim, const int *d_n_past,
                      float freq_base, cudaStream_t stream, int rope_type,
                      float freq_scale) {
  int half_dim = head_dim / 2;
  int total_pairs = batch_size * (num_heads + num_kv_heads) * half_dim;
  int threads = 256;
  int blocks = (total_pairs + threads - 1) / threads;

  BatchRoPEKernel<T><<<blocks, threads, 0, stream>>>(
      q, k, batch_size, num_heads, num_kv_heads, head_dim, d_n_past, freq_base,
      rope_type, freq_scale);
  return cudaGetLastError();
}

// ============================================================================
// SwiGLU: silu(gate) * up (templated)
// ============================================================================

template <typename T>
__global__ void SiluMulKernel(const T *__restrict__ gate,
                              const T *__restrict__ up, T *__restrict__ output,
                              int count) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count)
    return;

  float g = DtypeTraits<T>::to_float(gate[idx]);
  float u = DtypeTraits<T>::to_float(up[idx]);
  float silu = g / (1.0f + expf(-g));
  output[idx] = DtypeTraits<T>::from_float(silu * u);
}

template <typename T>
cudaError_t SiluMul(const T *gate, const T *up, T *output, int count,
                    cudaStream_t stream) {
  int threads = 256;
  int blocks = (count + threads - 1) / threads;
  SiluMulKernel<T><<<blocks, threads, 0, stream>>>(gate, up, output, count);
  return cudaGetLastError();
}

// ============================================================================
// Residual Addition (templated)
// ============================================================================

template <typename T>
__global__ void ResidualAddKernel(T *__restrict__ residual,
                                  const T *__restrict__ input, int count) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count)
    return;

  float r = DtypeTraits<T>::to_float(residual[idx]);
  float x = DtypeTraits<T>::to_float(input[idx]);
  residual[idx] = DtypeTraits<T>::from_float(r + x);
}

template <typename T>
cudaError_t ResidualAdd(T *residual, const T *input, int count,
                        cudaStream_t stream) {
  int threads = 256;
  int blocks = (count + threads - 1) / threads;
  ResidualAddKernel<T><<<blocks, threads, 0, stream>>>(residual, input, count);
  return cudaGetLastError();
}

// ============================================================================
// Embedding Lookup (templated)
// ============================================================================

template <typename T>
__global__ void EmbeddingLookupKernel(const T *__restrict__ table,
                                      const int *__restrict__ token_ids,
                                      T *__restrict__ output, int seq_len,
                                      int hidden_size) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = seq_len * hidden_size;
  if (idx >= total)
    return;

  int pos = idx / hidden_size;
  int dim = idx % hidden_size;
  int token_id = token_ids[pos];

  output[idx] = table[token_id * hidden_size + dim];
}

template <typename T>
cudaError_t EmbeddingLookup(const T *table, const int *token_ids, T *output,
                            int seq_len, int hidden_size, cudaStream_t stream) {
  int total = seq_len * hidden_size;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  EmbeddingLookupKernel<T><<<blocks, threads, 0, stream>>>(
      table, token_ids, output, seq_len, hidden_size);
  return cudaGetLastError();
}

// ============================================================================
// Half/BF16 to Float conversion (templated)
// ============================================================================

template <typename T>
__global__ void HalfToFloatKernel(const T *__restrict__ input,
                                  float *__restrict__ output, int count) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count)
    return;
  output[idx] = DtypeTraits<T>::to_float(input[idx]);
}

template <typename T>
cudaError_t HalfToFloat(const T *input, float *output, int count,
                        cudaStream_t stream) {
  int threads = 256;
  int blocks = (count + threads - 1) / threads;
  HalfToFloatKernel<T><<<blocks, threads, 0, stream>>>(input, output, count);
  return cudaGetLastError();
}

// ============================================================================
// Bias Add (templated)
// ============================================================================

template <typename T>
__global__ void BiasAddKernel(T *output, const T *bias, int total,
                              int bias_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total)
    return;
  float val = DtypeTraits<T>::to_float(output[idx]);
  float b = DtypeTraits<T>::to_float(bias[idx % bias_dim]);
  output[idx] = DtypeTraits<T>::from_float(val + b);
}

template <typename T>
cudaError_t BiasAdd(T *output, const T *bias, int rows, int bias_dim,
                    cudaStream_t stream) {
  int total = rows * bias_dim;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  BiasAddKernel<T>
      <<<blocks, threads, 0, stream>>>(output, bias, total, bias_dim);
  return cudaGetLastError();
}

// ============================================================================
// Batched KV Cache Append
// ============================================================================

template <typename T>
__global__ void BatchKvAppendKernel(
    const T *__restrict__ k_new, const T *__restrict__ v_new,
    T *__restrict__ kv_buffer, const int *__restrict__ d_seq_ids,
    const int *__restrict__ d_n_past, int kv_dim, size_t slot_stride,
    size_t layer_stride, size_t kv_stride, int layer) {
  int dim = blockIdx.x * blockDim.x + threadIdx.x;
  int b = blockIdx.y;
  int is_v = blockIdx.z; // 0=K, 1=V
  if (dim >= kv_dim)
    return;

  int seq_id = d_seq_ids[b];
  int n_past = d_n_past[b];
  const T *src = is_v ? (v_new + b * kv_dim) : (k_new + b * kv_dim);
  T *dst = kv_buffer + seq_id * slot_stride + layer * layer_stride +
           is_v * kv_stride + n_past * kv_dim;
  dst[dim] = src[dim];
}

template <typename T>
cudaError_t BatchKvAppend(const T *k_new, const T *v_new, T *kv_buffer,
                          const int *d_seq_ids, const int *d_n_past,
                          int batch_size, int kv_dim, size_t slot_stride,
                          size_t layer_stride, size_t kv_stride, int layer,
                          cudaStream_t stream) {
  int threads = 256;
  int blocks_x = (kv_dim + threads - 1) / threads;
  dim3 grid(blocks_x, batch_size, 2);
  BatchKvAppendKernel<T><<<grid, threads, 0, stream>>>(
      k_new, v_new, kv_buffer, d_seq_ids, d_n_past, kv_dim, slot_stride,
      layer_stride, kv_stride, layer);
  return cudaGetLastError();
}

// ============================================================================
// Logit Bias kernel
// ============================================================================

__global__ void LogitBiasKernel(float *__restrict__ logits,
                                const int *__restrict__ token_ids,
                                const float *__restrict__ biases,
                                int num_biases) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_biases)
    return;
  atomicAdd(&logits[token_ids[idx]], biases[idx]);
}

cudaError_t LogitBias(float *logits, const int *token_ids, const float *biases,
                      int num_biases, cudaStream_t stream) {
  if (num_biases <= 0)
    return cudaSuccess;
  int threads = 256;
  int blocks = (num_biases + threads - 1) / threads;
  LogitBiasKernel<<<blocks, threads, 0, stream>>>(logits, token_ids, biases,
                                                  num_biases);
  return cudaGetLastError();
}

// ============================================================================
// Repetition Penalty kernel
// ============================================================================

__global__ void RepetitionPenaltyKernel(float *__restrict__ logits,
                                        const int *__restrict__ history,
                                        int history_len, float penalty) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= history_len)
    return;
  int tok = history[idx];
  float val = logits[tok];
  logits[tok] = (val > 0.0f) ? (val / penalty) : (val * penalty);
}

cudaError_t RepetitionPenalty(float *logits, const int *history,
                              int history_len, float penalty,
                              cudaStream_t stream) {
  if (history_len <= 0 || penalty == 1.0f)
    return cudaSuccess;
  int threads = 256;
  int blocks = (history_len + threads - 1) / threads;
  RepetitionPenaltyKernel<<<blocks, threads, 0, stream>>>(logits, history,
                                                          history_len, penalty);
  return cudaGetLastError();
}

// ============================================================================
// Explicit template instantiations
// ============================================================================

template cudaError_t RmsNorm<half>(const half *, const half *, half *, int, int,
                                   float, cudaStream_t);
template cudaError_t RmsNorm<__nv_bfloat16>(const __nv_bfloat16 *,
                                            const __nv_bfloat16 *,
                                            __nv_bfloat16 *, int, int, float,
                                            cudaStream_t);

template cudaError_t RoPE<half>(half *, half *, int, int, int, int, int, float,
                                cudaStream_t, int, float);
template cudaError_t RoPE<__nv_bfloat16>(__nv_bfloat16 *, __nv_bfloat16 *, int,
                                         int, int, int, int, float,
                                         cudaStream_t, int, float);

template cudaError_t BatchRoPE<half>(half *, half *, int, int, int, int,
                                     const int *, float, cudaStream_t, int,
                                     float);
template cudaError_t BatchRoPE<__nv_bfloat16>(__nv_bfloat16 *, __nv_bfloat16 *,
                                              int, int, int, int, const int *,
                                              float, cudaStream_t, int, float);

template cudaError_t SiluMul<half>(const half *, const half *, half *, int,
                                   cudaStream_t);
template cudaError_t SiluMul<__nv_bfloat16>(const __nv_bfloat16 *,
                                            const __nv_bfloat16 *,
                                            __nv_bfloat16 *, int, cudaStream_t);

template cudaError_t ResidualAdd<half>(half *, const half *, int, cudaStream_t);
template cudaError_t ResidualAdd<__nv_bfloat16>(__nv_bfloat16 *,
                                                const __nv_bfloat16 *, int,
                                                cudaStream_t);

template cudaError_t EmbeddingLookup<half>(const half *, const int *, half *,
                                           int, int, cudaStream_t);
template cudaError_t EmbeddingLookup<__nv_bfloat16>(const __nv_bfloat16 *,
                                                    const int *,
                                                    __nv_bfloat16 *, int, int,
                                                    cudaStream_t);

template cudaError_t HalfToFloat<half>(const half *, float *, int,
                                       cudaStream_t);
template cudaError_t HalfToFloat<__nv_bfloat16>(const __nv_bfloat16 *, float *,
                                                int, cudaStream_t);

template cudaError_t BiasAdd<half>(half *, const half *, int, int,
                                   cudaStream_t);
template cudaError_t BiasAdd<__nv_bfloat16>(__nv_bfloat16 *,
                                            const __nv_bfloat16 *, int, int,
                                            cudaStream_t);

template cudaError_t BatchKvAppend<half>(const half *, const half *, half *,
                                         const int *, const int *, int, int,
                                         size_t, size_t, size_t, int,
                                         cudaStream_t);
template cudaError_t
BatchKvAppend<__nv_bfloat16>(const __nv_bfloat16 *, const __nv_bfloat16 *,
                             __nv_bfloat16 *, const int *, const int *, int,
                             int, size_t, size_t, size_t, int, cudaStream_t);

// ============================================================================
// Non-templated backward-compatible overloads (delegate to half instantiation)
// ============================================================================

cudaError_t RmsNorm(const half *input, const half *weight, half *output,
                    int count, int hidden_size, float eps,
                    cudaStream_t stream) {
  return RmsNorm<half>(input, weight, output, count, hidden_size, eps, stream);
}

cudaError_t RoPE(half *q, half *k, int seq_len, int num_heads, int num_kv_heads,
                 int head_dim, int n_past, float freq_base, cudaStream_t stream,
                 int rope_type, float freq_scale) {
  return RoPE<half>(q, k, seq_len, num_heads, num_kv_heads, head_dim, n_past,
                    freq_base, stream, rope_type, freq_scale);
}

cudaError_t BatchRoPE(half *q, half *k, int batch_size, int num_heads,
                      int num_kv_heads, int head_dim, const int *d_n_past,
                      float freq_base, cudaStream_t stream, int rope_type,
                      float freq_scale) {
  return BatchRoPE<half>(q, k, batch_size, num_heads, num_kv_heads, head_dim,
                         d_n_past, freq_base, stream, rope_type, freq_scale);
}

cudaError_t SiluMul(const half *gate, const half *up, half *output, int count,
                    cudaStream_t stream) {
  return SiluMul<half>(gate, up, output, count, stream);
}

cudaError_t ResidualAdd(half *residual, const half *input, int count,
                        cudaStream_t stream) {
  return ResidualAdd<half>(residual, input, count, stream);
}

cudaError_t EmbeddingLookup(const half *table, const int *token_ids,
                            half *output, int seq_len, int hidden_size,
                            cudaStream_t stream) {
  return EmbeddingLookup<half>(table, token_ids, output, seq_len, hidden_size,
                               stream);
}

cudaError_t HalfToFloat(const half *input, float *output, int count,
                        cudaStream_t stream) {
  return HalfToFloat<half>(input, output, count, stream);
}

} // namespace cuda_kernel
} // namespace inferflux
