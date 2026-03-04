#include "runtime/backends/cuda/native/kv_cache_gpu.h"

#include "server/logging/logger.h"
#include <cuda_bf16.h>
#include <cstring>

namespace inferflux {

template <typename T>
KvCacheGpuTyped<T>::~KvCacheGpuTyped() {
  if (buffer_) {
    cudaFree(buffer_);
    buffer_ = nullptr;
  }
}

template <typename T>
bool KvCacheGpuTyped<T>::Allocate(int num_layers, int num_kv_heads,
                                   int head_dim, int max_seq_len,
                                   int max_batch_size) {
  num_layers_ = num_layers;
  num_kv_heads_ = num_kv_heads;
  head_dim_ = head_dim;
  kv_dim_ = num_kv_heads * head_dim;
  max_seq_len_ = max_seq_len;
  max_batch_size_ = max_batch_size;

  // Layout: [batch][layer][2][seq_len][kv_dim]
  kv_stride_ = static_cast<size_t>(max_seq_len) * kv_dim_;
  layer_stride_ = 2 * kv_stride_;
  slot_stride_ = static_cast<size_t>(num_layers) * layer_stride_;

  size_t total_elements =
      static_cast<size_t>(max_batch_size) * slot_stride_;
  total_bytes_ = total_elements * sizeof(T);

  cudaError_t err = cudaMalloc(&buffer_, total_bytes_);
  if (err != cudaSuccess) {
    log::Error("kv_cache_gpu",
               "cudaMalloc failed for KV cache: " +
                   std::string(cudaGetErrorString(err)) +
                   " (size=" + std::to_string(total_bytes_ / (1024 * 1024)) +
                   " MB)");
    return false;
  }

  cudaMemset(buffer_, 0, total_bytes_);

  log::Info("kv_cache_gpu",
            "Allocated KV cache: " +
                std::to_string(total_bytes_ / (1024 * 1024)) +
                " MB (layers=" + std::to_string(num_layers) +
                ", kv_heads=" + std::to_string(num_kv_heads) +
                ", head_dim=" + std::to_string(head_dim) +
                ", max_seq=" + std::to_string(max_seq_len) +
                ", max_batch=" + std::to_string(max_batch_size) + ")");
  return true;
}

template <typename T>
T* KvCacheGpuTyped<T>::GetK(int layer, int seq_id) const {
  return buffer_ + seq_id * slot_stride_ + layer * layer_stride_;
}

template <typename T>
T* KvCacheGpuTyped<T>::GetV(int layer, int seq_id) const {
  return buffer_ + seq_id * slot_stride_ + layer * layer_stride_ + kv_stride_;
}

template <typename T>
cudaError_t KvCacheGpuTyped<T>::Append(int layer, int seq_id, int start_pos,
                                        int seq_len, const T* k_new,
                                        const T* v_new, cudaStream_t stream) {
  size_t copy_bytes = static_cast<size_t>(seq_len) * kv_dim_ * sizeof(T);

  T* k_dst = GetK(layer, seq_id) + static_cast<size_t>(start_pos) * kv_dim_;
  T* v_dst = GetV(layer, seq_id) + static_cast<size_t>(start_pos) * kv_dim_;

  cudaError_t err =
      cudaMemcpyAsync(k_dst, k_new, copy_bytes, cudaMemcpyDeviceToDevice,
                      stream);
  if (err != cudaSuccess) return err;

  err = cudaMemcpyAsync(v_dst, v_new, copy_bytes, cudaMemcpyDeviceToDevice,
                        stream);
  return err;
}

template <typename T>
void KvCacheGpuTyped<T>::ClearSequence(int seq_id) {
  if (!buffer_ || seq_id >= max_batch_size_) return;
  size_t bytes = slot_stride_ * sizeof(T);
  cudaMemset(buffer_ + seq_id * slot_stride_, 0, bytes);
}

// Explicit template instantiations
template class KvCacheGpuTyped<half>;
template class KvCacheGpuTyped<__nv_bfloat16>;

}  // namespace inferflux
