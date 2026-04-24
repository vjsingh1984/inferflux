#include "runtime/backends/cuda/native/hybrid_kv_cache_gpu.h"

#include "server/logging/logger.h"
#include <algorithm>
#include <cstring>
#include <cuda_bf16.h>

namespace inferflux {

template <typename T> HybridKvCacheGpuTyped<T>::~HybridKvCacheGpuTyped() {
  if (d_slot_base_ptrs_) {
    cudaFree(d_slot_base_ptrs_);
    d_slot_base_ptrs_ = nullptr;
  }
  if (h_slot_base_ptrs_) {
    cudaFreeHost(h_slot_base_ptrs_);
    h_slot_base_ptrs_ = nullptr;
  }
  for (auto *p : overflow_allocs_) {
    if (p) {
      cudaFree(p);
    }
  }
  overflow_allocs_.clear();
  if (base_buffer_) {
    cudaFree(base_buffer_);
    base_buffer_ = nullptr;
  }
}

template <typename T>
bool HybridKvCacheGpuTyped<T>::Allocate(int num_layers, int num_kv_heads,
                                        int head_dim, int max_seq_len,
                                        int max_batch_size, int base_slots) {
  num_layers_ = num_layers;
  num_kv_heads_ = num_kv_heads;
  head_dim_ = head_dim;
  kv_dim_ = num_kv_heads * head_dim;
  max_seq_len_ = max_seq_len;
  max_batch_size_ = max_batch_size;
  base_slots_ = std::min(base_slots, max_batch_size);

  // Layout per slot: [num_layers][2 (K/V)][max_seq_len][kv_dim]
  kv_stride_ = static_cast<size_t>(max_seq_len) * kv_dim_;
  layer_stride_ = 2 * kv_stride_;
  slot_stride_ = static_cast<size_t>(num_layers) * layer_stride_;
  const size_t slot_bytes = slot_stride_ * sizeof(T);

  // Tier 1: contiguous dense allocation
  const size_t base_bytes = static_cast<size_t>(base_slots_) * slot_bytes;
  cudaError_t err = cudaMalloc(&base_buffer_, base_bytes);
  if (err != cudaSuccess) {
    log::Error("hybrid_kv_cache",
               "cudaMalloc failed for base KV buffer: " +
                   std::string(cudaGetErrorString(err)) + " (size=" +
                   std::to_string(base_bytes / (1024 * 1024)) + " MB)");
    return false;
  }
  cudaMemset(base_buffer_, 0, base_bytes);

  // Tier 2: per-slot overflow allocations (LAZY - allocated on first use)
  const int overflow_count = max_batch_size - base_slots_;
  overflow_allocs_.resize(overflow_count, nullptr);

  // IMPORTANT: Allocate all overflow slots upfront to avoid nullptr in indirection table
  // Lazy allocation optimization disabled due to kernel access race conditions
  // TODO: Re-enable lazy allocation with proper slot initialization guards
  for (int i = 0; i < overflow_count; ++i) {
    err = cudaMalloc(&overflow_allocs_[i], slot_bytes);
    if (err != cudaSuccess) {
      log::Error("hybrid_kv_cache",
                 "cudaMalloc failed for overflow slot " + std::to_string(i) +
                     ": " + std::string(cudaGetErrorString(err)));
      // Free what we allocated so far
      for (int j = 0; j < i; ++j) {
        cudaFree(overflow_allocs_[j]);
      }
      overflow_allocs_.clear();
      cudaFree(base_buffer_);
      base_buffer_ = nullptr;
      cudaFreeHost(h_slot_base_ptrs_);
      h_slot_base_ptrs_ = nullptr;
      return false;
    }
    cudaMemset(overflow_allocs_[i], 0, slot_bytes);
  }

  total_bytes_ = base_bytes + (overflow_count * slot_bytes);

  // Build indirection table (pinned host + device)
  err = cudaMallocHost(&h_slot_base_ptrs_,
                       max_batch_size * sizeof(T *));
  if (err != cudaSuccess) {
    log::Error("hybrid_kv_cache", "cudaMallocHost failed for slot ptrs");
    return false;
  }
  for (int i = 0; i < base_slots_; ++i) {
    h_slot_base_ptrs_[i] = base_buffer_ + static_cast<size_t>(i) * slot_stride_;
  }
  for (int i = 0; i < overflow_count; ++i) {
    h_slot_base_ptrs_[base_slots_ + i] = overflow_allocs_[i];
  }

  err = cudaMalloc(&d_slot_base_ptrs_, max_batch_size * sizeof(T *));
  if (err != cudaSuccess) {
    log::Error("hybrid_kv_cache", "cudaMalloc failed for device slot ptrs");
    return false;
  }
  cudaMemcpy(d_slot_base_ptrs_, h_slot_base_ptrs_,
             max_batch_size * sizeof(T *), cudaMemcpyHostToDevice);

  log::Info("hybrid_kv_cache",
            "Allocated hybrid KV cache: " +
                std::to_string(total_bytes_ / (1024 * 1024)) +
                " MB (base_slots=" + std::to_string(base_slots_) +
                ", overflow_slots=" + std::to_string(overflow_count) +
                ", layers=" + std::to_string(num_layers) +
                ", kv_heads=" + std::to_string(num_kv_heads) +
                ", head_dim=" + std::to_string(head_dim) +
                ", max_seq=" + std::to_string(max_seq_len) +
                ", max_batch=" + std::to_string(max_batch_size) + ")");
  return true;
}

template <typename T>
bool HybridKvCacheGpuTyped<T>::EnsureOverflowSlot(int seq_id) {
  if (!IsOverflowSlot(seq_id)) {
    return true;  // Base slot, already allocated
  }

  const int overflow_idx = seq_id - base_slots_;
  if (overflow_allocs_[overflow_idx] != nullptr) {
    return true;  // Already allocated
  }

  // Allocate this overflow slot on-demand
  const size_t slot_bytes = slot_stride_ * sizeof(T);
  cudaError_t err = cudaMalloc(&overflow_allocs_[overflow_idx], slot_bytes);
  if (err != cudaSuccess) {
    log::Error("hybrid_kv_cache",
               "Lazy cudaMalloc failed for overflow slot " + std::to_string(seq_id) +
                   ": " + std::string(cudaGetErrorString(err)));
    return false;
  }
  cudaMemset(overflow_allocs_[overflow_idx], 0, slot_bytes);

  // Update indirection table (both host and device)
  h_slot_base_ptrs_[seq_id] = overflow_allocs_[overflow_idx];
  err = cudaMemcpy(d_slot_base_ptrs_ + seq_id, &h_slot_base_ptrs_[seq_id],
                  sizeof(T *), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    log::Error("hybrid_kv_cache",
               "Failed to update device slot ptr for overflow slot " +
                   std::to_string(seq_id));
    return false;
  }

  // Update memory accounting
  total_bytes_ += slot_bytes;
  log::Info("hybrid_kv_cache",
            "Lazy-allocated overflow slot " + std::to_string(seq_id) +
                " (" + std::to_string(slot_bytes / (1024 * 1024)) + " MB)");

  return true;
}

template <typename T>
T *HybridKvCacheGpuTyped<T>::GetK(int layer, int seq_id) const {
  return SlotBase(seq_id) + static_cast<size_t>(layer) * layer_stride_;
}

template <typename T>
T *HybridKvCacheGpuTyped<T>::GetV(int layer, int seq_id) const {
  return SlotBase(seq_id) + static_cast<size_t>(layer) * layer_stride_ +
         kv_stride_;
}

template <typename T>
cudaError_t
HybridKvCacheGpuTyped<T>::Append(int layer, int seq_id, int start_pos,
                                 int seq_len, const T *k_new, const T *v_new,
                                 cudaStream_t stream) {
  size_t copy_bytes = static_cast<size_t>(seq_len) * kv_dim_ * sizeof(T);
  T *k_dst = GetK(layer, seq_id) + static_cast<size_t>(start_pos) * kv_dim_;
  T *v_dst = GetV(layer, seq_id) + static_cast<size_t>(start_pos) * kv_dim_;

  cudaError_t err = cudaMemcpyAsync(k_dst, k_new, copy_bytes,
                                    cudaMemcpyDeviceToDevice, stream);
  if (err != cudaSuccess)
    return err;
  return cudaMemcpyAsync(v_dst, v_new, copy_bytes, cudaMemcpyDeviceToDevice,
                         stream);
}

template <typename T>
void HybridKvCacheGpuTyped<T>::ClearSequence(int seq_id) {
  if (seq_id < 0 || seq_id >= max_batch_size_)
    return;
  cudaMemset(SlotBase(seq_id), 0, slot_stride_ * sizeof(T));
}

template <typename T>
void HybridKvCacheGpuTyped<T>::ClearSequenceAsync(int seq_id,
                                                   cudaStream_t stream) {
  if (seq_id < 0 || seq_id >= max_batch_size_)
    return;
  cudaMemsetAsync(SlotBase(seq_id), 0, slot_stride_ * sizeof(T), stream);
}

template <typename T>
bool HybridKvCacheGpuTyped<T>::CopySequencePrefix(int src_seq, int dst_seq,
                                                   int n_tokens,
                                                   cudaStream_t stream) {
  if (src_seq < 0 || dst_seq < 0 || src_seq >= max_batch_size_ ||
      dst_seq >= max_batch_size_) {
    return false;
  }
  if (n_tokens <= 0 || src_seq == dst_seq) {
    return true;
  }
  const int copy_tokens = std::min(n_tokens, max_seq_len_);
  const size_t copy_bytes =
      static_cast<size_t>(copy_tokens) * kv_dim_ * sizeof(T);
  for (int layer = 0; layer < num_layers_; ++layer) {
    T *k_src = GetK(layer, src_seq);
    T *k_dst = GetK(layer, dst_seq);
    cudaError_t err = cudaMemcpyAsync(k_dst, k_src, copy_bytes,
                                      cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess)
      return false;
    T *v_src = GetV(layer, src_seq);
    T *v_dst = GetV(layer, dst_seq);
    err = cudaMemcpyAsync(v_dst, v_src, copy_bytes, cudaMemcpyDeviceToDevice,
                          stream);
    if (err != cudaSuccess)
      return false;
  }
  return cudaStreamSynchronize(stream) == cudaSuccess;
}

template <typename T>
bool HybridKvCacheGpuTyped<T>::SerializeSequence(
    int seq_id, std::vector<uint8_t> *out) const {
  if (!out || seq_id < 0 || seq_id >= max_batch_size_)
    return false;
  const size_t bytes = slot_stride_ * sizeof(T);
  out->resize(bytes);
  return cudaMemcpy(out->data(), SlotBase(seq_id), bytes,
                    cudaMemcpyDeviceToHost) == cudaSuccess;
}

template <typename T>
bool HybridKvCacheGpuTyped<T>::HydrateSequence(
    int seq_id, const std::vector<uint8_t> &blob, cudaStream_t stream) {
  if (seq_id < 0 || seq_id >= max_batch_size_)
    return false;
  const size_t bytes = slot_stride_ * sizeof(T);
  if (blob.size() != bytes)
    return false;
  if (cudaMemcpyAsync(SlotBase(seq_id), blob.data(), bytes,
                      cudaMemcpyHostToDevice, stream) != cudaSuccess) {
    return false;
  }
  return cudaStreamSynchronize(stream) == cudaSuccess;
}

template <typename T>
void HybridKvCacheGpuTyped<T>::GetBatchAppendPtrs(int layer,
                                                   const int *seq_ids,
                                                   const int *n_past,
                                                   int batch_size,
                                                   T **h_k_ptrs,
                                                   T **h_v_ptrs) const {
  for (int b = 0; b < batch_size; ++b) {
    h_k_ptrs[b] =
        GetK(layer, seq_ids[b]) + static_cast<size_t>(n_past[b]) * kv_dim_;
    h_v_ptrs[b] =
        GetV(layer, seq_ids[b]) + static_cast<size_t>(n_past[b]) * kv_dim_;
  }
}

template <typename T>
void HybridKvCacheGpuTyped<T>::GetBatchKVPtrs(int layer, const int *seq_ids,
                                              int batch_size,
                                              const T **h_k_ptrs,
                                              const T **h_v_ptrs) const {
  for (int b = 0; b < batch_size; ++b) {
    h_k_ptrs[b] = GetK(layer, seq_ids[b]);
    h_v_ptrs[b] = GetV(layer, seq_ids[b]);
  }
}

// Explicit template instantiations
template class HybridKvCacheGpuTyped<half>;
template class HybridKvCacheGpuTyped<__nv_bfloat16>;

} // namespace inferflux
