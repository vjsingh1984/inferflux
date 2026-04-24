#pragma once

#include "runtime/backends/cuda/native/kv_cache_gpu.h"

#include <vector>

namespace inferflux {

/**
 * HybridKvCacheGpuTyped<T>: Dense base + per-slot overflow KV cache.
 *
 * Tier 1 (Dense): First `base_slots` sequence slots live in one contiguous
 *   cudaMalloc, enabling fast pointer arithmetic.
 *
 * Tier 2 (Overflow): Remaining slots (base_slots..max_batch-1) are allocated
 *   as individual cudaMalloc's, either eagerly or lazily on first use.
 *
 * A device-resident indirection table `d_slot_base_ptrs_[max_batch]` maps
 * each seq_id to its slot's base pointer. Kernels replace
 *   `buffer + seq_id * slot_stride`
 * with
 *   `slot_base_ptrs[seq_id]`
 * — one pointer load instead of a multiply+add.
 *
 * Layout within each slot is identical to KvCacheGpuTyped:
 *   [num_layers][2 (K/V)][max_seq_len][kv_dim]
 */
template <typename T> class HybridKvCacheGpuTyped : public IKvCacheGpu {
public:
  HybridKvCacheGpuTyped() = default;
  ~HybridKvCacheGpuTyped() override;

  HybridKvCacheGpuTyped(const HybridKvCacheGpuTyped &) = delete;
  HybridKvCacheGpuTyped &operator=(const HybridKvCacheGpuTyped &) = delete;

  bool Allocate(int num_layers, int num_kv_heads, int head_dim, int max_seq_len,
                int max_batch_size, int base_slots);

  T *GetK(int layer, int seq_id) const;
  T *GetV(int layer, int seq_id) const;

  cudaError_t Append(int layer, int seq_id, int start_pos, int seq_len,
                     const T *k_new, const T *v_new, cudaStream_t stream);

  void GetBatchAppendPtrs(int layer, const int *seq_ids, const int *n_past,
                          int batch_size, T **h_k_ptrs, T **h_v_ptrs) const;
  void GetBatchKVPtrs(int layer, const int *seq_ids, int batch_size,
                      const T **h_k_ptrs, const T **h_v_ptrs) const;

  void ClearSequence(int seq_id) override;
  void ClearSequenceAsync(int seq_id, cudaStream_t stream) override;
  bool CopySequencePrefix(int src_seq, int dst_seq, int n_tokens,
                          cudaStream_t stream) override;
  bool SerializeSequence(int seq_id, std::vector<uint8_t> *out) const override;
  bool HydrateSequence(int seq_id, const std::vector<uint8_t> &blob,
                       cudaStream_t stream) override;

  size_t GetMemoryUsage() const override { return total_bytes_; }
  int MaxSeqLen() const override { return max_seq_len_; }
  int MaxBatchSize() const override { return max_batch_size_; }
  size_t SlotStride() const override { return slot_stride_; }
  size_t LayerStride() const override { return layer_stride_; }
  size_t KvStride() const override { return kv_stride_; }
  int KvDim() const override { return kv_dim_; }
  void *SlotBasePtrsDevice() const override { return d_slot_base_ptrs_; }

  void *GetKVoid(int layer, int seq_id) const override {
    return GetK(layer, seq_id);
  }
  void *GetVVoid(int layer, int seq_id) const override {
    return GetV(layer, seq_id);
  }
  cudaError_t AppendVoid(int layer, int seq_id, int start_pos, int seq_len,
                         const void *k_new, const void *v_new,
                         cudaStream_t stream) override {
    return Append(layer, seq_id, start_pos, seq_len,
                  static_cast<const T *>(k_new), static_cast<const T *>(v_new),
                  stream);
  }

  int BaseSlots() const { return base_slots_; }

private:
  T *SlotBase(int seq_id) const { return h_slot_base_ptrs_[seq_id]; }

  // Lazy allocation: allocate overflow slot on first use
  bool EnsureOverflowSlot(int seq_id);

  bool IsOverflowSlot(int seq_id) const { return seq_id >= base_slots_; }

  T *base_buffer_{nullptr};           // Tier 1: single contiguous alloc
  std::vector<T *> overflow_allocs_;   // Tier 2: per-slot allocs
  T **d_slot_base_ptrs_{nullptr};      // Device indirection table
  T **h_slot_base_ptrs_{nullptr};      // Pinned host mirror
  size_t total_bytes_{0};

  int num_layers_{0};
  int num_kv_heads_{0};
  int head_dim_{0};
  int kv_dim_{0};
  int max_seq_len_{0};
  int max_batch_size_{0};
  int base_slots_{0};

  size_t slot_stride_{0};
  size_t layer_stride_{0};
  size_t kv_stride_{0};
};

} // namespace inferflux
