#pragma once

#include <cstddef>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <vector>

namespace inferflux {

/**
 * IKvCacheGpu: non-templated base interface for sequence management.
 * Used by NativeKernelExecutor to call ClearSequence without knowing the dtype.
 */
class IKvCacheGpu {
public:
  virtual ~IKvCacheGpu() = default;
  virtual void ClearSequence(int seq_id, cudaStream_t stream = nullptr) = 0;
  virtual size_t GetMemoryUsage() const = 0;
  virtual int MaxSeqLen() const = 0;
  virtual int MaxBatchSize() const = 0;
};

/**
 * KvCacheGpuTyped<T>: pre-allocated contiguous GPU memory for KV cache.
 *
 * Layout: [max_batch][num_layers][2 (K/V)][max_seq_len][kv_dim]
 * where kv_dim = num_kv_heads * head_dim.
 */
template <typename T> class KvCacheGpuTyped : public IKvCacheGpu {
public:
  KvCacheGpuTyped() = default;
  ~KvCacheGpuTyped() override;

  KvCacheGpuTyped(const KvCacheGpuTyped &) = delete;
  KvCacheGpuTyped &operator=(const KvCacheGpuTyped &) = delete;

  bool Allocate(int num_layers, int num_kv_heads, int head_dim, int max_seq_len,
                int max_batch_size);

  T *GetK(int layer, int seq_id) const;
  T *GetV(int layer, int seq_id) const;

  cudaError_t Append(int layer, int seq_id, int start_pos, int seq_len,
                     const T *k_new, const T *v_new, cudaStream_t stream);

  void ClearSequence(int seq_id, cudaStream_t stream = nullptr) override;

  /**
   * Batched KV append: scatter B sequences' K/V into their cache slots with a
   * single kernel launch (replaces B x 2 cudaMemcpyAsync D2D calls).
   */
  cudaError_t BatchAppend(int layer, const int *d_seq_ids, const int *d_n_past,
                          int batch_size, const T *k_new, const T *v_new,
                          cudaStream_t stream);

  T *Buffer() const { return buffer_; }
  size_t SlotStride() const { return slot_stride_; }
  size_t LayerStride() const { return layer_stride_; }
  size_t KvStride() const { return kv_stride_; }
  int KvDim() const { return kv_dim_; }

  size_t GetMemoryUsage() const override { return total_bytes_; }
  int MaxSeqLen() const override { return max_seq_len_; }
  int MaxBatchSize() const override { return max_batch_size_; }

private:
  T *buffer_{nullptr};
  size_t total_bytes_{0};

  int num_layers_{0};
  int num_kv_heads_{0};
  int head_dim_{0};
  int kv_dim_{0};
  int max_seq_len_{0};
  int max_batch_size_{0};

  size_t slot_stride_{0};
  size_t layer_stride_{0};
  size_t kv_stride_{0};
};

// Backward-compatible alias
using KvCacheGpu = KvCacheGpuTyped<half>;

} // namespace inferflux
