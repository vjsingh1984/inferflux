#pragma once

#include "runtime/backends/cuda/native_cuda_executor.h"
#include "runtime/backends/cuda/kernels/flash_attention.cuh"

#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

namespace inferflux {

// Native CUDA kernel executor that uses custom attention kernels
// instead of delegating to llama.cpp
class NativeKernelExecutor final : public NativeCudaExecutor {
public:
  NativeKernelExecutor();
  ~NativeKernelExecutor() override;

  // NativeCudaExecutor interface
  std::string Name() const override {
    return "native_cuda_kernels";
  }

  bool IsFallback() const override {
    return false;  // We use native kernels!
  }

  const std::string &FallbackReason() const override {
    static const std::string no_reason;
    return no_reason;  // No fallback
  }

  bool LoadModel(const std::filesystem::path &model_path,
                 const LlamaBackendConfig &config) override;

  std::vector<UnifiedBatchOutput>
  ExecuteUnifiedBatch(const std::vector<UnifiedBatchInput> &inputs) override;

  bool SupportsAsyncUnifiedBatch() const override {
    return true;  // Native implementation supports async
  }

  UnifiedBatchHandle
  SubmitUnifiedBatchAsync(const std::vector<UnifiedBatchInput> &inputs,
                          UnifiedBatchLane lane) override;

  bool
  TryCollectUnifiedBatchAsync(UnifiedBatchHandle handle,
                              std::vector<UnifiedBatchOutput> *outputs) override;

  std::shared_ptr<LlamaCPUBackend> BackendHandle() const override;

  // Native-specific functionality
  bool HasFlashAttention2() const { return has_flash_attention_2_; }
  int GetDeviceId() const { return device_id_; }
  std::string GetDeviceName() const { return device_name_; }

private:
  // CUDA state
  int device_id_{0};
  std::string device_name_;
  cudaStream_t compute_stream_{nullptr};
  cudaStream_t copy_stream_{nullptr};
  cudaStream_t prefill_stream_{nullptr};   // Dedicated prefill stream
  cudaStream_t decode_stream_{nullptr};   // Dedicated decode stream

  // Async overlap tracking
  cudaEvent_t prefill_start_event_{nullptr};
  cudaEvent_t prefill_end_event_{nullptr};
  cudaEvent_t decode_start_event_{nullptr};
  cudaEvent_t decode_end_event_{nullptr};
  bool overlap_enabled_{true};  // Enable async overlap
  int min_prefill_tokens_{256};  // Minimum tokens to trigger prefill lane

  // Model parameters
  int num_heads_{0};
  int head_dim_{0};
  int vocab_size_{0};
  int context_length_{0};

  // Device memory pointers (owned)
  float *d_Q_{nullptr};
  float *d_K_{nullptr};
  float *d_V_{nullptr};
  float *d_O_{nullptr};

  // Size of allocated tensors
  size_t tensor_size_bytes_{0};

  // Kernel capabilities
  bool has_flash_attention_2_{true};
  bool use_flash_attention_{true};

  // Llama backend for tokenization and non-accelerated ops
  std::shared_ptr<LlamaCPUBackend> llama_backend_;

  // Async execution state
  struct AsyncJob {
    UnifiedBatchHandle handle{0};
    std::vector<UnifiedBatchInput> inputs;
    std::vector<UnifiedBatchOutput> outputs;
    cudaEvent_t completion_event{nullptr};
  };
  std::vector<AsyncJob> async_jobs_;
  UnifiedBatchHandle next_handle_{1};

  // Internal helpers
  bool InitializeCUDA();
  bool AllocateDeviceMemory(size_t tensor_size);
  void FreeDeviceMemory();
  bool LoadModelWeights(const std::filesystem::path &model_path);
  bool RunNativeAttention(const std::vector<UnifiedBatchInput> &inputs,
                         std::vector<UnifiedBatchOutput> *outputs);
  bool RunStandardAttention(const std::vector<UnifiedBatchInput> &inputs,
                           std::vector<UnifiedBatchOutput> *outputs);

  // Async overlap execution
  std::vector<UnifiedBatchOutput>
  ExecuteUnifiedBatchWithOverlap(const std::vector<UnifiedBatchInput> &inputs);
  bool HasMixedWorkload(const std::vector<UnifiedBatchInput> &inputs) const;
  void SplitBatchByType(const std::vector<UnifiedBatchInput> &inputs,
                       std::vector<size_t> &prefill_indices,
                       std::vector<size_t> &decode_indices) const;
};

} // namespace inferflux
