#pragma once

#include "runtime/backends/cuda/native_cuda_executor.h"
#include <atomic>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declarations for native kernel components
namespace inferflux {
class ModelForward;
class IKvCacheGpu;
template <typename T> class KvCacheGpuTyped;
using KvCacheGpu = KvCacheGpuTyped<half>;
class CublasGemm;
class GpuSampler;
template <typename T> class WeightMapTyped;
using WeightMap = WeightMapTyped<half>;
class NativeTokenizer;
} // namespace inferflux

namespace inferflux {

// Forward declaration
class SafetensorsParser;

/**
 * Safetensors model loader
 *
 * Loads HuggingFace safetensors format models directly.
 * No llama.cpp dependency.
 */
class SafetensorsLoader {
public:
  struct Tensor {
    std::string name;
    std::vector<size_t> shape;
    std::string dtype;    // "f16", "f32", "bf16", etc.
    size_t offset;        // File offset
    size_t size;          // Byte size
    void *cpu_data;       // Mapped CPU data
    void *gpu_data;       // GPU memory (after UploadToGPU)
    size_t gpu_offset{0}; // Offset within GPU buffer
  };

  struct ModelConfig {
    // Model architecture
    int hidden_size{0};
    int num_hidden_layers{0};
    int num_attention_heads{0};
    int num_key_value_heads{0}; // For GQA
    int head_dim{0};
    int intermediate_size{0};
    int vocab_size{0};
    int max_position_embeddings{0};

    // RoPE settings
    float rope_freq_base{10000.0f};
    float rope_freq_scale{1.0f};
    int rope_dim{0};

    // Model type
    std::string model_type; // "qwen2", "llama", etc.
    std::string activation; // "silu", "swiglu", etc.

    // Dtype and normalization
    std::string torch_dtype; // "bfloat16", "float16", etc.
    float rms_norm_eps{1e-6f};
  };

  SafetensorsLoader();
  ~SafetensorsLoader(); // Automatically frees CPU and GPU memory

  /**
   * Load model from safetensors directory
   * @param model_path Path containing model.safetensors.index.json
   */
  bool LoadModel(const std::string &model_path);

  /**
   * Get tensor by name
   */
  const Tensor *GetTensor(const std::string &name) const;

  /**
   * Get all tensor names
   */
  std::vector<std::string> GetTensorNames() const;

  /**
   * Get model configuration
   */
  const ModelConfig &GetConfig() const { return config_; }

  /**
   * Upload all tensors to GPU memory.
   * @param skip_bf16_conversion If true, upload BF16 weights as-is
   *        (for native BF16 inference pipeline).
   */
  bool UploadToGPU(cudaStream_t stream, bool skip_bf16_conversion = false);

  /**
   * Free CPU memory (after GPU upload)
   */
  void FreeCPUMemory();

  /**
   * Free GPU memory
   */
  void FreeGPUMemory();

  /**
   * Get model directory path
   */
  const std::string &GetModelPath() const { return model_path_; }

  /**
   * Get GPU weights buffer
   */
  void *GetGPUBuffer() const { return d_weights_buffer_; }

  /**
   * Get total GPU size
   */
  size_t GetGPUSize() const { return total_gpu_size_; }

private:
  bool LoadIndex(const std::string &index_path);
  bool LoadShard(const std::string &shard_path);
  bool ParseConfig(const std::string &config_path);

  std::unordered_map<std::string, Tensor> tensors_;
  ModelConfig config_;
  std::string model_path_;
  size_t total_size_bytes_{0};

  // Keep shard parsers alive to maintain valid CPU data mappings
  std::vector<std::unique_ptr<class SafetensorsParser>> shard_parsers_;

  // GPU memory
  void *d_weights_buffer_{nullptr}; // Contiguous GPU buffer for all weights
  size_t total_gpu_size_{0};        // Total size of GPU buffer
};

/**
 * Native CUDA Kernel Executor
 *
 * Implements native CUDA inference using:
 * - Safetensors format (direct loading)
 * - Custom FlashAttention/PagedAttention kernels
 * - CUTLASS/cuBLAS for GEMM
 * - NO llama.cpp dependency
 */
class NativeKernelExecutor final : public NativeCudaExecutor {
public:
  NativeKernelExecutor();
  ~NativeKernelExecutor() override;

  // NativeCudaExecutor interface
  std::string Name() const override { return "native_cuda_kernels"; }
  bool IsFallback() const override { return false; } // We use native kernels!
  const std::string &FallbackReason() const override {
    static const std::string no_reason;
    return no_reason;
  }

  /**
   * Load model from safetensors
   */
  bool LoadModel(const std::filesystem::path &model_path,
                 const LlamaBackendConfig &config) override;

  /**
   * Execute batch (to be implemented with native kernels)
   */
  std::vector<UnifiedBatchOutput>
  ExecuteUnifiedBatch(const std::vector<UnifiedBatchInput> &inputs) override;

  bool SupportsAsyncUnifiedBatch() const override;

  UnifiedBatchHandle
  SubmitUnifiedBatchAsync(const std::vector<UnifiedBatchInput> &inputs,
                          UnifiedBatchLane lane) override;

  bool TryCollectUnifiedBatchAsync(
      UnifiedBatchHandle handle,
      std::vector<UnifiedBatchOutput> *outputs) override;

  std::shared_ptr<LlamaCPUBackend> BackendHandle() const override {
    return nullptr; // No llama backend!
  }

  // Native perf
  NativePerfSnapshot NativeTakePerf() override;

  // Native* overrides — provide tokenization/readiness without llama.cpp
  std::vector<int> NativeTokenize(const std::string &prompt) const override;
  int NativeTokenCount(const std::string &text) const override;
  bool NativeIsReady() const override;
  void NativeFreeSequence(int sequence_id) override;
  void NativeCopySequencePrefix(int src_seq, int dst_seq,
                                int n_tokens) override;

  // Native-specific functionality
  bool HasFlashAttention2() const { return has_flash_attention_2_; }
  int GetDeviceId() const { return device_id_; }

  enum class InferenceDtype { kFP16, kBF16 };
  InferenceDtype GetInferenceDtype() const { return inference_dtype_; }

private:
  // Model loading
  std::unique_ptr<SafetensorsLoader> loader_;
  bool model_loaded_{false};

  // CUDA state
  int device_id_{0};
  cudaStream_t compute_stream_{nullptr};
  cudaStream_t copy_stream_{nullptr};

  // Model parameters (from safetensors)
  SafetensorsLoader::ModelConfig model_config_;

  // GPU memory pointers
  void *d_weights_{nullptr}; // All weights on GPU

  // Kernel capabilities
  bool has_flash_attention_2_{true};
  bool use_flash_attention_{true};
  InferenceDtype inference_dtype_{InferenceDtype::kFP16};

  // Native kernel pipeline components (only available with CUDA)
#ifdef INFERFLUX_NATIVE_KERNELS_READY
  std::unique_ptr<ModelForward> model_forward_;
  std::unique_ptr<IKvCacheGpu> kv_cache_; // Typed KV cache (FP16 or BF16)
  std::unique_ptr<CublasGemm> gemm_;
  std::unique_ptr<GpuSampler> sampler_;
  std::unique_ptr<WeightMap> weight_map_;
#endif
  std::unique_ptr<NativeTokenizer> tokenizer_;

  // Device logits buffer
#ifdef INFERFLUX_NATIVE_KERNELS_READY
  float *d_logits_{nullptr};
#endif

  // Async lane support
#ifdef INFERFLUX_NATIVE_KERNELS_READY
  cudaStream_t decode_stream_{nullptr};
  cudaStream_t prefill_stream_{nullptr};
  std::mutex lane_mutex_;
  struct AsyncBatchState {
    std::vector<UnifiedBatchOutput> outputs;
    cudaEvent_t completion_event{nullptr};
    bool is_decode{false};
  };
  std::unordered_map<UnifiedBatchHandle, AsyncBatchState> async_batches_;
  std::mutex async_batches_mutex_;
  std::atomic<UnifiedBatchHandle> next_handle_{1};
#endif

  // Performance timing via cudaEvents
#ifdef INFERFLUX_NATIVE_KERNELS_READY
  cudaEvent_t forward_start_{nullptr};
  cudaEvent_t forward_stop_{nullptr};
  cudaEvent_t sampling_start_{nullptr};
  cudaEvent_t sampling_stop_{nullptr};

  struct NativePerfAccumulator {
    std::atomic<double> prefill_ms{0.0};
    std::atomic<double> decode_ms{0.0};
    std::atomic<int32_t> prompt_tokens{0};
    std::atomic<int32_t> generated_tokens{0};
  };
  NativePerfAccumulator perf_accum_;
#endif

  // Internal helpers
  bool InitializeCUDA();
  bool InitializeNativePipeline();
  void FreeDeviceMemory();
  bool RunNativeInference(const std::vector<UnifiedBatchInput> &inputs,
                          std::vector<UnifiedBatchOutput> *outputs);
};

} // namespace inferflux
