#pragma once

#include "runtime/backends/cuda/native/model_loader.h"
#include "runtime/backends/cuda/native_kernel_executor.h"
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace inferflux {
namespace runtime {
namespace cuda {
namespace native {

/**
 * @brief Weight accessor for safetensors format
 *
 * Provides access to non-quantized FP16/BF16 weights from safetensors models.
 * Implements IWeightAccessor interface for compatibility with the abstraction layer.
 */
class SafetensorsWeightAccessor : public IWeightAccessor {
public:
  /**
   * @brief Construct weight accessor
   * @param tensor Pointer to tensor from SafetensorsLoader
   * @param gpu_base Base GPU buffer pointer
   */
  SafetensorsWeightAccessor(const SafetensorsLoader::Tensor *tensor,
                            void *gpu_base);

  // IWeightAccessor interface
  std::pair<size_t, size_t> GetDimensions() const override;
  std::string GetDataType() const override;
  bool IsQuantized() const override;
  void *GetGpuWeights(cudaStream_t stream) override;
  half *GetDequantizedGpuWeights(cudaStream_t stream) override;
  bool IsDequantizedCached() const override;

private:
  const SafetensorsLoader::Tensor *tensor_;
  void *gpu_base_;
  half *dequantized_cache_{nullptr}; // Cache for BF16→FP16 conversion
};

/**
 * @brief Adapter for SafetensorsLoader
 *
 * Adapter pattern implementation that wraps the existing SafetensorsLoader
 * to conform to the IModelLoader interface. This allows safetensors models
 * to work with the new abstraction layer without modifying existing code.
 *
 * Benefits:
 * - Backward compatibility: existing safetensors support remains unchanged
 * - Interface consistency: safetensors and GGUF use same interface
 * - Incremental migration: new code uses IModelLoader, old code unchanged
 */
class SafetensorsLoaderAdapter : public IModelLoader {
public:
  SafetensorsLoaderAdapter();
  ~SafetensorsLoaderAdapter() override;

  // IModelLoader interface
  bool Load(const std::filesystem::path &model_path) override;
  const ModelInfo &GetModelInfo() const override;
  std::string GetFormat() const override;
  bool IsQuantized() const override;
  std::string GetQuantizationType() const override;
  bool UploadToGPU(cudaStream_t stream) override;
  void FreeCPUMemory() override;
  void FreeGPUMemory() override;
  void *GetGPUBuffer() const override;
  size_t GetGPUSize() const override;

  /**
   * @brief Get weight accessor for a specific tensor
   * @param tensor_name Name of tensor (e.g., "model.layers.0.self_attn.q_proj.weight")
   * @return Shared pointer to weight accessor, or nullptr if not found
   */
  std::shared_ptr<IWeightAccessor> GetWeightAccessor(
      const std::string &tensor_name);

  /**
   * @brief Get access to the underlying SafetensorsLoader
   *
   * Provided for backward compatibility with existing code that
   * directly uses SafetensorsLoader (e.g., WeightMap).
   *
   * @return Pointer to underlying SafetensorsLoader
   */
  SafetensorsLoader *GetUnderlyingLoader() { return loader_.get(); }
  const SafetensorsLoader *GetUnderlyingLoader() const {
    return loader_.get();
  }

  /**
   * @brief Get all tensor names
   * @return Vector of tensor names
   */
  std::vector<std::string> GetTensorNames() const;

private:
  std::unique_ptr<SafetensorsLoader> loader_;
  ModelInfo model_info_;

  // Cache of weight accessors
  mutable std::unordered_map<std::string,
                             std::shared_ptr<IWeightAccessor>>
      weight_accessor_cache_;

  // Convert SafetensorsLoader::ModelConfig to ModelInfo
  void ConvertModelInfo();
};

/**
 * @brief Safetensors quantization handler (no-op)
 *
 * Since safetensors models use non-quantized FP16/BF16 weights,
 * this handler simply returns the weights as-is.
 *
 * This implements the Null Object pattern for quantization handlers.
 */
class SafetensorsQuantizationHandler : public IQuantizationHandler {
public:
  void DequantizeGpuToGpu(const void *quantized, half *dequantized,
                          size_t num_elements,
                          cudaStream_t stream) override;

  std::string GetType() const override { return "none"; }

  size_t GetDequantizedSize(size_t quantized_size) const override {
    // No dequantization needed
    return quantized_size;
  }

  double GetBitsPerValue() const override { return 16.0; } // FP16
};

} // namespace native
} // namespace cuda
} // namespace runtime
} // namespace inferflux
