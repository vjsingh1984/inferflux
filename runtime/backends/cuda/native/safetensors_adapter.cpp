#include "runtime/backends/cuda/native/safetensors_adapter.h"
#include "server/logging/logger.h"
#include <cstring>

namespace inferflux {
namespace runtime {
namespace cuda {
namespace native {

//==============================================================================
// SafetensorsWeightAccessor Implementation
//==============================================================================

SafetensorsWeightAccessor::SafetensorsWeightAccessor(
    const SafetensorsLoader::Tensor *tensor, void *gpu_base)
    : tensor_(tensor), gpu_base_(gpu_base) {
  if (!tensor_) {
    log::Error("safetensors_adapter", "WeightAccessor: null tensor");
  }
}

SafetensorsWeightAccessor::~SafetensorsWeightAccessor() {
  if (dequantized_cache_) {
    cudaFree(dequantized_cache_);
    dequantized_cache_ = nullptr;
  }
}

std::pair<size_t, size_t> SafetensorsWeightAccessor::GetDimensions() const {
  if (!tensor_ || tensor_->shape.size() < 2) {
    return {0, 0};
  }
  // Safetensors stores weights as [out_features, in_features]
  // Return {rows, cols} = {shape[0], shape[1]}
  return {tensor_->shape[0], tensor_->shape[1]};
}

std::string SafetensorsWeightAccessor::GetDataType() const {
  if (!tensor_) {
    return "unknown";
  }
  return tensor_->dtype;
}

bool SafetensorsWeightAccessor::IsQuantized() const {
  // Safetensors models use non-quantized FP16/BF16/F32 weights
  return false;
}

void *SafetensorsWeightAccessor::GetGpuWeights(cudaStream_t stream) {
  if (!tensor_) {
    return nullptr;
  }

  // GPU data is stored as offset from base buffer
  return static_cast<uint8_t *>(gpu_base_) + tensor_->gpu_offset;
}

half *SafetensorsWeightAccessor::GetDequantizedGpuWeights(cudaStream_t stream) {
  if (!tensor_) {
    return nullptr;
  }

  void *gpu_ptr = GetGpuWeights(stream);

  // For FP16, return directly
  if (tensor_->dtype == "f16" || tensor_->dtype == "F16") {
    return static_cast<half *>(gpu_ptr);
  }

  // For BF16, return cached converted data
  if (tensor_->dtype == "bf16" || tensor_->dtype == "BF16") {
    if (dequantized_cache_) {
      return dequantized_cache_;
    }

    // Allocate cache
    size_t num_elements = tensor_->shape[0] * tensor_->shape[1];
    size_t bytes = num_elements * sizeof(half);

    void *temp_cache;
    cudaMalloc(&temp_cache, bytes);
    dequantized_cache_ = static_cast<half *>(temp_cache);

    // BF16→FP16 conversion will be done by UploadToGPU
    // This just returns the converted data
    return dequantized_cache_;
  }

  // For F32, need to cast (not ideal, but works)
  if (tensor_->dtype == "f32" || tensor_->dtype == "F32") {
    return static_cast<half *>(gpu_ptr);
  }

  log::Warn("safetensors_adapter",
            "Unsupported dtype for dequantization: " + tensor_->dtype);
  return nullptr;
}

bool SafetensorsWeightAccessor::IsDequantizedCached() const {
  return dequantized_cache_ != nullptr;
}

//==============================================================================
// SafetensorsLoaderAdapter Implementation
//==============================================================================

SafetensorsLoaderAdapter::SafetensorsLoaderAdapter()
    : loader_(std::make_unique<SafetensorsLoader>()) {}

SafetensorsLoaderAdapter::~SafetensorsLoaderAdapter() {
  weight_accessor_cache_.clear();
}

bool SafetensorsLoaderAdapter::Load(const std::filesystem::path &model_path) {
  log::Info("safetensors_adapter",
            "Loading safetensors model from: " + model_path.string());

  if (!loader_->LoadModel(model_path.string())) {
    log::Error("safetensors_adapter", "Failed to load model");
    return false;
  }

  ConvertModelInfo();

  log::Info("safetensors_adapter",
            "Model loaded: " + std::string(model_info_.model_type) +
                " layers=" + std::to_string(model_info_.num_hidden_layers) +
                " hidden=" + std::to_string(model_info_.hidden_size) +
                " dtype=" + model_info_.torch_dtype);

  return true;
}

const ModelInfo &SafetensorsLoaderAdapter::GetModelInfo() const {
  return model_info_;
}

std::string SafetensorsLoaderAdapter::GetFormat() const {
  return "safetensors";
}

bool SafetensorsLoaderAdapter::IsQuantized() const {
  // Safetensors models use non-quantized weights
  return false;
}

std::string SafetensorsLoaderAdapter::GetQuantizationType() const { return ""; }

bool SafetensorsLoaderAdapter::UploadToGPU(cudaStream_t stream) {
  log::Info("safetensors_adapter", "Uploading weights to GPU...");

  // Check for BF16 weights - need conversion
  const auto &config = loader_->GetConfig();
  bool has_bf16 =
      (config.torch_dtype == "bfloat16" || config.torch_dtype == "bf16");

  bool success = loader_->UploadToGPU(stream, !has_bf16);

  if (!success) {
    log::Error("safetensors_adapter", "Failed to upload to GPU");
    return false;
  }

  log::Info("safetensors_adapter",
            "Upload complete: " +
                std::to_string(loader_->GetGPUSize() / (1024 * 1024)) + " MB");

  return true;
}

void SafetensorsLoaderAdapter::FreeCPUMemory() { loader_->FreeCPUMemory(); }

void SafetensorsLoaderAdapter::FreeGPUMemory() { loader_->FreeGPUMemory(); }

void *SafetensorsLoaderAdapter::GetGPUBuffer() const {
  return loader_->GetGPUBuffer();
}

size_t SafetensorsLoaderAdapter::GetGPUSize() const {
  return loader_->GetGPUSize();
}

void SafetensorsLoaderAdapter::SetDequantizedCachePolicy(
    DequantizedCachePolicy policy) {
  dequantized_cache_policy_ = policy;
}

DequantizedCachePolicy
SafetensorsLoaderAdapter::GetDequantizedCachePolicy() const {
  return dequantized_cache_policy_;
}

void SafetensorsLoaderAdapter::ClearDequantizedCache() {
  // Safetensors path is already dense FP16/BF16 and doesn't use the GGUF
  // dequant cache lifecycle contract.
}

std::shared_ptr<IWeightAccessor>
SafetensorsLoaderAdapter::GetWeightAccessor(const std::string &tensor_name) {
  // Check cache first
  auto it = weight_accessor_cache_.find(tensor_name);
  if (it != weight_accessor_cache_.end()) {
    return it->second;
  }

  // Get tensor from loader
  const auto *tensor = loader_->GetTensor(tensor_name);
  if (!tensor) {
    log::Warn("safetensors_adapter", "Tensor not found: " + tensor_name);
    return nullptr;
  }

  // Create accessor
  auto accessor = std::make_shared<SafetensorsWeightAccessor>(
      tensor, loader_->GetGPUBuffer());

  weight_accessor_cache_[tensor_name] = accessor;
  return accessor;
}

std::vector<std::string> SafetensorsLoaderAdapter::GetTensorNames() const {
  return loader_->GetTensorNames();
}

void SafetensorsLoaderAdapter::ConvertModelInfo() {
  const auto &config = loader_->GetConfig();

  model_info_.hidden_size = config.hidden_size;
  model_info_.num_hidden_layers = config.num_hidden_layers;
  model_info_.num_attention_heads = config.num_attention_heads;
  model_info_.num_key_value_heads = config.num_key_value_heads;
  model_info_.head_dim = config.head_dim;
  model_info_.intermediate_size = config.intermediate_size;
  model_info_.vocab_size = config.vocab_size;
  model_info_.max_position_embeddings = config.max_position_embeddings;
  model_info_.rope_freq_base = config.rope_freq_base;
  model_info_.rope_freq_scale = config.rope_freq_scale;
  model_info_.rope_dim = config.rope_dim;
  model_info_.model_type = config.model_type;
  model_info_.activation = config.activation;
  model_info_.torch_dtype = config.torch_dtype;
  model_info_.rms_norm_eps = config.rms_norm_eps;
}

//==============================================================================
// SafetensorsQuantizationHandler Implementation
//==============================================================================

void SafetensorsQuantizationHandler::DequantizeGpuToGpu(const void *quantized,
                                                        half *dequantized,
                                                        size_t num_elements,
                                                        cudaStream_t stream) {
  // No-op: safetensors weights are already FP16/BF16
  // Just copy if source and destination are different
  if (quantized != dequantized) {
    cudaMemcpyAsync(dequantized, quantized, num_elements * sizeof(half),
                    cudaMemcpyDeviceToDevice, stream);
  }
}

} // namespace native
} // namespace cuda
} // namespace runtime
} // namespace inferflux
