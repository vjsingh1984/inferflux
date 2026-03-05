#include "runtime/backends/cuda/native/gguf_model_loader.h"
#include "server/logging/logger.h"
#include <climits>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace inferflux {
namespace runtime {
namespace cuda {
namespace native {
namespace {

constexpr std::size_t kMiB = 1024ULL * 1024ULL;

bool SizeToLong(std::size_t value, long *out) {
  if (!out) {
    return false;
  }
  if (value > static_cast<std::size_t>(LONG_MAX)) {
    return false;
  }
  *out = static_cast<long>(value);
  return true;
}

int SaturatingToInt(std::size_t value) {
  if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return std::numeric_limits<int>::max();
  }
  return static_cast<int>(value);
}

bool CheckCudaStatus(cudaError_t status, const char *component,
                     const std::string &operation) {
  if (status == cudaSuccess) {
    return true;
  }
  log::Error(component, operation + " failed: " + cudaGetErrorString(status));
  return false;
}

} // namespace

//==============================================================================
// GGUFTensorData Implementation
//==============================================================================

std::shared_ptr<IQuantizationHandler>
GGUFTensorData::GetQuantizationHandler() const {
  std::string qtype = GetQuantizationType(info.type);
  return CreateQuantizationHandler(qtype);
}

//==============================================================================
// GGUFWeightAccessor Implementation
//==============================================================================

GGUFWeightAccessor::GGUFWeightAccessor(GGUFTensorData *tensor,
                                       GGUFModelLoader *loader)
    : tensor_(tensor), loader_(loader) {
  if (!tensor_) {
    log::Error("gguf_weight_accessor", "Null tensor");
  }
}

std::pair<size_t, size_t> GGUFWeightAccessor::GetDimensions() const {
  if (!tensor_ || tensor_->info.shape.size() < 2) {
    return {0, 0};
  }
  return {tensor_->info.shape[0], tensor_->info.shape[1]};
}

std::string GGUFWeightAccessor::GetDataType() const {
  if (!tensor_) {
    return "unknown";
  }
  return TensorTypeToString(tensor_->info.type);
}

bool GGUFWeightAccessor::IsQuantized() const {
  if (!tensor_) {
    return false;
  }
  return tensor_->info.is_quantized();
}

void *GGUFWeightAccessor::GetGpuWeights(cudaStream_t stream) {
  if (!tensor_) {
    return nullptr;
  }
  return tensor_->gpu_data;
}

half *GGUFWeightAccessor::GetDequantizedGpuWeights(cudaStream_t stream) {
  if (!tensor_) {
    return nullptr;
  }

  // Return cached dequantized data if available
  if (tensor_->dequantized_gpu) {
    return tensor_->dequantized_gpu;
  }

  // Perform lazy dequantization
  auto handler = tensor_->GetQuantizationHandler();
  if (!handler) {
    log::Error("gguf_weight_accessor", "No quantization handler");
    return nullptr;
  }

  // Calculate dequantized size
  size_t num_elements = 1;
  for (uint32_t dim : tensor_->info.shape) {
    num_elements *= dim;
  }

  // Allocate dequantized buffer
  half *d_dequantized = nullptr;
  if (!CheckCudaStatus(cudaMalloc(reinterpret_cast<void **>(&d_dequantized),
                                  num_elements * sizeof(half)),
                       "gguf_weight_accessor",
                       "cudaMalloc(dequantized buffer)")) {
    return nullptr;
  }

  // Dequantize
  handler->DequantizeGpuToGpu(tensor_->gpu_data, d_dequantized, num_elements,
                              stream);
  if (!CheckCudaStatus(cudaGetLastError(), "gguf_weight_accessor",
                       "DequantizeGpuToGpu launch")) {
    cudaFree(d_dequantized);
    return nullptr;
  }

  // Cache result
  tensor_->dequantized_gpu = d_dequantized;

  log::Debug("gguf_weight_accessor",
             "Dequantized tensor: " + tensor_->info.name + " (" +
                 std::to_string(num_elements) + " elements)");

  return d_dequantized;
}

bool GGUFWeightAccessor::IsDequantizedCached() const {
  return tensor_ && tensor_->IsDequantizedCached();
}

//==============================================================================
// GGUFModelLoader Implementation
//==============================================================================

GGUFModelLoader::GGUFModelLoader() = default;

GGUFModelLoader::~GGUFModelLoader() { FreeGPUMemoryImpl(); }

bool GGUFModelLoader::Load(const std::filesystem::path &model_path) {
  model_path_ = model_path;

  log::Info("gguf_loader", "Loading GGUF model: " + model_path.string());

  FILE *file = fopen(model_path.c_str(), "rb");
  if (!file) {
    log::Error("gguf_loader", "Failed to open file: " + model_path.string());
    return false;
  }

  // Parse header
  if (!ParseHeader(file)) {
    fclose(file);
    return false;
  }

  // Parse key-value pairs (metadata)
  if (!ParseKeyValuePairs(file)) {
    fclose(file);
    return false;
  }

  // Parse tensor info
  if (!ParseTensorInfo(file)) {
    fclose(file);
    return false;
  }

  // Load tensor data
  if (!LoadTensorData(file)) {
    fclose(file);
    return false;
  }

  fclose(file);

  // Extract model info
  if (!ExtractModelInfo()) {
    log::Error("gguf_loader", "Failed to extract model info");
    return false;
  }

  // Build tensor name mapping
  BuildTensorNameMapping();

  // Detect quantization
  for (const auto &[name, tensor] : tensors_) {
    if (tensor.info.is_quantized()) {
      quantization_type_ =
          ::inferflux::runtime::cuda::native::GetQuantizationType(
              tensor.info.type);
      if (!quantization_type_.empty()) {
        break;
      }
    }
  }

  log::Info("gguf_loader",
            "Model loaded: " + model_type_ +
                " tensors=" + std::to_string(tensors_.size()) +
                " quantization=" + quantization_type_ +
                " layers=" + std::to_string(model_info_.num_hidden_layers));

  return true;
}

bool GGUFModelLoader::ParseHeader(FILE *file) {
  if (fread(&header_, sizeof(GGUF::Header), 1, file) != 1) {
    log::Error("gguf_loader", "Failed to read header");
    return false;
  }

  if (!ValidateGGUFHeader(header_)) {
    return false;
  }

  log::Debug("gguf_loader",
             "GGUF header: magic=" + std::to_string(header_.magic) +
                 " version=" + std::to_string(header_.version) +
                 " tensors=" + std::to_string(header_.tensor_count) +
                 " kv_count=" + std::to_string(header_.kv_count));

  return true;
}

bool GGUFModelLoader::ParseKeyValuePairs(FILE *file) {
  log::Debug("gguf_loader", "Parsing " + std::to_string(header_.kv_count) +
                                " key-value pairs");

  for (uint32_t i = 0; i < header_.kv_count; ++i) {
    std::string key;
    if (!GGUFReader::ReadString(file, &key)) {
      log::Error("gguf_loader", "Failed to read key");
      return false;
    }

    uint32_t type_val;
    if (fread(&type_val, 4, 1, file) != 1) {
      log::Error("gguf_loader", "Failed to read value type");
      return false;
    }

    GGUF::ValueType type = static_cast<GGUF::ValueType>(type_val);

    // Skip value for now (we'll extract what we need in ExtractModelInfo)
    std::vector<uint8_t> value;
    if (!GGUFReader::ReadValue(file, type, &value)) {
      log::Error("gguf_loader", "Failed to read value for key: " + key);
      return false;
    }

    // Store model type from metadata
    if (key == "general.architecture") {
      if (!value.empty()) {
        model_type_ = std::string(value.begin(), value.end());
      }
    }
  }

  return true;
}

bool GGUFModelLoader::ParseTensorInfo(FILE *file) {
  log::Debug("gguf_loader", "Parsing " + std::to_string(header_.tensor_count) +
                                " tensor info entries");

  for (uint32_t i = 0; i < header_.tensor_count; ++i) {
    GGUFTensorData tensor;
    uint64_t offset;

    // Read name
    if (!GGUFReader::ReadString(file, &tensor.info.name)) {
      log::Error("gguf_loader", "Failed to read tensor name");
      return false;
    }

    // Read number of dimensions
    uint32_t n_dims;
    if (fread(&n_dims, 4, 1, file) != 1) {
      log::Error("gguf_loader", "Failed to read n_dims");
      return false;
    }

    // Read dimensions
    tensor.info.shape.resize(n_dims);
    for (uint32_t j = 0; j < n_dims; ++j) {
      uint32_t dim;
      if (fread(&dim, 4, 1, file) != 1) {
        log::Error("gguf_loader", "Failed to read dimension");
        return false;
      }
      tensor.info.shape[j] = dim;
    }

    // Read type
    uint32_t type_val;
    if (fread(&type_val, 4, 1, file) != 1) {
      log::Error("gguf_loader", "Failed to read tensor type");
      return false;
    }
    tensor.info.type = static_cast<GGUF::TensorType>(type_val);

    // Read offset
    if (fread(&offset, 8, 1, file) != 1) {
      log::Error("gguf_loader", "Failed to read tensor offset");
      return false;
    }
    tensor.info.offset = static_cast<size_t>(offset);

    // Calculate byte size
    tensor.info.byte_size = CalcTensorSize(tensor.info.type, tensor.info.shape);

    tensors_[tensor.info.name] = std::move(tensor);
  }

  return true;
}

bool GGUFModelLoader::LoadTensorData(FILE *file) {
  log::Info("gguf_loader", "Loading tensor data...");

  for (auto &[name, tensor] : tensors_) {
    // Seek to tensor data
    long seek_offset = 0;
    if (!SizeToLong(tensor.info.offset, &seek_offset)) {
      log::Error("gguf_loader", "Tensor offset exceeds seek range: " + name);
      return false;
    }
    if (fseek(file, seek_offset, SEEK_SET) != 0) {
      log::Error("gguf_loader", "Failed to seek to tensor: " + name);
      return false;
    }

    // Allocate buffer
    tensor.cpu_data.resize(tensor.info.byte_size);

    // Read data
    if (fread(tensor.cpu_data.data(), 1, tensor.info.byte_size, file) !=
        tensor.info.byte_size) {
      log::Error("gguf_loader", "Failed to read tensor data: " + name);
      return false;
    }
  }

  size_t total_size = 0;
  for (const auto &[name, tensor] : tensors_) {
    total_size += tensor.info.byte_size;
  }

  log::Info("gguf_loader", "Loaded " + std::to_string(tensors_.size()) +
                               " tensors, " +
                               std::to_string(total_size / kMiB) + " MB");

  return true;
}

bool GGUFModelLoader::ExtractModelInfo() {
  // Extract from parsed metadata
  // For now, use default values
  // TODO: Parse from GGUF metadata in ParseKeyValuePairs

  model_info_.model_type = model_type_.empty() ? "unknown" : model_type_;
  model_info_.torch_dtype = "float16"; // GGUF typically uses FP16
  model_info_.rms_norm_eps = 1e-6f;

  // Count layers from tensor names
  for (const auto &[name, tensor] : tensors_) {
    auto parts = GGUFReader::ParseTensorName(name);
    if (parts.layer >= 0 && parts.layer + 1 > model_info_.num_hidden_layers) {
      model_info_.num_hidden_layers = parts.layer + 1;
    }
  }

  // Get vocab size from token embedding
  auto tok_emb_it = tensors_.find("tok_emb.weight");
  if (tok_emb_it == tensors_.end()) {
    tok_emb_it = tensors_.find("token_embd.weight");
  }
  if (tok_emb_it != tensors_.end()) {
    if (tok_emb_it->second.info.shape.size() >= 1) {
      model_info_.vocab_size =
          SaturatingToInt(tok_emb_it->second.info.shape[0]);
    }
  }

  // Get hidden size from first attention layer
  for (const auto &[name, tensor] : tensors_) {
    if (name.find("attn_q.weight") != std::string::npos ||
        name.find("attn_q.bias") != std::string::npos) {
      if (tensor.info.shape.size() >= 2) {
        model_info_.hidden_size = SaturatingToInt(tensor.info.shape[1]);
        break;
      }
    }
  }

  log::Info("gguf_loader",
            "Model info: type=" + model_info_.model_type +
                " layers=" + std::to_string(model_info_.num_hidden_layers) +
                " hidden=" + std::to_string(model_info_.hidden_size) +
                " vocab=" + std::to_string(model_info_.vocab_size));

  return true;
}

void GGUFModelLoader::BuildTensorNameMapping() {
  for (const auto &[gguf_name, tensor] : tensors_) {
    std::string internal_name =
        GGUFReader::MapTensorName(gguf_name, model_type_);
    gguf_to_internal_name_map_[gguf_name] = internal_name;
    internal_to_gguf_name_map_[internal_name] = gguf_name;
  }
}

bool GGUFModelLoader::UploadToGPU(cudaStream_t stream) {
  log::Info("gguf_loader", "Uploading quantized weights to GPU...");

  // Calculate buffer size
  quantized_buffer_size_ = CalcGPUBufferSize();

  // Allocate unified buffer
  if (!CheckCudaStatus(cudaMalloc(&d_quantized_buffer_, quantized_buffer_size_),
                       "gguf_loader", "cudaMalloc(quantized buffer)")) {
    d_quantized_buffer_ = nullptr;
    return false;
  }

  // Copy quantized weights
  size_t offset = 0;
  uint8_t *buffer_base = static_cast<uint8_t *>(d_quantized_buffer_);
  for (auto &[name, tensor] : tensors_) {
    size_t size = tensor.cpu_data.size();
    if (!CheckCudaStatus(cudaMemcpyAsync(buffer_base + offset,
                                         tensor.cpu_data.data(), size,
                                         cudaMemcpyHostToDevice, stream),
                         "gguf_loader", "cudaMemcpyAsync(" + name + ")")) {
      cudaFree(d_quantized_buffer_);
      d_quantized_buffer_ = nullptr;
      return false;
    }

    tensor.gpu_data = buffer_base + offset;
    tensor.gpu_offset = offset;
    offset += size;
  }

  if (!CheckCudaStatus(cudaStreamSynchronize(stream), "gguf_loader",
                       "cudaStreamSynchronize(upload)")) {
    cudaFree(d_quantized_buffer_);
    d_quantized_buffer_ = nullptr;
    return false;
  }

  log::Info("gguf_loader", "Upload complete: " +
                               std::to_string(quantized_buffer_size_ / kMiB) +
                               " MB");

  // Free CPU memory
  FreeCPUMemory();

  return true;
}

void GGUFModelLoader::FreeCPUMemory() {
  for (auto &[name, tensor] : tensors_) {
    tensor.ClearCPUMemory();
  }
}

void GGUFModelLoader::FreeGPUMemory() { FreeGPUMemoryImpl(); }

void GGUFModelLoader::FreeGPUMemoryImpl() {
  if (d_quantized_buffer_) {
    CheckCudaStatus(cudaFree(d_quantized_buffer_), "gguf_loader",
                    "cudaFree(quantized buffer)");
    d_quantized_buffer_ = nullptr;
  }
  if (d_dequantized_buffer_) {
    CheckCudaStatus(cudaFree(d_dequantized_buffer_), "gguf_loader",
                    "cudaFree(dequantized buffer)");
    d_dequantized_buffer_ = nullptr;
  }
  for (auto &[name, tensor] : tensors_) {
    if (tensor.dequantized_gpu) {
      CheckCudaStatus(cudaFree(tensor.dequantized_gpu), "gguf_loader",
                      "cudaFree(tensor.dequantized_gpu:" + name + ")");
      tensor.dequantized_gpu = nullptr;
    }
  }
}

void *GGUFModelLoader::GetGPUBuffer() const { return d_quantized_buffer_; }

size_t GGUFModelLoader::GetGPUSize() const { return quantized_buffer_size_; }

const ModelInfo &GGUFModelLoader::GetModelInfo() const { return model_info_; }

std::string GGUFModelLoader::GetFormat() const { return "gguf"; }

bool GGUFModelLoader::IsQuantized() const {
  return !quantization_type_.empty();
}

std::string GGUFModelLoader::GetQuantizationType() const {
  return quantization_type_;
}

std::shared_ptr<IWeightAccessor>
GGUFModelLoader::GetWeightAccessor(const std::string &tensor_name) {
  // Check cache first
  auto it = weight_accessor_cache_.find(tensor_name);
  if (it != weight_accessor_cache_.end()) {
    return it->second;
  }

  // Map internal name to GGUF name
  auto gguf_it = internal_to_gguf_name_map_.find(tensor_name);
  if (gguf_it == internal_to_gguf_name_map_.end()) {
    log::Warn("gguf_loader", "Tensor not found: " + tensor_name);
    return nullptr;
  }

  // Get tensor data
  auto tensor_it = tensors_.find(gguf_it->second);
  if (tensor_it == tensors_.end()) {
    log::Warn("gguf_loader", "GGUF tensor not found: " + gguf_it->second);
    return nullptr;
  }

  // Create accessor
  auto accessor =
      std::make_shared<GGUFWeightAccessor>(&tensor_it->second, this);

  weight_accessor_cache_[tensor_name] = accessor;
  return accessor;
}

const GGUFTensorData *
GGUFModelLoader::GetTensorByGGUFName(const std::string &gguf_name) const {
  auto it = tensors_.find(gguf_name);
  if (it != tensors_.end()) {
    return &it->second;
  }
  return nullptr;
}

std::vector<std::string> GGUFModelLoader::GetTensorNames() const {
  std::vector<std::string> names;
  names.reserve(tensors_.size());
  for (const auto &[name, _] : tensors_) {
    names.push_back(name);
  }
  return names;
}

size_t GGUFModelLoader::CalcGPUBufferSize() const {
  size_t total = 0;
  for (const auto &[name, tensor] : tensors_) {
    total += tensor.cpu_data.size();
  }
  return total;
}

} // namespace native
} // namespace cuda
} // namespace runtime
} // namespace inferflux
