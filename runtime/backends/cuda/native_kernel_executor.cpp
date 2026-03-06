#include "runtime/backends/cuda/native_kernel_executor.h"
#include "runtime/backends/cuda/native/gguf_model_loader.h"
#include "runtime/backends/cuda/native/model_loader.h"
#include "runtime/backends/cuda/native/native_tokenizer.h"
#include "runtime/backends/cuda/native/nvtx_scoped.h"
#include "runtime/backends/cuda/native/quantized_forward.h"
#include "runtime/backends/cuda/native/safetensors_parser.h"
#include "server/logging/logger.h"
#include "server/metrics/metrics.h"

#ifdef INFERFLUX_NATIVE_KERNELS_READY
#include "runtime/backends/cuda/native/cublas_gemm.h"
#include "runtime/backends/cuda/native/gpu_sampler.h"
#include "runtime/backends/cuda/native/kv_cache_gpu.h"
#include "runtime/backends/cuda/native/model_forward.h"
#include "runtime/backends/cuda/native/model_forward_factory.h"
#include "runtime/backends/cuda/native/weight_map.h"
#endif

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <set>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <nlohmann/json.hpp>

namespace {

constexpr std::size_t kKiB = 1024ULL;
constexpr std::size_t kMiB = kKiB * kKiB;
constexpr std::size_t kGiB = kMiB * kKiB;

// BFloat16 → Float16 conversion (CPU-side)
// BF16: 1 sign + 8 exponent + 7 mantissa
// FP16: 1 sign + 5 exponent + 10 mantissa
inline uint16_t bf16_to_fp16(uint16_t bf16) {
  uint16_t sign = (bf16 >> 15) & 0x1;
  int32_t exponent = (bf16 >> 7) & 0xFF; // 8-bit BF16 exponent
  uint16_t mantissa = bf16 & 0x7F;       // 7-bit BF16 mantissa

  // Handle special cases
  if (exponent == 0xFF) {
    // Inf or NaN
    if (mantissa == 0) {
      return (sign << 15) | 0x7C00; // FP16 inf
    }
    return (sign << 15) | 0x7E00; // FP16 NaN
  }

  if (exponent == 0) {
    // Zero or denormal → FP16 zero (BF16 denormals are too small for FP16)
    return (sign << 15);
  }

  // Convert exponent: BF16 bias = 127, FP16 bias = 15
  int32_t fp16_exp = exponent - 127 + 15;

  if (fp16_exp >= 0x1F) {
    // Overflow → FP16 infinity
    return (sign << 15) | 0x7C00;
  }

  if (fp16_exp <= 0) {
    // Underflow → FP16 denormal or zero
    if (fp16_exp < -10) {
      return (sign << 15); // Too small, zero
    }
    // Denormalize: shift mantissa right
    uint32_t full_mantissa =
        (1 << 10) | (mantissa << 3); // implicit 1 + shift 7→10
    int shift = 1 - fp16_exp;
    full_mantissa >>= shift;
    return (sign << 15) | (full_mantissa & 0x3FF);
  }

  // Normal case: shift 7-bit mantissa to 10-bit (pad with zeros)
  uint16_t fp16_mantissa = mantissa << 3;
  return (sign << 15) | (fp16_exp << 10) | fp16_mantissa;
}

void ConvertBF16ToFP16(const void *src, size_t num_elements, void *dst) {
  const uint16_t *in = static_cast<const uint16_t *>(src);
  uint16_t *out = static_cast<uint16_t *>(dst);
  for (size_t i = 0; i < num_elements; ++i) {
    out[i] = bf16_to_fp16(in[i]);
  }
}

bool CheckedMulSize(std::size_t a, std::size_t b, std::size_t *out) {
  if (!out) {
    return false;
  }
  if (a != 0U && b > (std::numeric_limits<std::size_t>::max() / a)) {
    return false;
  }
  *out = a * b;
  return true;
}

std::string ToLowerAscii(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return value;
}

bool ParseBoolSetting(const char *raw, bool fallback) {
  if (!raw) {
    return fallback;
  }
  const std::string lowered = ToLowerAscii(raw);
  if (lowered == "1" || lowered == "true" || lowered == "yes" ||
      lowered == "on") {
    return true;
  }
  if (lowered == "0" || lowered == "false" || lowered == "no" ||
      lowered == "off") {
    return false;
  }
  return fallback;
}

std::string
MatmulModeToString(inferflux::runtime::cuda::native::MatmulExecutionMode mode) {
  switch (mode) {
  case inferflux::runtime::cuda::native::MatmulExecutionMode::
      kFusedDequantTileGemm:
    return "fused_dequant_tile_gemm";
  case inferflux::runtime::cuda::native::MatmulExecutionMode::
      kCompatDequantizeThenGemm:
    return "compat_dequantize_then_gemm";
  }
  return "unknown";
}

bool IsKnownQuantizationType(const std::string &quantization_type) {
  const std::string lowered = ToLowerAscii(quantization_type);
  return lowered == "q2_k" || lowered == "q3_k" || lowered == "q4_0" ||
         lowered == "q4_1" || lowered == "q4_k" || lowered == "q4_k_m" ||
         lowered == "q5_0" || lowered == "q5_1" || lowered == "q5_k" ||
         lowered == "q5_k_m" || lowered == "q6_k" || lowered == "q8_0" ||
         lowered == "q8_1" || lowered == "q8_k";
}

inferflux::SafetensorsLoader::ModelConfig
ConvertModelInfo(const inferflux::runtime::cuda::native::ModelInfo &info) {
  inferflux::SafetensorsLoader::ModelConfig cfg;
  cfg.hidden_size = info.hidden_size;
  cfg.num_hidden_layers = info.num_hidden_layers;
  cfg.num_attention_heads = info.num_attention_heads;
  cfg.num_key_value_heads = info.num_key_value_heads;
  cfg.head_dim = info.head_dim;
  if (cfg.head_dim == 0 && cfg.num_attention_heads > 0 && cfg.hidden_size > 0) {
    cfg.head_dim = cfg.hidden_size / cfg.num_attention_heads;
  }
  cfg.intermediate_size = info.intermediate_size;
  cfg.vocab_size = info.vocab_size;
  cfg.max_position_embeddings = info.max_position_embeddings;
  cfg.rope_freq_base = info.rope_freq_base;
  cfg.rope_freq_scale = info.rope_freq_scale;
  cfg.rope_dim = info.rope_dim;
  cfg.model_type = info.model_type;
  cfg.activation = info.activation;
  cfg.torch_dtype = info.torch_dtype;
  cfg.rms_norm_eps = info.rms_norm_eps;
  return cfg;
}

} // namespace

namespace inferflux {

#ifdef INFERFLUX_HAS_CUDA

using json = nlohmann::json;

//==============================================================================
// SafetensorsLoader Implementation
//==============================================================================

SafetensorsLoader::SafetensorsLoader() = default;

bool SafetensorsLoader::LoadModel(const std::string &model_path) {
  model_path_ = model_path;

  log::Info("safetensors_loader", "Loading model from: " + model_path);

  // Step 1: Parse config.json
  std::string config_path = model_path + "/config.json";
  if (!ParseConfig(config_path)) {
    log::Warn("safetensors_loader",
              "Failed to load config.json, using defaults");
  }

  // Step 2: Load and parse safetensors shard files
  std::vector<std::string> shard_files;

  std::string index_path = model_path + "/model.safetensors.index.json";

  if (access(index_path.c_str(), F_OK) == 0) {
    log::Info("safetensors_loader", "Loading from index file");

    try {
      std::ifstream f(index_path);
      json index = json::parse(f);

      std::set<std::string> unique_shards;
      if (index.contains("weight_map")) {
        auto weight_map = index["weight_map"];
        for (auto &[name, shard] : weight_map.items()) {
          unique_shards.insert(shard.get<std::string>());
        }
      }

      for (const auto &shard : unique_shards) {
        shard_files.push_back(model_path + "/" + shard);
      }

    } catch (const std::exception &e) {
      log::Error("safetensors_loader",
                 "Failed to parse index: " + std::string(e.what()));
      return false;
    }
  } else {
    // Try single model.safetensors file
    std::string single_path = model_path + "/model.safetensors";
    if (access(single_path.c_str(), F_OK) == 0) {
      shard_files.push_back(single_path);
    } else {
      log::Error("safetensors_loader",
                 "No safetensors files found in " + model_path);
      return false;
    }
  }

  log::Info("safetensors_loader",
            "Found " + std::to_string(shard_files.size()) + " shard files");

  // Step 3: Parse each shard file
  for (const auto &shard_path : shard_files) {
    log::Info("safetensors_loader", "Parsing shard: " + shard_path);

    auto parser = std::make_unique<SafetensorsParser>(shard_path);
    if (!parser->Parse()) {
      log::Error("safetensors_loader", "Failed to parse shard: " + shard_path);
      return false;
    }

    auto tensor_names = parser->GetTensorNames();
    for (const auto &tensor_name : tensor_names) {
      const auto *tensor_info = parser->GetTensor(tensor_name);
      if (tensor_info && tensors_.find(tensor_name) == tensors_.end()) {
        Tensor tensor;
        tensor.name = tensor_info->name;
        tensor.shape = tensor_info->shape;
        tensor.dtype = tensor_info->dtype;
        tensor.offset = tensor_info->offset;
        tensor.size = tensor_info->byte_size;
        tensor.cpu_data = tensor_info->data_ptr;
        tensor.gpu_data = nullptr;

        tensors_[tensor_name] = std::move(tensor);
      }
    }

    shard_parsers_.push_back(std::move(parser));
  }

  log::Info("safetensors_loader",
            "Model loaded successfully: " + std::to_string(tensors_.size()) +
                " tensors");

  // Log key tensors
  std::vector<std::string> key_tensors = {
      "model.embed_tokens.weight",
      "model.layers.0.input_layernorm.weight",
      "model.layers.0.self_attn.q_proj.weight",
      "model.layers.0.self_attn.k_proj.weight",
      "model.layers.0.self_attn.v_proj.weight",
      "model.layers.0.mlp.gate_proj.weight",
      "model.layers.0.mlp.down_proj.weight",
      "lm_head.weight"};

  for (const auto &tensor : key_tensors) {
    const auto *info = GetTensor(tensor);
    if (info) {
      std::string shape_str = "[";
      for (size_t i = 0; i < info->shape.size(); i++) {
        if (i > 0)
          shape_str += ", ";
        shape_str += std::to_string(info->shape[i]);
      }
      shape_str += "]";
      log::Info("safetensors_loader",
                "  Tensor: " + tensor + " dtype=" + info->dtype +
                    " shape=" + shape_str +
                    " size=" + std::to_string(info->size / kMiB) + " MB");
    }
  }

  return true;
}

bool SafetensorsLoader::LoadIndex(const std::string &index_path) {
  try {
    std::ifstream f(index_path);
    if (!f.is_open()) {
      log::Error("safetensors_loader", "Cannot open index: " + index_path);
      return false;
    }

    json index = json::parse(f);

    if (!index.contains("weight_map")) {
      log::Error("safetensors_loader", "Index missing weight_map");
      return false;
    }

    auto weight_map = index["weight_map"];

    std::unordered_map<std::string, std::vector<std::string>> shard_tensors;
    for (auto &[tensor_name, shard_file] : weight_map.items()) {
      shard_tensors[shard_file.get<std::string>()].push_back(tensor_name);
    }

    log::Info("safetensors_loader",
              "Found " + std::to_string(shard_tensors.size()) + " shard files");

    for (auto &[shard_file, tensor_names] : shard_tensors) {
      std::string full_path = model_path_ + "/" + shard_file;

      struct stat st;
      if (stat(full_path.c_str(), &st) != 0) {
        log::Warn("safetensors_loader", "Cannot stat shard: " + full_path);
        continue;
      }

      for (const auto &tensor_name : tensor_names) {
        Tensor tensor;
        tensor.name = tensor_name;
        tensor.dtype = "";
        tensor.offset = 0;
        tensor.size = 0;
        tensor.cpu_data = nullptr;
        tensor.gpu_data = nullptr;

        tensors_[tensor_name] = std::move(tensor);
      }

      log::Info("safetensors_loader",
                "Shard " + shard_file + ": " +
                    std::to_string(tensor_names.size()) + " tensors, " +
                    std::to_string(st.st_size / static_cast<off_t>(kGiB)) +
                    " GB");
    }

    return true;

  } catch (const std::exception &e) {
    log::Error("safetensors_loader",
               "JSON parse error: " + std::string(e.what()));
    return false;
  }
}

bool SafetensorsLoader::ParseConfig(const std::string &config_path) {
  try {
    std::ifstream f(config_path);
    if (!f.is_open()) {
      return false;
    }

    json config = json::parse(f);

    if (config.contains("hidden_size")) {
      config_.hidden_size = config["hidden_size"];
    }
    if (config.contains("num_hidden_layers")) {
      config_.num_hidden_layers = config["num_hidden_layers"];
    }
    if (config.contains("num_attention_heads")) {
      config_.num_attention_heads = config["num_attention_heads"];
    }
    if (config.contains("num_key_value_heads")) {
      config_.num_key_value_heads = config["num_key_value_heads"];
    }
    if (config.contains("head_dim")) {
      config_.head_dim = config["head_dim"];
    } else {
      config_.head_dim = config_.hidden_size / config_.num_attention_heads;
    }
    if (config.contains("intermediate_size")) {
      config_.intermediate_size = config["intermediate_size"];
    }
    if (config.contains("vocab_size")) {
      config_.vocab_size = config["vocab_size"];
    }
    if (config.contains("max_position_embeddings")) {
      config_.max_position_embeddings = config["max_position_embeddings"];
    }

    // RoPE settings
    if (config.contains("rope_theta")) {
      config_.rope_freq_base = config["rope_theta"];
    }

    // Model type
    if (config.contains("model_type")) {
      config_.model_type = config["model_type"];
    } else if (config.contains("architectures")) {
      auto archs = config["architectures"];
      if (archs.is_array() && archs.size() > 0) {
        config_.model_type = archs[0];
      }
    }

    // Activation
    if (config.contains("hidden_act")) {
      config_.activation = config["hidden_act"];
    }

    // Dtype
    if (config.contains("torch_dtype")) {
      config_.torch_dtype = config["torch_dtype"].get<std::string>();
    }

    // RMS norm epsilon
    if (config.contains("rms_norm_eps")) {
      config_.rms_norm_eps = config["rms_norm_eps"].get<float>();
    }

    log::Info("safetensors_loader",
              "Model config: " + config_.model_type +
                  ", hidden_size=" + std::to_string(config_.hidden_size) +
                  ", num_layers=" + std::to_string(config_.num_hidden_layers) +
                  ", num_heads=" + std::to_string(config_.num_attention_heads) +
                  ", head_dim=" + std::to_string(config_.head_dim));

    return true;

  } catch (const std::exception &e) {
    log::Error("safetensors_loader",
               "Config parse error: " + std::string(e.what()));
    return false;
  }
}

const SafetensorsLoader::Tensor *
SafetensorsLoader::GetTensor(const std::string &name) const {
  auto it = tensors_.find(name);
  if (it == tensors_.end()) {
    return nullptr;
  }
  return &it->second;
}

std::vector<std::string> SafetensorsLoader::GetTensorNames() const {
  std::vector<std::string> names;
  names.reserve(tensors_.size());
  for (const auto &[name, _] : tensors_) {
    names.push_back(name);
  }
  return names;
}

bool SafetensorsLoader::UploadToGPU(cudaStream_t stream,
                                    bool skip_bf16_conversion) {
  log::Info("safetensors_loader",
            "Uploading " + std::to_string(tensors_.size()) + " tensors to GPU");

  size_t total_size = 0;
  for (const auto &[name, tensor] : tensors_) {
    total_size += tensor.size;
  }

  log::Info("safetensors_loader",
            "Total GPU memory needed: " + std::to_string(total_size / kGiB) +
                " GB");

  cudaError_t err = cudaMalloc(&d_weights_buffer_, total_size);
  if (err != cudaSuccess) {
    log::Error("safetensors_loader",
               "cudaMalloc failed: " + std::string(cudaGetErrorString(err)) +
                   " (size=" + std::to_string(total_size) + " bytes)");
    return false;
  }

  log::Info("safetensors_loader",
            "Allocated GPU buffer at " +
                std::to_string(reinterpret_cast<uintptr_t>(d_weights_buffer_)));

  size_t offset = 0;
  size_t uploaded = 0;
  size_t bf16_converted = 0;
  std::vector<uint16_t> convert_buf; // Reusable buffer for BF16→FP16

  for (auto &[name, tensor] : tensors_) {
    if (!tensor.cpu_data) {
      log::Warn("safetensors_loader",
                "Tensor " + name + " has no CPU data, skipping");
      tensor.gpu_data = nullptr;
      continue;
    }

    const void *upload_data = tensor.cpu_data;
    bool needs_conversion = (tensor.dtype == "BF16") && !skip_bf16_conversion;

    if (needs_conversion) {
      // BF16→FP16 conversion on CPU before upload
      size_t num_elements = tensor.size / sizeof(uint16_t);
      if (convert_buf.size() < num_elements) {
        convert_buf.resize(num_elements);
      }
      ConvertBF16ToFP16(tensor.cpu_data, num_elements, convert_buf.data());
      upload_data = convert_buf.data();
      bf16_converted++;
    }

    err = cudaMemcpyAsync(static_cast<uint8_t *>(d_weights_buffer_) + offset,
                          upload_data, tensor.size, cudaMemcpyHostToDevice,
                          stream);

    if (err != cudaSuccess) {
      log::Error("safetensors_loader",
                 "cudaMemcpyAsync failed for " + name + ": " +
                     std::string(cudaGetErrorString(err)));
      cudaFree(d_weights_buffer_);
      d_weights_buffer_ = nullptr;
      return false;
    }

    tensor.gpu_data = static_cast<uint8_t *>(d_weights_buffer_) + offset;
    tensor.gpu_offset = offset;
    offset += tensor.size;
    uploaded++;

    if (uploaded <= 10 || uploaded % 50 == 0) {
      log::Info("safetensors_loader",
                "  Uploaded " + name + " (" +
                    std::to_string(tensor.size / kMiB) + " MB) -> GPU offset " +
                    std::to_string(tensor.gpu_offset) +
                    (needs_conversion ? " [BF16→FP16]" : ""));
    }
  }

  if (bf16_converted > 0) {
    log::Info("safetensors_loader", "Converted " +
                                        std::to_string(bf16_converted) +
                                        " BF16 tensors to FP16");
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    log::Error("safetensors_loader", "cudaStreamSynchronize failed: " +
                                         std::string(cudaGetErrorString(err)));
    cudaFree(d_weights_buffer_);
    d_weights_buffer_ = nullptr;
    return false;
  }

  log::Info("safetensors_loader",
            "Successfully uploaded " + std::to_string(uploaded) +
                " tensors to GPU (" + std::to_string(offset) + " bytes)");

  total_gpu_size_ = offset;

  // Free CPU memory after GPU upload
  log::Info("safetensors_loader", "Freeing CPU memory (keeping GPU memory)");
  for (auto &[name, tensor] : tensors_) {
    tensor.cpu_data = nullptr;
  }
  shard_parsers_.clear();

  return true;
}

void SafetensorsLoader::FreeCPUMemory() {
  log::Info("safetensors_loader", "Freeing CPU memory for " +
                                      std::to_string(tensors_.size()) +
                                      " tensors");

  for (auto &[name, tensor] : tensors_) {
    if (tensor.cpu_data && tensor.cpu_data != MAP_FAILED) {
      munmap(tensor.cpu_data, tensor.size);
      tensor.cpu_data = nullptr;
    }
  }

  log::Info("safetensors_loader", "CPU memory freed");
}

void SafetensorsLoader::FreeGPUMemory() {
  if (d_weights_buffer_) {
    log::Info("safetensors_loader", "Freeing GPU buffer (" +
                                        std::to_string(total_gpu_size_ / kMiB) +
                                        " MB)");
    cudaFree(d_weights_buffer_);
    d_weights_buffer_ = nullptr;
    total_gpu_size_ = 0;

    for (auto &[name, tensor] : tensors_) {
      tensor.gpu_data = nullptr;
      tensor.gpu_offset = 0;
    }
  }
}

SafetensorsLoader::~SafetensorsLoader() {
  FreeCPUMemory();
  FreeGPUMemory();
}

#else // !INFERFLUX_HAS_CUDA

// CPU-only stubs for SafetensorsLoader
SafetensorsLoader::SafetensorsLoader() = default;
SafetensorsLoader::~SafetensorsLoader() = default;
bool SafetensorsLoader::LoadModel(const std::string &) { return false; }
const SafetensorsLoader::Tensor *
SafetensorsLoader::GetTensor(const std::string &) const {
  return nullptr;
}
std::vector<std::string> SafetensorsLoader::GetTensorNames() const {
  return {};
}
bool SafetensorsLoader::UploadToGPU(cudaStream_t, bool) { return false; }
void SafetensorsLoader::FreeCPUMemory() {}
void SafetensorsLoader::FreeGPUMemory() {}
bool SafetensorsLoader::LoadIndex(const std::string &) { return false; }
bool SafetensorsLoader::LoadShard(const std::string &) { return false; }
bool SafetensorsLoader::ParseConfig(const std::string &) { return false; }

#endif // INFERFLUX_HAS_CUDA

//==============================================================================
// NativeKernelExecutor Implementation
//==============================================================================

NativeKernelExecutor::NativeKernelExecutor() = default;

NativeKernelExecutor::~NativeKernelExecutor() {
#ifdef INFERFLUX_NATIVE_KERNELS_READY
  model_forward_.reset();
  sampler_.reset();
  kv_cache_.reset();
  gemm_.reset();
  weight_map_.reset();
  if (d_logits_) {
    if (cudaFree(d_logits_) != cudaSuccess) {
      log::Warn("native_kernel_executor", "cudaFree(d_logits_) failed");
    }
    d_logits_ = nullptr;
  }
  if (forward_start_)
    cudaEventDestroy(forward_start_);
  if (forward_stop_)
    cudaEventDestroy(forward_stop_);
  if (sampling_start_)
    cudaEventDestroy(sampling_start_);
  if (sampling_stop_)
    cudaEventDestroy(sampling_stop_);
  if (prefill_start_event_)
    cudaEventDestroy(prefill_start_event_);
  if (prefill_end_event_)
    cudaEventDestroy(prefill_end_event_);
  if (decode_start_event_)
    cudaEventDestroy(decode_start_event_);
  if (decode_end_event_)
    cudaEventDestroy(decode_end_event_);
  if (decode_stream_)
    cudaStreamDestroy(decode_stream_);
  if (prefill_stream_)
    cudaStreamDestroy(prefill_stream_);
  // Cleanup async batch events
  {
    std::lock_guard<std::mutex> lock(async_batches_mutex_);
    for (auto &[handle, state] : async_batches_) {
      if (state.completion_event)
        cudaEventDestroy(state.completion_event);
    }
    async_batches_.clear();
  }
#endif
  tokenizer_.reset();
  loader_.reset();
#ifdef INFERFLUX_HAS_CUDA
  FreeDeviceMemory();
  if (compute_stream_) {
    cudaStreamDestroy(compute_stream_);
  }
  if (copy_stream_) {
    cudaStreamDestroy(copy_stream_);
  }
#endif
}

#ifdef INFERFLUX_HAS_CUDA

namespace {

bool CheckCudaStatus(cudaError_t status, const std::string &operation) {
  if (status == cudaSuccess) {
    return true;
  }
  log::Error("native_kernel_executor",
             operation + " failed: " + cudaGetErrorString(status));
  return false;
}

bool CheckBF16Support() {
  int device = 0;
  if (!CheckCudaStatus(cudaGetDevice(&device), "cudaGetDevice")) {
    return false;
  }
  cudaDeviceProp prop;
  if (!CheckCudaStatus(cudaGetDeviceProperties(&prop, device),
                       "cudaGetDeviceProperties")) {
    return false;
  }
  return prop.major >= 8;
}

} // namespace

bool NativeKernelExecutor::InitializeCUDA() {
  cudaError_t err = cudaGetDevice(&device_id_);
  if (err != cudaSuccess) {
    log::Error("native_kernel_executor",
               "cudaGetDevice failed: " + std::string(cudaGetErrorString(err)));
    return false;
  }

  err = cudaStreamCreateWithFlags(&compute_stream_, cudaStreamNonBlocking);
  if (err != cudaSuccess) {
    log::Error("native_kernel_executor",
               "cudaStreamCreate failed: " +
                   std::string(cudaGetErrorString(err)));
    return false;
  }

  err = cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking);
  if (err != cudaSuccess) {
    log::Error("native_kernel_executor",
               "cudaStreamCreate failed: " +
                   std::string(cudaGetErrorString(err)));
    return false;
  }

  log::Info("native_kernel_executor",
            "CUDA initialized on device " + std::to_string(device_id_));
  return true;
}

void NativeKernelExecutor::FreeDeviceMemory() {
  // GPU memory is managed by SafetensorsLoader and native components
}

bool NativeKernelExecutor::ConfigureDequantizedCachePolicy(
    const std::string &raw_policy) {
  std::string policy = ToLowerAscii(raw_policy);
  if (policy.empty()) {
    policy = "batch";
  }

  runtime::cuda::native::DequantizedCachePolicy parsed =
      runtime::cuda::native::DequantizedCachePolicy::kBatchLifetime;
  if (!runtime::cuda::native::ParseDequantizedCachePolicy(policy, &parsed)) {
    log::Warn("native_kernel_executor", "Invalid dequantized cache policy '" +
                                            policy +
                                            "'; falling back to batch");
    parsed = runtime::cuda::native::DequantizedCachePolicy::kBatchLifetime;
    policy = "batch";
  }
  dequantized_cache_policy_ = parsed;
  dequantized_cache_policy_hint_ = policy;

  if (model_loader_) {
    model_loader_->SetDequantizedCachePolicy(dequantized_cache_policy_);
  }
  return true;
}

void NativeKernelExecutor::ReleaseBatchScopedDequantizedCache() {
  if (dequantized_cache_policy_ !=
      runtime::cuda::native::DequantizedCachePolicy::kBatchLifetime) {
    return;
  }
  if (!model_loader_ || model_loader_->GetFormat() != "gguf") {
    return;
  }

  if (compute_stream_) {
    CheckCudaStatus(cudaStreamSynchronize(compute_stream_),
                    "cudaStreamSynchronize(compute_stream_,dequant_cleanup)");
  }
#ifdef INFERFLUX_NATIVE_KERNELS_READY
  if (decode_stream_) {
    CheckCudaStatus(cudaStreamSynchronize(decode_stream_),
                    "cudaStreamSynchronize(decode_stream_,dequant_cleanup)");
  }
  if (prefill_stream_) {
    CheckCudaStatus(cudaStreamSynchronize(prefill_stream_),
                    "cudaStreamSynchronize(prefill_stream_,dequant_cleanup)");
  }
#endif

#ifdef INFERFLUX_NATIVE_KERNELS_READY
  if (quantized_weight_map_) {
    quantized_weight_map_->ClearCache();
  }
#endif
  model_loader_->ClearDequantizedCache();
}

#ifdef INFERFLUX_NATIVE_KERNELS_READY
bool NativeKernelExecutor::InitializeNativePipeline() {
  const auto &config = model_config_;

  // Detect inference dtype from model config (overridable via env var)
  bool want_bf16 = (config.torch_dtype == "bfloat16");
  const char *dtype_override = std::getenv("INFERFLUX_NATIVE_DTYPE");
  if (dtype_override) {
    if (std::string(dtype_override) == "fp16") {
      want_bf16 = false;
      log::Info("native_kernel_executor",
                "INFERFLUX_NATIVE_DTYPE=fp16 override: forcing FP16 pipeline");
    } else if (std::string(dtype_override) == "bf16") {
      want_bf16 = true;
      log::Info("native_kernel_executor",
                "INFERFLUX_NATIVE_DTYPE=bf16 override: forcing BF16 pipeline");
    }
  }
  if (want_bf16 && !CheckBF16Support()) {
    log::Warn("native_kernel_executor",
              "BF16 requested but GPU SM < 80; falling back to FP16");
    want_bf16 = false;
  }
  inference_dtype_ = want_bf16 ? InferenceDtype::kBF16 : InferenceDtype::kFP16;

  std::string kv_precision_choice = kv_precision_hint_;
  if (const char *env_kv_precision = std::getenv("INFERFLUX_NATIVE_KV_DTYPE")) {
    kv_precision_choice = env_kv_precision;
  }
  kv_precision_choice = ToLowerAscii(kv_precision_choice);
  if (kv_precision_choice.empty()) {
    kv_precision_choice = "auto";
  }

  if (kv_precision_choice == "auto") {
    kv_precision_ = want_bf16 ? runtime::cuda::native::KvPrecision::kBf16
                              : runtime::cuda::native::KvPrecision::kFp16;
  } else if (!runtime::cuda::native::ParseKvPrecision(kv_precision_choice,
                                                      &kv_precision_)) {
    log::Warn("native_kernel_executor", "Invalid KV precision '" +
                                            kv_precision_choice +
                                            "', falling back to auto");
    kv_precision_ = want_bf16 ? runtime::cuda::native::KvPrecision::kBf16
                              : runtime::cuda::native::KvPrecision::kFp16;
  }

  if (kv_precision_ == runtime::cuda::native::KvPrecision::kBf16 &&
      !CheckBF16Support()) {
    log::Warn("native_kernel_executor",
              "BF16 KV cache requested but GPU SM < 80; using FP16 KV cache");
    kv_precision_ = runtime::cuda::native::KvPrecision::kFp16;
  }
  if (kv_precision_ == runtime::cuda::native::KvPrecision::kInt8 ||
      kv_precision_ == runtime::cuda::native::KvPrecision::kFp8) {
    log::Warn("native_kernel_executor",
              "KV precision '" +
                  runtime::cuda::native::KvPrecisionToString(kv_precision_) +
                  "' is not implemented yet; using FP16 KV cache");
    kv_precision_ = runtime::cuda::native::KvPrecision::kFp16;
  }

  log::Info("native_kernel_executor",
            "Inference dtype: " + std::string(want_bf16 ? "bf16" : "fp16") +
                " (torch_dtype=" + config.torch_dtype + "), kv_dtype=" +
                runtime::cuda::native::KvPrecisionToString(kv_precision_));

  // 1. Initialize cuBLAS wrapper
  gemm_ = std::make_unique<CublasGemm>();
  if (!gemm_->Initialize(compute_stream_)) {
    log::Error("native_kernel_executor", "Failed to initialize cuBLAS");
    return false;
  }

  // 2. Allocate KV cache (independent precision policy)
  int max_batch = 32;
  int max_seq = 4096;
  if (config.max_position_embeddings > 0 &&
      config.max_position_embeddings < max_seq) {
    max_seq = config.max_position_embeddings;
  }

  if (kv_precision_ == runtime::cuda::native::KvPrecision::kBf16) {
    auto cache = std::make_unique<KvCacheGpuTyped<__nv_bfloat16>>();
    if (!cache->Allocate(config.num_hidden_layers, config.num_key_value_heads,
                         config.head_dim, max_seq, max_batch)) {
      log::Error("native_kernel_executor", "Failed to allocate BF16 KV cache");
      return false;
    }
    kv_cache_ = std::move(cache);
  } else {
    auto cache = std::make_unique<KvCacheGpu>();
    if (!cache->Allocate(config.num_hidden_layers, config.num_key_value_heads,
                         config.head_dim, max_seq, max_batch)) {
      log::Error("native_kernel_executor", "Failed to allocate FP16 KV cache");
      return false;
    }
    kv_cache_ = std::move(cache);
  }

  // 2.5. Strategy selection (foundation layer for native quantized runtime).
  runtime::cuda::native::QuantizedRuntimeStrategyRegistry &registry =
      runtime::cuda::native::QuantizedRuntimeStrategyRegistry::Instance();
  registry.RegisterDefaults();

  if (model_loader_ && model_loader_->GetFormat() == "gguf") {
    cudaDeviceProp prop{};
    int sm_major = 0;
    int sm_minor = 0;
    if (CheckCudaStatus(cudaGetDeviceProperties(&prop, device_id_),
                        "cudaGetDeviceProperties(strategy selection)")) {
      sm_major = prop.major;
      sm_minor = prop.minor;
    }

    std::string quantization_type = model_loader_->GetQuantizationType();
    const bool is_quantized_model = model_loader_->IsQuantized();
    if (!quantization_type.empty() &&
        IsKnownQuantizationType(quantization_type)) {
      const auto tensor_type =
          runtime::cuda::native::StringToTensorType(quantization_type);
      const auto selection =
          registry.Select(tensor_type, kv_precision_, sm_major, sm_minor);
      if (selection.weight_layout && selection.matmul && selection.attention) {
        const auto mode = selection.matmul->Mode();
        log::Info("native_kernel_executor",
                  "Selected GGUF strategies: layout=" +
                      selection.weight_layout->Id() + ", matmul=" +
                      selection.matmul->Id() + " (" + MatmulModeToString(mode) +
                      ")" + ", attention=" + selection.attention->Id() + " (" +
                      selection.reason + ")");
        if (require_fused_quantized_matmul_ && is_quantized_model &&
            mode != runtime::cuda::native::MatmulExecutionMode::
                        kFusedDequantTileGemm) {
          log::Error("native_kernel_executor",
                     "Strict quantized runtime policy rejected startup: "
                     "fused dequant-tile GEMM required but selected matmul '" +
                         selection.matmul->Id() + "' (" +
                         MatmulModeToString(mode) + ")");
          return false;
        }
      } else {
        const std::string reason =
            "Incomplete GGUF strategy selection: " + selection.reason;
        if (require_fused_quantized_matmul_ && is_quantized_model) {
          log::Error("native_kernel_executor",
                     "Strict quantized runtime policy rejected startup: " +
                         reason);
          return false;
        }
        log::Warn("native_kernel_executor", reason);
      }
    } else if (is_quantized_model) {
      const std::string reason =
          "Quantized GGUF model loaded but quantization type '" +
          quantization_type + "' is unknown; strategy selection skipped";
      if (require_fused_quantized_matmul_) {
        log::Error("native_kernel_executor",
                   "Strict quantized runtime policy rejected startup: " +
                       reason);
        return false;
      }
      log::Warn("native_kernel_executor", reason);
    }
  }

  // 3. Create weights + forward implementation
  if (model_loader_) {
    quantized_weight_map_ = std::make_unique<QuantizedWeightMap>();
    if (!quantized_weight_map_->Build(model_loader_.get(), model_info_,
                                      compute_stream_)) {
      log::Error("native_kernel_executor",
                 "Failed to build quantized weight map");
      return false;
    }
    model_forward_ = CreateQuantizedForwardAsModelForward(config.model_type);
    auto *quantized_forward =
        dynamic_cast<QuantizedForward *>(model_forward_.get());
    if (!quantized_forward) {
      log::Error("native_kernel_executor",
                 "Failed to create quantized forward for model_type: " +
                     config.model_type);
      return false;
    }
    if (!quantized_forward->Initialize(model_info_, quantized_weight_map_.get(),
                                       kv_cache_.get(), gemm_.get(),
                                       compute_stream_)) {
      log::Error("native_kernel_executor",
                 "Failed to initialize quantized forward");
      return false;
    }
  } else {
    weight_map_ = std::make_unique<WeightMap>();
    if (!weight_map_->Build(*loader_, config)) {
      log::Error("native_kernel_executor", "Failed to build weight map");
      return false;
    }
    if (want_bf16) {
      model_forward_ =
          CreateModelForwardTyped<__nv_bfloat16>(config.model_type);
    } else {
      model_forward_ = CreateModelForward(config.model_type);
    }
    if (!model_forward_) {
      log::Error("native_kernel_executor",
                 "Unsupported model_type: " + config.model_type);
      return false;
    }
    if (!model_forward_->Initialize(config, *weight_map_, kv_cache_.get(),
                                    gemm_.get(), compute_stream_)) {
      log::Error("native_kernel_executor", "Failed to initialize forward pass");
      return false;
    }
  }

  // 4. Initialize GPU sampler
  sampler_ = std::make_unique<GpuSampler>();
  if (!sampler_->Initialize(config.vocab_size, compute_stream_)) {
    log::Error("native_kernel_executor", "Failed to initialize GPU sampler");
    return false;
  }

  // 5. Allocate device logits buffer (sized for batched decode)
  if (config.vocab_size <= 0) {
    log::Error("native_kernel_executor",
               "Invalid vocab_size for logits buffer");
    return false;
  }
  std::size_t logits_elements = 0;
  std::size_t logits_bytes = 0;
  if (!CheckedMulSize(static_cast<std::size_t>(max_batch),
                      static_cast<std::size_t>(config.vocab_size),
                      &logits_elements) ||
      !CheckedMulSize(logits_elements, sizeof(float), &logits_bytes)) {
    log::Error("native_kernel_executor", "Logits buffer size overflow");
    return false;
  }
  cudaError_t err = cudaMalloc(&d_logits_, logits_bytes);
  if (err != cudaSuccess) {
    log::Error("native_kernel_executor", "Failed to allocate logits buffer");
    return false;
  }

  // 6. Load NativeTokenizer from model directory
  tokenizer_ = std::make_unique<NativeTokenizer>();
  const std::string tokenizer_path =
      model_loader_ ? loaded_model_path_.string() : loader_->GetModelPath();
  bool tokenizer_ready = tokenizer_->Load(tokenizer_path);
  if (!tokenizer_ready && model_loader_ &&
      model_loader_->GetFormat() == "gguf") {
    auto *gguf_loader = dynamic_cast<runtime::cuda::native::GGUFModelLoader *>(
        model_loader_.get());
    if (gguf_loader && !gguf_loader->TokenizerPieces().empty()) {
      tokenizer_ready = tokenizer_->LoadFromPieces(
          gguf_loader->TokenizerPieces(), gguf_loader->TokenizerEosTokenId(),
          gguf_loader->TokenizerBosTokenId());
    }
  }
  if (!tokenizer_ready) {
    log::Error("native_kernel_executor",
               "Failed to initialize tokenizer for model path: " +
                   tokenizer_path);
    return false;
  }

  // 9. Create cudaEvent pairs for forward/sampling timing
  if (!CheckCudaStatus(cudaEventCreate(&forward_start_),
                       "cudaEventCreate(forward_start_)") ||
      !CheckCudaStatus(cudaEventCreate(&forward_stop_),
                       "cudaEventCreate(forward_stop_)") ||
      !CheckCudaStatus(cudaEventCreate(&sampling_start_),
                       "cudaEventCreate(sampling_start_)") ||
      !CheckCudaStatus(cudaEventCreate(&sampling_stop_),
                       "cudaEventCreate(sampling_stop_)")) {
    return false;
  }

  // 9.5. Create cudaEvent pairs for phase overlap tracking
  if (!CheckCudaStatus(cudaEventCreate(&prefill_start_event_),
                       "cudaEventCreate(prefill_start_event_)") ||
      !CheckCudaStatus(cudaEventCreate(&prefill_end_event_),
                       "cudaEventCreate(prefill_end_event_)") ||
      !CheckCudaStatus(cudaEventCreate(&decode_start_event_),
                       "cudaEventCreate(decode_start_event_)") ||
      !CheckCudaStatus(cudaEventCreate(&decode_end_event_),
                       "cudaEventCreate(decode_end_event_)")) {
    return false;
  }

  // 10. Create lane-specific streams for async overlap
  if (!CheckCudaStatus(
          cudaStreamCreateWithFlags(&decode_stream_, cudaStreamNonBlocking),
          "cudaStreamCreateWithFlags(decode_stream_)") ||
      !CheckCudaStatus(
          cudaStreamCreateWithFlags(&prefill_stream_, cudaStreamNonBlocking),
          "cudaStreamCreateWithFlags(prefill_stream_)")) {
    return false;
  }

  log::Info("native_kernel_executor",
            "Native inference pipeline initialized successfully");
  return true;
}
#else
bool NativeKernelExecutor::InitializeNativePipeline() {
  log::Warn("native_kernel_executor",
            "Native kernels not compiled; pipeline unavailable");
  return false;
}
#endif

bool NativeKernelExecutor::LoadModel(const std::filesystem::path &model_path,
                                     const LlamaBackendConfig &config) {
  log::Info("native_kernel_executor",
            "Loading native CUDA model from: " + model_path.string());
  loaded_model_path_ = model_path;
  loader_.reset();
  model_loader_.reset();
#ifdef INFERFLUX_NATIVE_KERNELS_READY
  weight_map_.reset();
  quantized_weight_map_.reset();
#endif

  // Initialize phase overlap settings from config
  overlap_enabled_ = config.cuda_phase_overlap_scaffold;
  min_prefill_tokens_ = config.cuda_phase_overlap_min_prefill_tokens;
  kv_precision_hint_ = config.native_kv_cache_dtype;
  dequantized_cache_policy_hint_ = config.native_dequantized_cache_policy;
  require_fused_quantized_matmul_ =
      config.native_require_fused_quantized_matmul;
  if (const char *env_require_fused =
          std::getenv("INFERFLUX_NATIVE_REQUIRE_FUSED_MATMUL")) {
    require_fused_quantized_matmul_ =
        ParseBoolSetting(env_require_fused, require_fused_quantized_matmul_);
  }
  if (const char *env_dequant_policy =
          std::getenv("INFERFLUX_NATIVE_DEQUANT_CACHE_POLICY")) {
    dequantized_cache_policy_hint_ = env_dequant_policy;
  }
  ConfigureDequantizedCachePolicy(dequantized_cache_policy_hint_);
  log::Info(
      "native_kernel_executor",
      "Phase overlap: " +
          std::string(overlap_enabled_ ? "enabled" : "disabled") +
          ", min_prefill_tokens=" + std::to_string(min_prefill_tokens_) +
          ", kv_dtype_hint=" + kv_precision_hint_ + ", dequant_cache_policy=" +
          dequantized_cache_policy_hint_ + ", require_fused_quantized_matmul=" +
          std::string(require_fused_quantized_matmul_ ? "true" : "false"));

  // Initialize CUDA
  if (!InitializeCUDA()) {
    return false;
  }

  auto detected_loader = runtime::cuda::native::CreateModelLoader(model_path);
  if (!detected_loader) {
    log::Error("native_kernel_executor",
               "Failed to detect model loader for path: " +
                   model_path.string());
    return false;
  }

  const std::string detected_format = detected_loader->GetFormat();
  detected_loader->SetDequantizedCachePolicy(dequantized_cache_policy_);
  if (detected_format == "safetensors") {
    // Keep legacy safetensors loader path until the generic loader path is
    // feature-parity validated for BF16/FP16 conversion controls.
    loader_ = std::make_unique<SafetensorsLoader>();
    if (!loader_->LoadModel(model_path.string())) {
      log::Error("native_kernel_executor", "Failed to load safetensors model");
      return false;
    }
    model_config_ = loader_->GetConfig();

    // Decide whether to skip BF16→FP16 conversion
    // If model is BF16 and GPU supports it, keep BF16 for native pipeline
    // Env var INFERFLUX_NATIVE_DTYPE=fp16 forces FP16 conversion
    bool skip_bf16 = false;
#ifdef INFERFLUX_NATIVE_KERNELS_READY
    {
      const char *dtype_override = std::getenv("INFERFLUX_NATIVE_DTYPE");
      bool force_fp16 = dtype_override && std::string(dtype_override) == "fp16";
      if (model_config_.torch_dtype == "bfloat16" && CheckBF16Support() &&
          !force_fp16) {
        skip_bf16 = true;
        log::Info(
            "native_kernel_executor",
            "BF16 model detected with SM >= 80; uploading weights as BF16");
      } else if (model_config_.torch_dtype == "bfloat16" && force_fp16) {
        log::Info("native_kernel_executor",
                  "BF16 model with FP16 override; converting BF16→FP16 on CPU");
      }
    }
#endif

    // Upload weights to GPU
    log::Info("native_kernel_executor", "Uploading weights to GPU...");
    if (!loader_->UploadToGPU(compute_stream_, skip_bf16)) {
      log::Error("native_kernel_executor", "Failed to upload weights to GPU");
      return false;
    }
  } else {
    model_loader_ = std::move(detected_loader);
    model_loader_->SetDequantizedCachePolicy(dequantized_cache_policy_);
    if (!model_loader_->Load(model_path)) {
      log::Error("native_kernel_executor",
                 "Failed to load model via loader: " + detected_format);
      return false;
    }
    model_info_ = model_loader_->GetModelInfo();
    model_config_ = ConvertModelInfo(model_info_);
    log::Info("native_kernel_executor", "Uploading model weights to GPU via " +
                                            detected_format + " loader...");
    if (!model_loader_->UploadToGPU(compute_stream_)) {
      log::Error("native_kernel_executor",
                 "Failed to upload model weights via " + detected_format +
                     " loader");
      return false;
    }
  }

  // Initialize native inference pipeline
  if (!InitializeNativePipeline()) {
    log::Error("native_kernel_executor",
               "Native pipeline initialization failed");
    return false;
  }

  // Report native backend uses FA2 attention kernel
  GlobalMetrics().SetCudaAttentionKernel("fa2");

  log::Info("native_kernel_executor", "Native CUDA model loaded successfully");
  model_loaded_ = true;
  return true;
}

#ifdef INFERFLUX_NATIVE_KERNELS_READY

// ==========================================================================
// Phase Overlap Helper Methods
// ==========================================================================

bool NativeKernelExecutor::IsPrefillLikeInput(
    const UnifiedBatchInput &input) const {
  // Prefill requests have multiple tokens or don't request logits
  return input.tokens.size() > 1 || !input.request_logits;
}

bool NativeKernelExecutor::HasMixedWorkload(
    const std::vector<UnifiedBatchInput> &inputs) const {
  if (!overlap_enabled_) {
    return false;
  }

  bool has_prefill = false;
  bool has_decode = false;

  for (const auto &input : inputs) {
    if (IsPrefillLikeInput(input)) {
      has_prefill = true;
    } else {
      has_decode = true;
    }
    if (has_prefill && has_decode) {
      return true;
    }
  }

  return false;
}

void NativeKernelExecutor::SplitBatchByType(
    const std::vector<UnifiedBatchInput> &inputs,
    std::vector<size_t> &prefill_indices,
    std::vector<size_t> &decode_indices) const {
  prefill_indices.clear();
  decode_indices.clear();

  for (size_t i = 0; i < inputs.size(); ++i) {
    if (IsPrefillLikeInput(inputs[i])) {
      prefill_indices.push_back(i);
    } else {
      decode_indices.push_back(i);
    }
  }
}

std::vector<LlamaCPUBackend::UnifiedBatchOutput>
NativeKernelExecutor::ExecuteUnifiedBatchWithOverlap(
    const std::vector<UnifiedBatchInput> &inputs) {
  std::vector<UnifiedBatchOutput> outputs;
  outputs.resize(inputs.size());

  // Split batch into prefill and decode subsets
  std::vector<size_t> prefill_indices;
  std::vector<size_t> decode_indices;
  SplitBatchByType(inputs, prefill_indices, decode_indices);

  // Check if prefill is large enough to warrant overlap
  int total_prefill_tokens = 0;
  for (size_t idx : prefill_indices) {
    total_prefill_tokens += static_cast<int>(inputs[idx].tokens.size());
  }

  if (total_prefill_tokens < min_prefill_tokens_ || prefill_indices.empty() ||
      decode_indices.empty()) {
    // Fall back to standard execution
    log::Debug("native_kernel_executor",
               "Skipping overlap: prefill_tokens=" +
                   std::to_string(total_prefill_tokens) +
                   ", min=" + std::to_string(min_prefill_tokens_) +
                   ", prefill_count=" + std::to_string(prefill_indices.size()) +
                   ", decode_count=" + std::to_string(decode_indices.size()));
    return ExecuteUnifiedBatch(inputs);
  }

  log::Info("native_kernel_executor",
            "Using async overlap for mixed batch (prefill=" +
                std::to_string(prefill_indices.size()) +
                ", decode=" + std::to_string(decode_indices.size()) +
                ", prefill_tokens=" + std::to_string(total_prefill_tokens) +
                ")");

  // Record start events on both streams
  if (!CheckCudaStatus(cudaEventRecord(prefill_start_event_, prefill_stream_),
                       "cudaEventRecord(prefill_start_event_)") ||
      !CheckCudaStatus(cudaEventRecord(decode_start_event_, decode_stream_),
                       "cudaEventRecord(decode_start_event_)")) {
    return outputs;
  }

  // Execute decode on decode_stream_ (concurrently with prefill)
  struct DecodeEntry {
    size_t input_idx;
    int token_id;
    int n_past;
    int sequence_id;
    float temperature;
    int top_k;
    float top_p;
    uint32_t seed;
  };
  std::vector<DecodeEntry> decode_group;
  decode_group.reserve(decode_indices.size());

  for (size_t idx : decode_indices) {
    const auto &input = inputs[idx];
    decode_group.push_back({idx, input.tokens[0], input.n_past,
                            input.sequence_id, input.sampling.temperature,
                            input.sampling.top_k, input.sampling.top_p,
                            input.sampling.seed});
  }

  // Switch to decode stream and launch decode kernels
  model_forward_->SetStream(decode_stream_);
  gemm_->SetStream(decode_stream_);

  if (!decode_group.empty()) {
    int B = static_cast<int>(decode_group.size());

    // Collect batch vectors
    std::vector<int> batch_tokens(B);
    std::vector<int> batch_n_past(B);
    std::vector<int> batch_seq_ids(B);
    std::vector<float> batch_temps(B);
    std::vector<int> batch_top_ks(B);
    std::vector<float> batch_top_ps(B);

    for (int b = 0; b < B; ++b) {
      batch_tokens[b] = decode_group[b].token_id;
      batch_n_past[b] = decode_group[b].n_past;
      batch_seq_ids[b] = decode_group[b].sequence_id;
      batch_temps[b] = decode_group[b].temperature;
      batch_top_ks[b] = decode_group[b].top_k;
      batch_top_ps[b] = decode_group[b].top_p;
    }

    // Batched forward pass on decode stream (non-blocking)
    bool fwd_ok = model_forward_->BatchForward(batch_tokens, batch_n_past,
                                               batch_seq_ids, d_logits_, B);

    if (fwd_ok) {
      // Batched sampling on decode stream (non-blocking)
      std::vector<int> sampled_tokens;
      sampler_->SampleBatch(d_logits_, B, batch_temps, batch_top_ks,
                            batch_top_ps, &sampled_tokens);

      // Fill outputs
      for (int b = 0; b < B; ++b) {
        const auto &de = decode_group[b];
        auto &output = outputs[de.input_idx];
        output.token = sampled_tokens[b];
        output.piece =
            tokenizer_ ? tokenizer_->IdToString(sampled_tokens[b]) : "";
        output.ok = true;
        perf_accum_.generated_tokens.fetch_add(1, std::memory_order_relaxed);
      }
    } else {
      for (const auto &de : decode_group) {
        outputs[de.input_idx].ok = false;
        outputs[de.input_idx].token = -1;
      }
    }
  }

  // Execute prefill on prefill_stream_ (concurrently with decode)
  model_forward_->SetStream(prefill_stream_);
  gemm_->SetStream(prefill_stream_);

  for (size_t idx : prefill_indices) {
    const auto &input = inputs[idx];
    auto &output = outputs[idx];
    output.ok = false;
    output.token = -1;

    int batch_tokens = static_cast<int>(input.tokens.size());

    // Forward pass on prefill stream (non-blocking)
    if (!model_forward_->Forward(input.tokens, input.n_past, input.sequence_id,
                                 d_logits_)) {
      log::Error("native_kernel_executor", "Prefill forward pass failed");
      continue;
    }

    if (input.request_logits) {
      // Sampling on prefill stream (non-blocking)
      int token_id = sampler_->Sample(
          d_logits_, input.sampling.temperature, input.sampling.top_k,
          input.sampling.top_p, input.sampling.seed);
      output.token = token_id;
      output.piece = tokenizer_ ? tokenizer_->IdToString(token_id) : "";
      perf_accum_.generated_tokens.fetch_add(1, std::memory_order_relaxed);
    }

    perf_accum_.prompt_tokens.fetch_add(batch_tokens,
                                        std::memory_order_relaxed);
    output.ok = true;
  }

  // Record end events on both streams
  if (!CheckCudaStatus(cudaEventRecord(prefill_end_event_, prefill_stream_),
                       "cudaEventRecord(prefill_end_event_)") ||
      !CheckCudaStatus(cudaEventRecord(decode_end_event_, decode_stream_),
                       "cudaEventRecord(decode_end_event_)")) {
    return outputs;
  }

  // Synchronize both streams before returning
  if (!CheckCudaStatus(cudaStreamSynchronize(prefill_stream_),
                       "cudaStreamSynchronize(prefill_stream_)") ||
      !CheckCudaStatus(cudaStreamSynchronize(decode_stream_),
                       "cudaStreamSynchronize(decode_stream_)")) {
    return outputs;
  }

  // Calculate overlap duration
  float prefill_ms = 0.0f;
  float decode_ms = 0.0f;
  if (!CheckCudaStatus(cudaEventElapsedTime(&prefill_ms, prefill_start_event_,
                                            prefill_end_event_),
                       "cudaEventElapsedTime(prefill)") ||
      !CheckCudaStatus(cudaEventElapsedTime(&decode_ms, decode_start_event_,
                                            decode_end_event_),
                       "cudaEventElapsedTime(decode)")) {
    return outputs;
  }

  // Record metrics
  GlobalMetrics().RecordNativeForwardPass(/*is_decode=*/false,
                                          total_prefill_tokens, prefill_ms);
  GlobalMetrics().RecordNativeForwardPass(
      /*is_decode=*/true, static_cast<int>(decode_indices.size()), decode_ms);
  perf_accum_.prefill_ms.store(
      perf_accum_.prefill_ms.load(std::memory_order_relaxed) + prefill_ms,
      std::memory_order_relaxed);
  perf_accum_.decode_ms.store(
      perf_accum_.decode_ms.load(std::memory_order_relaxed) + decode_ms,
      std::memory_order_relaxed);

  // Calculate overlap (how much time was saved by concurrent execution)
  float total_sequential_ms = prefill_ms + decode_ms;
  float actual_concurrent_ms = std::max(prefill_ms, decode_ms);
  float overlap_ms = total_sequential_ms - actual_concurrent_ms;

  if (overlap_ms > 0) {
    GlobalMetrics().RecordCudaLaneOverlap(overlap_ms);
    log::Info("native_kernel_executor",
              "Phase overlap: " + std::to_string(overlap_ms) + "ms saved (" +
                  std::to_string(static_cast<int>(100.0 * overlap_ms /
                                                  total_sequential_ms)) +
                  "% reduction)");
  }

  // Restore compute stream for next execution
  model_forward_->SetStream(compute_stream_);
  gemm_->SetStream(compute_stream_);

  return outputs;
}

#endif // INFERFLUX_NATIVE_KERNELS_READY

std::vector<LlamaCPUBackend::UnifiedBatchOutput>
NativeKernelExecutor::ExecuteUnifiedBatch(
    const std::vector<LlamaCPUBackend::UnifiedBatchInput> &inputs) {
  if (!model_loaded_) {
    log::Error("native_kernel_executor", "Model not loaded");
    return {};
  }

#ifdef INFERFLUX_NATIVE_KERNELS_READY
  if (!model_forward_) {
    log::Warn("native_kernel_executor", "Native pipeline not initialized");
    return {};
  }

  struct ScopedDequantCacheCleanup {
    NativeKernelExecutor *executor{nullptr};
    ~ScopedDequantCacheCleanup() {
      if (executor) {
        executor->ReleaseBatchScopedDequantizedCache();
      }
    }
  } scoped_cleanup{this};

  NVTX_SCOPE("NativeExecuteUnifiedBatch");

  // Check for mixed workload and use overlap path if enabled
  if (HasMixedWorkload(inputs)) {
    return ExecuteUnifiedBatchWithOverlap(inputs);
  }

  // Standard execution path for non-mixed workloads
  std::vector<UnifiedBatchOutput> outputs;
  outputs.resize(inputs.size());

  // Partition inputs into decode (tokens.size()==1, request_logits) and others
  struct DecodeEntry {
    int input_idx;
    int token_id;
    int n_past;
    int sequence_id;
    float temperature;
    int top_k;
    float top_p;
    uint32_t seed;
  };
  std::vector<DecodeEntry> decode_group;
  std::vector<int> prefill_indices;

  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto &input = inputs[i];
    if (input.tokens.size() == 1 && input.request_logits) {
      decode_group.push_back({static_cast<int>(i), input.tokens[0],
                              input.n_past, input.sequence_id,
                              input.sampling.temperature, input.sampling.top_k,
                              input.sampling.top_p, input.sampling.seed});
    } else {
      prefill_indices.push_back(static_cast<int>(i));
    }
  }

  // === Batched decode group ===
  if (!decode_group.empty()) {
    int B = static_cast<int>(decode_group.size());
    NVTX_SCOPE("BatchedDecode");

    // Collect batch vectors
    std::vector<int> batch_tokens(B);
    std::vector<int> batch_n_past(B);
    std::vector<int> batch_seq_ids(B);
    std::vector<float> batch_temps(B);
    std::vector<int> batch_top_ks(B);
    std::vector<float> batch_top_ps(B);

    for (int b = 0; b < B; ++b) {
      batch_tokens[b] = decode_group[b].token_id;
      batch_n_past[b] = decode_group[b].n_past;
      batch_seq_ids[b] = decode_group[b].sequence_id;
      batch_temps[b] = decode_group[b].temperature;
      batch_top_ks[b] = decode_group[b].top_k;
      batch_top_ps[b] = decode_group[b].top_p;
    }

    // Batched forward pass
    if (!CheckCudaStatus(cudaEventRecord(forward_start_, compute_stream_),
                         "cudaEventRecord(forward_start_,decode_batch)")) {
      for (const auto &de : decode_group) {
        outputs[de.input_idx].ok = false;
        outputs[de.input_idx].token = -1;
      }
      return outputs;
    }
    bool fwd_ok = model_forward_->BatchForward(batch_tokens, batch_n_past,
                                               batch_seq_ids, d_logits_, B);
    if (!CheckCudaStatus(cudaEventRecord(forward_stop_, compute_stream_),
                         "cudaEventRecord(forward_stop_,decode_batch)") ||
        !CheckCudaStatus(cudaEventSynchronize(forward_stop_),
                         "cudaEventSynchronize(forward_stop_,decode_batch)")) {
      for (const auto &de : decode_group) {
        outputs[de.input_idx].ok = false;
        outputs[de.input_idx].token = -1;
      }
      return outputs;
    }
    float fwd_ms = 0.0f;
    if (!CheckCudaStatus(
            cudaEventElapsedTime(&fwd_ms, forward_start_, forward_stop_),
            "cudaEventElapsedTime(forward,decode_batch)")) {
      for (const auto &de : decode_group) {
        outputs[de.input_idx].ok = false;
        outputs[de.input_idx].token = -1;
      }
      return outputs;
    }
    GlobalMetrics().RecordNativeForwardPass(/*is_decode=*/true, B, fwd_ms);
    perf_accum_.decode_ms.store(
        perf_accum_.decode_ms.load(std::memory_order_relaxed) + fwd_ms,
        std::memory_order_relaxed);

    if (!fwd_ok) {
      log::Error("native_kernel_executor", "BatchForward failed");
      for (const auto &de : decode_group) {
        outputs[de.input_idx].ok = false;
        outputs[de.input_idx].token = -1;
      }
    } else {
      // Batched sampling
      if (!CheckCudaStatus(cudaEventRecord(sampling_start_, compute_stream_),
                           "cudaEventRecord(sampling_start_,decode_batch)")) {
        for (const auto &de : decode_group) {
          outputs[de.input_idx].ok = false;
          outputs[de.input_idx].token = -1;
        }
        return outputs;
      }
      std::vector<int> sampled_tokens;
      sampler_->SampleBatch(d_logits_, B, batch_temps, batch_top_ks,
                            batch_top_ps, &sampled_tokens);
      if (!CheckCudaStatus(cudaEventRecord(sampling_stop_, compute_stream_),
                           "cudaEventRecord(sampling_stop_,decode_batch)") ||
          !CheckCudaStatus(
              cudaEventSynchronize(sampling_stop_),
              "cudaEventSynchronize(sampling_stop_,decode_batch)")) {
        for (const auto &de : decode_group) {
          outputs[de.input_idx].ok = false;
          outputs[de.input_idx].token = -1;
        }
        return outputs;
      }
      float samp_ms = 0.0f;
      if (!CheckCudaStatus(
              cudaEventElapsedTime(&samp_ms, sampling_start_, sampling_stop_),
              "cudaEventElapsedTime(sampling,decode_batch)")) {
        for (const auto &de : decode_group) {
          outputs[de.input_idx].ok = false;
          outputs[de.input_idx].token = -1;
        }
        return outputs;
      }
      GlobalMetrics().RecordNativeSampling(B, samp_ms);

      // Fill outputs
      for (int b = 0; b < B; ++b) {
        int idx = decode_group[b].input_idx;
        int token_id = sampled_tokens[b];
        if (tokenizer_ && token_id == tokenizer_->EosTokenId()) {
          outputs[idx].token = -1;
          outputs[idx].piece = "";
        } else {
          outputs[idx].token = token_id;
          outputs[idx].piece =
              tokenizer_ ? tokenizer_->IdToString(token_id) : "";
          perf_accum_.generated_tokens.fetch_add(1, std::memory_order_relaxed);
        }
        outputs[idx].ok = true;
      }
    }
  }

  // === Prefill group (sequential, variable lengths) ===
  for (int idx : prefill_indices) {
    const auto &input = inputs[idx];
    UnifiedBatchOutput &output = outputs[idx];
    output.ok = false;
    output.token = -1;

    bool is_decode = (input.tokens.size() == 1);
    int batch_tokens = static_cast<int>(input.tokens.size());

    if (!input.request_logits) {
      NVTX_SCOPE("ForwardPass");
      if (!CheckCudaStatus(
              cudaEventRecord(forward_start_, compute_stream_),
              "cudaEventRecord(forward_start_,prefill_no_logits)")) {
        continue;
      }
      if (!model_forward_->Forward(input.tokens, input.n_past,
                                   input.sequence_id, d_logits_)) {
        log::Error("native_kernel_executor", "Forward pass failed");
      }
      if (!CheckCudaStatus(
              cudaEventRecord(forward_stop_, compute_stream_),
              "cudaEventRecord(forward_stop_,prefill_no_logits)") ||
          !CheckCudaStatus(
              cudaEventSynchronize(forward_stop_),
              "cudaEventSynchronize(forward_stop_,prefill_no_logits)")) {
        continue;
      }
      float fwd_ms = 0.0f;
      if (!CheckCudaStatus(
              cudaEventElapsedTime(&fwd_ms, forward_start_, forward_stop_),
              "cudaEventElapsedTime(forward,prefill_no_logits)")) {
        continue;
      }
      GlobalMetrics().RecordNativeForwardPass(is_decode, batch_tokens, fwd_ms);
      perf_accum_.prefill_ms.store(
          perf_accum_.prefill_ms.load(std::memory_order_relaxed) + fwd_ms,
          std::memory_order_relaxed);
      perf_accum_.prompt_tokens.fetch_add(batch_tokens,
                                          std::memory_order_relaxed);
      output.ok = true;
      continue;
    }

    // Forward pass
    {
      NVTX_SCOPE("ForwardPass");
      if (!CheckCudaStatus(cudaEventRecord(forward_start_, compute_stream_),
                           "cudaEventRecord(forward_start_,prefill_logits)")) {
        continue;
      }
      if (!model_forward_->Forward(input.tokens, input.n_past,
                                   input.sequence_id, d_logits_)) {
        log::Error("native_kernel_executor", "Forward pass failed");
        continue;
      }
      if (!CheckCudaStatus(cudaEventRecord(forward_stop_, compute_stream_),
                           "cudaEventRecord(forward_stop_,prefill_logits)") ||
          !CheckCudaStatus(
              cudaEventSynchronize(forward_stop_),
              "cudaEventSynchronize(forward_stop_,prefill_logits)")) {
        continue;
      }
      float fwd_ms = 0.0f;
      if (!CheckCudaStatus(
              cudaEventElapsedTime(&fwd_ms, forward_start_, forward_stop_),
              "cudaEventElapsedTime(forward,prefill_logits)")) {
        continue;
      }
      GlobalMetrics().RecordNativeForwardPass(is_decode, batch_tokens, fwd_ms);
      if (is_decode) {
        perf_accum_.decode_ms.store(
            perf_accum_.decode_ms.load(std::memory_order_relaxed) + fwd_ms,
            std::memory_order_relaxed);
      } else {
        perf_accum_.prefill_ms.store(
            perf_accum_.prefill_ms.load(std::memory_order_relaxed) + fwd_ms,
            std::memory_order_relaxed);
        perf_accum_.prompt_tokens.fetch_add(batch_tokens,
                                            std::memory_order_relaxed);
      }
    }

    // Sample
    {
      NVTX_SCOPE("Sampling");
      if (!CheckCudaStatus(cudaEventRecord(sampling_start_, compute_stream_),
                           "cudaEventRecord(sampling_start_,prefill_logits)")) {
        continue;
      }
      int token_id = sampler_->Sample(
          d_logits_, input.sampling.temperature, input.sampling.top_k,
          input.sampling.top_p, input.sampling.seed);
      if (!CheckCudaStatus(cudaEventRecord(sampling_stop_, compute_stream_),
                           "cudaEventRecord(sampling_stop_,prefill_logits)") ||
          !CheckCudaStatus(
              cudaEventSynchronize(sampling_stop_),
              "cudaEventSynchronize(sampling_stop_,prefill_logits)")) {
        continue;
      }
      float samp_ms = 0.0f;
      if (!CheckCudaStatus(
              cudaEventElapsedTime(&samp_ms, sampling_start_, sampling_stop_),
              "cudaEventElapsedTime(sampling,prefill_logits)")) {
        continue;
      }
      GlobalMetrics().RecordNativeSampling(1, samp_ms);

      if (tokenizer_ && token_id == tokenizer_->EosTokenId()) {
        output.token = -1;
        output.piece = "";
      } else {
        output.token = token_id;
        output.piece = tokenizer_ ? tokenizer_->IdToString(token_id) : "";
        perf_accum_.generated_tokens.fetch_add(1, std::memory_order_relaxed);
      }
      output.ok = true;
    }
  }

  return outputs;
#else
  log::Warn("native_kernel_executor",
            "Native inference not compiled (INFERFLUX_NATIVE_KERNELS_READY=0)");
  return {};
#endif
}

bool NativeKernelExecutor::SupportsAsyncUnifiedBatch() const {
#ifdef INFERFLUX_NATIVE_KERNELS_READY
  return model_forward_ != nullptr;
#else
  return false;
#endif
}

UnifiedBatchHandle NativeKernelExecutor::SubmitUnifiedBatchAsync(
    const std::vector<UnifiedBatchInput> &inputs, UnifiedBatchLane lane) {
#ifdef INFERFLUX_NATIVE_KERNELS_READY
  if (!model_forward_)
    return 0;

  bool is_decode = (lane == UnifiedBatchLane::kDecode) ||
                   (lane == UnifiedBatchLane::kAuto && !inputs.empty() &&
                    inputs[0].tokens.size() == 1);
  cudaStream_t target_stream = is_decode ? decode_stream_ : prefill_stream_;

  // Lock the pipeline, switch stream, execute, restore
  std::lock_guard<std::mutex> lock(lane_mutex_);
  model_forward_->SetStream(target_stream);
  gemm_->SetStream(target_stream);

  auto results = ExecuteUnifiedBatch(inputs);

  // Restore compute stream
  model_forward_->SetStream(compute_stream_);
  gemm_->SetStream(compute_stream_);

  // Record completion event
  cudaEvent_t event;
  if (!CheckCudaStatus(cudaEventCreate(&event),
                       "cudaEventCreate(async_completion_event)")) {
    return 0;
  }
  if (!CheckCudaStatus(cudaEventRecord(event, target_stream),
                       "cudaEventRecord(async_completion_event)")) {
    cudaEventDestroy(event);
    return 0;
  }

  UnifiedBatchHandle handle = next_handle_.fetch_add(1);
  {
    std::lock_guard<std::mutex> alock(async_batches_mutex_);
    async_batches_[handle] = {std::move(results), event, is_decode};
  }

  GlobalMetrics().RecordCudaLaneSubmission(is_decode);
  return handle;
#else
  (void)inputs;
  (void)lane;
  return 0;
#endif
}

bool NativeKernelExecutor::TryCollectUnifiedBatchAsync(
    UnifiedBatchHandle handle, std::vector<UnifiedBatchOutput> *outputs) {
#ifdef INFERFLUX_NATIVE_KERNELS_READY
  std::lock_guard<std::mutex> lock(async_batches_mutex_);
  auto it = async_batches_.find(handle);
  if (it == async_batches_.end())
    return false;

  cudaError_t status = cudaEventQuery(it->second.completion_event);
  if (status == cudaErrorNotReady) {
    return false;
  }
  if (status != cudaSuccess) {
    log::Error("native_kernel_executor",
               "cudaEventQuery(async_completion_event) failed: " +
                   std::string(cudaGetErrorString(status)));
    cudaEventDestroy(it->second.completion_event);
    async_batches_.erase(it);
    return false;
  }

  // Completed
  if (outputs) {
    *outputs = std::move(it->second.outputs);
  }
  GlobalMetrics().RecordCudaLaneCompletion(it->second.is_decode);
  if (!CheckCudaStatus(cudaEventDestroy(it->second.completion_event),
                       "cudaEventDestroy(async_completion_event)")) {
    async_batches_.erase(it);
    return false;
  }
  async_batches_.erase(it);
  return true;
#else
  (void)handle;
  (void)outputs;
  return false;
#endif
}

bool NativeKernelExecutor::RunNativeInference(
    const std::vector<UnifiedBatchInput> &inputs,
    std::vector<UnifiedBatchOutput> *outputs) {
  auto results = ExecuteUnifiedBatch(inputs);
  const bool has_results = !results.empty();
  if (outputs) {
    *outputs = std::move(results);
  }
  return has_results;
}

#else // !INFERFLUX_HAS_CUDA

// CPU-only stubs for NativeKernelExecutor CUDA-dependent methods.
bool NativeKernelExecutor::LoadModel(const std::filesystem::path &,
                                     const LlamaBackendConfig &) {
  log::Error("native_kernel_executor", "CUDA not available");
  return false;
}

std::vector<LlamaCPUBackend::UnifiedBatchOutput>
NativeKernelExecutor::ExecuteUnifiedBatch(
    const std::vector<LlamaCPUBackend::UnifiedBatchInput> &) {
  return {};
}

bool NativeKernelExecutor::SupportsAsyncUnifiedBatch() const { return false; }

UnifiedBatchHandle NativeKernelExecutor::SubmitUnifiedBatchAsync(
    const std::vector<UnifiedBatchInput> &, UnifiedBatchLane) {
  return 0;
}

bool NativeKernelExecutor::TryCollectUnifiedBatchAsync(
    UnifiedBatchHandle, std::vector<UnifiedBatchOutput> *) {
  return false;
}

bool NativeKernelExecutor::RunNativeInference(
    const std::vector<UnifiedBatchInput> &, std::vector<UnifiedBatchOutput> *) {
  return false;
}

#endif // INFERFLUX_HAS_CUDA

// ==========================================================================
// NativeTakePerf (works on all build paths)
// ==========================================================================

NativeCudaRuntime::NativePerfSnapshot NativeKernelExecutor::NativeTakePerf() {
  NativePerfSnapshot snap;
#ifdef INFERFLUX_NATIVE_KERNELS_READY
  snap.prefill_ms = perf_accum_.prefill_ms.exchange(0.0);
  snap.decode_ms = perf_accum_.decode_ms.exchange(0.0);
  snap.prompt_tokens = perf_accum_.prompt_tokens.exchange(0);
  snap.generated_tokens = perf_accum_.generated_tokens.exchange(0);
#endif
  return snap;
}

// ==========================================================================
// Native* method overrides (no CUDA dependency)
// ==========================================================================

std::vector<int>
NativeKernelExecutor::NativeTokenize(const std::string &prompt) const {
  if (tokenizer_) {
    return tokenizer_->Encode(prompt);
  }
  return {};
}

int NativeKernelExecutor::NativeTokenCount(const std::string &text) const {
  if (tokenizer_) {
    return static_cast<int>(tokenizer_->Encode(text).size());
  }
  return 0;
}

bool NativeKernelExecutor::NativeIsReady() const { return model_loaded_; }

void NativeKernelExecutor::NativeFreeSequence(int sequence_id) {
#ifdef INFERFLUX_NATIVE_KERNELS_READY
  if (kv_cache_) {
    kv_cache_->ClearSequence(sequence_id);
  }
#endif
}

void NativeKernelExecutor::NativeCopySequencePrefix(int /*src_seq*/,
                                                    int /*dst_seq*/,
                                                    int /*n_tokens*/) {
  // No-op stub — full implementation needs KV cache copy support
}

} // namespace inferflux
