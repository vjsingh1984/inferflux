#include "runtime/backends/cuda/native_kernel_executor.h"
#include "runtime/backends/cuda/native/native_tokenizer.h"
#include "runtime/backends/cuda/native/nvtx_scoped.h"
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

#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <set>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <nlohmann/json.hpp>

namespace {

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

void ConvertBF16ToFP16(const void *src, void *dst, size_t num_elements) {
  const uint16_t *in = static_cast<const uint16_t *>(src);
  uint16_t *out = static_cast<uint16_t *>(dst);
  for (size_t i = 0; i < num_elements; ++i) {
    out[i] = bf16_to_fp16(in[i]);
  }
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
                    " shape=" + shape_str + " size=" +
                    std::to_string(info->size / (1024 * 1024)) + " MB");
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
                    std::to_string(st.st_size / (1024 * 1024 * 1024)) + " GB");
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
            "Total GPU memory needed: " +
                std::to_string(total_size / (1024 * 1024 * 1024)) + " GB");

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
      ConvertBF16ToFP16(tensor.cpu_data, convert_buf.data(), num_elements);
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
                    std::to_string(tensor.size / (1024 * 1024)) +
                    " MB) -> GPU offset " + std::to_string(tensor.gpu_offset) +
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
    log::Info("safetensors_loader",
              "Freeing GPU buffer (" +
                  std::to_string(total_gpu_size_ / (1024 * 1024)) + " MB)");
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
    cudaFree(d_logits_);
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

bool CheckBF16Support() {
  int device = 0;
  cudaGetDevice(&device);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
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

  log::Info("native_kernel_executor",
            "Inference dtype: " + std::string(want_bf16 ? "bf16" : "fp16") +
                " (torch_dtype=" + config.torch_dtype + ")");

  // 1. Build WeightMap from loader (always FP16 typed — gpu_data is raw)
  weight_map_ = std::make_unique<WeightMap>();
  if (!weight_map_->Build(*loader_, config)) {
    log::Error("native_kernel_executor", "Failed to build weight map");
    return false;
  }

  // 2. Initialize cuBLAS wrapper
  gemm_ = std::make_unique<CublasGemm>();
  if (!gemm_->Initialize(compute_stream_)) {
    log::Error("native_kernel_executor", "Failed to initialize cuBLAS");
    return false;
  }

  // 3. Allocate KV cache (typed based on inference dtype)
  int max_batch = 32;
  int max_seq = 4096;
  if (config.max_position_embeddings > 0 &&
      config.max_position_embeddings < max_seq) {
    max_seq = config.max_position_embeddings;
  }

  if (want_bf16) {
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

  // 4. Create ModelForward via typed factory
  if (want_bf16) {
    model_forward_ = CreateModelForwardTyped<__nv_bfloat16>(config.model_type);
  } else {
    model_forward_ = CreateModelForward(config.model_type);
  }
  if (!model_forward_) {
    log::Error("native_kernel_executor",
               "Unsupported model_type: " + config.model_type);
    return false;
  }

  // 5. Initialize transformer forward pass
  if (!model_forward_->Initialize(config, *weight_map_, kv_cache_.get(),
                                  gemm_.get(), compute_stream_)) {
    log::Error("native_kernel_executor", "Failed to initialize forward pass");
    return false;
  }

  // 6. Initialize GPU sampler
  sampler_ = std::make_unique<GpuSampler>();
  if (!sampler_->Initialize(config.vocab_size, compute_stream_)) {
    log::Error("native_kernel_executor", "Failed to initialize GPU sampler");
    return false;
  }

  // 7. Allocate device logits buffer (sized for batched decode)
  int max_batch = 32;
  cudaError_t err =
      cudaMalloc(&d_logits_, static_cast<size_t>(max_batch) *
                                 config.vocab_size * sizeof(float));
  if (err != cudaSuccess) {
    log::Error("native_kernel_executor", "Failed to allocate logits buffer");
    return false;
  }

  // 8. Load NativeTokenizer from model directory
  tokenizer_ = std::make_unique<NativeTokenizer>();
  if (!tokenizer_->Load(loader_->GetModelPath())) {
    log::Warn("native_kernel_executor",
              "Failed to load tokenizer from " + loader_->GetModelPath());
  }

  // 9. Create cudaEvent pairs for forward/sampling timing
  cudaEventCreate(&forward_start_);
  cudaEventCreate(&forward_stop_);
  cudaEventCreate(&sampling_start_);
  cudaEventCreate(&sampling_stop_);

  // 10. Create lane-specific streams for async overlap
  cudaStreamCreateWithFlags(&decode_stream_, cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&prefill_stream_, cudaStreamNonBlocking);

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

  // Initialize CUDA
  if (!InitializeCUDA()) {
    return false;
  }

  // Create loader
  loader_ = std::make_unique<SafetensorsLoader>();
  if (!loader_->LoadModel(model_path.string())) {
    log::Error("native_kernel_executor", "Failed to load safetensors model");
    return false;
  }

  // Get model config
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
      log::Info("native_kernel_executor",
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

  // Initialize native inference pipeline
  if (!InitializeNativePipeline()) {
    log::Warn("native_kernel_executor",
              "Native pipeline init failed; inference will return empty");
  }

  // Report native backend uses FA2 attention kernel
  GlobalMetrics().SetCudaAttentionKernel("fa2");

  log::Info("native_kernel_executor", "Native CUDA model loaded successfully");
  model_loaded_ = true;
  return true;
}

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

  NVTX_SCOPE("NativeExecuteUnifiedBatch");
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
    cudaEventRecord(forward_start_, compute_stream_);
    bool fwd_ok = model_forward_->BatchForward(batch_tokens, batch_n_past,
                                               batch_seq_ids, d_logits_, B);
    cudaEventRecord(forward_stop_, compute_stream_);
    cudaEventSynchronize(forward_stop_);
    float fwd_ms = 0.0f;
    cudaEventElapsedTime(&fwd_ms, forward_start_, forward_stop_);
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
      cudaEventRecord(sampling_start_, compute_stream_);
      std::vector<int> sampled_tokens;
      sampler_->SampleBatch(d_logits_, B, batch_temps, batch_top_ks,
                            batch_top_ps, &sampled_tokens);
      cudaEventRecord(sampling_stop_, compute_stream_);
      cudaEventSynchronize(sampling_stop_);
      float samp_ms = 0.0f;
      cudaEventElapsedTime(&samp_ms, sampling_start_, sampling_stop_);
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
      cudaEventRecord(forward_start_, compute_stream_);
      if (!model_forward_->Forward(input.tokens, input.n_past,
                                   input.sequence_id, d_logits_)) {
        log::Error("native_kernel_executor", "Forward pass failed");
      }
      cudaEventRecord(forward_stop_, compute_stream_);
      cudaEventSynchronize(forward_stop_);
      float fwd_ms = 0.0f;
      cudaEventElapsedTime(&fwd_ms, forward_start_, forward_stop_);
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
      cudaEventRecord(forward_start_, compute_stream_);
      if (!model_forward_->Forward(input.tokens, input.n_past,
                                   input.sequence_id, d_logits_)) {
        log::Error("native_kernel_executor", "Forward pass failed");
        continue;
      }
      cudaEventRecord(forward_stop_, compute_stream_);
      cudaEventSynchronize(forward_stop_);
      float fwd_ms = 0.0f;
      cudaEventElapsedTime(&fwd_ms, forward_start_, forward_stop_);
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
      cudaEventRecord(sampling_start_, compute_stream_);
      int token_id = sampler_->Sample(
          d_logits_, input.sampling.temperature, input.sampling.top_k,
          input.sampling.top_p, input.sampling.seed);
      cudaEventRecord(sampling_stop_, compute_stream_);
      cudaEventSynchronize(sampling_stop_);
      float samp_ms = 0.0f;
      cudaEventElapsedTime(&samp_ms, sampling_start_, sampling_stop_);
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
  cudaEventCreate(&event);
  cudaEventRecord(event, target_stream);

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

  // Completed
  if (outputs) {
    *outputs = std::move(it->second.outputs);
  }
  GlobalMetrics().RecordCudaLaneCompletion(it->second.is_decode);
  cudaEventDestroy(it->second.completion_event);
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
  if (outputs) {
    *outputs = std::move(results);
  }
  return !results.empty();
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

NativeCudaExecutor::NativePerfSnapshot NativeKernelExecutor::NativeTakePerf() {
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
