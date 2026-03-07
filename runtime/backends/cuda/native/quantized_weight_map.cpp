#include "runtime/backends/cuda/native/quantized_weight_map.h"
#include "runtime/backends/cuda/native/gguf_util.h"
#include "server/logging/logger.h"

namespace inferflux {

QuantizedWeightMap::~QuantizedWeightMap() {
  // Note: GPU memory is managed by IModelLoader, we don't own it
}

bool QuantizedWeightMap::Build(IModelLoader *loader, const ModelInfo &config,
                               cudaStream_t stream) {
  if (!loader) {
    log::Error("quantized_weight_map", "Null loader");
    return false;
  }

  loader_ = loader;
  stream_ = stream;
  num_layers_ = config.num_hidden_layers;
  is_quantized_ = loader->IsQuantized();
  quantization_type_ = loader->GetQuantizationType();

  log::Info("quantized_weight_map",
            "Building weight map: layers=" + std::to_string(num_layers_) +
                " quantized=" + std::string(is_quantized_ ? "true" : "false") +
                " type=" + quantization_type_);

  // Allocate layer structures
  layers_.resize(num_layers_);

  // Build tensor names for each layer
  for (int layer = 0; layer < num_layers_; ++layer) {
    auto &lw = layers_[layer];

    // Get weight accessors for all layer tensors
    lw.q_proj_accessor = loader->GetWeightAccessor(
        GetLayerTensorName(layer, "self_attn", "q_proj"));
    lw.k_proj_accessor = loader->GetWeightAccessor(
        GetLayerTensorName(layer, "self_attn", "k_proj"));
    lw.v_proj_accessor = loader->GetWeightAccessor(
        GetLayerTensorName(layer, "self_attn", "v_proj"));
    lw.o_proj_accessor = loader->GetWeightAccessor(
        GetLayerTensorName(layer, "self_attn", "o_proj"));

    lw.input_norm_accessor =
        loader->GetWeightAccessor(GetLayerTensorName(layer, "input_layernorm"));
    lw.post_attn_norm_accessor = loader->GetWeightAccessor(
        GetLayerTensorName(layer, "post_attention_layernorm"));

    lw.gate_proj_accessor = loader->GetWeightAccessor(
        GetLayerTensorName(layer, "mlp", "gate_proj"));
    lw.up_proj_accessor =
        loader->GetWeightAccessor(GetLayerTensorName(layer, "mlp", "up_proj"));
    lw.down_proj_accessor = loader->GetWeightAccessor(
        GetLayerTensorName(layer, "mlp", "down_proj"));

    // Biases (optional)
    lw.q_proj_bias_accessor = loader->GetWeightAccessor(
        GetLayerTensorName(layer, "self_attn", "q_proj", "bias"));
    lw.k_proj_bias_accessor = loader->GetWeightAccessor(
        GetLayerTensorName(layer, "self_attn", "k_proj", "bias"));
    lw.v_proj_bias_accessor = loader->GetWeightAccessor(
        GetLayerTensorName(layer, "self_attn", "v_proj", "bias"));
  }

  // Global tensors
  embed_tokens_accessor =
      loader->GetWeightAccessor("model.embed_tokens.weight");
  final_norm_accessor = loader->GetWeightAccessor("model.norm.weight");
  lm_head_accessor = loader->GetWeightAccessor("lm_head.weight");
  if (!lm_head_accessor && embed_tokens_accessor) {
    log::Info("quantized_weight_map",
              "lm_head.weight not found, using tied embeddings");
    lm_head_accessor = embed_tokens_accessor;
  }

  log::Info("quantized_weight_map", "Weight map built successfully");
  return true;
}

std::string
QuantizedWeightMap::GetLayerTensorName(int layer, const std::string &component,
                                       const std::string &type,
                                       const std::string &suffix) const {

  // HuggingFace style naming:
  // model.layers.{layer}.{component}[.{type}].{suffix} Examples:
  // - model.layers.0.self_attn.q_proj.weight (with type)
  // - model.layers.0.input_layernorm.weight (without type)
  // - model.layers.0.self_attn.q_proj.bias (with bias suffix)

  if (type.empty()) {
    // Simple component: model.layers.{layer}.{component}.{suffix}
    return "model.layers." + std::to_string(layer) + "." + component + "." +
           suffix;
  }
  // Complex component: model.layers.{layer}.{component}.{type}.{suffix}
  return "model.layers." + std::to_string(layer) + "." + component + "." +
         type + "." + suffix;
}

const half *QuantizedWeightMap::GetDequantizedWeights(
    std::shared_ptr<IWeightAccessor> accessor, const half *&cache_ptr) const {

  if (!accessor) {
    return nullptr;
  }

  // Return cached if available
  if (cache_ptr) {
    return cache_ptr;
  }

  // Lazy dequantization
  cache_ptr = accessor->GetDequantizedGpuWeights(stream_);

  if (!cache_ptr) {
    log::Warn("quantized_weight_map", "Failed to get dequantized weights");
  }

  return cache_ptr;
}

// --- Per-layer accessors ---

const half *QuantizedWeightMap::LayerQProj(int layer) const {
  if (layer < 0 || layer >= num_layers_) {
    return nullptr;
  }
  return GetDequantizedWeights(layers_[layer].q_proj_accessor,
                               layers_[layer].q_proj);
}

const half *QuantizedWeightMap::LayerKProj(int layer) const {
  if (layer < 0 || layer >= num_layers_) {
    return nullptr;
  }
  return GetDequantizedWeights(layers_[layer].k_proj_accessor,
                               layers_[layer].k_proj);
}

const half *QuantizedWeightMap::LayerVProj(int layer) const {
  if (layer < 0 || layer >= num_layers_) {
    return nullptr;
  }
  return GetDequantizedWeights(layers_[layer].v_proj_accessor,
                               layers_[layer].v_proj);
}

const half *QuantizedWeightMap::LayerOProj(int layer) const {
  if (layer < 0 || layer >= num_layers_) {
    return nullptr;
  }
  return GetDequantizedWeights(layers_[layer].o_proj_accessor,
                               layers_[layer].o_proj);
}

const half *QuantizedWeightMap::LayerInputNorm(int layer) const {
  if (layer < 0 || layer >= num_layers_) {
    return nullptr;
  }
  return GetDequantizedWeights(layers_[layer].input_norm_accessor,
                               layers_[layer].input_norm);
}

const half *QuantizedWeightMap::LayerPostAttnNorm(int layer) const {
  if (layer < 0 || layer >= num_layers_) {
    return nullptr;
  }
  return GetDequantizedWeights(layers_[layer].post_attn_norm_accessor,
                               layers_[layer].post_attn_norm);
}

const half *QuantizedWeightMap::LayerGateProj(int layer) const {
  if (layer < 0 || layer >= num_layers_) {
    return nullptr;
  }
  return GetDequantizedWeights(layers_[layer].gate_proj_accessor,
                               layers_[layer].gate_proj);
}

const half *QuantizedWeightMap::LayerUpProj(int layer) const {
  if (layer < 0 || layer >= num_layers_) {
    return nullptr;
  }
  return GetDequantizedWeights(layers_[layer].up_proj_accessor,
                               layers_[layer].up_proj);
}

const half *QuantizedWeightMap::LayerDownProj(int layer) const {
  if (layer < 0 || layer >= num_layers_) {
    return nullptr;
  }
  return GetDequantizedWeights(layers_[layer].down_proj_accessor,
                               layers_[layer].down_proj);
}

// --- Bias accessors ---

const half *QuantizedWeightMap::LayerQProjBias(int layer) const {
  if (layer < 0 || layer >= num_layers_) {
    return nullptr;
  }
  return GetDequantizedWeights(layers_[layer].q_proj_bias_accessor,
                               layers_[layer].q_proj_bias);
}

const half *QuantizedWeightMap::LayerKProjBias(int layer) const {
  if (layer < 0 || layer >= num_layers_) {
    return nullptr;
  }
  return GetDequantizedWeights(layers_[layer].k_proj_bias_accessor,
                               layers_[layer].k_proj_bias);
}

const half *QuantizedWeightMap::LayerVProjBias(int layer) const {
  if (layer < 0 || layer >= num_layers_) {
    return nullptr;
  }
  return GetDequantizedWeights(layers_[layer].v_proj_bias_accessor,
                               layers_[layer].v_proj_bias);
}

// --- Global accessors ---

const half *QuantizedWeightMap::EmbedTokens() const {
  return GetDequantizedWeights(embed_tokens_accessor, embed_tokens_);
}

const half *QuantizedWeightMap::FinalNorm() const {
  return GetDequantizedWeights(final_norm_accessor, final_norm_);
}

const half *QuantizedWeightMap::LmHead() const {
  return GetDequantizedWeights(lm_head_accessor, lm_head_);
}

// --- Raw quantized weight accessors ---

namespace {
QuantizedWeightInfo MakeRawInfo(std::shared_ptr<IWeightAccessor> accessor,
                                cudaStream_t stream) {
  if (!accessor || !accessor->IsQuantized()) {
    return {};
  }
  auto dims = accessor->GetDimensions();
  auto type_str = accessor->GetDataType();
  auto tensor_type = runtime::cuda::native::StringToTensorType(type_str);
  QuantizedWeightInfo info;
  info.data = accessor->GetGpuWeights(stream);
  info.quant_type = static_cast<int>(tensor_type);
  info.num_elements = static_cast<int64_t>(dims.first * dims.second);
  return info;
}
} // namespace

QuantizedWeightInfo QuantizedWeightMap::GetRawLayerQProj(int layer) const {
  if (layer < 0 || layer >= num_layers_)
    return {};
  return MakeRawInfo(layers_[layer].q_proj_accessor, stream_);
}

QuantizedWeightInfo QuantizedWeightMap::GetRawLayerKProj(int layer) const {
  if (layer < 0 || layer >= num_layers_)
    return {};
  return MakeRawInfo(layers_[layer].k_proj_accessor, stream_);
}

QuantizedWeightInfo QuantizedWeightMap::GetRawLayerVProj(int layer) const {
  if (layer < 0 || layer >= num_layers_)
    return {};
  return MakeRawInfo(layers_[layer].v_proj_accessor, stream_);
}

QuantizedWeightInfo QuantizedWeightMap::GetRawLayerOProj(int layer) const {
  if (layer < 0 || layer >= num_layers_)
    return {};
  return MakeRawInfo(layers_[layer].o_proj_accessor, stream_);
}

QuantizedWeightInfo QuantizedWeightMap::GetRawLayerGateProj(int layer) const {
  if (layer < 0 || layer >= num_layers_)
    return {};
  return MakeRawInfo(layers_[layer].gate_proj_accessor, stream_);
}

QuantizedWeightInfo QuantizedWeightMap::GetRawLayerUpProj(int layer) const {
  if (layer < 0 || layer >= num_layers_)
    return {};
  return MakeRawInfo(layers_[layer].up_proj_accessor, stream_);
}

QuantizedWeightInfo QuantizedWeightMap::GetRawLayerDownProj(int layer) const {
  if (layer < 0 || layer >= num_layers_)
    return {};
  return MakeRawInfo(layers_[layer].down_proj_accessor, stream_);
}

QuantizedWeightInfo QuantizedWeightMap::GetRawLmHead() const {
  return MakeRawInfo(lm_head_accessor, stream_);
}

// --- Utility ---

bool QuantizedWeightMap::HasTensor(const std::string &name) const {
  return loader_->GetWeightAccessor(name) != nullptr;
}

std::shared_ptr<IWeightAccessor>
QuantizedWeightMap::GetWeightAccessor(const std::string &name) {
  return loader_->GetWeightAccessor(name);
}

void QuantizedWeightMap::ClearCache() {
  log::Info("quantized_weight_map", "Clearing dequantized weight cache");

  // Clear layer caches
  for (auto &lw : layers_) {
    lw.q_proj = nullptr;
    lw.k_proj = nullptr;
    lw.v_proj = nullptr;
    lw.o_proj = nullptr;
    lw.input_norm = nullptr;
    lw.post_attn_norm = nullptr;
    lw.gate_proj = nullptr;
    lw.up_proj = nullptr;
    lw.down_proj = nullptr;
    lw.q_proj_bias = nullptr;
    lw.k_proj_bias = nullptr;
    lw.v_proj_bias = nullptr;
  }

  // Clear global caches
  embed_tokens_ = nullptr;
  final_norm_ = nullptr;
  lm_head_ = nullptr;
}

} // namespace inferflux
