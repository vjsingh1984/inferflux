#include "runtime/backends/cuda/native/weight_map.h"
#include "server/logging/logger.h"

namespace inferflux {

template <typename T>
const T *WeightMapTyped<T>::Resolve(const SafetensorsLoader &loader,
                                    const std::string &name) const {
  const auto *tensor = loader.GetTensor(name);
  if (!tensor || !tensor->gpu_data) {
    return nullptr;
  }
  return static_cast<const T *>(tensor->gpu_data);
}

template <typename T>
bool WeightMapTyped<T>::Build(const SafetensorsLoader &loader,
                              const SafetensorsLoader::ModelConfig &config) {
  num_layers_ = config.num_hidden_layers;

  embed_tokens_ = Resolve(loader, "model.embed_tokens.weight");
  if (!embed_tokens_) {
    log::Error("weight_map", "Missing model.embed_tokens.weight");
    return false;
  }

  final_norm_ = Resolve(loader, "model.norm.weight");
  if (!final_norm_) {
    log::Error("weight_map", "Missing model.norm.weight");
    return false;
  }

  lm_head_ = Resolve(loader, "lm_head.weight");
  if (!lm_head_) {
    log::Info("weight_map", "lm_head.weight not found, using tied embeddings");
    lm_head_ = embed_tokens_;
  }

  layers_.resize(num_layers_);
  for (int i = 0; i < num_layers_; ++i) {
    auto prefix = "model.layers." + std::to_string(i);
    auto &lw = layers_[i];

    lw.q_proj = Resolve(loader, prefix + ".self_attn.q_proj.weight");
    lw.k_proj = Resolve(loader, prefix + ".self_attn.k_proj.weight");
    lw.v_proj = Resolve(loader, prefix + ".self_attn.v_proj.weight");
    lw.o_proj = Resolve(loader, prefix + ".self_attn.o_proj.weight");
    lw.input_norm = Resolve(loader, prefix + ".input_layernorm.weight");
    lw.post_attn_norm =
        Resolve(loader, prefix + ".post_attention_layernorm.weight");
    lw.gate_proj = Resolve(loader, prefix + ".mlp.gate_proj.weight");
    lw.up_proj = Resolve(loader, prefix + ".mlp.up_proj.weight");
    lw.down_proj = Resolve(loader, prefix + ".mlp.down_proj.weight");

    // Optional biases (Qwen2 has q/k/v biases, Llama does not)
    lw.q_proj_bias = Resolve(loader, prefix + ".self_attn.q_proj.bias");
    lw.k_proj_bias = Resolve(loader, prefix + ".self_attn.k_proj.bias");
    lw.v_proj_bias = Resolve(loader, prefix + ".self_attn.v_proj.bias");

    if (!lw.q_proj || !lw.k_proj || !lw.v_proj || !lw.o_proj ||
        !lw.input_norm || !lw.post_attn_norm || !lw.gate_proj || !lw.up_proj ||
        !lw.down_proj) {
      log::Error("weight_map",
                 "Missing weights for layer " + std::to_string(i));
      return false;
    }
  }

  bool has_biases = num_layers_ > 0 && layers_[0].q_proj_bias != nullptr;
  log::Info("weight_map",
            "Built weight map: " + std::to_string(num_layers_) +
                " layers, embed=" + std::to_string(config.hidden_size) + "d" +
                (has_biases ? ", attn_biases=true" : ""));
  return true;
}

template <typename T> const T *WeightMapTyped<T>::LayerQProj(int layer) const {
  return layers_[layer].q_proj;
}
template <typename T> const T *WeightMapTyped<T>::LayerKProj(int layer) const {
  return layers_[layer].k_proj;
}
template <typename T> const T *WeightMapTyped<T>::LayerVProj(int layer) const {
  return layers_[layer].v_proj;
}
template <typename T> const T *WeightMapTyped<T>::LayerOProj(int layer) const {
  return layers_[layer].o_proj;
}
template <typename T>
const T *WeightMapTyped<T>::LayerInputNorm(int layer) const {
  return layers_[layer].input_norm;
}
template <typename T>
const T *WeightMapTyped<T>::LayerPostAttnNorm(int layer) const {
  return layers_[layer].post_attn_norm;
}
template <typename T>
const T *WeightMapTyped<T>::LayerGateProj(int layer) const {
  return layers_[layer].gate_proj;
}
template <typename T> const T *WeightMapTyped<T>::LayerUpProj(int layer) const {
  return layers_[layer].up_proj;
}
template <typename T>
const T *WeightMapTyped<T>::LayerDownProj(int layer) const {
  return layers_[layer].down_proj;
}
template <typename T>
const T *WeightMapTyped<T>::LayerQProjBias(int layer) const {
  return layers_[layer].q_proj_bias;
}
template <typename T>
const T *WeightMapTyped<T>::LayerKProjBias(int layer) const {
  return layers_[layer].k_proj_bias;
}
template <typename T>
const T *WeightMapTyped<T>::LayerVProjBias(int layer) const {
  return layers_[layer].v_proj_bias;
}

// Explicit template instantiations
template class WeightMapTyped<half>;
template class WeightMapTyped<__nv_bfloat16>;

} // namespace inferflux
