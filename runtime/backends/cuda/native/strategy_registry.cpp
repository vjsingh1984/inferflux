#include "runtime/backends/cuda/native/strategy_registry.h"

#include <algorithm>
#include <cctype>
#include <utility>

namespace inferflux {
namespace runtime {
namespace cuda {
namespace native {
namespace {

std::string ToLower(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return value;
}

class StaticWeightLayoutStrategy final : public IWeightLayoutStrategy {
public:
  StaticWeightLayoutStrategy(std::string id,
                             std::vector<GGUF::TensorType> supported_types,
                             std::size_t block_elements,
                             std::size_t block_bytes)
      : id_(std::move(id)), supported_types_(std::move(supported_types)),
        block_elements_(block_elements), block_bytes_(block_bytes) {}

  std::string Id() const override { return id_; }

  bool Supports(GGUF::TensorType tensor_type) const override {
    return std::find(supported_types_.begin(), supported_types_.end(),
                     tensor_type) != supported_types_.end();
  }

  std::size_t BlockElements() const override { return block_elements_; }
  std::size_t BlockBytes() const override { return block_bytes_; }

private:
  std::string id_;
  std::vector<GGUF::TensorType> supported_types_;
  std::size_t block_elements_{0};
  std::size_t block_bytes_{0};
};

class StaticMatmulStrategy final : public IMatmulStrategy {
public:
  StaticMatmulStrategy(std::string id, MatmulExecutionMode mode,
                       std::vector<GGUF::TensorType> supported_types,
                       int min_sm_major)
      : id_(std::move(id)), mode_(mode),
        supported_types_(std::move(supported_types)),
        min_sm_major_(min_sm_major) {}

  std::string Id() const override { return id_; }
  MatmulExecutionMode Mode() const override { return mode_; }

  bool Supports(GGUF::TensorType tensor_type, int sm_major,
                int sm_minor) const override {
    (void)sm_minor;
    if (sm_major < min_sm_major_) {
      return false;
    }
    return std::find(supported_types_.begin(), supported_types_.end(),
                     tensor_type) != supported_types_.end();
  }

private:
  std::string id_;
  MatmulExecutionMode mode_{MatmulExecutionMode::kCompatDequantizeThenGemm};
  std::vector<GGUF::TensorType> supported_types_;
  int min_sm_major_{0};
};

class StaticAttentionStrategy final : public IAttentionStrategy {
public:
  StaticAttentionStrategy(std::string id, KvPrecision mode, int min_sm_major)
      : id_(std::move(id)), mode_(mode), min_sm_major_(min_sm_major) {}

  std::string Id() const override { return id_; }
  KvPrecision KvMode() const override { return mode_; }

  bool Supports(KvPrecision requested_mode, int sm_major,
                int sm_minor) const override {
    (void)sm_minor;
    return requested_mode == mode_ && sm_major >= min_sm_major_;
  }

private:
  std::string id_;
  KvPrecision mode_{KvPrecision::kFp16};
  int min_sm_major_{0};
};

} // namespace

std::string KvPrecisionToString(KvPrecision precision) {
  switch (precision) {
  case KvPrecision::kFp16:
    return "fp16";
  case KvPrecision::kBf16:
    return "bf16";
  case KvPrecision::kInt8:
    return "int8";
  case KvPrecision::kFp8:
    return "fp8";
  }
  return "fp16";
}

bool ParseKvPrecision(const std::string &raw, KvPrecision *out) {
  if (!out) {
    return false;
  }
  const std::string lowered = ToLower(raw);
  if (lowered == "fp16") {
    *out = KvPrecision::kFp16;
    return true;
  }
  if (lowered == "bf16") {
    *out = KvPrecision::kBf16;
    return true;
  }
  if (lowered == "int8") {
    *out = KvPrecision::kInt8;
    return true;
  }
  if (lowered == "fp8") {
    *out = KvPrecision::kFp8;
    return true;
  }
  return false;
}

QuantizedRuntimeStrategyRegistry &QuantizedRuntimeStrategyRegistry::Instance() {
  static QuantizedRuntimeStrategyRegistry registry;
  return registry;
}

void QuantizedRuntimeStrategyRegistry::RegisterWeightLayout(
    std::unique_ptr<IWeightLayoutStrategy> strategy) {
  if (strategy) {
    weight_layouts_.push_back(std::move(strategy));
  }
}

void QuantizedRuntimeStrategyRegistry::RegisterMatmul(
    std::unique_ptr<IMatmulStrategy> strategy) {
  if (strategy) {
    matmul_strategies_.push_back(std::move(strategy));
  }
}

void QuantizedRuntimeStrategyRegistry::RegisterAttention(
    std::unique_ptr<IAttentionStrategy> strategy) {
  if (strategy) {
    attention_strategies_.push_back(std::move(strategy));
  }
}

const IWeightLayoutStrategy *
QuantizedRuntimeStrategyRegistry::SelectWeightLayout(
    GGUF::TensorType tensor_type) const {
  for (const auto &candidate : weight_layouts_) {
    if (candidate && candidate->Supports(tensor_type)) {
      return candidate.get();
    }
  }
  return nullptr;
}

const IMatmulStrategy *QuantizedRuntimeStrategyRegistry::SelectMatmul(
    GGUF::TensorType tensor_type, int sm_major, int sm_minor) const {
  for (const auto &candidate : matmul_strategies_) {
    if (candidate && candidate->Supports(tensor_type, sm_major, sm_minor)) {
      return candidate.get();
    }
  }
  return nullptr;
}

const IAttentionStrategy *QuantizedRuntimeStrategyRegistry::SelectAttention(
    KvPrecision requested_mode, int sm_major, int sm_minor) const {
  for (const auto &candidate : attention_strategies_) {
    if (candidate && candidate->Supports(requested_mode, sm_major, sm_minor)) {
      return candidate.get();
    }
  }
  return nullptr;
}

StrategySelection
QuantizedRuntimeStrategyRegistry::Select(GGUF::TensorType tensor_type,
                                         KvPrecision requested_kv_mode,
                                         int sm_major, int sm_minor) const {
  StrategySelection result;
  result.weight_layout = SelectWeightLayout(tensor_type);
  result.matmul = SelectMatmul(tensor_type, sm_major, sm_minor);
  result.attention = SelectAttention(requested_kv_mode, sm_major, sm_minor);

  result.reason = "tensor=" + TensorTypeToString(tensor_type) +
                  ", sm=" + std::to_string(sm_major) + "." +
                  std::to_string(sm_minor) +
                  ", kv=" + KvPrecisionToString(requested_kv_mode);
  return result;
}

void QuantizedRuntimeStrategyRegistry::RegisterDefaults() {
  if (defaults_registered_) {
    return;
  }

  // Quantized K-block layouts (Q*_K family).
  RegisterWeightLayout(std::make_unique<StaticWeightLayoutStrategy>(
      "layout.gguf.kblock.256",
      std::vector<GGUF::TensorType>{
          GGUF::TensorType::Q2_K,
          GGUF::TensorType::Q3_K,
          GGUF::TensorType::Q4_K,
          GGUF::TensorType::Q5_K,
          GGUF::TensorType::Q6_K,
          GGUF::TensorType::Q8_K,
      },
      256U, 0U));

  // Quantized 32-element block layouts.
  RegisterWeightLayout(std::make_unique<StaticWeightLayoutStrategy>(
      "layout.gguf.block.32",
      std::vector<GGUF::TensorType>{
          GGUF::TensorType::Q4_0,
          GGUF::TensorType::Q4_1,
          GGUF::TensorType::Q5_0,
          GGUF::TensorType::Q5_1,
          GGUF::TensorType::Q8_0,
          GGUF::TensorType::Q8_1,
      },
      32U, 0U));

  // Dense floating-point tensors.
  RegisterWeightLayout(std::make_unique<StaticWeightLayoutStrategy>(
      "layout.gguf.dense",
      std::vector<GGUF::TensorType>{
          GGUF::TensorType::F16,
          GGUF::TensorType::F32,
      },
      1U, 0U));

  // Preferred fused strategy (selected when SM >= 80 and quantized type).
  RegisterMatmul(std::make_unique<StaticMatmulStrategy>(
      "matmul.fused.dequant_tile_gemm.v1",
      MatmulExecutionMode::kFusedDequantTileGemm,
      std::vector<GGUF::TensorType>{
          GGUF::TensorType::Q8_0, GGUF::TensorType::Q4_K,
          GGUF::TensorType::Q6_K, GGUF::TensorType::Q8_K},
      8));

  // Compatibility strategy (existing dequantize-then-GEMM path).
  RegisterMatmul(std::make_unique<StaticMatmulStrategy>(
      "matmul.compat.dequantize_then_gemm",
      MatmulExecutionMode::kCompatDequantizeThenGemm,
      std::vector<GGUF::TensorType>{
          GGUF::TensorType::F16, GGUF::TensorType::F32, GGUF::TensorType::Q4_0,
          GGUF::TensorType::Q4_1, GGUF::TensorType::Q5_0,
          GGUF::TensorType::Q5_1, GGUF::TensorType::Q8_0,
          GGUF::TensorType::Q8_1, GGUF::TensorType::Q2_K,
          GGUF::TensorType::Q3_K, GGUF::TensorType::Q4_K,
          GGUF::TensorType::Q5_K, GGUF::TensorType::Q6_K,
          GGUF::TensorType::Q8_K},
      0));

  // Attention strategies keyed by KV precision. FP16 is universally supported.
  RegisterAttention(std::make_unique<StaticAttentionStrategy>(
      "attention.paged_kv.fp16", KvPrecision::kFp16, 0));

  // BF16 KV requires SM80+ for stable throughput/quality profile.
  RegisterAttention(std::make_unique<StaticAttentionStrategy>(
      "attention.paged_kv.bf16", KvPrecision::kBf16, 8));

  defaults_registered_ = true;
}

} // namespace native
} // namespace cuda
} // namespace runtime
} // namespace inferflux
