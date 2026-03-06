#pragma once

#include "runtime/backends/cuda/native/gguf_util.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace inferflux {
namespace runtime {
namespace cuda {
namespace native {

enum class KvPrecision {
  kFp16,
  kBf16,
  kInt8,
  kFp8,
};

std::string KvPrecisionToString(KvPrecision precision);

// Parses fp16|bf16|int8|fp8 (case-insensitive). Returns false on invalid input.
bool ParseKvPrecision(const std::string &raw, KvPrecision *out);

enum class MatmulExecutionMode {
  kFusedDequantTileGemm,
  kCompatDequantizeThenGemm,
};

class IWeightLayoutStrategy {
public:
  virtual ~IWeightLayoutStrategy() = default;

  virtual std::string Id() const = 0;
  virtual bool Supports(GGUF::TensorType tensor_type) const = 0;
  virtual std::size_t BlockElements() const = 0;
  virtual std::size_t BlockBytes() const = 0;
};

class IMatmulStrategy {
public:
  virtual ~IMatmulStrategy() = default;

  virtual std::string Id() const = 0;
  virtual MatmulExecutionMode Mode() const = 0;
  virtual bool Supports(GGUF::TensorType tensor_type, int sm_major,
                        int sm_minor) const = 0;
};

class IAttentionStrategy {
public:
  virtual ~IAttentionStrategy() = default;

  virtual std::string Id() const = 0;
  virtual KvPrecision KvMode() const = 0;
  virtual bool Supports(KvPrecision requested_mode, int sm_major,
                        int sm_minor) const = 0;
};

struct StrategySelection {
  const IWeightLayoutStrategy *weight_layout{nullptr};
  const IMatmulStrategy *matmul{nullptr};
  const IAttentionStrategy *attention{nullptr};
  std::string reason;
};

class QuantizedRuntimeStrategyRegistry {
public:
  static QuantizedRuntimeStrategyRegistry &Instance();

  void RegisterWeightLayout(std::unique_ptr<IWeightLayoutStrategy> strategy);
  void RegisterMatmul(std::unique_ptr<IMatmulStrategy> strategy);
  void RegisterAttention(std::unique_ptr<IAttentionStrategy> strategy);

  const IWeightLayoutStrategy *
  SelectWeightLayout(GGUF::TensorType tensor_type) const;
  const IMatmulStrategy *SelectMatmul(GGUF::TensorType tensor_type,
                                      int sm_major, int sm_minor) const;
  const IAttentionStrategy *SelectAttention(KvPrecision requested_mode,
                                            int sm_major,
                                            int sm_minor) const;

  StrategySelection Select(GGUF::TensorType tensor_type,
                           KvPrecision requested_kv_mode,
                           int sm_major,
                           int sm_minor) const;

  void RegisterDefaults();

private:
  QuantizedRuntimeStrategyRegistry() = default;

  bool defaults_registered_{false};
  std::vector<std::unique_ptr<IWeightLayoutStrategy>> weight_layouts_;
  std::vector<std::unique_ptr<IMatmulStrategy>> matmul_strategies_;
  std::vector<std::unique_ptr<IAttentionStrategy>> attention_strategies_;
};

} // namespace native
} // namespace cuda
} // namespace runtime
} // namespace inferflux
