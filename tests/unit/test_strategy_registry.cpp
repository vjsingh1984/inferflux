#include <catch2/catch_amalgamated.hpp>

#include "runtime/backends/cuda/native/model_loader.h"
#include "runtime/backends/cuda/native/strategy_registry.h"

#ifdef INFERFLUX_HAS_CUDA

namespace {

using inferflux::runtime::cuda::native::DequantizedCachePolicy;
using inferflux::runtime::cuda::native::KvPrecision;
using inferflux::runtime::cuda::native::ParseDequantizedCachePolicy;
using inferflux::runtime::cuda::native::ParseKvPrecision;
using inferflux::runtime::cuda::native::QuantizedRuntimeStrategyRegistry;
namespace gguf = inferflux::runtime::cuda::native::GGUF;

} // namespace

TEST_CASE("Kv precision parser accepts supported values", "[quantization]") {
  KvPrecision value = KvPrecision::kFp16;
  REQUIRE(ParseKvPrecision("fp16", &value));
  CHECK(value == KvPrecision::kFp16);

  REQUIRE(ParseKvPrecision("BF16", &value));
  CHECK(value == KvPrecision::kBf16);

  REQUIRE(ParseKvPrecision("int8", &value));
  CHECK(value == KvPrecision::kInt8);

  REQUIRE(ParseKvPrecision("fp8", &value));
  CHECK(value == KvPrecision::kFp8);
}

TEST_CASE("Kv precision parser rejects invalid values", "[quantization]") {
  KvPrecision value = KvPrecision::kFp16;
  REQUIRE_FALSE(ParseKvPrecision("auto", &value));
  REQUIRE_FALSE(ParseKvPrecision("bad_value", &value));
}

TEST_CASE("Dequantized cache policy parser accepts supported values",
          "[quantization]") {
  DequantizedCachePolicy value = DequantizedCachePolicy::kModelLifetime;
  REQUIRE(ParseDequantizedCachePolicy("batch", &value));
  CHECK(value == DequantizedCachePolicy::kBatchLifetime);

  REQUIRE(ParseDequantizedCachePolicy("model", &value));
  CHECK(value == DequantizedCachePolicy::kModelLifetime);
}

TEST_CASE("Dequantized cache policy parser rejects invalid values",
          "[quantization]") {
  DequantizedCachePolicy value = DequantizedCachePolicy::kBatchLifetime;
  REQUIRE_FALSE(ParseDequantizedCachePolicy("auto", &value));
  REQUIRE_FALSE(ParseDequantizedCachePolicy("unknown", &value));
}

TEST_CASE("Strategy registry prefers fused quantized matmul on SM80+",
          "[quantization]") {
  auto &registry = QuantizedRuntimeStrategyRegistry::Instance();
  registry.RegisterDefaults();

  const auto selection =
      registry.Select(gguf::TensorType::Q4_K, KvPrecision::kFp16, 8, 9);

  REQUIRE(selection.weight_layout != nullptr);
  REQUIRE(selection.matmul != nullptr);
  REQUIRE(selection.attention != nullptr);
  CHECK(selection.matmul->Id() == "matmul.fused.dequant_tile_gemm.v1");
}

TEST_CASE("Strategy registry falls back to compatibility matmul on pre-SM80",
          "[quantization]") {
  auto &registry = QuantizedRuntimeStrategyRegistry::Instance();
  registry.RegisterDefaults();

  const auto selection =
      registry.Select(gguf::TensorType::Q4_K, KvPrecision::kFp16, 7, 5);

  REQUIRE(selection.matmul != nullptr);
  CHECK(selection.matmul->Id() == "matmul.compat.dequantize_then_gemm");
}

TEST_CASE("BF16 attention strategy requires SM80+", "[quantization]") {
  auto &registry = QuantizedRuntimeStrategyRegistry::Instance();
  registry.RegisterDefaults();

  const auto supported =
      registry.Select(gguf::TensorType::Q6_K, KvPrecision::kBf16, 8, 0);
  REQUIRE(supported.attention != nullptr);
  CHECK(supported.attention->Id() == "attention.paged_kv.bf16");

  const auto unsupported =
      registry.Select(gguf::TensorType::Q6_K, KvPrecision::kBf16, 7, 5);
  CHECK(unsupported.attention == nullptr);
}

#else

TEST_CASE("Strategy registry tests require CUDA build", "[quantization]") {
  SUCCEED();
}

#endif
