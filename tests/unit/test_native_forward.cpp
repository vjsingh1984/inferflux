#include <catch2/catch_amalgamated.hpp>

#include "runtime/backends/cuda/native_kernel_executor.h"

#ifdef INFERFLUX_NATIVE_KERNELS_READY
#include "runtime/backends/cuda/native/cublas_gemm.h"
#include "runtime/backends/cuda/native/fused_quant_gemm.h"
#include "runtime/backends/cuda/native/model_forward_factory.h"
#include "runtime/backends/cuda/native/quantized_weight_map_adapter.h"
#include "runtime/backends/cuda/native/weight_map.h"
#endif

namespace inferflux {

// ============================================================================
// WeightMap Tests (compile-time only, no GPU required)
// ============================================================================

TEST_CASE("WeightMap: Build returns false on empty loader",
          "[native_forward]") {
  // WeightMap::Build requires a loaded SafetensorsLoader with GPU data.
  // Without actual safetensors files, we verify the interface compiles
  // and handles the empty case.
  SafetensorsLoader loader;
  SafetensorsLoader::ModelConfig config;
  config.num_hidden_layers = 2;
  config.hidden_size = 128;

#ifdef INFERFLUX_NATIVE_KERNELS_READY
  WeightMap wm;
  // Should fail because no tensors are loaded
  REQUIRE_FALSE(wm.Build(loader, config));
#else
  // Without native kernels, just verify types compile
  REQUIRE(config.hidden_size == 128);
#endif
}

// ============================================================================
// ModelForwardFactory Tests
// ============================================================================

TEST_CASE("ModelForwardFactory: supported model types", "[native_forward]") {
#ifdef INFERFLUX_NATIVE_KERNELS_READY
  // Llama-family models should return a valid ModelForward
  REQUIRE(CreateModelForward("llama") != nullptr);
  REQUIRE(CreateModelForward("qwen2") != nullptr);
  REQUIRE(CreateModelForward("qwen3") != nullptr);
  REQUIRE(CreateModelForward("mistral") != nullptr);
  REQUIRE(CreateModelForward("gemma") != nullptr);
  REQUIRE(CreateModelForward("phi3") != nullptr);

  // Unsupported models should return nullptr
  REQUIRE(CreateModelForward("unknown_model") == nullptr);
  REQUIRE(CreateModelForward("") == nullptr);
#else
  REQUIRE(true); // Placeholder when native kernels not compiled
#endif
}

// ============================================================================
// NativeKernelExecutor Interface Tests
// ============================================================================

TEST_CASE("NativeKernelExecutor: Name and fallback state", "[native_forward]") {
  NativeKernelExecutor executor;
  REQUIRE(executor.Name() == "native_cuda");
  REQUIRE_FALSE(executor.IsFallback());
  REQUIRE(executor.FallbackReason().empty());
  REQUIRE(executor.BackendHandle() == nullptr);
}

TEST_CASE("NativeKernelExecutor: ExecuteUnifiedBatch returns empty when no "
          "model loaded",
          "[native_forward]") {
  NativeKernelExecutor executor;
  std::vector<LlamaCPUBackend::UnifiedBatchInput> inputs;
  auto outputs = executor.ExecuteUnifiedBatch(inputs);
  REQUIRE(outputs.empty());
}

// ============================================================================
// SafetensorsLoader Config Parsing Tests
// ============================================================================

TEST_CASE("SafetensorsLoader: GetConfig returns defaults for no config",
          "[native_forward]") {
  SafetensorsLoader loader;
  const auto &config = loader.GetConfig();
  REQUIRE(config.hidden_size == 0);
  REQUIRE(config.num_hidden_layers == 0);
  REQUIRE(config.vocab_size == 0);
  REQUIRE(config.model_type.empty());
}

TEST_CASE("SafetensorsLoader: GetTensor returns nullptr for unknown name",
          "[native_forward]") {
  SafetensorsLoader loader;
  REQUIRE(loader.GetTensor("nonexistent") == nullptr);
  REQUIRE(loader.GetTensorNames().empty());
}

// ============================================================================
// NativeKernelExecutor Native* Method Tests
// ============================================================================

TEST_CASE("NativeKernelExecutor: NativeIsReady returns false when no model",
          "[native_forward]") {
  NativeKernelExecutor executor;
  REQUIRE_FALSE(executor.NativeIsReady());
}

TEST_CASE("NativeKernelExecutor: NativeTokenize returns empty when no model",
          "[native_forward]") {
  NativeKernelExecutor executor;
  auto tokens = executor.NativeTokenize("hello world");
  REQUIRE(tokens.empty());
}

TEST_CASE("NativeKernelExecutor: NativeTokenCount returns 0 when no model",
          "[native_forward]") {
  NativeKernelExecutor executor;
  REQUIRE(executor.NativeTokenCount("hello world") == 0);
}

TEST_CASE("NativeKernelExecutor: NativeFreeSequence does not crash when no "
          "model",
          "[native_forward]") {
  NativeKernelExecutor executor;
  executor.NativeFreeSequence(0); // Should not crash
  REQUIRE(true);
}

TEST_CASE(
    "NativeKernelExecutor: NativeCopySequencePrefix is safe with no model",
    "[native_forward]") {
  NativeKernelExecutor executor;
  executor.NativeCopySequencePrefix(0, 1, 10); // Should not crash
  REQUIRE(true);
}

TEST_CASE("NativeKernelExecutor: NativeSerializeSequence returns empty with no "
          "model",
          "[native_forward]") {
  NativeKernelExecutor executor;
  auto blob = executor.NativeSerializeSequence(0);
  REQUIRE(blob.empty());
}

TEST_CASE("NativeKernelExecutor: NativeHydrateSequence returns false with no "
          "model",
          "[native_forward]") {
  NativeKernelExecutor executor;
  REQUIRE_FALSE(executor.NativeHydrateSequence(0, {}));
}

// ============================================================================
// SafetensorsLoader ModelConfig Extended Fields Tests
// ============================================================================

TEST_CASE("SafetensorsLoader: ModelConfig defaults for torch_dtype and "
          "rms_norm_eps",
          "[native_forward]") {
  SafetensorsLoader::ModelConfig config;
  REQUIRE(config.torch_dtype.empty());
  REQUIRE(config.rms_norm_eps == Catch::Approx(1e-6f));
}

// ============================================================================
// QuantizedWeightMapAdapter Tests
// ============================================================================

#ifdef INFERFLUX_NATIVE_KERNELS_READY

// ============================================================================
// QuantizedWeightInfo & Raw Accessor Tests
// ============================================================================

TEST_CASE("QuantizedWeightInfo: default-constructed has null data",
          "[native_forward]") {
  QuantizedWeightInfo info;
  REQUIRE(info.data == nullptr);
  REQUIRE(info.quant_type == -1);
  REQUIRE(info.num_elements == 0);
}

TEST_CASE("WeightMapTyped: HasQuantizedWeights returns false by default",
          "[native_forward]") {
  WeightMap wm;
  REQUIRE_FALSE(wm.HasQuantizedWeights());
  REQUIRE(wm.LayerQProjRaw(0).data == nullptr);
  REQUIRE(wm.LmHeadRaw().data == nullptr);
}

TEST_CASE(
    "QuantizedWeightMapAdapter: HasQuantizedWeights delegates to underlying",
    "[native_forward]") {
  QuantizedWeightMap qwm;
  QuantizedWeightMapAdapter adapter(&qwm);

  // Without Build(), IsQuantized() returns false
  REQUIRE_FALSE(adapter.HasQuantizedWeights());

  // Raw accessors should return empty info
  REQUIRE(adapter.LayerQProjRaw(0).data == nullptr);
  REQUIRE(adapter.LayerKProjRaw(0).data == nullptr);
  REQUIRE(adapter.LayerVProjRaw(0).data == nullptr);
  REQUIRE(adapter.LayerOProjRaw(0).data == nullptr);
  REQUIRE(adapter.LayerGateProjRaw(0).data == nullptr);
  REQUIRE(adapter.LayerUpProjRaw(0).data == nullptr);
  REQUIRE(adapter.LayerDownProjRaw(0).data == nullptr);
  REQUIRE(adapter.LmHeadRaw().data == nullptr);
}

// ============================================================================
// FusedQuantGemm Dispatch Tests
// ============================================================================

TEST_CASE("FusedQuantGemm: returns false for null data", "[native_forward]") {
  QuantizedWeightInfo info;
  // Should return false (no data), not crash
  bool used = FusedQuantGemm::Gemv(info, nullptr, nullptr, 1, 128, 64, nullptr);
  REQUIRE_FALSE(used);
}

TEST_CASE("FusedQuantGemm: returns false for unsupported quant type",
          "[native_forward]") {
  QuantizedWeightInfo info;
  info.data = reinterpret_cast<const void *>(0x1); // Non-null dummy
  info.quant_type = 0; // F32 — not a quantized type we support
  info.num_elements = 128 * 64;

  bool used = FusedQuantGemm::Gemv(info, nullptr, nullptr, 1, 128, 64, nullptr);
  REQUIRE_FALSE(used);
}

TEST_CASE("FusedQuantGemm: returns false for large M", "[native_forward]") {
  QuantizedWeightInfo info;
  info.data = reinterpret_cast<const void *>(0x1);
  info.quant_type = 12; // Q4_K
  info.num_elements = 128 * 64;

  // M=64 always exceeds the adaptive threshold (capped at kFusedGemmMaxM=32)
  bool used =
      FusedQuantGemm::Gemv(info, nullptr, nullptr, 64, 128, 64, nullptr);
  REQUIRE_FALSE(used);
}

TEST_CASE("FusedQuantGemm: adaptive threshold varies by quant type",
          "[native_forward]") {
  // Lower bits per weight → higher threshold (fused saves more bandwidth)
  int q4k_threshold = FusedQuantGemm::GetAdaptiveThreshold(12); // Q4_K
  int q8_0_threshold = FusedQuantGemm::GetAdaptiveThreshold(8); // Q8_0
  REQUIRE(q4k_threshold >= q8_0_threshold);
  REQUIRE(q4k_threshold >= 4);
  REQUIRE(q4k_threshold <= 32);
}

// ============================================================================
// QuantizedWeightMapAdapter Tests
// ============================================================================

TEST_CASE("QuantizedWeightMapAdapter: delegates NumLayers to underlying map",
          "[native_forward]") {
  // QuantizedWeightMap without Build() returns 0 layers
  QuantizedWeightMap qwm;
  QuantizedWeightMapAdapter adapter(&qwm);

  // Adapter should delegate to the underlying map
  REQUIRE(adapter.NumLayers() == 0);
  REQUIRE(adapter.EmbedTokens() == nullptr);
  REQUIRE(adapter.FinalNorm() == nullptr);
  REQUIRE(adapter.LmHead() == nullptr);
}

TEST_CASE("QuantizedWeightMapAdapter: is a WeightMapTyped<half>",
          "[native_forward]") {
  QuantizedWeightMap qwm;
  QuantizedWeightMapAdapter adapter(&qwm);

  // Verify polymorphism: adapter can be used as WeightMapTyped<half>*
  const WeightMap *base = &adapter;
  REQUIRE(base->NumLayers() == 0);
  REQUIRE(base->EmbedTokens() == nullptr);
}

#endif // INFERFLUX_NATIVE_KERNELS_READY

} // namespace inferflux
