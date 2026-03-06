#include <catch2/catch_amalgamated.hpp>

#include "runtime/backends/cuda/native_kernel_executor.h"

#ifdef INFERFLUX_NATIVE_KERNELS_READY
#include "runtime/backends/cuda/native/cublas_gemm.h"
#include "runtime/backends/cuda/native/model_forward_factory.h"
#include "runtime/backends/cuda/native/native_tokenizer.h"
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
// NativeTokenizer Tests
// ============================================================================

TEST_CASE("NativeTokenizer: Load returns false for missing file",
          "[native_forward]") {
#ifdef INFERFLUX_NATIVE_KERNELS_READY
  NativeTokenizer tok;
  REQUIRE_FALSE(tok.Load("/nonexistent/path"));
  REQUIRE(tok.VocabSize() == 0);
  REQUIRE(tok.EosTokenId() == -1);
  REQUIRE(tok.BosTokenId() == -1);
  REQUIRE(tok.IdToString(0).empty());
#else
  REQUIRE(true);
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

TEST_CASE("NativeKernelExecutor: NativeCopySequencePrefix is a no-op stub",
          "[native_forward]") {
  NativeKernelExecutor executor;
  executor.NativeCopySequencePrefix(0, 1, 10); // Should not crash
  REQUIRE(true);
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

} // namespace inferflux
