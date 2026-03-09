#include <catch2/catch_amalgamated.hpp>

#include "runtime/backends/cuda/native_kernel_executor.h"

#ifdef INFERFLUX_NATIVE_KERNELS_READY
#include "runtime/backends/cuda/native/cublas_gemm.h"
#include "runtime/backends/cuda/native/fused_quant_gemm.h"
#include "runtime/backends/cuda/native/gguf_util.h"
#include "runtime/backends/cuda/native/model_forward_factory.h"
#include "runtime/backends/cuda/native/quantized_weight_map_adapter.h"
#include "runtime/backends/cuda/native/safetensors_adapter.h"
#include "runtime/backends/cuda/native/weight_map.h"
#endif

#include <array>
#include <cstdlib>

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
  REQUIRE(CreateModelForwardTyped<__nv_bfloat16>("llama") != nullptr);
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

TEST_CASE("NativeKernelExecutor: async unified-batch contract is gated by "
          "runtime readiness",
          "[native_forward]") {
  NativeKernelExecutor executor;
  REQUIRE_FALSE(executor.SupportsAsyncUnifiedBatch());

  LlamaCPUBackend::UnifiedBatchInput input;
  input.sequence_id = 0;
  input.n_past = 0;
  input.tokens = {1};
  input.request_logits = true;

  const auto handle = executor.SubmitUnifiedBatchAsync(
      {input}, LlamaCPUBackend::UnifiedBatchLane::kAuto);
  REQUIRE(handle == 0);

  std::vector<LlamaCPUBackend::UnifiedBatchOutput> outputs;
  REQUIRE_FALSE(executor.TryCollectUnifiedBatchAsync(1, &outputs));
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

#ifdef INFERFLUX_NATIVE_KERNELS_READY
TEST_CASE("SafetensorsWeightAccessor: FP16 path is direct non-quantized "
          "access",
          "[native_forward]") {
  SafetensorsLoader::Tensor tensor;
  tensor.name = "w";
  tensor.shape = {4, 8};
  tensor.dtype = "f16";
  tensor.gpu_offset = 16;

  std::array<uint8_t, 128> gpu_buffer{};
  runtime::cuda::native::SafetensorsWeightAccessor accessor(
      &tensor, gpu_buffer.data());

  REQUIRE_FALSE(accessor.IsQuantized());
  REQUIRE(accessor.GetDataType() == "f16");
  REQUIRE(accessor.GetDimensions() == std::make_pair<size_t, size_t>(4, 8));

  void *raw = accessor.GetGpuWeights(nullptr);
  REQUIRE(raw == (gpu_buffer.data() + tensor.gpu_offset));
  half *dequant = accessor.GetDequantizedGpuWeights(nullptr);
  REQUIRE(reinterpret_cast<void *>(dequant) == raw);
}
#endif

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

TEST_CASE("FusedQuantGemm: adaptive threshold remains bounded for all target "
          "quantized GGUF types",
          "[native_forward]") {
  const std::array<int, 4> target_quant_types = {
      12, // Q4_K
      14, // Q6_K
      8,  // Q8_0
      15, // Q8_K
  };
  for (const int quant_type : target_quant_types) {
    const int threshold = FusedQuantGemm::GetAdaptiveThreshold(quant_type);
    REQUIRE(threshold >= 4);
    REQUIRE(threshold <= 32);
  }
}

TEST_CASE("FusedQuantGemm: deterministic fallback policy by batch size for "
          "target quant types",
          "[native_forward]") {
  const std::array<int, 4> target_quant_types = {
      12, // Q4_K
      14, // Q6_K
      8,  // Q8_0
      15, // Q8_K
  };

  for (const int quant_type : target_quant_types) {
    const int threshold = FusedQuantGemm::GetAdaptiveThreshold(quant_type);
    const int short_prefill_m = std::max(2, threshold);
    const int batched_decode_m = threshold + 1;

    // Decode and short-prefill stay on fused path.
    REQUIRE(FusedQuantGemm::ShouldUseFusedPath(quant_type, 1));
    REQUIRE(FusedQuantGemm::ShouldUseFusedPath(quant_type, short_prefill_m));

    // Larger M deterministically falls back to compatibility/cuBLAS path.
    REQUIRE_FALSE(
        FusedQuantGemm::ShouldUseFusedPath(quant_type, batched_decode_m));
  }
}

TEST_CASE("FusedQuantGemm: deterministic fallback policy rejects invalid "
          "inputs",
          "[native_forward]") {
  REQUIRE_FALSE(FusedQuantGemm::ShouldUseFusedPath(0, 1));  // F32
  REQUIRE_FALSE(FusedQuantGemm::ShouldUseFusedPath(1, 1));  // F16
  REQUIRE_FALSE(FusedQuantGemm::ShouldUseFusedPath(12, 0)); // non-positive M
}

TEST_CASE("FusedQuantGemm::Gemv respects deterministic fallback threshold for "
          "all target quant types",
          "[native_forward]") {
  const std::array<int, 4> target_quant_types = {
      12, // Q4_K
      14, // Q6_K
      8,  // Q8_0
      15, // Q8_K
  };

  for (const int quant_type : target_quant_types) {
    QuantizedWeightInfo info;
    info.data = reinterpret_cast<const void *>(0x1); // Non-null sentinel
    info.quant_type = quant_type;
    info.num_elements = 128 * 64;

    const int threshold = FusedQuantGemm::GetAdaptiveThreshold(quant_type);
    const int m_fallback = threshold + 1;

    // M > threshold must return false before any kernel dispatch.
    const bool used = FusedQuantGemm::Gemv(info, nullptr, nullptr, m_fallback,
                                           128, 64, nullptr);
    REQUIRE_FALSE(used);
  }
}

TEST_CASE(
    "FusedQuantGemm::RmsNormGemv respects deterministic fallback threshold for "
    "all target quant types",
    "[native_forward]") {
  const std::array<int, 4> target_quant_types = {
      12, // Q4_K
      14, // Q6_K
      8,  // Q8_0
      15, // Q8_K
  };

  for (const int quant_type : target_quant_types) {
    QuantizedWeightInfo info;
    info.data = reinterpret_cast<const void *>(0x1); // Non-null sentinel
    info.quant_type = quant_type;
    info.num_elements = 128 * 64;

    const int threshold = FusedQuantGemm::GetAdaptiveThreshold(quant_type);
    const int m_fallback = threshold + 1;

    // M > threshold must return false before any kernel dispatch.
    const bool used = FusedQuantGemm::RmsNormGemv(
        info, nullptr, nullptr, nullptr, m_fallback, 128, 64, 1e-6f, nullptr);
    REQUIRE_FALSE(used);
  }
}

TEST_CASE("FusedQuantGemm: CUDA runtime contract launches fused decode path "
          "and falls back above threshold",
          "[native_forward][cuda_runtime_contract]") {
  const char *disable_fused_env = std::getenv("INFERFLUX_DISABLE_FUSED_GEMV");
  if (disable_fused_env &&
      (std::string(disable_fused_env) == "1" ||
       std::string(disable_fused_env) == "true")) {
    SUCCEED("Fused kernels disabled via INFERFLUX_DISABLE_FUSED_GEMV.");
    return;
  }

  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    SUCCEED("No CUDA device available; skipping CUDA runtime contract.");
    return;
  }

  const auto run_contract = [](runtime::cuda::native::GGUF::TensorType qtype,
                               int K, int N) {
    const int quant_type = static_cast<int>(qtype);
    const int threshold = FusedQuantGemm::GetAdaptiveThreshold(quant_type);
    const int m_fused = 1;
    const int m_fallback = threshold + 1;

    const size_t weight_bytes =
        runtime::cuda::native::CalcTensorSize(qtype, {static_cast<uint64_t>(N),
                                                      static_cast<uint64_t>(K)});
    REQUIRE(weight_bytes > 0);

    void *d_weight = nullptr;
    half *d_activation = nullptr;
    half *d_output = nullptr;

    REQUIRE(cudaMalloc(&d_weight, weight_bytes) == cudaSuccess);
    REQUIRE(cudaMemset(d_weight, 0, weight_bytes) == cudaSuccess);
    REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_activation),
                       static_cast<size_t>(m_fallback) * K * sizeof(half)) ==
            cudaSuccess);
    REQUIRE(cudaMemset(d_activation, 0,
                       static_cast<size_t>(m_fallback) * K * sizeof(half)) ==
            cudaSuccess);
    REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d_output),
                       static_cast<size_t>(m_fallback) * N * sizeof(half)) ==
            cudaSuccess);
    REQUIRE(cudaMemset(d_output, 0,
                       static_cast<size_t>(m_fallback) * N * sizeof(half)) ==
            cudaSuccess);

    QuantizedWeightInfo info;
    info.data = d_weight;
    info.quant_type = quant_type;
    info.num_elements = static_cast<int64_t>(N) * static_cast<int64_t>(K);

    (void)cudaGetLastError(); // clear sticky errors before launch checks
    const bool used_fused = FusedQuantGemm::Gemv(info, d_activation, d_output,
                                                 m_fused, N, K, nullptr);
    REQUIRE(used_fused);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    REQUIRE(cudaGetLastError() == cudaSuccess);

    const bool used_fallback = FusedQuantGemm::Gemv(
        info, d_activation, d_output, m_fallback, N, K, nullptr);
    REQUIRE_FALSE(used_fallback);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    REQUIRE(cudaFree(d_output) == cudaSuccess);
    REQUIRE(cudaFree(d_activation) == cudaSuccess);
    REQUIRE(cudaFree(d_weight) == cudaSuccess);
  };

  // Keep dimensions minimal but valid for each quant block layout.
  run_contract(runtime::cuda::native::GGUF::TensorType::Q4_K, 256, 32);
  run_contract(runtime::cuda::native::GGUF::TensorType::Q6_K, 256, 32);
  run_contract(runtime::cuda::native::GGUF::TensorType::Q8_0, 32, 32);
  run_contract(runtime::cuda::native::GGUF::TensorType::Q8_K, 256, 32);
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
