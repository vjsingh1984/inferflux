#include <catch2/catch_amalgamated.hpp>

#include "runtime/backends/backend_factory.h"
#include "runtime/backends/cuda/cuda_backend.h"
#include "runtime/backends/cuda/native_cuda_backend.h"
#include "runtime/backends/llama/llama_backend_traits.h"

using namespace inferflux;

namespace {

class MockLoadBackend : public LlamaCPUBackend {
public:
  bool called{false};

  bool LoadModel(const std::filesystem::path &,
                 const LlamaBackendConfig &) override {
    called = true;
    return true;
  }
};

} // namespace

TEST_CASE("LoadModel dispatches virtually through backend interface",
          "[backend_factory]") {
  auto backend = std::make_shared<MockLoadBackend>();
  std::shared_ptr<LlamaCPUBackend> base = backend;

  REQUIRE(base->LoadModel("/tmp/not-used", {}));
  REQUIRE(backend->called);
}

TEST_CASE("BackendFactory stores and returns exposure policy",
          "[backend_factory]") {
  BackendFactory::SetExposurePolicy({false, true});
  auto policy = BackendFactory::ExposurePolicy();
  REQUIRE_FALSE(policy.prefer_native);
  REQUIRE(policy.allow_llama_cpp_fallback);
  REQUIRE_FALSE(policy.strict_native_request);
  BackendFactory::SetExposurePolicy({true, true});
}

TEST_CASE("BackendFactory cpu hint resolves to CPU backend",
          "[backend_factory]") {
  BackendFactory::SetExposurePolicy({true, true});
  auto selection = BackendFactory::Create("cpu");

  REQUIRE(selection.backend != nullptr);
  REQUIRE(selection.backend_label == "cpu");
  REQUIRE(selection.provider == BackendProvider::kLlamaCpp);
  REQUIRE_FALSE(selection.used_fallback);
  REQUIRE(selection.config.gpu_layers == 0);
  REQUIRE(selection.capabilities.supports_logprobs);
  REQUIRE(selection.capabilities.supports_structured_output);
  REQUIRE(dynamic_cast<CudaBackend *>(selection.backend.get()) == nullptr);
}

TEST_CASE("BackendFactory cuda hint resolves predictably",
          "[backend_factory]") {
  BackendFactory::SetExposurePolicy({true, true});
  auto selection = BackendFactory::Create("cuda");

#ifdef INFERFLUX_HAS_CUDA
  REQUIRE(selection.backend != nullptr);
  REQUIRE(selection.backend_label == "cuda");
  if (NativeCudaBackend::NativeKernelsReady()) {
    REQUIRE(selection.provider == BackendProvider::kNative);
    REQUIRE_FALSE(selection.used_fallback);
    REQUIRE(selection.fallback_reason.empty());
    REQUIRE(dynamic_cast<NativeCudaBackend *>(selection.backend.get()) !=
            nullptr);
  } else {
    REQUIRE(selection.provider == BackendProvider::kLlamaCpp);
    REQUIRE(selection.used_fallback);
    REQUIRE(selection.fallback_reason.find("native backend unavailable") !=
            std::string::npos);
    REQUIRE(dynamic_cast<CudaBackend *>(selection.backend.get()) != nullptr);
  }
  REQUIRE(selection.config.gpu_layers > 0);
  REQUIRE(selection.capabilities.supports_streaming);
  REQUIRE(selection.capabilities.supports_logprobs);
#else
  REQUIRE(selection.backend_label == "cpu");
  REQUIRE(selection.config.gpu_layers == 0);
  REQUIRE(dynamic_cast<CudaBackend *>(selection.backend.get()) == nullptr);
#endif
}

TEST_CASE(
    "BackendFactory explicit cuda_llama_cpp hint forces llama.cpp provider",
    "[backend_factory]") {
  BackendFactory::SetExposurePolicy({true, false});
  auto selection = BackendFactory::Create("cuda_llama_cpp");

  REQUIRE(selection.provider == BackendProvider::kLlamaCpp);
#ifdef INFERFLUX_HAS_CUDA
  REQUIRE(selection.backend != nullptr);
  REQUIRE(selection.backend_label == "cuda");
  REQUIRE_FALSE(selection.used_fallback);
  REQUIRE(dynamic_cast<CudaBackend *>(selection.backend.get()) != nullptr);
#else
  REQUIRE(selection.backend != nullptr);
  REQUIRE(selection.backend_label == "cpu");
  REQUIRE(dynamic_cast<CudaBackend *>(selection.backend.get()) == nullptr);
#endif

  BackendFactory::SetExposurePolicy({true, true});
}

TEST_CASE("BackendFactory explicit cuda_native hint uses native or "
          "deterministic fallback",
          "[backend_factory]") {
  BackendFactory::SetExposurePolicy({false, true});
  auto selection = BackendFactory::Create("cuda_native");

#ifdef INFERFLUX_HAS_CUDA
  if (NativeCudaBackend::NativeKernelsReady()) {
    REQUIRE(selection.provider == BackendProvider::kNative);
    REQUIRE(selection.backend != nullptr);
    REQUIRE(selection.backend_label == "cuda");
    REQUIRE(dynamic_cast<NativeCudaBackend *>(selection.backend.get()) !=
            nullptr);
    REQUIRE_FALSE(selection.used_fallback);
    REQUIRE(selection.fallback_reason.empty());
  } else {
    REQUIRE(selection.provider == BackendProvider::kLlamaCpp);
    REQUIRE(selection.backend != nullptr);
    REQUIRE(selection.backend_label == "cuda");
    REQUIRE(dynamic_cast<CudaBackend *>(selection.backend.get()) != nullptr);
    REQUIRE(selection.used_fallback);
    REQUIRE(selection.fallback_reason.find("explicitly requested") !=
            std::string::npos);
  }
#else
  REQUIRE(selection.provider == BackendProvider::kLlamaCpp);
  REQUIRE(selection.backend != nullptr);
  REQUIRE(selection.backend_label == "cpu");
  REQUIRE(dynamic_cast<CudaBackend *>(selection.backend.get()) == nullptr);
  REQUIRE(selection.used_fallback);
  REQUIRE(selection.fallback_reason.find("explicitly requested") !=
          std::string::npos);
#endif
}

TEST_CASE(
    "BackendFactory strict native policy rejects explicit cuda_native when"
    " native kernels are not ready",
    "[backend_factory]") {
  if (NativeCudaBackend::NativeKernelsReady()) {
    SUCCEED("native kernels ready; strict rejection no longer applies");
    return;
  }

  BackendFactory::SetExposurePolicy({true, true, true});
  auto selection = BackendFactory::Create("cuda_native");

  REQUIRE(selection.provider == BackendProvider::kNative);
  REQUIRE(selection.backend == nullptr);
  REQUIRE(selection.fallback_reason.find("backend_policy_violation") !=
          std::string::npos);
  REQUIRE(selection.fallback_reason.find("strict_native_request") !=
          std::string::npos);

  BackendFactory::SetExposurePolicy({true, true, false});
}

TEST_CASE("BackendFactory auto hint follows compiled accelerators",
          "[backend_factory]") {
  BackendFactory::SetExposurePolicy({true, true});
  auto selection = BackendFactory::Create("auto");

  REQUIRE(selection.backend != nullptr);
#ifdef INFERFLUX_HAS_CUDA
  REQUIRE(selection.backend_label == "cuda");
#elif INFERFLUX_HAS_MLX
  REQUIRE(selection.backend_label == "mlx");
#else
  REQUIRE(selection.backend_label == "cpu");
#endif
}

TEST_CASE("MergeBackendConfig disables accelerator-only flags on CPU",
          "[backend_factory]") {
  BackendFactory::SetExposurePolicy({true, true});
  LlamaBackendConfig defaults;
  defaults.gpu_layers = 42;
  defaults.use_flash_attention = true;
  defaults.flash_attention_tile = 64;

  BackendFactoryResult selection;
  selection.backend = std::make_shared<LlamaCPUBackend>();
  selection.backend_label = "cpu";
  selection.config.gpu_layers = 0;

  auto merged = MergeBackendConfig(defaults, selection);
  REQUIRE(merged.gpu_layers == 0);
  REQUIRE_FALSE(merged.use_flash_attention);
  REQUIRE(merged.flash_attention_tile == 64);
}

TEST_CASE("MergeBackendConfig applies CUDA defaults when unset",
          "[backend_factory]") {
  BackendFactory::SetExposurePolicy({true, true});
  LlamaBackendConfig defaults;
  defaults.gpu_layers = 0;
  defaults.use_flash_attention = true;
  defaults.flash_attention_tile = 0;

  BackendFactoryResult selection;
  selection.target = LlamaBackendTarget::kCuda;
  selection.traits = DescribeLlamaBackendTarget(selection.target);
  selection.backend = std::make_shared<LlamaCPUBackend>();
  selection.backend_label = "cuda";
  selection.config.gpu_layers = 99;
  selection.config.flash_attention_tile = 128;

  auto merged = MergeBackendConfig(defaults, selection);
  REQUIRE(merged.gpu_layers == 99);
  REQUIRE(merged.use_flash_attention);
  REQUIRE(merged.flash_attention_tile == 128);
}

TEST_CASE("MergeBackendConfig keeps flash attention for MPS target",
          "[backend_factory]") {
  BackendFactory::SetExposurePolicy({true, true});
  LlamaBackendConfig defaults;
  defaults.gpu_layers = 0;
  defaults.use_flash_attention = true;
  defaults.flash_attention_tile = 0;

  BackendFactoryResult selection;
  selection.target = LlamaBackendTarget::kMps;
  selection.traits = DescribeLlamaBackendTarget(selection.target);
  selection.backend = std::make_shared<LlamaCPUBackend>();
  selection.backend_label = "mps";

  auto merged = MergeBackendConfig(defaults, selection);
  REQUIRE(merged.gpu_layers == 99);
  REQUIRE(merged.use_flash_attention);
  REQUIRE(merged.flash_attention_tile == 128);
}

TEST_CASE("ParseLlamaBackendTarget normalizes known hints",
          "[backend_factory]") {
  BackendFactory::SetExposurePolicy({true, true});
  REQUIRE(ParseLlamaBackendTarget("CUDA") == LlamaBackendTarget::kCuda);
  REQUIRE(ParseLlamaBackendTarget("mps") == LlamaBackendTarget::kMps);
  REQUIRE(ParseLlamaBackendTarget("rocm") == LlamaBackendTarget::kRocm);
  REQUIRE(ParseLlamaBackendTarget("vulkan") == LlamaBackendTarget::kVulkan);
  REQUIRE(ParseLlamaBackendTarget("unknown") == LlamaBackendTarget::kCpu);
}

TEST_CASE("TuneLlamaBackendConfig disables acceleration knobs for CPU",
          "[backend_factory]") {
  BackendFactory::SetExposurePolicy({true, true});
  LlamaBackendConfig cfg;
  cfg.gpu_layers = 32;
  cfg.use_flash_attention = true;
  cfg.flash_attention_tile = 256;
  cfg.cuda_phase_overlap_scaffold = true;
  cfg.cuda_phase_overlap_prefill_replica = true;
  cfg.cuda_attention_kernel = "fa3";

  auto tuned = TuneLlamaBackendConfig(LlamaBackendTarget::kCpu, cfg);
  REQUIRE(tuned.gpu_layers == 0);
  REQUIRE_FALSE(tuned.use_flash_attention);
  REQUIRE(tuned.flash_attention_tile == 256);
  REQUIRE_FALSE(tuned.cuda_phase_overlap_scaffold);
  REQUIRE_FALSE(tuned.cuda_phase_overlap_prefill_replica);
  REQUIRE(tuned.cuda_attention_kernel == "standard");
}

TEST_CASE("TuneLlamaBackendConfig preserves flash attention for MPS",
          "[backend_factory]") {
  BackendFactory::SetExposurePolicy({true, true});
  LlamaBackendConfig cfg;
  cfg.gpu_layers = 0;
  cfg.use_flash_attention = true;
  cfg.flash_attention_tile = 0;
  cfg.cuda_phase_overlap_scaffold = true;
  cfg.cuda_phase_overlap_prefill_replica = true;

  auto tuned = TuneLlamaBackendConfig(LlamaBackendTarget::kMps, cfg);
  REQUIRE(tuned.gpu_layers == 99);
  REQUIRE(tuned.use_flash_attention);
  REQUIRE(tuned.flash_attention_tile == 128);
  REQUIRE_FALSE(tuned.cuda_phase_overlap_scaffold);
  REQUIRE_FALSE(tuned.cuda_phase_overlap_prefill_replica);
}

TEST_CASE("TuneLlamaBackendConfig keeps CUDA overlap scaffold settings",
          "[backend_factory]") {
  BackendFactory::SetExposurePolicy({true, true});
  LlamaBackendConfig cfg;
  cfg.cuda_phase_overlap_scaffold = true;
  cfg.cuda_phase_overlap_prefill_replica = true;
  cfg.cuda_phase_overlap_min_prefill_tokens = 0;
  cfg.cuda_attention_kernel = "FA3";

  auto tuned = TuneLlamaBackendConfig(LlamaBackendTarget::kCuda, cfg);
  REQUIRE(tuned.cuda_phase_overlap_scaffold);
  REQUIRE(tuned.cuda_phase_overlap_prefill_replica);
  REQUIRE(tuned.cuda_phase_overlap_min_prefill_tokens == 256);
  REQUIRE(tuned.cuda_attention_kernel == "fa3");
}

TEST_CASE("TuneLlamaBackendConfig normalizes unknown CUDA attention kernel",
          "[backend_factory]") {
  BackendFactory::SetExposurePolicy({true, true});
  LlamaBackendConfig cfg;
  cfg.cuda_attention_kernel = "unknown-kernel";

  auto tuned = TuneLlamaBackendConfig(LlamaBackendTarget::kCuda, cfg);
  REQUIRE(tuned.cuda_attention_kernel == "auto");
}

TEST_CASE("BackendFactory can disable llama.cpp fallback for native policy",
          "[backend_factory]") {
  BackendFactory::SetExposurePolicy({true, false});
  auto selection = BackendFactory::Create("cuda");

  if (NativeCudaBackend::NativeKernelsReady()) {
    REQUIRE(selection.provider == BackendProvider::kNative);
    REQUIRE(selection.backend != nullptr);
    REQUIRE(selection.fallback_reason.empty());
  } else {
    REQUIRE(selection.provider == BackendProvider::kNative);
    REQUIRE(selection.backend == nullptr);
    REQUIRE(selection.fallback_reason.find("fallback disabled") !=
            std::string::npos);
  }

  BackendFactory::SetExposurePolicy({true, true});
}

TEST_CASE("BackendFactory NormalizeHintList deduplicates and falls back",
          "[backend_factory]") {
  auto hints =
      BackendFactory::NormalizeHintList({"CUDA", "cuda", "mps"}, "cpu");
  REQUIRE(hints.size() == 2);
  REQUIRE(hints[0] == "cuda");
  REQUIRE(hints[1] == "mps");

  auto fallback = BackendFactory::NormalizeHintList({}, "ROCM");
  REQUIRE(fallback.size() == 1);
  REQUIRE(fallback[0] == "rocm");
}

TEST_CASE("BackendFactory NormalizeHint keeps explicit provider hints",
          "[backend_factory]") {
  REQUIRE(BackendFactory::NormalizeHint("CUDA_NATIVE") == "cuda_native");
  REQUIRE(BackendFactory::NormalizeHint("CUDA_LLAMA_CPP") == "cuda_llama_cpp");
  REQUIRE(BackendFactory::NormalizeHint("cuda_llama") == "cuda_llama");
}
