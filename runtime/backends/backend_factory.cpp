#include "runtime/backends/backend_factory.h"

#include "runtime/backends/cuda/cuda_backend.h"
#include "server/logging/logger.h"

#if INFERFLUX_HAS_MLX
#include "runtime/backends/mlx/mlx_backend.h"
#endif

#include <algorithm>
#include <cctype>
#include <mutex>
#include <unordered_set>
#include <vector>

namespace inferflux {

namespace {

BackendExposurePolicy g_policy{};
std::mutex g_policy_mutex;

BackendExposurePolicy LoadPolicy() {
  std::lock_guard<std::mutex> lock(g_policy_mutex);
  return g_policy;
}

bool SupportsNativeBackend(LlamaBackendTarget target) {
  // Native CUDA/ROCm/MPS runtime backends are intentionally not wired yet.
  // We keep this gate explicit so the factory can expose policy-driven
  // fallback behavior (native-preferred -> universal llama fallback).
  (void)target;
  return false;
}

BackendFactoryResult CpuFallback(const std::string &reason) {
  if (!reason.empty()) {
    log::Warn("backend_factory", reason);
  }
  BackendFactoryResult out;
  out.target = LlamaBackendTarget::kCpu;
  out.traits = DescribeLlamaBackendTarget(out.target);
  out.capabilities = out.traits.capabilities;
  out.backend = std::make_shared<LlamaCPUBackend>();
  out.backend_label = out.traits.label;
  out.provider = BackendProvider::kUniversalLlama;
  out.config = TuneLlamaBackendConfig(out.target, {});
  return out;
}

BackendFactoryResult UniversalLlamaForTarget(LlamaBackendTarget target,
                                             const std::string &hint) {
  BackendFactoryResult out;
  out.target = target;
  out.traits = DescribeLlamaBackendTarget(target);
  out.capabilities = out.traits.capabilities;
  out.backend_label = out.traits.label;
  out.provider = BackendProvider::kUniversalLlama;
  out.config = TuneLlamaBackendConfig(target, {});

  if (hint == "cuda") {
#ifdef INFERFLUX_HAS_CUDA
    out.backend = std::make_shared<CudaBackend>();
#else
    return CpuFallback(
        "CUDA backend requested but binary was built without CUDA support. "
        "Falling back to CPU backend.");
#endif
    return out;
  }

  if (hint == "mlx") {
#if INFERFLUX_HAS_MLX
    out.target = LlamaBackendTarget::kMps;
    out.traits = DescribeLlamaBackendTarget(out.target);
    out.capabilities = out.traits.capabilities;
    out.backend = std::make_shared<MlxBackend>();
    out.backend_label = "mlx";
    out.config = TuneLlamaBackendConfig(out.target, {});
    return out;
#else
    return CpuFallback(
        "MLX backend requested but binary was built without ENABLE_MLX. "
        "Falling back to CPU backend.");
#endif
  }

  if (hint == "mps" || hint == "rocm" || hint == "vulkan") {
    out.backend = std::make_shared<LlamaCPUBackend>();
    return out;
  }

  if (hint == "cpu") {
    return CpuFallback("");
  }

  return CpuFallback("Unknown backend hint '" + hint +
                     "'. Falling back to CPU backend.");
}

} // namespace

std::string BackendFactory::NormalizeHint(const std::string &backend_hint) {
  if (backend_hint.empty()) {
    return "cpu";
  }
  std::string lowered = backend_hint;
  std::transform(
      lowered.begin(), lowered.end(), lowered.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (lowered == "auto" || lowered == "mlx") {
    return lowered;
  }
  return DescribeLlamaBackendTarget(ParseLlamaBackendTarget(lowered)).label;
}

std::vector<std::string>
BackendFactory::NormalizeHintList(const std::vector<std::string> &backend_hints,
                                  const std::string &default_hint) {
  std::vector<std::string> out;
  out.reserve(backend_hints.size() + 1);
  std::unordered_set<std::string> seen;

  const auto append_unique = [&](const std::string &raw_hint,
                                 std::vector<std::string> *dest) {
    const std::string normalized = NormalizeHint(raw_hint);
    if (normalized.empty()) {
      return;
    }
    if (seen.insert(normalized).second) {
      dest->push_back(normalized);
    }
  };

  for (const auto &hint : backend_hints) {
    append_unique(hint, &out);
  }
  if (out.empty()) {
    append_unique(default_hint.empty() ? "cpu" : default_hint, &out);
  }
  return out;
}

BackendFactoryResult BackendFactory::Create(const std::string &backend_hint) {
  std::string hint = NormalizeHint(backend_hint);
  if (hint == "auto") {
#ifdef INFERFLUX_HAS_CUDA
    hint = "cuda";
#elif INFERFLUX_HAS_MLX
    hint = "mlx";
#else
    hint = "cpu";
#endif
  }

  const auto target = ParseLlamaBackendTarget(hint);
  const auto traits = DescribeLlamaBackendTarget(target);
  const auto policy = LoadPolicy();

  if (hint == "mlx") {
    // MLX has its own backend implementation today.
    return UniversalLlamaForTarget(LlamaBackendTarget::kMps, hint);
  }

  if (hint == "cpu") {
    return CpuFallback("");
  }

  if (hint == "cuda" || hint == "mps" || hint == "rocm" || hint == "vulkan") {
    if (policy.prefer_native && SupportsNativeBackend(target)) {
      // Reserved for future native backend wiring.
      BackendFactoryResult out;
      out.target = target;
      out.traits = traits;
      out.capabilities = traits.capabilities;
      out.backend_label = traits.label;
      out.provider = BackendProvider::kNative;
      out.config = TuneLlamaBackendConfig(target, {});
      return out;
    }

    if (policy.prefer_native && !policy.allow_universal_fallback) {
      BackendFactoryResult out;
      out.target = target;
      out.traits = traits;
      out.capabilities = traits.capabilities;
      out.provider = BackendProvider::kNative;
      out.backend_label = traits.label;
      out.fallback_reason =
          "native backend requested but unavailable; universal fallback "
          "disabled";
      log::Error("backend_factory",
                 "No backend exposed for '" + hint +
                     "': native preferred, universal fallback disabled.");
      return out;
    }

    BackendFactoryResult out;
    out = UniversalLlamaForTarget(target, hint);
    if (policy.prefer_native && !SupportsNativeBackend(target)) {
      out.used_fallback = true;
      out.fallback_reason =
          "native backend unavailable; exposed universal llama backend";
      log::Warn("backend_factory",
                "Using universal llama backend for '" + hint +
                    "' because native backend is not available yet.");
    }
    return out;
  }

  return CpuFallback("Unknown backend hint '" + backend_hint +
                     "'. Falling back to CPU backend.");
}

void BackendFactory::SetExposurePolicy(const BackendExposurePolicy &policy) {
  std::lock_guard<std::mutex> lock(g_policy_mutex);
  g_policy = policy;
}

BackendExposurePolicy BackendFactory::ExposurePolicy() { return LoadPolicy(); }

LlamaBackendConfig MergeBackendConfig(const LlamaBackendConfig &defaults,
                                      const BackendFactoryResult &selection) {
  return TuneLlamaBackendConfig(selection.target, defaults);
}

} // namespace inferflux
