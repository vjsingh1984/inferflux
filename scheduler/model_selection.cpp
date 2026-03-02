#include "scheduler/model_selection.h"

#include "runtime/backends/cpu/llama_backend.h"

#include <algorithm>
#include <cctype>
#include <optional>
#include <tuple>
#include <vector>

namespace inferflux {

namespace {

BackendCapabilities EffectiveCapabilities(const ModelInfo &info) {
  BackendCapabilities capabilities = info.capabilities;
  if (!info.supports_structured_output) {
    capabilities.supports_structured_output = false;
  }
  return capabilities;
}

int BackendPreferenceRank(const std::string &backend_label) {
  std::string lowered = backend_label;
  std::transform(
      lowered.begin(), lowered.end(), lowered.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (lowered == "cuda") {
    return 0;
  }
  if (lowered == "rocm") {
    return 1;
  }
  if (lowered == "mps") {
    return 2;
  }
  if (lowered == "mlx") {
    return 3;
  }
  if (lowered == "vulkan") {
    return 4;
  }
  if (lowered == "cpu") {
    return 5;
  }
  return 6;
}

std::optional<ModelSelectionResult>
PickFallbackCandidate(ModelRouter *router, const ModelInfo &primary,
                      const BackendFeatureRequirements &requirements,
                      bool require_ready_backend,
                      const std::string &fallback_feature) {
  if (!router) {
    return std::nullopt;
  }

  auto models = router->ListModels();
  std::optional<ModelSelectionResult> best;
  std::tuple<int, int, std::string> best_key{999, 1, ""};

  for (const auto &candidate_info : models) {
    if (candidate_info.id.empty() || candidate_info.id == primary.id) {
      continue;
    }
    if (!primary.path.empty() && candidate_info.path != primary.path) {
      continue;
    }

    auto candidate_check = CheckBackendCapabilities(
        EffectiveCapabilities(candidate_info), requirements);
    if (!candidate_check.supported) {
      continue;
    }

    auto candidate_backend = router->GetBackend(candidate_info.id);
    if (require_ready_backend &&
        (!candidate_backend || !candidate_backend->IsReady())) {
      continue;
    }

    std::tuple<int, int, std::string> candidate_key{
        BackendPreferenceRank(candidate_info.backend),
        candidate_info.backend_fallback ? 1 : 0, candidate_info.id};
    if (best.has_value() && !(candidate_key < best_key)) {
      continue;
    }

    ModelSelectionResult selected;
    selected.status = ModelSelectionStatus::kSelected;
    selected.info = candidate_info;
    selected.backend = std::move(candidate_backend);
    selected.used_fallback = true;
    selected.fallback_from_backend = primary.backend;
    selected.fallback_feature = fallback_feature;
    best = std::move(selected);
    best_key = candidate_key;
  }

  return best;
}

} // namespace

ModelSelectionResult
SelectModelForRequest(ModelRouter *router, const std::string &requested_model,
                      const BackendFeatureRequirements &requirements,
                      const ModelSelectionOptions &options) {
  ModelSelectionResult result;
  const bool explicit_model_requested = !requested_model.empty();
  const bool allow_default_fallback =
      options.allow_capability_fallback_for_default &&
      !explicit_model_requested;

  if (!router) {
    result.status = ModelSelectionStatus::kNotFound;
    result.reason = "router unavailable";
    return result;
  }

  auto *primary = router->Resolve(requested_model);
  if (!primary) {
    result.status = ModelSelectionStatus::kNotFound;
    result.reason = "model not found";
    return result;
  }
  result.info = *primary;

  auto primary_check =
      CheckBackendCapabilities(EffectiveCapabilities(*primary), requirements);
  if (!primary_check.supported) {
    if (allow_default_fallback) {
      auto fallback = PickFallbackCandidate(
          router, *primary, requirements, options.require_ready_backend,
          primary_check.missing_feature.empty()
              ? "unsupported_feature"
              : primary_check.missing_feature);
      if (fallback.has_value()) {
        return *fallback;
      }
    }

    result.status = ModelSelectionStatus::kUnsupported;
    result.missing_feature = primary_check.missing_feature;
    result.reason = primary_check.reason.empty()
                        ? "Selected model does not support requested features"
                        : primary_check.reason;
    return result;
  }

  auto backend = router->GetBackend(primary->id);
  if (options.require_ready_backend && (!backend || !backend->IsReady())) {
    if (allow_default_fallback) {
      auto fallback = PickFallbackCandidate(router, *primary, requirements,
                                            options.require_ready_backend,
                                            "backend_unavailable");
      if (fallback.has_value()) {
        return *fallback;
      }
    }

    result.status = ModelSelectionStatus::kBackendUnavailable;
    result.reason = "Selected model backend is not ready";
    return result;
  }

  result.status = ModelSelectionStatus::kSelected;
  result.backend = std::move(backend);
  return result;
}

} // namespace inferflux
