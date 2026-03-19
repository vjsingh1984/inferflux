#include "server/http/model_json.h"

#include "runtime/backends/backend_factory.h"

using json = nlohmann::json;

namespace inferflux {

json BuildCapabilitiesJson(const BackendCapabilities &capabilities) {
  return json{
      {"streaming", capabilities.supports_streaming},
      {"logprobs", capabilities.supports_logprobs},
      {"structured_output", capabilities.supports_structured_output},
      {"embeddings", capabilities.supports_embeddings},
      {"vision", capabilities.supports_vision},
      {"speculative_decoding", capabilities.supports_speculative_decoding},
      {"fairness_preemption", capabilities.supports_fairness_preemption},
      {"kv_prefix_transfer", capabilities.supports_kv_prefix_transfer},
  };
}

std::string ModelSourcePath(const ModelInfo &info) {
  return info.source_path.empty() ? info.path : info.source_path;
}

std::string ModelEffectiveLoadPath(const ModelInfo &info) {
  const std::string source = ModelSourcePath(info);
  return info.effective_load_path.empty() ? source : info.effective_load_path;
}

json BuildBackendExposureJson(const ModelInfo &info) {
  const std::string requested =
      info.requested_backend.empty() ? info.backend : info.requested_backend;
  const std::string provider =
      info.backend_provider.empty()
          ? BackendFactory::ProviderLabel(BackendProvider::kLlamaCpp)
          : BackendFactory::ProviderLabel(
                BackendFactory::ParseProviderLabel(info.backend_provider));
  return json{
      {"requested_backend", requested},
      {"exposed_backend", info.backend},
      {"provider", provider},
      {"fallback", info.backend_fallback},
      {"fallback_reason", info.backend_fallback_reason},
  };
}

json BuildModelIdentityJson(const ModelInfo &info) {
  return json{
      {"id", info.id},
      {"path", info.path},
      {"source_path", ModelSourcePath(info)},
      {"effective_load_path", ModelEffectiveLoadPath(info)},
      {"format", info.format},
      {"requested_format", info.requested_format},
      {"backend", info.backend},
      {"backend_exposure", BuildBackendExposureJson(info)},
      {"ready", info.ready},
      {"capabilities", BuildCapabilitiesJson(info.capabilities)},
  };
}

json BuildOpenAIModelJson(const ModelInfo &info, int64_t created_ts) {
  json model = BuildModelIdentityJson(info);
  model["object"] = "model";
  model["created"] = created_ts;
  model["owned_by"] = "inferflux";
  return model;
}

json BuildAdminModelJson(const ModelInfo &info, const std::string &default_id) {
  json model = BuildModelIdentityJson(info);
  model["requested_backend"] =
      info.requested_backend.empty() ? info.backend : info.requested_backend;
  model["backend_provider"] =
      info.backend_provider.empty()
          ? BackendFactory::ProviderLabel(BackendProvider::kLlamaCpp)
          : BackendFactory::ProviderLabel(
                BackendFactory::ParseProviderLabel(info.backend_provider));
  model["default"] = (info.id == default_id);
  return model;
}

} // namespace inferflux
