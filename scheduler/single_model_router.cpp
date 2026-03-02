#include "scheduler/single_model_router.h"

#include "runtime/backends/backend_factory.h"
#include "server/logging/logger.h"
#include "server/metrics/metrics.h"

#include <filesystem>
#include <vector>

namespace inferflux {

namespace {

std::string ProviderLabel(BackendProvider provider) {
  switch (provider) {
  case BackendProvider::kNative:
    return "native";
  case BackendProvider::kUniversalLlama:
  default:
    return "universal";
  }
}

} // namespace

SingleModelRouter::SingleModelRouter(std::shared_ptr<LlamaCPUBackend> backend,
                                     ModelInfo info) {
  RegisterModel(info, std::move(backend));
}

SingleModelRouter::SingleModelRouter() = default;

SingleModelRouter::SingleModelRouter(
    const LlamaBackendConfig &default_backend_config,
    const std::string &default_backend_hint,
    const std::vector<std::string> &backend_priority)
    : default_backend_config_(default_backend_config),
      default_backend_hint_(BackendFactory::NormalizeHint(
          default_backend_hint.empty() ? "cpu" : default_backend_hint)),
      backend_priority_(BackendFactory::NormalizeHintList(
          backend_priority, default_backend_hint_)) {}

bool SingleModelRouter::RegisterModel(
    const ModelInfo &info, std::shared_ptr<LlamaCPUBackend> backend) {
  if (!backend) {
    return false;
  }
  if (info.path.empty() && info.id.empty()) {
    return false;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  Entry entry;
  entry.info = info;
  entry.info.path = info.path;
  entry.info.backend = BackendFactory::NormalizeHint(
      info.backend.empty() ? "cpu" : info.backend);
  entry.info.requested_backend =
      info.requested_backend.empty()
          ? entry.info.backend
          : BackendFactory::NormalizeHint(info.requested_backend);
  entry.info.backend_provider =
      info.backend_provider.empty() ? "universal" : info.backend_provider;
  entry.info.backend_fallback = info.backend_fallback;
  entry.info.backend_fallback_reason = info.backend_fallback_reason;
  auto register_target = ParseLlamaBackendTarget(entry.info.backend);
  entry.info.capabilities =
      DescribeLlamaBackendTarget(register_target).capabilities;
  entry.info.capabilities.supports_vision = backend->SupportsVision();
  entry.info.supports_structured_output =
      entry.info.capabilities.supports_structured_output;
  entry.info.is_moe = backend->IsMoE();
  entry.info.n_experts = backend->ExpertCount();
  entry.info.n_active_experts = backend->ActiveExperts();
  entry.info.id =
      info.id.empty()
          ? GenerateModelIdLocked(info.path.empty() ? "model" : info.path)
          : EnsureUniqueIdLocked(info.id);
  entry.info.ready = backend->IsReady();
  entry.backend = std::move(backend);
  entry.load_time = std::chrono::steady_clock::now();
  models_[entry.info.id] = entry;
  if (default_model_id_.empty()) {
    default_model_id_ = entry.info.id;
  }
  GlobalMetrics().RecordModelLoad(entry.info.id, entry.info.backend, 0.0);
  GlobalMetrics().RecordBackendExposure(
      entry.info.requested_backend, entry.info.backend,
      entry.info.backend_provider, entry.info.backend_fallback);
  RecordModelReadyLocked(entry, entry.info.ready);
  return true;
}

std::vector<ModelInfo> SingleModelRouter::ListModels() const {
  std::vector<ModelInfo> out;
  std::lock_guard<std::mutex> lock(mutex_);
  out.reserve(models_.size());
  for (const auto &[id, entry] : models_) {
    ModelInfo copy = entry.info;
    copy.ready = entry.backend && entry.backend->IsReady();
    out.push_back(copy);
  }
  return out;
}

std::string SingleModelRouter::LoadModel(const std::string &path,
                                         const std::string &backend_hint,
                                         const std::string &requested_id) {
  const auto backend_candidates = BuildBackendCandidates(backend_hint);
  BackendFactoryResult selection;
  std::string selected_requested_backend;
  std::string failure_reason;

  auto start = std::chrono::steady_clock::now();
  for (const auto &candidate : backend_candidates) {
    auto candidate_selection = BackendFactory::Create(candidate);
    selected_requested_backend = candidate;
    if (!candidate_selection.backend) {
      if (!candidate_selection.fallback_reason.empty()) {
        failure_reason = candidate_selection.fallback_reason;
      } else {
        failure_reason =
            "backend candidate '" + candidate + "' was not available";
      }
      continue;
    }

    auto cfg = MergeBackendConfig(default_backend_config_, candidate_selection);
    if (!candidate_selection.backend->LoadModel(path, cfg)) {
      failure_reason =
          "backend candidate '" + candidate + "' failed to load model";
      continue;
    }
    selection = std::move(candidate_selection);
    break;
  }

  if (!selection.backend) {
    if (!failure_reason.empty()) {
      log::Error("single_model_router", failure_reason);
    }
    return "";
  }

  auto load_finished = std::chrono::steady_clock::now();
  double load_seconds =
      std::chrono::duration<double>(load_finished - start).count();

  ModelInfo info;
  info.path = path;
  info.backend = selection.backend_label;
  info.requested_backend = selected_requested_backend.empty()
                               ? selection.backend_label
                               : selected_requested_backend;
  info.backend_provider = ProviderLabel(selection.provider);
  info.backend_fallback = selection.used_fallback;
  info.backend_fallback_reason = selection.fallback_reason;
  info.ready = selection.backend->IsReady();
  info.capabilities = selection.capabilities;
  info.capabilities.supports_vision = selection.backend->SupportsVision();
  info.supports_structured_output =
      info.capabilities.supports_structured_output;
  info.is_moe = selection.backend->IsMoE();
  info.n_experts = selection.backend->ExpertCount();
  info.n_active_experts = selection.backend->ActiveExperts();
  if (info.backend_fallback) {
    log::Warn("single_model_router",
              "Backend fallback for model '" + path + "': requested=" +
                  info.requested_backend + ", selected=" + info.backend +
                  (info.backend_fallback_reason.empty()
                       ? ""
                       : ", reason=" + info.backend_fallback_reason));
  }

  std::lock_guard<std::mutex> lock(mutex_);
  std::string preferred = !requested_id.empty()
                              ? requested_id
                              : std::filesystem::path(path).stem().string();
  info.id = EnsureUniqueIdLocked(preferred);

  Entry entry;
  entry.info = info;
  entry.backend = std::move(selection.backend);
  entry.load_time = load_finished;
  models_[info.id] = entry;
  if (default_model_id_.empty()) {
    default_model_id_ = info.id;
  }
  GlobalMetrics().RecordModelLoad(info.id, info.backend, load_seconds);
  GlobalMetrics().RecordBackendExposure(info.requested_backend, info.backend,
                                        info.backend_provider,
                                        info.backend_fallback);
  RecordModelReadyLocked(entry, entry.info.ready);
  return info.id;
}

bool SingleModelRouter::UnloadModel(const std::string &id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = models_.find(id);
  if (it == models_.end()) {
    return false;
  }
  RecordModelReadyLocked(it->second, false);
  models_.erase(it);
  if (default_model_id_ == id) {
    default_model_id_.clear();
    UpdateDefaultModelLocked();
  }
  return true;
}

ModelInfo *SingleModelRouter::Resolve(const std::string &requested_model) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (models_.empty()) {
    return nullptr;
  }
  std::string target =
      requested_model.empty() ? default_model_id_ : requested_model;
  auto it = models_.find(target);
  if (it == models_.end()) {
    if (!requested_model.empty() && !default_model_id_.empty()) {
      it = models_.find(default_model_id_);
      if (it == models_.end()) {
        return nullptr;
      }
    } else {
      return nullptr;
    }
  }
  bool ready = it->second.backend && it->second.backend->IsReady();
  it->second.info.ready = ready;
  RecordModelReadyLocked(it->second, ready);
  return &it->second.info;
}

std::shared_ptr<LlamaCPUBackend>
SingleModelRouter::GetBackend(const std::string &model_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = models_.find(model_id);
  if (it == models_.end()) {
    return nullptr;
  }
  return it->second.backend;
}

bool SingleModelRouter::SetDefaultModel(const std::string &model_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (models_.find(model_id) == models_.end()) {
    return false;
  }
  default_model_id_ = model_id;
  return true;
}

std::string SingleModelRouter::DefaultModelId() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return default_model_id_;
}

std::shared_ptr<LlamaCPUBackend> SingleModelRouter::Backend() const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (default_model_id_.empty()) {
    return nullptr;
  }
  auto it = models_.find(default_model_id_);
  if (it == models_.end()) {
    return nullptr;
  }
  return it->second.backend;
}

std::string
SingleModelRouter::GenerateModelIdLocked(const std::string &path) const {
  std::string base = std::filesystem::path(path).stem().string();
  if (base.empty()) {
    base = "model";
  }
  return EnsureUniqueIdLocked(base);
}

std::string
SingleModelRouter::EnsureUniqueIdLocked(const std::string &preferred) const {
  std::string base = preferred.empty() ? "model" : preferred;
  std::string candidate = base;
  int suffix = 1;
  while (models_.find(candidate) != models_.end()) {
    candidate = base + "-" + std::to_string(suffix++);
  }
  return candidate;
}

void SingleModelRouter::RecordModelReadyLocked(const Entry &entry,
                                               bool ready) const {
  GlobalMetrics().RecordModelReady(entry.info.id, entry.info.backend, ready);
}

void SingleModelRouter::UpdateDefaultModelLocked() {
  if (!default_model_id_.empty()) {
    return;
  }
  if (models_.empty()) {
    return;
  }
  default_model_id_ = models_.begin()->first;
}

std::vector<std::string> SingleModelRouter::BuildBackendCandidates(
    const std::string &backend_hint) const {
  if (!backend_hint.empty()) {
    const std::string normalized = BackendFactory::NormalizeHint(backend_hint);
    if (normalized != "auto") {
      return {normalized};
    }
  }
  if (!backend_priority_.empty()) {
    return backend_priority_;
  }
  return {default_backend_hint_};
}

} // namespace inferflux
