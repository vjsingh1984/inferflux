#include "scheduler/single_model_router.h"

#include "server/metrics/metrics.h"

#include <filesystem>

namespace inferflux {

namespace {

LlamaBackendConfig ConfigForBackendHint(const std::string& hint) {
  LlamaBackendConfig cfg;
  if (hint == "mps" || hint == "cuda") {
    cfg.gpu_layers = 99;
  }
  return cfg;
}

std::string BackendLabelOrDefault(const std::string& hint) {
  if (hint.empty()) {
    return "cpu";
  }
  return hint;
}

}  // namespace

SingleModelRouter::SingleModelRouter(std::shared_ptr<LlamaCPUBackend> backend,
                                     ModelInfo info) {
  RegisterModel(info, std::move(backend));
}

SingleModelRouter::SingleModelRouter() = default;

bool SingleModelRouter::RegisterModel(const ModelInfo& info,
                                      std::shared_ptr<LlamaCPUBackend> backend) {
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
  entry.info.backend = info.backend.empty() ? "cpu" : info.backend;
  entry.info.supports_structured_output = true;
  entry.info.is_moe = backend->IsMoE();
  entry.info.n_experts = backend->ExpertCount();
  entry.info.n_active_experts = backend->ActiveExperts();
  entry.info.id = info.id.empty()
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
  RecordModelReadyLocked(entry, entry.info.ready);
  return true;
}

std::vector<ModelInfo> SingleModelRouter::ListModels() const {
  std::vector<ModelInfo> out;
  std::lock_guard<std::mutex> lock(mutex_);
  out.reserve(models_.size());
  for (const auto& [id, entry] : models_) {
    ModelInfo copy = entry.info;
    copy.ready = entry.backend && entry.backend->IsReady();
    out.push_back(copy);
  }
  return out;
}

std::string SingleModelRouter::LoadModel(const std::string& path,
                                         const std::string& backend_hint,
                                         const std::string& requested_id) {
  auto new_backend = std::make_shared<LlamaCPUBackend>();
  auto start = std::chrono::steady_clock::now();
  LlamaBackendConfig cfg = ConfigForBackendHint(backend_hint);
  if (!new_backend->LoadModel(path, cfg)) {
    return "";
  }

  auto load_finished = std::chrono::steady_clock::now();
  double load_seconds = std::chrono::duration<double>(load_finished - start).count();

  ModelInfo info;
  info.path = path;
  info.backend = BackendLabelOrDefault(backend_hint);
  info.ready = new_backend->IsReady();
  info.supports_structured_output = true;
  info.is_moe = new_backend->IsMoE();
  info.n_experts = new_backend->ExpertCount();
  info.n_active_experts = new_backend->ActiveExperts();

  std::lock_guard<std::mutex> lock(mutex_);
  std::string preferred = !requested_id.empty()
                              ? requested_id
                              : std::filesystem::path(path).stem().string();
  info.id = EnsureUniqueIdLocked(preferred);

  Entry entry;
  entry.info = info;
  entry.backend = std::move(new_backend);
  entry.load_time = load_finished;
  models_[info.id] = entry;
  if (default_model_id_.empty()) {
    default_model_id_ = info.id;
  }
  GlobalMetrics().RecordModelLoad(info.id, info.backend, load_seconds);
  RecordModelReadyLocked(entry, entry.info.ready);
  return info.id;
}

bool SingleModelRouter::UnloadModel(const std::string& id) {
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

ModelInfo* SingleModelRouter::Resolve(const std::string& requested_model) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (models_.empty()) {
    return nullptr;
  }
  std::string target = requested_model.empty() ? default_model_id_ : requested_model;
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

std::shared_ptr<LlamaCPUBackend> SingleModelRouter::GetBackend(const std::string& model_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = models_.find(model_id);
  if (it == models_.end()) {
    return nullptr;
  }
  return it->second.backend;
}

bool SingleModelRouter::SetDefaultModel(const std::string& model_id) {
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

std::string SingleModelRouter::GenerateModelIdLocked(const std::string& path) const {
  std::string base = std::filesystem::path(path).stem().string();
  if (base.empty()) {
    base = "model";
  }
  return EnsureUniqueIdLocked(base);
}

std::string SingleModelRouter::EnsureUniqueIdLocked(const std::string& preferred) const {
  std::string base = preferred.empty() ? "model" : preferred;
  std::string candidate = base;
  int suffix = 1;
  while (models_.find(candidate) != models_.end()) {
    candidate = base + "-" + std::to_string(suffix++);
  }
  return candidate;
}

void SingleModelRouter::RecordModelReadyLocked(const Entry& entry, bool ready) const {
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

}  // namespace inferflux
