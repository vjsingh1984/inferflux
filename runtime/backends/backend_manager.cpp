#include "runtime/backends/backend_manager.h"
#include "server/logging/logger.h"

namespace inferflux {

std::shared_ptr<LlamaCPUBackend>
BackendManager::LoadBackend(const std::string &name, const std::string &path,
                            const LlamaBackendConfig &config,
                            bool prefer_cuda) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = backends_.find(name);
  if (it != backends_.end() && it->second && it->second->IsReady()) {
    return it->second;
  }
  if (prefer_cuda) {
#ifdef INFERFLUX_HAS_CUDA
    log::Info("backend_manager",
              "CUDA support detected; GPU backend integration pending.");
#else
    log::Error("backend_manager",
               "CUDA requested but binary was built without CUDA support.");
#endif
  }
  auto backend = std::make_shared<LlamaCPUBackend>();
  if (!backend->LoadModel(path, config)) {
    log::Error("backend_manager",
               "failed to load model " + path + " for " + name);
    return nullptr;
  }
  backends_[name] = backend;
  return backend;
}

std::shared_ptr<LlamaCPUBackend>
BackendManager::GetBackend(const std::string &name) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = backends_.find(name);
  if (it == backends_.end()) {
    return nullptr;
  }
  return it->second;
}

} // namespace inferflux
