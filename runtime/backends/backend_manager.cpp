#include "runtime/backends/backend_manager.h"

#include <iostream>

namespace inferflux {

std::shared_ptr<LlamaCPUBackend> BackendManager::LoadBackend(const std::string& name,
                                                             const std::string& path,
                                                             int gpu_layers) {
  auto existing = GetBackend(name);
  if (existing && existing->IsReady()) {
    return existing;
  }
  auto backend = std::make_shared<LlamaCPUBackend>();
  inferflux::LlamaBackendConfig config;
  config.gpu_layers = gpu_layers;
  if (!backend->LoadModel(path, config)) {
    std::cerr << "BackendManager: failed to load model " << path << " for " << name << std::endl;
    return nullptr;
  }
  backends_[name] = backend;
  return backend;
}

std::shared_ptr<LlamaCPUBackend> BackendManager::GetBackend(const std::string& name) const {
  auto it = backends_.find(name);
  if (it == backends_.end()) {
    return nullptr;
  }
  return it->second;
}

}  // namespace inferflux
