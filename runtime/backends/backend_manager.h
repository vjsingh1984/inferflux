#pragma once

#include "runtime/backends/cpu/llama_backend.h"

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace inferflux {

class BackendManager {
public:
  BackendManager() = default;

  std::shared_ptr<LlamaCPUBackend> LoadBackend(const std::string &name,
                                               const std::string &path,
                                               const LlamaBackendConfig &config,
                                               bool prefer_cuda = false);
  std::shared_ptr<LlamaCPUBackend> GetBackend(const std::string &name) const;

private:
  mutable std::mutex mutex_;
  std::unordered_map<std::string, std::shared_ptr<LlamaCPUBackend>> backends_;
};

} // namespace inferflux
