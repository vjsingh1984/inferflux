#pragma once

#include "runtime/backends/cpu/llama_backend.h"

#include <memory>
#include <string>
#include <unordered_map>

namespace inferflux {

class BackendManager {
 public:
  BackendManager() = default;

  std::shared_ptr<LlamaCPUBackend> LoadBackend(const std::string& name,
                                               const std::string& path,
                                               int gpu_layers);
  std::shared_ptr<LlamaCPUBackend> GetBackend(const std::string& name) const;

 private:
  std::unordered_map<std::string, std::shared_ptr<LlamaCPUBackend>> backends_;
};

}  // namespace inferflux
