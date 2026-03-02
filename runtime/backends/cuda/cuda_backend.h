#pragma once

#include "runtime/backends/cpu/llama_backend.h"

#include <filesystem>
#include <string>

namespace inferflux {

class CudaBackend : public LlamaCPUBackend {
public:
  CudaBackend() = default;

  bool LoadModel(const std::filesystem::path &model_path,
                 const LlamaBackendConfig &config = {}) override;
};

} // namespace inferflux
