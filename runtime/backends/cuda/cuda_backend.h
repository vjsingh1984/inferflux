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

#ifdef INFERFLUX_USE_COMMON_BACKEND_TYPES
  std::string Name() const override { return "cuda_llama_cpp"; }
#endif
};

} // namespace inferflux
