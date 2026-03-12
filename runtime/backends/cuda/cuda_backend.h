#pragma once

#include "runtime/backends/cpu/llama_cpp_backend.h"

#include <filesystem>
#include <string>

namespace inferflux {

class CudaBackend : public LlamaCppBackend {
public:
  CudaBackend() = default;

  bool LoadModel(const std::filesystem::path &model_path,
                 const LlamaBackendConfig &config = {}) override;

#ifdef INFERFLUX_USE_COMMON_BACKEND_TYPES
  std::string Name() const override { return "llama_cpp_cuda"; }
#endif
};

} // namespace inferflux
