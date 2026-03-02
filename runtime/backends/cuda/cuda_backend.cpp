#include "runtime/backends/cuda/cuda_backend.h"
#include "server/logging/logger.h"

namespace inferflux {

CudaBackend::CudaBackend() : backend_(std::make_shared<LlamaCPUBackend>()) {}

bool CudaBackend::LoadModel(const std::filesystem::path &model_path,
                            const LlamaBackendConfig &config) {
  if (!backend_) {
    backend_ = std::make_shared<LlamaCPUBackend>();
  }
  LlamaBackendConfig tuned = config;
  if (tuned.gpu_layers <= 0) {
    tuned.gpu_layers = 99;
  }
  if (!backend_->LoadModel(model_path, tuned)) {
    log::Error("cuda_backend",
               "Failed to load model at " + model_path.string() +
                   ". Build with CUDA to enable true GPU execution.");
    return false;
  }
  if (flash_attention_enabled_) {
    log::Info("cuda_backend",
              "FlashAttention placeholder active. Integrate FA3 kernels here.");
  }
  return true;
}

std::string CudaBackend::Generate(const std::string &prompt, int max_tokens) {
  if (!backend_ || !backend_->IsReady()) {
    return {};
  }
  return backend_->Generate(prompt, max_tokens);
}

bool CudaBackend::IsReady() const { return backend_ && backend_->IsReady(); }

int CudaBackend::TokenCount(const std::string &text) const {
  if (!backend_) {
    return 0;
  }
  return backend_->TokenCount(text);
}

} // namespace inferflux
