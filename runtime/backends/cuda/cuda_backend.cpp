#include "runtime/backends/cuda/cuda_backend.h"
#include "runtime/backends/llama/llama_backend_traits.h"
#include "server/logging/logger.h"

namespace inferflux {

bool CudaBackend::LoadModel(const std::filesystem::path &model_path,
                            const LlamaBackendConfig &config) {
#ifdef INFERFLUX_HAS_CUDA
  LlamaBackendConfig tuned =
      TuneLlamaBackendConfig(LlamaBackendTarget::kCuda, config);
  if (!LlamaCPUBackend::LoadModel(model_path, tuned)) {
    log::Error("cuda_backend",
               "failed to load CUDA model at " + model_path.string());
    return false;
  }
  if (tuned.use_flash_attention) {
    log::Info("cuda_backend",
              "FlashAttention enabled (tile=" +
                  std::to_string(tuned.flash_attention_tile) + ")");
  }
  return true;
#else
  (void)model_path;
  (void)config;
  log::Error("cuda_backend",
             "CUDA backend requested but binary was built without CUDA "
             "support.");
  return false;
#endif
}

} // namespace inferflux
