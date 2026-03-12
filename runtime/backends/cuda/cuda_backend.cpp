#include "runtime/backends/cuda/cuda_backend.h"
#include "runtime/backends/llama/llama_backend_traits.h"
#include "server/logging/logger.h"
#include "server/metrics/metrics.h"

namespace inferflux {

bool CudaBackend::LoadModel(const std::filesystem::path &model_path,
                            const LlamaBackendConfig &config) {
#ifdef INFERFLUX_HAS_CUDA
  LlamaBackendConfig tuned =
      TuneLlamaBackendConfig(LlamaBackendTarget::kCuda, config);
  if (!LlamaCppBackend::LoadModel(model_path, tuned)) {
    log::Error("cuda_backend",
               "failed to load CUDA model at " + model_path.string());
    return false;
  }

  // Determine attention kernel to report in metrics
  std::string attention_kernel = "standard";
  if (tuned.use_flash_attention) {
    // FlashAttention is enabled - report the kernel type
    // Note: When cuda_attention_kernel is "auto", llama.cpp will automatically
    // choose between fa2/fa3/standard based on GPU compatibility. We report
    // "fa2" as the default since that's what will be tried first.
    if (tuned.cuda_attention_kernel == "auto") {
      attention_kernel = "fa2"; // llama.cpp tries FA2 first when auto
    } else if (tuned.cuda_attention_kernel == "fa2" ||
               tuned.cuda_attention_kernel == "fa3") {
      attention_kernel = tuned.cuda_attention_kernel;
    } else {
      attention_kernel = "fa2"; // Default FA2 when enabled
    }

    log::Info("cuda_backend",
              "FlashAttention enabled (kernel=" + attention_kernel +
                  ", tile=" + std::to_string(tuned.flash_attention_tile) + ")");
  }

  // Update metrics with the selected attention kernel
  GlobalMetrics().SetCudaAttentionKernel(attention_kernel);
  if (tuned.use_flash_attention) {
    GlobalMetrics().RecordFlashAttentionRequest(attention_kernel);
  }
  GlobalMetrics().SetFlashAttentionEnabled(tuned.use_flash_attention);

  log::Info("cuda_backend",
            "CUDA model loaded successfully (attention_kernel=" +
                attention_kernel + ")");
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
