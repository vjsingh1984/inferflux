#pragma once

#include "runtime/backends/cpu/llama_cpp_backend.h"
#include "runtime/device_context.h"
#include <memory>
#include <string>

namespace inferflux {

/**
 * RocmBackend provides AMD GPU support through ROCm/HIP.
 *
 * This backend enables InferFlux to run on AMD GPUs (MI200, MI250X, MI300X)
 * using the ROCm software stack. It leverages llama.cpp's HIP backend which
 * provides FlashAttention-2 support on AMD GPUs.
 *
 * Key features:
 * - HIP device context for AMD GPU memory management
 * - FlashAttention-2 support on GFX9+ architecture (MI200, MI250X, MI300X)
 * - Reuses llama.cpp's ggml-hip backend
 * - Compatible with ROCm 6.1+
 */
class RocmBackend : public LlamaCppBackend {
public:
  RocmBackend();
  ~RocmBackend() override;

  bool LoadModel(const std::filesystem::path &model_path,
                 const LlamaBackendConfig &config) override;

  std::string GetBackendType() const override { return "rocm"; }

  bool IsReady() const override;

  // FlashAttention support for ROCm
  bool SupportsFlashAttention() const;
  std::string GetFlashAttentionVersion() const;
  std::string GetSelectedAttentionKernel() const {
    return selected_attention_kernel_;
  }

private:
  bool hip_initialized_{false};
  std::string selected_attention_kernel_{"standard"};
  int hip_device_{0};
  bool supports_flash_attention_{false};
  std::string device_arch_; // GFX90A, GFX942, etc.
  size_t total_memory_mb_{0};

  bool InitializeHip();
  bool DetectFlashAttentionSupport();
  bool ConfigureFlashAttention(const LlamaBackendConfig &config);
};

} // namespace inferflux