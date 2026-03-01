#pragma once

#include "runtime/backends/cpu/llama_backend.h"

#include <filesystem>
#include <memory>
#include <string>

namespace inferflux {

class CudaBackend {
 public:
  CudaBackend();

  bool LoadModel(const std::filesystem::path& model_path,
                 const LlamaBackendConfig& config = {});
  std::string Generate(const std::string& prompt, int max_tokens);
  bool IsReady() const;
  int TokenCount(const std::string& text) const;

  void SetFlashAttentionEnabled(bool enabled) { flash_attention_enabled_ = enabled; }
  bool FlashAttentionEnabled() const { return flash_attention_enabled_; }

 private:
  std::shared_ptr<LlamaCPUBackend> backend_;
  bool flash_attention_enabled_{false};
};

}  // namespace inferflux
