#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <llama.h>

namespace inferflux {

struct LlamaBackendConfig {
  int32_t ctx_size = 2048;
  int32_t batch_size = 512;
  int gpu_layers = 0;
};

class LlamaCPUBackend {
 public:
  LlamaCPUBackend();
  ~LlamaCPUBackend();

  bool LoadModel(const std::filesystem::path& model_path, const LlamaBackendConfig& config = {});
  std::string Generate(const std::string& prompt, int max_tokens);
  bool IsReady() const { return context_ != nullptr; }

 private:
  std::vector<int> Tokenize(const std::string& prompt, bool add_bos) const;
  std::string TokenToString(int token) const;
  int SampleGreedy() const;

  llama_model* model_{nullptr};
  llama_context* context_{nullptr};
  const struct llama_vocab* vocab_{nullptr};
  int32_t n_vocab_{0};
  LlamaBackendConfig config_;
};

}  // namespace inferflux
