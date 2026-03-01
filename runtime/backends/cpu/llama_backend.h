#pragma once

#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <llama.h>

namespace inferflux {

struct LlamaBackendConfig {
  int32_t ctx_size = 2048;
  int32_t batch_size = 512;
  int gpu_layers = 0;
  bool use_flash_attention = false;
  int flash_attention_tile = 128;
};

class LlamaCPUBackend {
 public:
  LlamaCPUBackend();
  ~LlamaCPUBackend();

  bool LoadModel(const std::filesystem::path& model_path, const LlamaBackendConfig& config = {});
  std::string Generate(const std::string& prompt,
                       int max_tokens,
                       const std::function<bool(const std::string&)>& on_chunk = {},
                       const std::function<bool()>& should_stop = {});
  bool IsReady() const { return context_ != nullptr || test_ready_; }

  // Returns the number of tokens in `text` using the loaded model's vocabulary.
  // Falls back to 0 if no model is loaded. Does not include the BOS token.
  int TokenCount(const std::string& text) const;

  // Testing hook: marks the backend as ready without loading a model.
  // Only used in unit tests to exercise router logic without GGUF weights.
  void ForceReadyForTests() { test_ready_ = true; }

 private:
  std::vector<int> Tokenize(const std::string& prompt, bool add_bos) const;
  std::string TokenToString(int token) const;
  int SampleGreedy() const;

  llama_model* model_{nullptr};
  llama_context* context_{nullptr};
  const struct llama_vocab* vocab_{nullptr};
  int32_t n_vocab_{0};
  LlamaBackendConfig config_;
  bool test_ready_{false};
};

}  // namespace inferflux
