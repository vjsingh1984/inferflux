#pragma once

#include "runtime/backends/cpu/llama_backend.h"
#include "runtime/backends/mlx/mlx_execution.h"
#include "runtime/backends/mlx/mlx_loader.h"
#include "runtime/backends/mlx/mlx_tokenizer.h"

namespace inferflux {

// MLX backend: loads MLX-native model directories (config.json + *.safetensors)
// and runs inference on Apple Silicon via Metal.
// Falls back to LlamaCPUBackend for GGUF files.
class MlxBackend : public LlamaCPUBackend {
public:
  MlxBackend();
  ~MlxBackend();

  bool LoadModel(const std::filesystem::path &model_path,
                 const LlamaBackendConfig &config = {});

  // Tokenizer accessors — available after a successful LoadModel on a
  // directory.
  bool TokenizerLoaded() const { return tokenizer_.Loaded(); }

  MlxTokenizerResult Tokenize(const std::string &text,
                              bool add_bos = true) const {
    return tokenizer_.Encode(text, add_bos);
  }

  std::string Detokenize(const std::vector<int32_t> &ids,
                         bool skip_special = true) const {
    return tokenizer_.Decode(ids, skip_special);
  }

  // Run autoregressive inference on text using the MLX engine.
  // Returns an empty string when the engine or tokenizer is not ready.
  std::string InferText(const std::string &prompt, int max_new_tokens = 200,
                        bool add_bos = true) const;

  // --- LlamaCPUBackend virtual overrides ---

  // True when the MLX engine is ready, or when a GGUF model is loaded in the
  // base llama.cpp backend.
  bool IsReady() const override;

  // Token count via the HF tokenizer when loaded; falls back to base class.
  int TokenCount(const std::string &text) const override;

  // Full generation: routes through the MLX engine when engine_ready_,
  // otherwise delegates to LlamaCPUBackend::Generate().
  std::string
  Generate(const std::string &prompt, int max_tokens,
           const std::function<bool(const std::string &)> &on_chunk = {},
           const std::function<bool()> &should_stop = {}, int logprob_top_n = 0,
           std::vector<TokenLogprob> *out_logprobs = nullptr,
           const std::vector<std::string> &stop_seqs = {}) override;

private:
  MlxWeightLoader loader_;
  MlxModelDescriptor descriptor_;
  MlxWeightStore weight_store_;
  MlxTokenizer tokenizer_;
  MlxExecutionEngine engine_;
  bool engine_ready_{false};
};

} // namespace inferflux
