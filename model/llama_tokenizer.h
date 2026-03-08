#pragma once

#include "model/tokenizer.h"

#include <string>
#include <utility>
#include <vector>

struct llama_model;
struct llama_vocab;

namespace inferflux {

/**
 * LlamaTokenizer: production tokenizer using llama.cpp internals.
 *
 * Loads a GGUF model with 0 GPU layers (CPU-only metadata parse) to access
 * llama.cpp's production SentencePiece/BPE tokenizer and chat template engine.
 * Default ITokenizer implementation for all backends.
 */
class LlamaTokenizer final : public ITokenizer {
public:
  LlamaTokenizer();
  ~LlamaTokenizer() override;

  LlamaTokenizer(const LlamaTokenizer &) = delete;
  LlamaTokenizer &operator=(const LlamaTokenizer &) = delete;

  bool Load(const std::string &model_path) override;
  std::vector<int> Tokenize(const std::string &text,
                            bool add_bos = true) const override;
  std::string Detokenize(const std::vector<int> &tokens) const override;
  std::string TokenToString(int token_id) const override;
  int TokenCount(const std::string &text) const override;
  ChatResult ApplyChatTemplate(
      const std::vector<std::pair<std::string, std::string>> &messages,
      bool add_assistant_prefix = true) const override;
  int BosTokenId() const override;
  int EosTokenId() const override;
  int VocabSize() const override;
  bool IsLoaded() const override { return model_ != nullptr; }

private:
  llama_model *model_{nullptr};
  const llama_vocab *vocab_{nullptr};
  int n_vocab_{0};
  std::string chat_template_; // Extracted from GGUF metadata
};

} // namespace inferflux
