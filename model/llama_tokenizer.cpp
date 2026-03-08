#include "model/llama_tokenizer.h"
#include "server/logging/logger.h"

#include <llama.h>

#include <algorithm>
#include <cstring>

namespace inferflux {

LlamaTokenizer::LlamaTokenizer() = default;

LlamaTokenizer::~LlamaTokenizer() {
  if (model_) {
    llama_model_free(model_);
    model_ = nullptr;
  }
}

bool LlamaTokenizer::Load(const std::string &model_path) {
  if (model_) {
    llama_model_free(model_);
    model_ = nullptr;
    vocab_ = nullptr;
  }

  llama_model_params params = llama_model_default_params();
  params.n_gpu_layers = 0; // CPU-only: no weight upload, just metadata

  model_ = llama_model_load_from_file(model_path.c_str(), params);
  if (!model_) {
    log::Error("llama_tokenizer",
               "Failed to load model for tokenization: " + model_path);
    return false;
  }

  vocab_ = llama_model_get_vocab(model_);
  if (!vocab_) {
    log::Error("llama_tokenizer", "Failed to obtain vocabulary");
    llama_model_free(model_);
    model_ = nullptr;
    return false;
  }

  n_vocab_ = llama_vocab_n_tokens(vocab_);

  // Extract chat template from GGUF metadata
  {
    char tmpl_buf[8192];
    int32_t tmpl_len = llama_model_meta_val_str(
        model_, "tokenizer.chat_template", tmpl_buf, sizeof(tmpl_buf));
    if (tmpl_len > 0) {
      chat_template_.assign(tmpl_buf, static_cast<size_t>(tmpl_len));
      log::Info("llama_tokenizer", "Chat template loaded (" +
                                       std::to_string(tmpl_len) + " chars)");
    } else {
      log::Warn("llama_tokenizer",
                "No chat_template in GGUF metadata — will use chatml default");
    }
  }

  log::Info("llama_tokenizer",
            "Loaded tokenizer: vocab_size=" + std::to_string(n_vocab_) +
                " bos=" + std::to_string(BosTokenId()) +
                " eos=" + std::to_string(EosTokenId()));
  return true;
}

std::vector<int> LlamaTokenizer::Tokenize(const std::string &text,
                                          bool add_bos) const {
  if (!vocab_ || text.empty()) {
    return {};
  }

  int prompt_len = static_cast<int>(text.size());
  // Over-allocate: at most 1 token per character + BOS
  std::vector<int> tokens(prompt_len + 2);

  int n = llama_tokenize(vocab_, text.c_str(), prompt_len, tokens.data(),
                         static_cast<int>(tokens.size()), add_bos, true);
  if (n < 0) {
    // Buffer too small — resize and retry
    tokens.resize(static_cast<size_t>(-n) + 1);
    n = llama_tokenize(vocab_, text.c_str(), prompt_len, tokens.data(),
                       static_cast<int>(tokens.size()), add_bos, true);
    if (n < 0) {
      log::Error("llama_tokenizer", "Tokenization failed");
      return {};
    }
  }

  tokens.resize(static_cast<size_t>(n));
  return tokens;
}

std::string LlamaTokenizer::Detokenize(const std::vector<int> &tokens) const {
  if (!vocab_ || tokens.empty()) {
    return "";
  }

  std::string result;
  result.reserve(tokens.size() * 4); // Rough estimate

  for (int token_id : tokens) {
    result += TokenToString(token_id);
  }
  return result;
}

std::string LlamaTokenizer::TokenToString(int token_id) const {
  if (!vocab_ || token_id < 0 || token_id >= n_vocab_) {
    return "";
  }

  char buf[256];
  int n = llama_token_to_piece(vocab_, token_id, buf, sizeof(buf), 0, true);
  if (n < 0) {
    // Buffer too small
    std::vector<char> big_buf(static_cast<size_t>(-n) + 1);
    n = llama_token_to_piece(vocab_, token_id, big_buf.data(),
                             static_cast<int>(big_buf.size()), 0, true);
    if (n > 0) {
      return std::string(big_buf.data(), static_cast<size_t>(n));
    }
    return "";
  }
  if (n > 0) {
    return std::string(buf, static_cast<size_t>(n));
  }
  return "";
}

int LlamaTokenizer::TokenCount(const std::string &text) const {
  return static_cast<int>(Tokenize(text, false).size());
}

ITokenizer::ChatResult LlamaTokenizer::ApplyChatTemplate(
    const std::vector<std::pair<std::string, std::string>> &messages,
    bool add_assistant_prefix) const {
  ChatResult result;
  if (!model_ || messages.empty()) {
    return result;
  }

  // Keep content strings alive for the C-struct array
  std::vector<std::string> contents;
  contents.reserve(messages.size());
  std::vector<llama_chat_message> chat;
  chat.reserve(messages.size());
  for (const auto &[role, content] : messages) {
    contents.push_back(content);
    chat.push_back({role.c_str(), contents.back().c_str()});
  }

  int total_chars = 0;
  for (const auto &[r, c] : messages) {
    total_chars += static_cast<int>(r.size() + c.size());
  }
  int buf_size = std::max(4096, total_chars * 2 + 512);
  std::vector<char> buf(buf_size);

  // Pass nullptr to llama_chat_apply_template to use chatml default.
  // The GGUF Jinja2 template string can be mismatched by the C function's
  // pattern-matching, so we always use the well-tested chatml default which
  // covers TinyLlama, Qwen, and most ChatML-based models.
  // TODO: implement proper Jinja2 template rendering for non-ChatML models.
  int32_t n =
      llama_chat_apply_template(nullptr, chat.data(), chat.size(),
                                add_assistant_prefix, buf.data(), buf_size);
  if (n < 0) {
    return result; // Template not supported
  }
  if (n > buf_size) {
    buf.resize(static_cast<size_t>(n) + 1);
    n = llama_chat_apply_template(nullptr, chat.data(), chat.size(),
                                  add_assistant_prefix, buf.data(), n);
    if (n < 0) {
      return result;
    }
  }

  result.prompt = std::string(buf.data(), static_cast<size_t>(n));
  result.valid = true;
  log::Info("llama_tokenizer", "ApplyChatTemplate result (" +
                                   std::to_string(n) + " chars): [" +
                                   result.prompt + "]");
  return result;
}

int LlamaTokenizer::BosTokenId() const {
  if (!vocab_) {
    return -1;
  }
  return llama_vocab_bos(vocab_);
}

int LlamaTokenizer::EosTokenId() const {
  if (!vocab_) {
    return -1;
  }
  return llama_vocab_eos(vocab_);
}

int LlamaTokenizer::VocabSize() const { return n_vocab_; }

} // namespace inferflux
