#include "model/tokenizer_factory.h"
#include "model/hf_tokenizer.h"
#include "model/llama_tokenizer.h"
#include "model/model_format.h"
#include "server/logging/logger.h"

#include <memory>

namespace inferflux {

std::unique_ptr<ITokenizer> CreateTokenizer(const std::string &model_path,
                                            const std::string &format) {
  if (model_path.empty()) {
    return nullptr;
  }

  const std::string resolved = ResolveModelFormat(model_path, format);

  // GGUF → LlamaTokenizer (proper BPE via llama.cpp)
  if (resolved == "gguf") {
    auto tok = std::make_unique<LlamaTokenizer>();
    if (tok->Load(model_path)) {
      log::Info("tokenizer_factory", "Created LlamaTokenizer for GGUF model");
      return tok;
    }
    log::Error("tokenizer_factory",
               "LlamaTokenizer failed to load: " + model_path);
    return nullptr;
  }

  // safetensors / hf → try HFTokenizer first, then GGUF sidecar fallback
  if (resolved == "safetensors" || resolved == "hf") {
    auto hf_tok = std::make_unique<HFTokenizer>();
    if (hf_tok->Load(model_path)) {
      log::Info("tokenizer_factory",
                "Created HFTokenizer for " + resolved + " model");
      return hf_tok;
    }

    // Fallback: look for a GGUF sidecar file
    const std::string llama_path = ResolveLlamaLoadPath(model_path, resolved);
    if (!llama_path.empty()) {
      auto llama_tok = std::make_unique<LlamaTokenizer>();
      if (llama_tok->Load(llama_path)) {
        log::Info("tokenizer_factory",
                  "Created LlamaTokenizer via GGUF sidecar: " + llama_path);
        return llama_tok;
      }
    }

    log::Error("tokenizer_factory",
               "All tokenizer strategies failed for: " + model_path);
    return nullptr;
  }

  // Unknown format — try LlamaTokenizer as last resort
  auto tok = std::make_unique<LlamaTokenizer>();
  if (tok->Load(model_path)) {
    log::Info("tokenizer_factory",
              "Created LlamaTokenizer (fallback) for: " + model_path);
    return tok;
  }

  log::Error("tokenizer_factory",
             "No tokenizer could be created for: " + model_path);
  return nullptr;
}

} // namespace inferflux
