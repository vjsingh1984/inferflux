#include "model/gguf_tokenizer.h"

#include "runtime/backends/cuda/native/gguf_model_loader.h"
#include "server/logging/logger.h"

namespace inferflux {

bool GGUFTokenizer::Load(const std::string &model_path) {
  runtime::cuda::native::GGUFModelLoader loader;
  if (!loader.Load(model_path)) {
    log::Error("gguf_tokenizer",
               "Failed to load GGUF metadata from: " + model_path);
    return false;
  }
  return InitializeFromMetadata(
      loader.TokenizerPieces(), loader.TokenizerMerges(),
      loader.TokenizerPreTokenizer(), loader.TokenizerBosTokenId(),
      loader.TokenizerEosTokenId(), loader.TokenizerAddBosToken(),
      loader.TokenizerChatTemplate());
}

bool GGUFTokenizer::InitializeFromMetadata(
    const std::vector<std::string> &pieces,
    const std::vector<std::string> &merges,
    const std::string &pre_tokenizer_hint, int bos_token_id, int eos_token_id,
    bool add_bos_token, const std::string &chat_template) {
  if (!inner_.InitializeFromBpeData(pieces, merges, pre_tokenizer_hint,
                                    bos_token_id, eos_token_id, chat_template,
                                    add_bos_token)) {
    return false;
  }
  chat_template_ = chat_template;
  return true;
}

std::vector<int> GGUFTokenizer::Tokenize(const std::string &text,
                                         bool add_bos) const {
  auto encoded = inner_.Encode(text, add_bos);
  if (!encoded.ok) {
    return {};
  }
  return std::vector<int>(encoded.ids.begin(), encoded.ids.end());
}

std::string GGUFTokenizer::Detokenize(const std::vector<int> &tokens) const {
  std::vector<int32_t> ids(tokens.begin(), tokens.end());
  return inner_.Decode(ids, /*skip_special=*/true);
}

std::string GGUFTokenizer::TokenToString(int token_id) const {
  return inner_.Decode({static_cast<int32_t>(token_id)},
                       /*skip_special=*/false);
}

bool GGUFTokenizer::IsSpecialToken(int token_id) const {
  return inner_.IsSpecial(static_cast<int32_t>(token_id));
}

int GGUFTokenizer::TokenCount(const std::string &text) const {
  return static_cast<int>(Tokenize(text, /*add_bos=*/false).size());
}

// ── Chat template strategies ────────────────────────────────────────────────
// Each strategy renders messages into the format expected by a model family.
// The correct strategy is selected from the Jinja2 template string stored
// in GGUF metadata (tokenizer.chat_template). The GGUF loader provides
// this string at initialization time.

namespace {

// Strategy: ChatML (Qwen, TinyLlama, Yi, Phi, OpenChat, Gemma, etc.)
// Format: <|im_start|>role\ncontent<|im_end|>\n
std::string RenderChatML(
    const std::vector<std::pair<std::string, std::string>> &messages,
    bool add_assistant_prefix) {
  std::string out;
  for (const auto &[role, content] : messages) {
    out += "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
  }
  if (add_assistant_prefix) {
    out += "<|im_start|>assistant\n";
  }
  return out;
}

// Strategy: Llama-2 / Llama-3 instruct
// Format: [INST] <<SYS>>\nsystem\n<</SYS>>\n\nuser [/INST] assistant </s>
std::string RenderLlama(
    const std::vector<std::pair<std::string, std::string>> &messages,
    bool add_assistant_prefix) {
  std::string out;
  bool first_user = true;
  for (const auto &[role, content] : messages) {
    if (role == "system") {
      out += "[INST] <<SYS>>\n" + content + "\n<</SYS>>\n\n";
    } else if (role == "user") {
      if (first_user && out.empty()) {
        out += "[INST] ";
      } else if (!first_user) {
        out += "[INST] ";
      }
      out += content + " [/INST]";
      first_user = false;
    } else if (role == "assistant") {
      out += " " + content + " </s>";
    }
  }
  (void)add_assistant_prefix; // Llama format implicit
  return out;
}

// Strategy: Mistral / Mixtral
// Format: [INST] content [/INST] (no system wrapping)
std::string RenderMistral(
    const std::vector<std::pair<std::string, std::string>> &messages,
    bool add_assistant_prefix) {
  std::string out;
  for (const auto &[role, content] : messages) {
    if (role == "user") {
      out += "[INST] " + content + " [/INST]";
    } else if (role == "assistant") {
      out += content + "</s> ";
    } else if (role == "system") {
      out += "[INST] " + content + "\n\n";
    }
  }
  (void)add_assistant_prefix;
  return out;
}

// Strategy: Gemma (Google)
// Format: <start_of_turn>role\ncontent<end_of_turn>\n
std::string RenderGemma(
    const std::vector<std::pair<std::string, std::string>> &messages,
    bool add_assistant_prefix) {
  std::string out;
  for (const auto &[role, content] : messages) {
    out += "<start_of_turn>" + role + "\n" + content + "<end_of_turn>\n";
  }
  if (add_assistant_prefix) {
    out += "<start_of_turn>model\n";
  }
  return out;
}

// Detect template family from Jinja2 template string in GGUF metadata.
// Returns a function pointer to the matching renderer.
using TemplateRenderer = std::string (*)(
    const std::vector<std::pair<std::string, std::string>> &, bool);

TemplateRenderer DetectTemplateFamily(const std::string &jinja_template) {
  if (jinja_template.find("im_start") != std::string::npos)
    return &RenderChatML;
  if (jinja_template.find("start_of_turn") != std::string::npos)
    return &RenderGemma;
  if (jinja_template.find("[INST]") != std::string::npos) {
    // Distinguish Llama (has <<SYS>>) from Mistral (no <<SYS>>)
    if (jinja_template.find("<<SYS>>") != std::string::npos ||
        jinja_template.find("bos_token") != std::string::npos)
      return &RenderLlama;
    return &RenderMistral;
  }
  // Default: ChatML is the safest fallback for instruct models
  return &RenderChatML;
}

} // namespace

ITokenizer::ChatResult GGUFTokenizer::ApplyChatTemplate(
    const std::vector<std::pair<std::string, std::string>> &messages,
    bool add_assistant_prefix) const {
  if (messages.empty()) {
    return {};
  }

  auto renderer = DetectTemplateFamily(chat_template_);
  std::string prompt = renderer(messages, add_assistant_prefix);

  ChatResult result;
  result.prompt = prompt;
  result.valid = !prompt.empty();
  return result;
}

int GGUFTokenizer::BosTokenId() const { return inner_.BosId(); }

int GGUFTokenizer::EosTokenId() const { return inner_.EosId(); }

int GGUFTokenizer::VocabSize() const { return inner_.VocabSize(); }

} // namespace inferflux
