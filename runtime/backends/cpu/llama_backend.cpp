#include "runtime/backends/cpu/llama_backend.h"

#include <llama.h>

#include <algorithm>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace {

void BatchClear(llama_batch& batch) { batch.n_tokens = 0; }

void BatchAdd(llama_batch& batch, llama_token id, llama_pos pos, bool logits) {
  if (!batch.seq_id[batch.n_tokens]) {
    throw std::runtime_error("llama_batch capacity exceeded");
  }
  batch.token[batch.n_tokens] = id;
  batch.pos[batch.n_tokens] = pos;
  batch.n_seq_id[batch.n_tokens] = 1;
  batch.seq_id[batch.n_tokens][0] = 0;
  batch.logits[batch.n_tokens] = logits ? 1 : 0;
  batch.n_tokens++;
}

}  // namespace

namespace inferflux {

namespace {
std::mutex g_llama_init_mutex;
int g_llama_init_refcount = 0;

void LlamaBackendAcquire() {
  std::lock_guard<std::mutex> lock(g_llama_init_mutex);
  if (g_llama_init_refcount++ == 0) {
    llama_backend_init();
  }
}

void LlamaBackendRelease() {
  std::lock_guard<std::mutex> lock(g_llama_init_mutex);
  if (--g_llama_init_refcount == 0) {
    llama_backend_free();
  }
}
}  // namespace

LlamaCPUBackend::LlamaCPUBackend() { LlamaBackendAcquire(); }

LlamaCPUBackend::~LlamaCPUBackend() {
  if (context_ != nullptr) {
    llama_free(context_);
    context_ = nullptr;
  }
  if (model_ != nullptr) {
    llama_model_free(model_);
    model_ = nullptr;
  }
  vocab_ = nullptr;
  LlamaBackendRelease();
}

bool LlamaCPUBackend::LoadModel(const std::filesystem::path& model_path, const LlamaBackendConfig& config) {
  test_ready_ = false;
  if (!std::filesystem::exists(model_path)) {
    std::cerr << "[LlamaCPUBackend] model path does not exist: " << model_path << std::endl;
    return false;
  }
  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = config.gpu_layers;

  model_ = llama_model_load_from_file(model_path.string().c_str(), model_params);
  if (!model_) {
    std::cerr << "[LlamaCPUBackend] failed to load model from " << model_path << std::endl;
    return false;
  }

  llama_context_params ctx_params = llama_context_default_params();
  ctx_params.n_ctx = config.ctx_size;
  ctx_params.n_batch = config.batch_size;

  if (config.use_flash_attention) {
    std::cout << "[LlamaCPUBackend] FlashAttention requested (tile=" << config.flash_attention_tile
              << "). Integrate FA3 kernels via llama.cpp CUDA build to enable GPU acceleration.\n";
  }

  context_ = llama_init_from_model(model_, ctx_params);
  if (!context_) {
    std::cerr << "[LlamaCPUBackend] failed to create context" << std::endl;
    return false;
  }
  vocab_ = llama_model_get_vocab(model_);
  if (!vocab_) {
    std::cerr << "[LlamaCPUBackend] failed to obtain vocabulary" << std::endl;
    return false;
  }
  n_vocab_ = llama_vocab_n_tokens(vocab_);
  config_ = config;
  return true;
}

int LlamaCPUBackend::TokenCount(const std::string& text) const {
  auto tokens = Tokenize(text, /*add_bos=*/false);
  return static_cast<int>(tokens.size());
}

std::vector<int> LlamaCPUBackend::Tokenize(const std::string& prompt, bool add_bos) const {
  if (!model_) {
    return {};
  }
  if (!vocab_) {
    return {};
  }
  std::vector<llama_token> tokens;
  tokens.resize(prompt.size() + 8);
  int n = llama_tokenize(vocab_, prompt.c_str(), prompt.size(), tokens.data(), tokens.size(), add_bos, true);
  if (n < 0) {
    tokens.resize(-n);
    n = llama_tokenize(vocab_, prompt.c_str(), prompt.size(), tokens.data(), tokens.size(), add_bos, true);
  }
  tokens.resize(n);
  return {tokens.begin(), tokens.end()};
}

int LlamaCPUBackend::SampleGreedy() const {
  const float* logits = llama_get_logits(context_);
  if (!logits) {
    return vocab_ ? llama_vocab_eos(vocab_) : 0;
  }
  auto max_it = std::max_element(logits, logits + n_vocab_);
  return static_cast<int>(std::distance(logits, max_it));
}

std::string LlamaCPUBackend::TokenToString(int token) const {
  if (!vocab_) {
    return {};
  }
  std::string buf;
  buf.resize(16);
  int written = llama_token_to_piece(vocab_, token, buf.data(), buf.size(), 0, false);
  if (written < 0) {
    buf.resize(static_cast<std::size_t>(-written));
    if (llama_token_to_piece(vocab_, token, buf.data(), buf.size(), 0, false) < 0) {
      return {};
    }
  } else {
    buf.resize(static_cast<std::size_t>(written));
  }
  return buf;
}

std::string LlamaCPUBackend::Generate(const std::string& prompt,
                                      int max_tokens,
                                      const std::function<bool(const std::string&)>& on_chunk,
                                      const std::function<bool()>& should_stop) {
  if (!IsReady()) {
    return {};
  }
  auto prompt_tokens = Tokenize(prompt, true);
  if (prompt_tokens.empty()) {
    return {};
  }

  int32_t batch_cap = std::max<int32_t>(config_.batch_size,
                                        static_cast<int32_t>(prompt_tokens.size() + std::max(max_tokens, 1)));
  llama_batch batch = llama_batch_init(batch_cap, 0, 1);
  llama_pos position = 0;
  for (size_t i = 0; i < prompt_tokens.size(); ++i) {
    BatchAdd(batch, prompt_tokens[i], position++, i == prompt_tokens.size() - 1);
  }
  if (llama_decode(context_, batch) != 0) {
    std::cerr << "[LlamaCPUBackend] llama_decode failed for prompt" << std::endl;
    llama_batch_free(batch);
    return {};
  }
  BatchClear(batch);

  std::string output;
  llama_token eos = llama_vocab_eos(vocab_);
  int tokens_remaining = std::max(max_tokens, 1);

  while (tokens_remaining-- > 0) {
    if (should_stop && should_stop()) {
      break;
    }
    int token = SampleGreedy();
    if (token == eos) {
      break;
    }
    std::string piece = TokenToString(token);
    output += piece;
    if (on_chunk) {
      if (!on_chunk(piece)) {
        break;
      }
    }
    if (should_stop && should_stop()) {
      break;
    }
    BatchAdd(batch, token, position++, true);
    if (llama_decode(context_, batch) != 0) {
      std::cerr << "[LlamaCPUBackend] llama_decode failed while generating" << std::endl;
      break;
    }
    BatchClear(batch);
  }

  llama_batch_free(batch);
  return output;
}

}  // namespace inferflux
