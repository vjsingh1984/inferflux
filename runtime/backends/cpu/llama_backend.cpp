#include "runtime/backends/cpu/llama_backend.h"

#include <llama.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

void BatchClear(llama_batch &batch) { batch.n_tokens = 0; }

void BatchAdd(llama_batch &batch, llama_token id, llama_pos pos, bool logits) {
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

// Sequence-aware variant for phased prefill/decode (§2.5 Option A).
void BatchAddSeq(llama_batch &batch, llama_token id, llama_pos pos,
                 llama_seq_id seq_id, bool logits) {
  if (!batch.seq_id[batch.n_tokens]) {
    throw std::runtime_error("llama_batch capacity exceeded");
  }
  batch.token[batch.n_tokens] = id;
  batch.pos[batch.n_tokens] = pos;
  batch.n_seq_id[batch.n_tokens] = 1;
  batch.seq_id[batch.n_tokens][0] = seq_id;
  batch.logits[batch.n_tokens] = logits ? 1 : 0;
  batch.n_tokens++;
}

} // namespace

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
} // namespace

LlamaCPUBackend::LlamaCPUBackend() { LlamaBackendAcquire(); }

LlamaCPUBackend::~LlamaCPUBackend() {
#ifdef INFERFLUX_HAS_MTMD
  if (mtmd_ctx_) {
    mtmd_free(mtmd_ctx_);
    mtmd_ctx_ = nullptr;
  }
#endif
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

bool LlamaCPUBackend::LoadModel(const std::filesystem::path &model_path,
                                const LlamaBackendConfig &config) {
  test_ready_ = false;
  if (!std::filesystem::exists(model_path)) {
    std::cerr << "[LlamaCPUBackend] model path does not exist: " << model_path
              << std::endl;
    return false;
  }
  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = config.gpu_layers;

  model_ =
      llama_model_load_from_file(model_path.string().c_str(), model_params);
  if (!model_) {
    std::cerr << "[LlamaCPUBackend] failed to load model from " << model_path
              << std::endl;
    return false;
  }

  llama_context_params ctx_params = llama_context_default_params();
  ctx_params.n_ctx = config.ctx_size;
  ctx_params.n_batch = config.batch_size;
  // Pre-allocate logit output rows for all concurrent sequences so that
  // BatchDecodeStep() can call llama_get_logits_ith(ctx, i) correctly.
  ctx_params.n_seq_max =
      static_cast<uint32_t>(std::max(config.max_parallel_sequences, 1));

  // Wire Flash Attention (§2.7): set llama.cpp context parameter directly.
  // LLAMA_FLASH_ATTN_TYPE_ENABLED lets llama.cpp choose FA on any supported
  // backend (Metal, CUDA); the tile parameter is stored for future FA3 CUDA
  // integration.
  ctx_params.flash_attn_type = config.use_flash_attention
                                   ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                                   : LLAMA_FLASH_ATTN_TYPE_DISABLED;
  if (config.use_flash_attention) {
    std::cout << "[LlamaCPUBackend] FlashAttention enabled (tile="
              << config.flash_attention_tile << ")\n";
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

int LlamaCPUBackend::TokenCount(const std::string &text) const {
  auto tokens = Tokenize(text, /*add_bos=*/false);
  return static_cast<int>(tokens.size());
}

std::vector<int>
LlamaCPUBackend::TokenizeForCache(const std::string &prompt) const {
  // Returns the BPE token sequence (with BOS) for use in the KV prefix store.
  // Returns empty when no model is loaded so the caller can skip prefix reuse.
  return Tokenize(prompt, /*add_bos=*/true);
}

std::vector<int> LlamaCPUBackend::Tokenize(const std::string &prompt,
                                           bool add_bos) const {
  if (!model_) {
    return {};
  }
  if (!vocab_) {
    return {};
  }
  std::vector<llama_token> tokens;
  tokens.resize(prompt.size() + 8);
  int n = llama_tokenize(vocab_, prompt.c_str(), prompt.size(), tokens.data(),
                         tokens.size(), add_bos, true);
  if (n < 0) {
    tokens.resize(-n);
    n = llama_tokenize(vocab_, prompt.c_str(), prompt.size(), tokens.data(),
                       tokens.size(), add_bos, true);
  }
  tokens.resize(n);
  return {tokens.begin(), tokens.end()};
}

int LlamaCPUBackend::SampleGreedy() const {
  const float *logits = llama_get_logits(context_);
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
  int written =
      llama_token_to_piece(vocab_, token, buf.data(), buf.size(), 0, false);
  if (written < 0) {
    buf.resize(static_cast<std::size_t>(-written));
    if (llama_token_to_piece(vocab_, token, buf.data(), buf.size(), 0, false) <
        0) {
      return {};
    }
  } else {
    buf.resize(static_cast<std::size_t>(written));
  }
  return buf;
}

std::string LlamaCPUBackend::Generate(
    const std::string &prompt, int max_tokens,
    const std::function<bool(const std::string &)> &on_chunk,
    const std::function<bool()> &should_stop, int logprob_top_n,
    std::vector<TokenLogprob> *out_logprobs) {
  if (!IsReady()) {
    return {};
  }
  auto prompt_tokens = Tokenize(prompt, true);
  if (prompt_tokens.empty()) {
    return {};
  }

  int32_t batch_cap = std::max<int32_t>(
      config_.batch_size,
      static_cast<int32_t>(prompt_tokens.size() + std::max(max_tokens, 1)));
  llama_batch batch = llama_batch_init(batch_cap, 0, 1);
  llama_pos position = 0;
  for (size_t i = 0; i < prompt_tokens.size(); ++i) {
    BatchAdd(batch, prompt_tokens[i], position++,
             i == prompt_tokens.size() - 1);
  }
  if (llama_decode(context_, batch) != 0) {
    std::cerr << "[LlamaCPUBackend] llama_decode failed for prompt"
              << std::endl;
    llama_batch_free(batch);
    return {};
  }
  BatchClear(batch);

  std::string output;
  llama_token eos = llama_vocab_eos(vocab_);
  int tokens_remaining = std::max(max_tokens, 1);

  if (grammar_sampler_) {
    llama_sampler_reset(grammar_sampler_);
  }

  while (tokens_remaining-- > 0) {
    if (should_stop && should_stop()) {
      break;
    }
    int token = 0;
    if (grammar_sampler_) {
      token = llama_sampler_sample(grammar_sampler_, context_, -1);
    } else {
      token = SampleGreedy();
    }
    if (token == eos) {
      break;
    }
    std::string piece = TokenToString(token);
    // Collect logprob before the next llama_decode invalidates the logits ptr.
    if (out_logprobs) {
      out_logprobs->push_back(CollectLogprob(token, piece, logprob_top_n));
    }
    output += piece;
    if (on_chunk) {
      if (!on_chunk(piece)) {
        break;
      }
    }
    if (should_stop && should_stop()) {
      break;
    }
    // Context-window management: sliding-window KV eviction when the next
    // token would exceed n_ctx.  We discard the oldest half of the KV cache
    // and shift the remaining positions so generation can continue.
    {
      llama_pos n_ctx = static_cast<llama_pos>(llama_n_ctx(context_));
      if (position >= n_ctx - 1 && n_ctx > 1) {
        llama_pos keep = n_ctx / 2;
        llama_pos discard = position - keep + 1;
        if (discard > 0) {
          llama_memory_seq_rm(llama_get_memory(context_), 0, 0, discard);
          llama_memory_seq_add(llama_get_memory(context_), 0, discard,
                               static_cast<llama_pos>(-1), -discard);
          position -= discard;
        }
      }
    }
    BatchAdd(batch, token, position++, true);
    if (llama_decode(context_, batch) != 0) {
      std::cerr << "[LlamaCPUBackend] llama_decode failed while generating"
                << std::endl;
      break;
    }
    BatchClear(batch);
  }

  llama_batch_free(batch);
  if (grammar_sampler_) {
    llama_sampler_reset(grammar_sampler_);
  }
  return output;
}

bool LlamaCPUBackend::LoadMmproj(const std::filesystem::path &mmproj_path) {
#ifdef INFERFLUX_HAS_MTMD
  if (!model_) {
    std::cerr
        << "[LlamaCPUBackend] LoadMmproj: text model must be loaded first\n";
    return false;
  }
  if (mtmd_ctx_) {
    mtmd_free(mtmd_ctx_);
    mtmd_ctx_ = nullptr;
    vision_ready_ = false;
  }
  auto params = mtmd_context_params_default();
  params.use_gpu = (config_.gpu_layers > 0);
  params.n_threads = 4;
  params.print_timings = false;
  params.warmup = false;
  mtmd_ctx_ = mtmd_init_from_file(mmproj_path.string().c_str(), model_, params);
  if (!mtmd_ctx_) {
    std::cerr << "[LlamaCPUBackend] failed to load mmproj from " << mmproj_path
              << "\n";
    return false;
  }
  vision_ready_ = mtmd_support_vision(mtmd_ctx_);
  std::cout << "[LlamaCPUBackend] mmproj loaded from " << mmproj_path
            << " (vision=" << vision_ready_ << ")\n";
  return vision_ready_;
#else
  (void)mmproj_path;
  std::cerr
      << "[LlamaCPUBackend] ENABLE_MTMD=OFF; vision support unavailable\n";
  return false;
#endif
}

std::string LlamaCPUBackend::GenerateWithImages(
    const std::string &prompt, const std::vector<DecodedImage> &images,
    int max_tokens, const std::function<bool(const std::string &)> &on_chunk,
    const std::function<bool()> &should_stop) {
#ifdef INFERFLUX_HAS_MTMD
  if (!IsReady() || !vision_ready_ || !mtmd_ctx_ || images.empty()) {
    return Generate(prompt, max_tokens, on_chunk, should_stop);
  }

  // Build bitmap list from raw image bytes.
  std::vector<mtmd_bitmap *> bitmaps;
  bitmaps.reserve(images.size());
  for (const auto &img : images) {
    if (img.raw_bytes.empty())
      continue;
    auto *bmp = mtmd_helper_bitmap_init_from_buf(
        mtmd_ctx_, img.raw_bytes.data(), img.raw_bytes.size());
    if (!bmp) {
      std::cerr << "[LlamaCPUBackend] failed to decode image bitmap; falling "
                   "back to text-only\n";
      for (auto *b : bitmaps)
        mtmd_bitmap_free(b);
      return Generate(prompt, max_tokens, on_chunk, should_stop);
    }
    if (!img.image_id.empty()) {
      mtmd_bitmap_set_id(bmp, img.image_id.c_str());
    }
    bitmaps.push_back(bmp);
  }

  if (bitmaps.empty()) {
    return Generate(prompt, max_tokens, on_chunk, should_stop);
  }

  // Tokenize prompt + image bitmaps into interleaved chunks.
  auto *chunks = mtmd_input_chunks_init();
  mtmd_input_text input_text;
  input_text.text = prompt.c_str();
  input_text.add_special = true;
  input_text.parse_special = true;
  const mtmd_bitmap **bitmap_ptr =
      const_cast<const mtmd_bitmap **>(bitmaps.data());
  int32_t rc =
      mtmd_tokenize(mtmd_ctx_, chunks, &input_text, bitmap_ptr, bitmaps.size());
  for (auto *b : bitmaps)
    mtmd_bitmap_free(b);
  if (rc != 0) {
    std::cerr << "[LlamaCPUBackend] mtmd_tokenize failed (rc=" << rc
              << "); falling back\n";
    mtmd_input_chunks_free(chunks);
    return Generate(prompt, max_tokens, on_chunk, should_stop);
  }

  // Evaluate all chunks (text decode + image encode) through the model.
  llama_pos n_past = 0;
  int32_t n_batch = config_.batch_size;
  rc = mtmd_helper_eval_chunks(mtmd_ctx_, context_, chunks, n_past, 0, n_batch,
                               /*logits_last=*/true, &n_past);
  mtmd_input_chunks_free(chunks);
  if (rc != 0) {
    std::cerr << "[LlamaCPUBackend] mtmd_helper_eval_chunks failed (rc=" << rc
              << ")\n";
    return {};
  }

  // Decode loop: generate tokens starting from the updated n_past position.
  int32_t batch_cap = std::max<int32_t>(config_.batch_size, max_tokens + 1);
  llama_batch batch = llama_batch_init(batch_cap, 0, 1);

  std::string output;
  llama_token eos = llama_vocab_eos(vocab_);
  int tokens_remaining = std::max(max_tokens, 1);

  if (grammar_sampler_) {
    llama_sampler_reset(grammar_sampler_);
  }

  while (tokens_remaining-- > 0) {
    if (should_stop && should_stop())
      break;
    int token = 0;
    if (grammar_sampler_) {
      token = llama_sampler_sample(grammar_sampler_, context_, -1);
    } else {
      token = SampleGreedy();
    }
    if (token == eos)
      break;
    std::string piece = TokenToString(token);
    output += piece;
    if (on_chunk && !on_chunk(piece))
      break;
    if (should_stop && should_stop())
      break;
    BatchAdd(batch, token, n_past++, true);
    if (llama_decode(context_, batch) != 0) {
      std::cerr
          << "[LlamaCPUBackend] llama_decode failed (vision decode loop)\n";
      break;
    }
    BatchClear(batch);
  }

  llama_batch_free(batch);
  if (grammar_sampler_) {
    llama_sampler_reset(grammar_sampler_);
  }
  return output;
#else
  (void)images;
  return Generate(prompt, max_tokens, on_chunk, should_stop);
#endif
}

LlamaCPUBackend::PrefillResult
LlamaCPUBackend::Prefill(const std::string &prompt, int sequence_id) {
  if (!context_ || !vocab_) {
    return {};
  }
  auto prompt_tokens = Tokenize(prompt, /*add_bos=*/true);
  if (prompt_tokens.empty()) {
    return {};
  }
  // Clear any previous KV state for this sequence slot before filling it.
  llama_memory_seq_rm(llama_get_memory(context_),
                      static_cast<llama_seq_id>(sequence_id), -1, -1);

  int32_t batch_cap = std::max<int32_t>(
      config_.batch_size, static_cast<int32_t>(prompt_tokens.size() + 1));
  llama_batch batch = llama_batch_init(batch_cap, 0, 1);
  for (std::size_t i = 0; i < prompt_tokens.size(); ++i) {
    BatchAddSeq(batch, prompt_tokens[i], static_cast<llama_pos>(i),
                static_cast<llama_seq_id>(sequence_id),
                /*logits=*/i == prompt_tokens.size() - 1);
  }
  if (llama_decode(context_, batch) != 0) {
    std::cerr << "[LlamaCPUBackend] Prefill: llama_decode failed for seq "
              << sequence_id << std::endl;
    llama_batch_free(batch);
    return {};
  }
  // Sample the first output token while the logit buffer is fresh.
  // This must happen before any subsequent Prefill() call, which would
  // overwrite the buffer and cause a logit-buffer race in multi-seq batches.
  PrefillResult result;
  result.n_past = static_cast<int>(prompt_tokens.size());
  result.ok = true;
  llama_token eos = llama_vocab_eos(vocab_);
  int first_tok = SampleGreedy();
  if (first_tok >= 0 && first_tok != eos) {
    result.first_token = first_tok;
    result.first_piece = TokenToString(first_tok);
  }
  llama_batch_free(batch);
  return result;
}

void LlamaCPUBackend::CopySequencePrefix(int src_seq, int dst_seq,
                                         int n_tokens) {
  if (!context_)
    return;
  // Clear dst slot first so no stale KV cells survive from a previous request.
  llama_memory_seq_rm(llama_get_memory(context_),
                      static_cast<llama_seq_id>(dst_seq), -1, -1);
  // Copy positions [0, n_tokens) from src to dst.
  llama_memory_seq_cp(
      llama_get_memory(context_), static_cast<llama_seq_id>(src_seq),
      static_cast<llama_seq_id>(dst_seq), static_cast<llama_pos>(0),
      static_cast<llama_pos>(n_tokens));
}

LlamaCPUBackend::PrefillResult
LlamaCPUBackend::PrefillPartial(const std::string &prompt, int sequence_id,
                                int n_past_start) {
  if (!context_ || !vocab_)
    return {};
  auto prompt_tokens = Tokenize(prompt, /*add_bos=*/true);
  if (prompt_tokens.empty())
    return {};
  int n_total = static_cast<int>(prompt_tokens.size());
  if (n_past_start >= n_total) {
    // Prefix covers the entire prompt; no suffix tokens to evaluate and no
    // fresh logits from which to sample first_token.
    PrefillResult r;
    r.ok = true;
    r.n_past = n_total;
    return r;
  }
  int32_t suffix_len = n_total - n_past_start;
  int32_t batch_cap = std::max<int32_t>(config_.batch_size, suffix_len + 1);
  llama_batch batch = llama_batch_init(batch_cap, 0, 1);
  for (int i = n_past_start; i < n_total; ++i) {
    BatchAddSeq(batch, prompt_tokens[i], static_cast<llama_pos>(i),
                static_cast<llama_seq_id>(sequence_id),
                /*logits=*/i == n_total - 1);
  }
  if (llama_decode(context_, batch) != 0) {
    std::cerr << "[LlamaCPUBackend] PrefillPartial: llama_decode failed seq "
              << sequence_id << std::endl;
    llama_batch_free(batch);
    return {};
  }
  PrefillResult result;
  result.n_past = n_total;
  result.ok = true;
  llama_token eos = llama_vocab_eos(vocab_);
  int first_tok = SampleGreedy();
  if (first_tok >= 0 && first_tok != eos) {
    result.first_token = first_tok;
    result.first_piece = TokenToString(first_tok);
  }
  llama_batch_free(batch);
  return result;
}

std::string LlamaCPUBackend::Decode(
    int n_past, int sequence_id, int max_tokens,
    const std::function<bool(const std::string &)> &on_chunk,
    const std::function<bool()> &should_stop, int logprob_top_n,
    std::vector<TokenLogprob> *out_logprobs, int first_token) {
  if (!context_ || !vocab_ || n_past < 0) {
    return {};
  }
  int32_t batch_cap = std::max<int32_t>(config_.batch_size, max_tokens + 1);
  llama_batch batch = llama_batch_init(batch_cap, 0, 1);
  llama_pos position = static_cast<llama_pos>(n_past);

  std::string output;
  llama_token eos = llama_vocab_eos(vocab_);
  int tokens_remaining = std::max(max_tokens, 1);

  if (grammar_sampler_) {
    llama_sampler_reset(grammar_sampler_);
  }

  // If a first token was pre-sampled by Prefill() (to avoid the logit-buffer
  // race in multi-sequence prefill), emit and feed it before the loop.
  if (first_token >= 0 && first_token != eos && tokens_remaining > 0) {
    std::string piece = TokenToString(first_token);
    if (out_logprobs) {
      out_logprobs->push_back(
          CollectLogprob(first_token, piece, logprob_top_n));
    }
    output += piece;
    tokens_remaining--;
    if (on_chunk && !on_chunk(piece)) {
      llama_batch_free(batch);
      if (grammar_sampler_)
        llama_sampler_reset(grammar_sampler_);
      return output;
    }
    if (should_stop && should_stop()) {
      llama_batch_free(batch);
      if (grammar_sampler_)
        llama_sampler_reset(grammar_sampler_);
      return output;
    }
    BatchAddSeq(batch, first_token, position++,
                static_cast<llama_seq_id>(sequence_id), true);
    if (llama_decode(context_, batch) != 0) {
      llama_batch_free(batch);
      if (grammar_sampler_)
        llama_sampler_reset(grammar_sampler_);
      return output;
    }
    BatchClear(batch);
  }

  while (tokens_remaining-- > 0) {
    if (should_stop && should_stop())
      break;
    int token = 0;
    if (grammar_sampler_) {
      token = llama_sampler_sample(grammar_sampler_, context_, -1);
    } else {
      token = SampleGreedy();
    }
    if (token == eos)
      break;
    std::string piece = TokenToString(token);
    if (out_logprobs) {
      out_logprobs->push_back(CollectLogprob(token, piece, logprob_top_n));
    }
    output += piece;
    if (on_chunk && !on_chunk(piece))
      break;
    if (should_stop && should_stop())
      break;
    // Context-window management: sliding-window KV eviction (same as
    // Generate(), but scoped to the sequence slot for phased decode).
    {
      llama_pos n_ctx = static_cast<llama_pos>(llama_n_ctx(context_));
      if (position >= n_ctx - 1 && n_ctx > 1) {
        llama_pos keep = n_ctx / 2;
        llama_pos discard = position - keep + 1;
        if (discard > 0) {
          llama_memory_seq_rm(llama_get_memory(context_),
                              static_cast<llama_seq_id>(sequence_id), 0,
                              discard);
          llama_memory_seq_add(llama_get_memory(context_),
                               static_cast<llama_seq_id>(sequence_id), discard,
                               static_cast<llama_pos>(-1), -discard);
          position -= discard;
        }
      }
    }
    BatchAddSeq(batch, token, position++,
                static_cast<llama_seq_id>(sequence_id), true);
    if (llama_decode(context_, batch) != 0) {
      std::cerr << "[LlamaCPUBackend] Decode: llama_decode failed for seq "
                << sequence_id << std::endl;
      break;
    }
    BatchClear(batch);
  }

  llama_batch_free(batch);
  if (grammar_sampler_) {
    llama_sampler_reset(grammar_sampler_);
  }
  return output;
}

std::vector<LlamaCPUBackend::BatchDecodeOutput>
LlamaCPUBackend::BatchDecodeStep(std::vector<BatchDecodeInput> &inputs) {
  if (!context_ || !vocab_ || inputs.empty()) {
    return {};
  }
  int n = static_cast<int>(inputs.size());
  llama_batch batch = llama_batch_init(n, 0, 1);

  for (int i = 0; i < n; ++i) {
    auto &inp = inputs[i];
    // Per-sequence context-window eviction (mirrors the logic in Decode()).
    llama_pos n_ctx = static_cast<llama_pos>(llama_n_ctx(context_));
    if (inp.n_past >= static_cast<int>(n_ctx) - 1 && n_ctx > 1) {
      llama_pos keep = n_ctx / 2;
      llama_pos discard = static_cast<llama_pos>(inp.n_past) - keep + 1;
      if (discard > 0) {
        llama_memory_seq_rm(llama_get_memory(context_),
                            static_cast<llama_seq_id>(inp.sequence_id), 0,
                            discard);
        llama_memory_seq_add(llama_get_memory(context_),
                             static_cast<llama_seq_id>(inp.sequence_id),
                             discard, static_cast<llama_pos>(-1), -discard);
        inp.n_past -= static_cast<int>(discard);
      }
    }
    BatchAddSeq(batch, inp.feed_token, static_cast<llama_pos>(inp.n_past),
                static_cast<llama_seq_id>(inp.sequence_id), /*logits=*/true);
  }

  if (llama_decode(context_, batch) != 0) {
    std::cerr << "[LlamaCPUBackend] BatchDecodeStep: llama_decode failed\n";
    llama_batch_free(batch);
    return {};
  }
  llama_batch_free(batch);

  llama_token eos = llama_vocab_eos(vocab_);
  std::vector<BatchDecodeOutput> results(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    inputs[i].n_past++; // advance position for the next step
    const float *logits = llama_get_logits_ith(context_, i);
    if (!logits) {
      results[static_cast<std::size_t>(i)].token = -1;
      continue;
    }
    auto max_it = std::max_element(logits, logits + n_vocab_);
    int tok = static_cast<int>(std::distance(logits, max_it));
    if (tok == eos) {
      results[static_cast<std::size_t>(i)].token = -1;
    } else {
      results[static_cast<std::size_t>(i)].token = tok;
      results[static_cast<std::size_t>(i)].piece = TokenToString(tok);
    }
  }
  return results;
}

void LlamaCPUBackend::FreeSequence(int sequence_id) {
  if (!context_)
    return;
  llama_memory_seq_rm(llama_get_memory(context_),
                      static_cast<llama_seq_id>(sequence_id), -1, -1);
}

std::vector<uint8_t> LlamaCPUBackend::SerializeSequence(int sequence_id) {
  if (!context_)
    return {};
  std::size_t size = llama_state_seq_get_size(
      context_, static_cast<llama_seq_id>(sequence_id));
  if (size == 0)
    return {};
  std::vector<uint8_t> buf(size);
  std::size_t written = llama_state_seq_get_data(
      context_, buf.data(), buf.size(), static_cast<llama_seq_id>(sequence_id));
  if (written == 0)
    return {};
  buf.resize(written);
  return buf;
}

bool LlamaCPUBackend::HydrateSequence(int dest_sequence_id,
                                      const std::vector<uint8_t> &blob) {
  if (!context_ || blob.empty())
    return false;
  std::size_t read =
      llama_state_seq_set_data(context_, blob.data(), blob.size(),
                               static_cast<llama_seq_id>(dest_sequence_id));
  return read > 0;
}

bool LlamaCPUBackend::IsMoE() const { return ExpertCount() > 0; }

int LlamaCPUBackend::ExpertCount() const {
  if (!model_)
    return 0;
  char buf[32] = {};
  int32_t len =
      llama_model_meta_val_str(model_, "llm.expert_count", buf, sizeof(buf));
  if (len <= 0)
    return 0;
  return std::atoi(buf);
}

int LlamaCPUBackend::ActiveExperts() const {
  if (!model_)
    return 0;
  char buf[32] = {};
  int32_t len = llama_model_meta_val_str(model_, "llm.expert_used_count", buf,
                                         sizeof(buf));
  if (len <= 0)
    return 0;
  return std::atoi(buf);
}

void LlamaCPUBackend::EnableGrammarConstraint(const std::string &grammar,
                                              const std::string &root) {
  if (grammar.empty() || !context_ || !vocab_) {
    return;
  }
  DisableGrammarConstraint();
  auto params = llama_sampler_chain_default_params();
  auto *chain = llama_sampler_chain_init(params);
  llama_sampler_chain_add(
      chain, llama_sampler_init_grammar(vocab_, grammar.c_str(), root.c_str()));
  llama_sampler_chain_add(chain, llama_sampler_init_greedy());
  grammar_sampler_ = chain;
}

void LlamaCPUBackend::DisableGrammarConstraint() {
  if (grammar_sampler_) {
    llama_sampler_free(grammar_sampler_);
    grammar_sampler_ = nullptr;
  }
}

TokenLogprob LlamaCPUBackend::CollectLogprob(int token_id,
                                             const std::string &token_str,
                                             int top_n) const {
  const float *logits = llama_get_logits(context_);
  TokenLogprob tlp;
  tlp.token = token_str;

  if (!logits || n_vocab_ <= 0) {
    return tlp;
  }
  if (token_id < 0 || token_id >= n_vocab_) {
    return tlp; // sampler returned out-of-range token; return zero logprob
  }

  // Numerically stable log-softmax: subtract max before exp.
  float max_l = logits[0];
  for (int i = 1; i < n_vocab_; ++i) {
    if (logits[i] > max_l)
      max_l = logits[i];
  }
  double sum_exp = 0.0;
  for (int i = 0; i < n_vocab_; ++i) {
    sum_exp += std::exp(static_cast<double>(logits[i] - max_l));
  }
  float log_denom = static_cast<float>(std::log(sum_exp)) + max_l;

  tlp.logprob = logits[token_id] - log_denom;

  // UTF-8 bytes of the token string.
  tlp.bytes.reserve(token_str.size());
  for (unsigned char c : token_str) {
    tlp.bytes.push_back(static_cast<int>(c));
  }

  if (top_n > 0) {
    int k = std::min(top_n, n_vocab_);
    std::vector<int> indices(n_vocab_);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });
    tlp.top_logprobs.reserve(k);
    for (int i = 0; i < k; ++i) {
      int alt_id = indices[i];
      tlp.top_logprobs.push_back(
          {TokenToString(alt_id), logits[alt_id] - log_denom});
    }
  }
  return tlp;
}

// §2.3 — model-native chat template formatting.
// Uses llama_chat_apply_template() from llama.h (built into the llama lib,
// no extra compilation units required).  The function reads the model's
// built-in chat template from GGUF metadata (NULL tmpl argument selects the
// model's own template).  Supported models include Llama 3.x, Mistral/Mixtral,
// Qwen 2/2.5, Hermes, DeepSeek, Phi-3, ChatML, and others in the predefined
// list; for unsupported models it returns valid=false and the caller falls
// back to the plain text preamble.
LlamaCPUBackend::ChatTemplateResult LlamaCPUBackend::FormatChatMessages(
    const std::vector<std::pair<std::string, std::string>> &messages,
    bool add_assistant_prefix) {
  ChatTemplateResult result;
  if (!model_ || messages.empty()) {
    return result;
  }

  // Keep content strings alive for the duration of the C-struct array.
  std::vector<std::string> contents;
  contents.reserve(messages.size());
  std::vector<llama_chat_message> chat;
  chat.reserve(messages.size());
  for (const auto &[role, content] : messages) {
    contents.push_back(content);
    chat.push_back({role.c_str(), contents.back().c_str()});
  }

  // Initial buffer: 2× total content length + headroom for role tokens.
  int total_chars = 0;
  for (const auto &[r, c] : messages) {
    total_chars += static_cast<int>(r.size() + c.size());
  }
  int buf_size = std::max(4096, total_chars * 2 + 512);
  std::vector<char> buf(buf_size);

  // NULL template → use the model's built-in template.
  int32_t n =
      llama_chat_apply_template(nullptr, chat.data(), chat.size(),
                                add_assistant_prefix, buf.data(), buf_size);
  if (n < 0) {
    // Template not in the predefined list; caller falls back to preamble.
    return result;
  }
  if (n > buf_size) {
    // Buffer was too small; retry with exact size.
    buf.resize(static_cast<std::size_t>(n) + 1);
    n = llama_chat_apply_template(nullptr, chat.data(), chat.size(),
                                  add_assistant_prefix, buf.data(), n);
    if (n < 0) {
      return result;
    }
  }

  result.prompt = std::string(buf.data(), static_cast<std::size_t>(n));
  result.valid = true;
  return result;
}

} // namespace inferflux
