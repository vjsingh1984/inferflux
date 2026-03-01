#pragma once

#include "runtime/logprob.h"
#include "runtime/multimodal/image_preprocessor.h"

#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <llama.h>

#ifdef INFERFLUX_HAS_MTMD
#include "tools/mtmd/mtmd-helper.h"
#include "tools/mtmd/mtmd.h"
#endif

namespace inferflux {

struct LlamaBackendConfig {
  int32_t ctx_size = 2048;
  int32_t batch_size = 512;
  int gpu_layers = 0;
  bool use_flash_attention = false;
  int flash_attention_tile = 128;
  std::string
      mmproj_path; // Path to multimodal projector; empty = vision disabled.
  // Maximum number of KV-cache sequences that can be live simultaneously.
  // Must be >= kMaxSequenceSlots (16) for multi-sequence batch decode.
  int max_parallel_sequences{16};
};

class LlamaCPUBackend {
public:
  LlamaCPUBackend();
  ~LlamaCPUBackend();

  bool LoadModel(const std::filesystem::path &model_path,
                 const LlamaBackendConfig &config = {});

  // Load a multimodal projector (mmproj) GGUF alongside the text model.
  // Must be called after LoadModel(). Returns false if ENABLE_MTMD is off.
  bool LoadMmproj(const std::filesystem::path &mmproj_path);

  // True when a mmproj has been loaded and supports vision input.
  bool SupportsVision() const { return vision_ready_; }

  // Result of a phased prefill pass (§2.5 Option A).
  struct PrefillResult {
    int n_past{
        0}; // KV position after prompt evaluation (= prompt token count).
    bool ok{
        false}; // false on error: context not ready, or llama_decode failed.
    // First output token sampled from prefill logits while they are fresh.
    // -1 means EOS was the first token (empty generation) or an error occurred.
    // Storing this here avoids a logit-buffer race when multiple sequences are
    // prefilled sequentially: the second Prefill() overwrites the logit buffer,
    // so seq0's first token must be captured before seq1 is prefilled.
    int first_token{-1};
    std::string first_piece; // text of first_token; empty when first_token==-1
  };

  // Input/output for one step of multi-sequence batch decode.
  // BatchDecodeStep() feeds feed_token at n_past for each sequence, runs one
  // llama_decode() call covering all sequences, then samples the next token for
  // each sequence using llama_get_logits_ith().  n_past is updated in-place.
  struct BatchDecodeInput {
    int sequence_id;
    int n_past; // current KV position; incremented in-place by BatchDecodeStep
    int feed_token; // token to insert at n_past (from prior step / Prefill)
  };
  struct BatchDecodeOutput {
    int token{-1};     // next sampled token; -1 = EOS or error
    std::string piece; // text of token; empty when token == -1
  };

  // Phased prefill/decode API (§2.5 Option A — in-process disaggregated
  // execution). Each concurrent request gets a unique sequence_id (use
  // request_id % kMaxSequenceSlots). Prefill() clears the sequence slot first,
  // so reuse of the same id is safe.

  // Evaluate all prompt tokens for sequence_id and populate the KV cache.
  // Returns {ok=false} when the context is not loaded or the prompt is empty.
  PrefillResult Prefill(const std::string &prompt, int sequence_id);

  // Autoregressive decode starting from n_past (returned by Prefill) for
  // sequence_id. Grammar constraints must be set via EnableGrammarConstraint
  // before calling.  logprob_top_n / out_logprobs work identically to Generate.
  // first_token: when >= 0, this token (pre-sampled by Prefill while logits
  // were fresh) is emitted and fed back before the auto-regressive loop starts,
  // correcting the logit-buffer race that arises after multi-sequence prefill.
  std::string
  Decode(int n_past, int sequence_id, int max_tokens,
         const std::function<bool(const std::string &)> &on_chunk = {},
         const std::function<bool()> &should_stop = {}, int logprob_top_n = 0,
         std::vector<TokenLogprob> *out_logprobs = nullptr,
         int first_token = -1);

  // Execute one shared decode step for N sequences simultaneously.
  // For each input, inserts feed_token at n_past, calls one llama_decode()
  // covering all N sequences, then samples the next token per sequence using
  // llama_get_logits_ith().  inputs[i].n_past is incremented in-place.
  // Returns empty vector on context-not-loaded or llama_decode failure.
  std::vector<BatchDecodeOutput>
  BatchDecodeStep(std::vector<BatchDecodeInput> &inputs);

  // Copy KV cache entries for positions [0, n_tokens) from src_seq to dst_seq.
  // dst_seq is cleared first so no stale cells remain.  Used by the KV prefix
  // reuse path: CopySequencePrefix + PrefillPartial replaces a full Prefill
  // call when a warm matching prefix is available in the prefix store.
  // No-op when the context is not loaded.
  void CopySequencePrefix(int src_seq, int dst_seq, int n_tokens);

  // Partial prefill that evaluates only the suffix of prompt starting at
  // n_past_start.  Positions [0, n_past_start) must already be populated in
  // sequence_id's KV slot (via CopySequencePrefix).  Returns {ok=false} when
  // the context is not loaded, the prompt is empty, or llama_decode fails.
  // If n_past_start >= total prompt token count, returns
  // {ok=true, n_past=count, first_token=-1}: the prefix already covers the
  // full prompt so no suffix evaluation is needed (and no fresh logits are
  // available to sample a first token).
  PrefillResult PrefillPartial(const std::string &prompt, int sequence_id,
                               int n_past_start);

  // Release KV cache slots for the given sequence_id.
  void FreeSequence(int sequence_id);

  // Serialize the KV cache state for sequence_id to a byte buffer.
  // Returns an empty vector when the context is not loaded or serialization
  // fails. Used for cross-process KV transfer (§2.5 disaggregated path).
  std::vector<uint8_t> SerializeSequence(int sequence_id);

  // Restore KV cache state for dest_sequence_id from a previously serialized
  // buffer. Returns false when the context is not loaded, the buffer is empty,
  // or restore fails.
  bool HydrateSequence(int dest_sequence_id, const std::vector<uint8_t> &blob);

  // MoE detection helpers (§2.6).  Return 0 / false when the model is not
  // loaded or the GGUF metadata key is absent (i.e., the model is not a MoE
  // model).
  bool IsMoE() const;
  int ExpertCount() const;
  int ActiveExperts() const;

  // Generate completion for `prompt`.  When `logprob_top_n > 0`, one
  // TokenLogprob entry per generated token is appended to *out_logprobs
  // (logprob_top_n alternatives are stored in TokenLogprob::top_logprobs).
  // Pass nullptr to disable collection (default).
  std::string
  Generate(const std::string &prompt, int max_tokens,
           const std::function<bool(const std::string &)> &on_chunk = {},
           const std::function<bool()> &should_stop = {}, int logprob_top_n = 0,
           std::vector<TokenLogprob> *out_logprobs = nullptr);

  // Vision-aware generation. Prompt must contain <__media__> markers matching
  // images. Falls back to Generate() when vision is not ready or images is
  // empty.
  std::string GenerateWithImages(
      const std::string &prompt, const std::vector<DecodedImage> &images,
      int max_tokens,
      const std::function<bool(const std::string &)> &on_chunk = {},
      const std::function<bool()> &should_stop = {});

  void EnableGrammarConstraint(const std::string &grammar,
                               const std::string &root);
  void DisableGrammarConstraint();
  bool IsReady() const { return context_ != nullptr || test_ready_; }

  // Returns the effective context size (in tokens) for the loaded model.
  // Returns 0 when no model is loaded.
  int ContextSize() const {
    return context_ ? static_cast<int>(llama_n_ctx(context_)) : 0;
  }

  // Flash Attention (§2.7): returns true when FA was requested in
  // LlamaBackendConfig and the context was successfully created with
  // LLAMA_FLASH_ATTN_TYPE_ENABLED.
  bool FlashAttentionEnabled() const { return config_.use_flash_attention; }

  // §2.3 — model-native chat template formatting.
  // Format a sequence of {role, content} message pairs using the model's
  // built-in chat template (read from GGUF metadata via
  // llama_chat_apply_template). When add_assistant_prefix=true the returned
  // string ends with the model-specific prefix tokens that start the assistant
  // turn, which is what inference expects.  Returns {valid=false} when no model
  // is loaded, the template is unsupported, or messages is empty.
  struct ChatTemplateResult {
    bool valid{false};
    std::string prompt;
  };
  ChatTemplateResult FormatChatMessages(
      const std::vector<std::pair<std::string, std::string>> &messages,
      bool add_assistant_prefix = true);

  int TokenCount(const std::string &text) const;
  void ForceReadyForTests() { test_ready_ = true; }

  // Returns the BPE token ID vector for `prompt` (with BOS prepended).
  // Used by the scheduler's KV prefix store so that prefix matching is done
  // in BPE-token space instead of via the SimpleTokenizer proxy, avoiding the
  // boundary-mismatch that arises when the two tokenizers disagree on word
  // splits.  Returns an empty vector when no model is loaded (null context).
  std::vector<int> TokenizeForCache(const std::string &prompt) const;

private:
  std::vector<int> Tokenize(const std::string &prompt, bool add_bos) const;
  std::string TokenToString(int token) const;
  int SampleGreedy() const;

  // Build a TokenLogprob from the current context_ logits for `token_id`.
  // Computes log-softmax and optionally finds top-`top_n` alternatives.
  // Must be called immediately after llama_decode() and before the next decode.
  TokenLogprob CollectLogprob(int token_id, const std::string &token_str,
                              int top_n) const;

  llama_model *model_{nullptr};
  llama_context *context_{nullptr};
  const struct llama_vocab *vocab_{nullptr};
  int32_t n_vocab_{0};
  LlamaBackendConfig config_;
  bool test_ready_{false};
  struct llama_sampler *grammar_sampler_{nullptr};

#ifdef INFERFLUX_HAS_MTMD
  mtmd_context *mtmd_ctx_{nullptr};
#endif
  bool vision_ready_{false};
};

} // namespace inferflux
