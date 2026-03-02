#pragma once

#include "runtime/backends/ep_dispatch.h"
#include "runtime/logprob.h"
#include "runtime/multimodal/image_preprocessor.h"
#include "scheduler/request_batch.h"

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

  // Distributed Parallelism Degrees (§P1e).
  int tp_degree{1}; // Tensor Parallel degree
  int pp_degree{1}; // Pipeline Parallel degree
};

class LlamaCPUBackend {
public:
  LlamaCPUBackend();
  virtual ~LlamaCPUBackend();

  virtual bool LoadModel(const std::filesystem::path &model_path,
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

  // Input/output for one step of unified batch execution (§P1b).
  // A single ExecuteUnifiedBatch() call can mix prefill (multiple tokens,
  // n_past=0) and decode (one token, n_past>0) sequences in the same
  // forward pass.
  struct UnifiedBatchInput {
    int sequence_id;
    int n_past;
    std::vector<int> tokens;
    bool request_logits{true}; // true to sample a token after this step
    SamplingParams sampling;   // Per-request sampling parameters (§P1b)
  };
  struct UnifiedBatchOutput {
    int token{-1};
    std::string piece;
    bool ok{false};
  };

  // Execute a mixed batch of prefill and decode sequences.
  // Returns results for all inputs where request_logits=true.
  virtual std::vector<UnifiedBatchOutput>
  ExecuteUnifiedBatch(const std::vector<UnifiedBatchInput> &inputs);

  // Evaluate all prompt tokens for sequence_id and populate the KV cache.
  virtual PrefillResult Prefill(const std::string &prompt, int sequence_id);

  // Autoregressive decode starting from n_past (returned by Prefill) for
  // sequence_id. Grammar constraints must be set via EnableGrammarConstraint
  // before calling.  logprob_top_n / out_logprobs work identically to Generate.
  virtual std::string
  Decode(int n_past, int sequence_id, int max_tokens,
         const std::function<bool(const std::string &, const TokenLogprob *)>
             &on_chunk = {},
         const std::function<bool()> &should_stop = {}, int logprob_top_n = 0,
         std::vector<TokenLogprob> *out_logprobs = nullptr,
         int first_token = -1, const std::vector<std::string> &stop_seqs = {});

  // Execute one shared decode step for N sequences simultaneously.
  std::vector<BatchDecodeOutput>
  BatchDecodeStep(std::vector<BatchDecodeInput> &inputs);

  // Copy KV cache entries for positions [0, n_tokens) from src_seq to dst_seq.
  virtual void CopySequencePrefix(int src_seq, int dst_seq, int n_tokens);

  // Partial prefill that evaluates only the suffix of prompt starting at
  // n_past_start.
  virtual PrefillResult PrefillPartial(const std::string &prompt,
                                       int sequence_id, int n_past_start);

  // Release KV cache slots for the given sequence_id.
  virtual void FreeSequence(int sequence_id);

  // Serialize the KV cache state for sequence_id to a byte buffer.
  std::vector<uint8_t> SerializeSequence(int sequence_id);

  // Restore KV cache state for dest_sequence_id from a previously serialized
  // buffer.
  bool HydrateSequence(int dest_sequence_id, const std::vector<uint8_t> &blob);

  bool IsMoE() const;
  int ExpertCount() const;
  int ActiveExperts() const;

  // Generate completion for `prompt`.
  virtual std::string
  Generate(const std::string &prompt, int max_tokens,
           const std::function<bool(const std::string &, const TokenLogprob *)>
               &on_chunk = {},
           const std::function<bool()> &should_stop = {}, int logprob_top_n = 0,
           std::vector<TokenLogprob> *out_logprobs = nullptr,
           const std::vector<std::string> &stop_seqs = {});

  // Vision-aware generation.
  std::string GenerateWithImages(
      const std::string &prompt, const std::vector<DecodedImage> &images,
      int max_tokens,
      const std::function<bool(const std::string &, const TokenLogprob *)>
          &on_chunk = {},
      const std::function<bool()> &should_stop = {},
      const std::vector<std::string> &stop_seqs = {});

  // Set up a unified sampler chain for grammar + sampling params.
  virtual void SetupSampler(const std::string &grammar, const std::string &root,
                            const SamplingParams &sp);
  virtual void TeardownSampler();

  // Backward-compat wrappers (grammar-only, default SamplingParams).
  void EnableGrammarConstraint(const std::string &grammar,
                               const std::string &root);
  void DisableGrammarConstraint();

  virtual bool IsReady() const { return context_ != nullptr || test_ready_; }

  // Returns the effective context size (in tokens) for the loaded model.
  int ContextSize() const {
    return context_ ? static_cast<int>(llama_n_ctx(context_)) : 0;
  }

  bool FlashAttentionEnabled() const { return config_.use_flash_attention; }

  struct PerfSnapshot {
    double prefill_ms{0};
    double decode_ms{0};
    int32_t prompt_tokens{0};
    int32_t generated_tokens{0};
  };

  virtual PerfSnapshot TakePerf();

  struct ChatTemplateResult {
    bool valid{false};
    std::string prompt;
  };
  virtual ChatTemplateResult FormatChatMessages(
      const std::vector<std::pair<std::string, std::string>> &messages,
      bool add_assistant_prefix = true);

  virtual int TokenCount(const std::string &text) const;
  void ForceReadyForTests() { test_ready_ = true; }

  std::vector<float> Embed(const std::string &text);
  int EmbedDims() const;

  virtual std::vector<int> TokenizeForCache(const std::string &prompt) const;

  // Rank within the Tensor Parallel group.
  int TPRank() const { return tp_rank_; }

protected:
  struct llama_sampler *active_sampler_{nullptr};
  std::shared_ptr<EPDispatch> ep_dispatch_;
  int tp_rank_{0};

private:
  std::vector<int> Tokenize(const std::string &prompt, bool add_bos) const;
  std::string TokenToString(int token) const;

  TokenLogprob CollectLogprob(int token_id, const std::string &token_str,
                              int top_n) const;

  llama_model *model_{nullptr};
  llama_context *context_{nullptr};
  const struct llama_vocab *vocab_{nullptr};
  int32_t n_vocab_{0};
  LlamaBackendConfig config_;
  bool test_ready_{false};
  PerfSnapshot last_perf_{};
  llama_context *embed_ctx_{nullptr};
  bool EnsureEmbedCtx();

#ifdef INFERFLUX_HAS_MTMD
  mtmd_context *mtmd_ctx_{nullptr};
#endif
  bool vision_ready_{false};
};

} // namespace inferflux
