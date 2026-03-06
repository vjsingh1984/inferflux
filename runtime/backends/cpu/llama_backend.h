#pragma once

#include "runtime/backends/common/backend_interface.h"
#include "runtime/backends/ep_dispatch.h"
#include "runtime/logprob.h"
#include "runtime/multimodal/image_preprocessor.h"
#include "scheduler/request_batch.h"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
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
  // CUDA attention kernel policy:
  // auto | fa3 | fa2 | standard
  std::string cuda_attention_kernel{"auto"};
  // CUDA phase-overlap scaffold (foundation for native async overlap).
  // When enabled, CUDA backend can split mixed unified batches into decode-
  // first and prefill lanes to reduce decode head-of-line blocking.
  bool cuda_phase_overlap_scaffold{false};
  // Minimum count of prefill tokens in a mixed batch before split kicks in.
  int cuda_phase_overlap_min_prefill_tokens{256};
  // Optional dual-context overlap mode: run prefill lane on a separate CUDA
  // context and hand off KV to decode lane via
  // SerializeSequence/HydrateSequence. Disabled by default because it increases
  // memory footprint.
  bool cuda_phase_overlap_prefill_replica{false};
  // KV cache precision policy for native CUDA runtime:
  // auto | fp16 | bf16 | int8 | fp8
  // `auto` keeps current behavior (match inference dtype).
  std::string native_kv_cache_dtype{"auto"};
  std::string
      mmproj_path; // Path to multimodal projector; empty = vision disabled.
  // Maximum number of KV-cache sequences that can be live simultaneously.
  // Increased from 16 to 128 for production concurrent workloads.
  // Managed by SequenceSlotManager for timeout-based eviction.
  int max_parallel_sequences{128};

  // Distributed Parallelism Degrees (§P1e).
  int tp_degree{1}; // Tensor Parallel degree
  int pp_degree{1}; // Pipeline Parallel degree
};

// When INFERFLUX_USE_COMMON_BACKEND_TYPES is enabled, LlamaCPUBackend
// inherits from BackendInterface and uses common types. When disabled,
// it maintains backward compatibility with nested types.
#ifdef INFERFLUX_USE_COMMON_BACKEND_TYPES
class LlamaCPUBackend : public BackendInterface {
#else
class LlamaCPUBackend {
#endif
public:
  LlamaCPUBackend();
  virtual ~LlamaCPUBackend();

#ifdef INFERFLUX_USE_COMMON_BACKEND_TYPES
  // Type aliases for backward compatibility with code that references
  // LlamaCPUBackend::UnifiedBatchInput, etc.
  using UnifiedBatchInput = ::inferflux::UnifiedBatchInput;
  using UnifiedBatchOutput = ::inferflux::UnifiedBatchOutput;
  using UnifiedBatchLane = ::inferflux::UnifiedBatchLane;
  using UnifiedBatchHandle = ::inferflux::UnifiedBatchHandle;
  using PrefillResult = ::inferflux::PrefillResult;
#endif

#ifdef INFERFLUX_USE_COMMON_BACKEND_TYPES
  bool LoadModel(const std::filesystem::path &model_path,
                 const LlamaBackendConfig &config = {}) override;
#else
  virtual bool LoadModel(const std::filesystem::path &model_path,
                         const LlamaBackendConfig &config = {});
#endif

  // Load a multimodal projector (mmproj) GGUF alongside the text model.
  // Must be called after LoadModel(). Returns false if ENABLE_MTMD is off.
  bool LoadMmproj(const std::filesystem::path &mmproj_path);

  // True when a mmproj has been loaded and supports vision input.
  bool SupportsVision() const { return vision_ready_; }

#ifndef INFERFLUX_USE_COMMON_BACKEND_TYPES
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
#endif // INFERFLUX_USE_COMMON_BACKEND_TYPES

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

#ifndef INFERFLUX_USE_COMMON_BACKEND_TYPES
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

  // Execution lane hint for async unified-batch submission.
  // kDecode should be favored for lower token latency.
  enum class UnifiedBatchLane {
    kAuto,
    kDecode,
    kPrefill,
  };
  using UnifiedBatchHandle = uint64_t;
#endif // INFERFLUX_USE_COMMON_BACKEND_TYPES

  // Execute a mixed batch of prefill and decode sequences.
  // Returns results for all inputs where request_logits=true.
#ifdef INFERFLUX_USE_COMMON_BACKEND_TYPES
  std::vector<UnifiedBatchOutput>
  ExecuteUnifiedBatch(const std::vector<UnifiedBatchInput> &inputs) override;
#else
  virtual std::vector<UnifiedBatchOutput>
  ExecuteUnifiedBatch(const std::vector<UnifiedBatchInput> &inputs);
#endif

  // Backend-agnostic async unified batch contract.
  // Base implementation executes synchronously and stores the result for
  // later collection through TryCollectUnifiedBatchAsync().
#ifdef INFERFLUX_USE_COMMON_BACKEND_TYPES
  bool SupportsAsyncUnifiedBatch() const override;
  UnifiedBatchHandle SubmitUnifiedBatchAsync(
      const std::vector<UnifiedBatchInput> &inputs,
      UnifiedBatchLane lane = UnifiedBatchLane::kAuto) override;
  bool TryCollectUnifiedBatchAsync(
      UnifiedBatchHandle handle,
      std::vector<UnifiedBatchOutput> *outputs) override;
  // Maximum number of tokens the backend can safely accept in one unified
  // batch step. Used by scheduler/executor-side chunking guards.
  int UnifiedBatchTokenCapacity() const override;
#else
  virtual bool SupportsAsyncUnifiedBatch() const;
  virtual UnifiedBatchHandle
  SubmitUnifiedBatchAsync(const std::vector<UnifiedBatchInput> &inputs,
                          UnifiedBatchLane lane = UnifiedBatchLane::kAuto);
  virtual bool
  TryCollectUnifiedBatchAsync(UnifiedBatchHandle handle,
                              std::vector<UnifiedBatchOutput> *outputs);
  // Maximum number of tokens the backend can safely accept in one unified
  // batch step. Used by scheduler/executor-side chunking guards.
  virtual int UnifiedBatchTokenCapacity() const;
#endif

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
         int first_token = -1,
         const std::vector<std::string> &stop_seqs =
             {}); // NOLINT(bugprone-easily-swappable-parameters)

  // Execute one shared decode step for N sequences simultaneously.
  std::vector<BatchDecodeOutput>
  BatchDecodeStep(std::vector<BatchDecodeInput> &inputs);

  // Copy KV cache entries for positions [0, n_tokens) from src_seq to dst_seq.
  virtual void CopySequencePrefix(int src_seq, int dst_seq, int n_tokens);

  // Partial prefill that evaluates only the suffix of prompt starting at
  // n_past_start.
  virtual PrefillResult PrefillPartial(
      const std::string &prompt, int sequence_id,
      int n_past_start); // NOLINT(bugprone-easily-swappable-parameters)

  // Release KV cache slots for the given sequence_id.
  virtual void FreeSequence(int sequence_id);

  // Serialize the KV cache state for sequence_id to a byte buffer.
  virtual std::vector<uint8_t> SerializeSequence(int sequence_id);

  // Restore KV cache state for dest_sequence_id from a previously serialized
  // buffer.
  virtual bool HydrateSequence(int dest_sequence_id,
                               const std::vector<uint8_t> &blob);

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

#ifdef INFERFLUX_USE_COMMON_BACKEND_TYPES
  // BackendInterface implementation
  std::string Name() const override;
  bool IsFallback() const override { return false; }
  const std::string &FallbackReason() const override {
    static const std::string empty_reason;
    return empty_reason;
  }
#endif

protected:
  struct llama_sampler *active_sampler_{nullptr};
  std::shared_ptr<EPDispatch> ep_dispatch_;
  int tp_rank_{0};

private:
  void TeardownSamplerImpl();
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
  mutable std::mutex async_results_mutex_;
  UnifiedBatchHandle next_async_handle_{1};
  std::unordered_map<UnifiedBatchHandle, std::vector<UnifiedBatchOutput>>
      async_results_;

#ifdef INFERFLUX_HAS_MTMD
  mtmd_context *mtmd_ctx_{nullptr};
#endif
  bool vision_ready_{false};
};

} // namespace inferflux
