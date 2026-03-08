#pragma once

#include "runtime/backends/cpu/llama_backend.h"

#include <atomic>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>

namespace inferflux {

class NativeCudaRuntime;

// NativeCudaBackend is the provider entry point for the InferFlux native CUDA
// runtime.
// PHASE 1 NOTE: Still inherits from LlamaCPUBackend due to extensive method
// set.
// TODO: Phase 1.5 - Extract ILlamaBackend interface with full method set
class NativeCudaBackend : public LlamaCPUBackend {
public:
  NativeCudaBackend();
  ~NativeCudaBackend() override;

  bool LoadModel(const std::filesystem::path &model_path,
                 const LlamaBackendConfig &config = {}) override;

#ifdef INFERFLUX_USE_COMMON_BACKEND_TYPES
  std::string Name() const override { return "native_cuda"; }
#endif

  std::vector<UnifiedBatchOutput>
  ExecuteUnifiedBatch(const std::vector<UnifiedBatchInput> &inputs) override;

  bool SupportsAsyncUnifiedBatch() const override;
  UnifiedBatchHandle SubmitUnifiedBatchAsync(
      const std::vector<UnifiedBatchInput> &inputs,
      UnifiedBatchLane lane = UnifiedBatchLane::kAuto) override;
  bool TryCollectUnifiedBatchAsync(
      UnifiedBatchHandle handle,
      std::vector<UnifiedBatchOutput> *outputs) override;
  int UnifiedBatchTokenCapacity() const override;
  PrefillResult Prefill(const std::string &prompt, int sequence_id) override;
  PrefillResult PrefillPartial(const std::string &prompt, int sequence_id,
                               int n_past_start) override;
  void CopySequencePrefix(int src_seq, int dst_seq, int n_tokens) override;
  void FreeSequence(int sequence_id) override;
  std::vector<uint8_t> SerializeSequence(int sequence_id) override;
  bool HydrateSequence(int dest_sequence_id,
                       const std::vector<uint8_t> &blob) override;
  std::string
  Decode(int n_past, int sequence_id, int max_tokens,
         const std::function<bool(const std::string &, const TokenLogprob *)>
             &on_chunk = {},
         const std::function<bool()> &should_stop = {}, int logprob_top_n = 0,
         std::vector<TokenLogprob> *out_logprobs = nullptr,
         int first_token = -1,
         const std::vector<std::string> &stop_seqs = {}) override;
  std::string
  Generate(const std::string &prompt, int max_tokens,
           const std::function<bool(const std::string &, const TokenLogprob *)>
               &on_chunk = {},
           const std::function<bool()> &should_stop = {}, int logprob_top_n = 0,
           std::vector<TokenLogprob> *out_logprobs = nullptr,
           const std::vector<std::string> &stop_seqs = {}) override;
  void SetupSampler(const std::string &grammar, const std::string &root,
                    const SamplingParams &sp) override;
  void TeardownSampler() override;
  PerfSnapshot TakePerf() override;
  ChatTemplateResult FormatChatMessages(
      const std::vector<std::pair<std::string, std::string>> &messages,
      bool add_assistant_prefix = true) override;
  std::vector<float> EmbedForParity(const std::string &text);
  int EmbedDimsForParity() const;
  int TokenCount(const std::string &text) const override;
  std::vector<int> TokenizeForCache(const std::string &prompt) const override;
  bool IsReady() const override;

  std::string ExecutorKind() const { return runtime_kind_; }
  bool IsFallbackExecutor() const { return fallback_mode_; }
  const std::string &FallbackReason() const { return fallback_reason_; }

  // Endpoint capability contracts for native provider routing.
  virtual bool SupportsLogprobsContract() const;
  virtual bool SupportsStructuredOutputContract() const;
  virtual bool SupportsEmbeddingsContract() const;
  virtual bool SupportsSpeculativeDecodingContract() const;

  // Returns true when native CUDA kernels are compiled and CUDA runtime is
  // available.
  static bool NativeKernelsReady();

private:
  bool IsParityDelegateAvailable() const;
  std::shared_ptr<LlamaCPUBackend> EnsureParityBackend() const;
  std::shared_ptr<LlamaCPUBackend> DelegateBackend() const;
  bool UsesStructuredConstraintSampler() const;
  SamplingParams SnapshotSamplingParams() const;
  static int AcquireEphemeralSequenceId();

  std::unique_ptr<NativeCudaRuntime> runtime_;
  std::filesystem::path loaded_model_path_;
  LlamaBackendConfig loaded_config_{};
  std::filesystem::path parity_load_path_;
  bool parity_delegate_enabled_{true};
  mutable bool parity_delegate_available_{false};
  mutable bool parity_delegate_init_attempted_{false};
  mutable std::shared_ptr<LlamaCPUBackend> parity_backend_;
  std::string runtime_kind_{"native_cuda"};
  bool fallback_mode_{true};
  std::string fallback_reason_;
  mutable std::mutex parity_backend_mutex_;
  mutable std::mutex sampling_mutex_;
  SamplingParams active_sampling_{};
  bool sampling_active_{false};
  bool structured_constraint_sampler_active_{false};
  static std::atomic<int> next_ephemeral_sequence_id_;
};

} // namespace inferflux
