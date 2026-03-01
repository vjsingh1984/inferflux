#pragma once

#include "runtime/multimodal/image_preprocessor.h"

#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <llama.h>

#ifdef INFERFLUX_HAS_MTMD
#include "tools/mtmd/mtmd.h"
#include "tools/mtmd/mtmd-helper.h"
#endif

namespace inferflux {

struct LlamaBackendConfig {
  int32_t ctx_size = 2048;
  int32_t batch_size = 512;
  int gpu_layers = 0;
  bool use_flash_attention = false;
  int flash_attention_tile = 128;
  std::string mmproj_path;  // Path to multimodal projector; empty = vision disabled.
};

class LlamaCPUBackend {
 public:
  LlamaCPUBackend();
  ~LlamaCPUBackend();

  bool LoadModel(const std::filesystem::path& model_path, const LlamaBackendConfig& config = {});

  // Load a multimodal projector (mmproj) GGUF alongside the text model.
  // Must be called after LoadModel(). Returns false if ENABLE_MTMD is off.
  bool LoadMmproj(const std::filesystem::path& mmproj_path);

  // True when a mmproj has been loaded and supports vision input.
  bool SupportsVision() const { return vision_ready_; }

  // Result of a phased prefill pass (§2.5 Option A).
  struct PrefillResult {
    int n_past{0};   // KV position after prompt evaluation (= prompt token count).
    bool ok{false};  // false on error: context not ready, or llama_decode failed.
  };

  // Phased prefill/decode API (§2.5 Option A — in-process disaggregated execution).
  // Each concurrent request gets a unique sequence_id (use request_id % kMaxSequenceSlots).
  // Prefill() clears the sequence slot first, so reuse of the same id is safe.

  // Evaluate all prompt tokens for sequence_id and populate the KV cache.
  // Returns {ok=false} when the context is not loaded or the prompt is empty.
  PrefillResult Prefill(const std::string& prompt, int sequence_id);

  // Autoregressive decode starting from n_past (returned by Prefill) for sequence_id.
  // Grammar constraints must be set via EnableGrammarConstraint before calling.
  std::string Decode(int n_past,
                     int sequence_id,
                     int max_tokens,
                     const std::function<bool(const std::string&)>& on_chunk = {},
                     const std::function<bool()>& should_stop = {});

  // Release KV cache slots for the given sequence_id.
  void FreeSequence(int sequence_id);

  // Serialize the KV cache state for sequence_id to a byte buffer.
  // Returns an empty vector when the context is not loaded or serialization fails.
  // Used for cross-process KV transfer (§2.5 disaggregated path).
  std::vector<uint8_t> SerializeSequence(int sequence_id);

  // Restore KV cache state for dest_sequence_id from a previously serialized buffer.
  // Returns false when the context is not loaded, the buffer is empty, or restore fails.
  bool HydrateSequence(int dest_sequence_id, const std::vector<uint8_t>& blob);

  // MoE detection helpers (§2.6).  Return 0 / false when the model is not loaded
  // or the GGUF metadata key is absent (i.e., the model is not a MoE model).
  bool IsMoE() const;
  int ExpertCount() const;
  int ActiveExperts() const;

  std::string Generate(const std::string& prompt,
                       int max_tokens,
                       const std::function<bool(const std::string&)>& on_chunk = {},
                       const std::function<bool()>& should_stop = {});

  // Vision-aware generation. Prompt must contain <__media__> markers matching images.
  // Falls back to Generate() when vision is not ready or images is empty.
  std::string GenerateWithImages(const std::string& prompt,
                                 const std::vector<DecodedImage>& images,
                                 int max_tokens,
                                 const std::function<bool(const std::string&)>& on_chunk = {},
                                 const std::function<bool()>& should_stop = {});

  void EnableGrammarConstraint(const std::string& grammar, const std::string& root);
  void DisableGrammarConstraint();
  bool IsReady() const { return context_ != nullptr || test_ready_; }

  // Flash Attention (§2.7): returns true when FA was requested in LlamaBackendConfig
  // and the context was successfully created with LLAMA_FLASH_ATTN_TYPE_ENABLED.
  bool FlashAttentionEnabled() const { return config_.use_flash_attention; }

  int TokenCount(const std::string& text) const;
  void ForceReadyForTests() { test_ready_ = true; }

 private:
  std::vector<int> Tokenize(const std::string& prompt, bool add_bos) const;
  std::string TokenToString(int token) const;
  int SampleGreedy() const;

  llama_model* model_{nullptr};
  llama_context* context_{nullptr};
  const struct llama_vocab* vocab_{nullptr};
  int32_t n_vocab_{0};
  LlamaBackendConfig config_;
  bool test_ready_{false};
  struct llama_sampler* grammar_sampler_{nullptr};

#ifdef INFERFLUX_HAS_MTMD
  mtmd_context* mtmd_ctx_{nullptr};
#endif
  bool vision_ready_{false};
};

}  // namespace inferflux
