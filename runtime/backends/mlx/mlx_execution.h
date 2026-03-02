#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "runtime/backends/mlx/mlx_loader.h"
#include "runtime/logprob.h"
#include "scheduler/request_batch.h"

namespace inferflux {

struct MlxSamplerOutput {
  std::string text;
  std::vector<int32_t> ids;
  bool ok{false};
};

// ---------------------------------------------------------------------------
// MlxExecutionEngine
//
// Implements the LLaMA/Mistral transformer forward pass via the mlx-c API.
// Holds a per-layer KV cache that grows with each Step() call.
//
// Single-sequence lifecycle (Generate() path):
//   Initialize()              — acquire GPU stream
//   LoadWeights(store, cfg)   — wire in weight tensors + model config
//   Reset()                   — clear KV cache (call before each new prompt)
//   Step(token_ids, sp)       — forward pass, returns sampled token id
//   Shutdown()                — release GPU resources
//
// Multi-sequence phased lifecycle (Prefill/Decode path, INF-8):
//   AllocSlot(seq_id)         — create an empty KV slot for a sequence
//   StepSeq(seq_id, ...)      — forward pass bound to a specific slot
//   SeqNPast(seq_id)          — n_past for a specific slot
//   CopySlotPrefix(src,dst,n) — copy first n tokens of KV from src to dst
//   FreeSlot(seq_id)          — release a slot's GPU memory
// ---------------------------------------------------------------------------
class MlxExecutionEngine {
public:
  MlxExecutionEngine() = default;
  ~MlxExecutionEngine();

  bool Initialize();
  void Shutdown();

  // Wire materialised weights and model config into the engine.
  // `store` must outlive the engine.
  bool LoadWeights(const MlxWeightStore &store, const MlxModelConfig &cfg);

  // True after a successful LoadWeights() call.
  bool WeightsLoaded() const { return weights_ != nullptr; }

  // Clear the single-sequence KV cache and reset position.
  // Does NOT touch phased sequence slots.
  void Reset();

  // Run one forward pass over token_ids starting at position n_past_.
  // Returns the sampled next token ID (greedy when temperature <= 0), or -1
  // on error. When logprob_top_n > 0, logprob entries are appended to
  // *out_logprobs.
  int32_t Step(const std::vector<int32_t> &token_ids,
               const SamplingParams &sp = {}, int logprob_top_n = 0,
               std::vector<TokenLogprob> *out_logprobs = nullptr);

  // Current KV cache length (tokens processed so far) for the single-seq path.
  int NPast() const { return n_past_; }

  // ── Phased / multi-sequence slot API (INF-8) ──────────────────────────────

  // Allocate an empty KV slot for seq_id. No-op if already allocated.
  void AllocSlot(int seq_id);

  // Free a slot's GPU arrays and remove it from the slot table.
  void FreeSlot(int seq_id);

  // Copy the first n_tokens of KV state from src_seq into dst_seq.
  // dst_seq is overwritten. Both slots must be allocated (or src == active).
  // Sets dst_seq.n_past = n_tokens.
  void CopySlotPrefix(int src_seq, int dst_seq, int n_tokens);

  // Run a forward pass for a specific sequence slot.
  // Saves the current active slot (if any) and loads seq_id's slot before
  // calling Forward().  Returns the sampled token or -1 on error.
  int32_t StepSeq(int seq_id, const std::vector<int32_t> &token_ids,
                  const SamplingParams &sp = {}, int logprob_top_n = 0,
                  std::vector<TokenLogprob> *out_logprobs = nullptr);

  // Return n_past for a specific sequence slot (-1 if not allocated).
  int SeqNPast(int seq_id) const;

private:
#ifdef INFERFLUX_HAS_MLX
  int32_t Forward(const std::vector<int32_t> &token_ids,
                  const SamplingParams &sp, int logprob_top_n,
                  std::vector<TokenLogprob> *out_logprobs);

  const MlxWeightStore *weights_{nullptr};
  MlxModelConfig config_;
  mlx_stream stream_{};

  // Flat (single-sequence) KV cache: [1, n_kv_heads, n_past, head_dim].
  // Also used as the "working" cache when a phased slot is active.
  std::vector<mlx_array> key_cache_;
  std::vector<mlx_array> val_cache_;

  int n_past_{0};
  bool initialized_{false};

  // Sampling state: PRNG + penalty lookback window.
  std::mt19937 rng_;
  bool rng_seeded_{false};
  std::vector<int32_t> token_history_; // penalty lookback window

  // ── Phased slot management ─────────────────────────────────────────────
  struct SlotState {
    std::vector<mlx_array>
        key_cache; // per-layer KV (may be empty = not yet used)
    std::vector<mlx_array> val_cache;
    int n_past{0};
    std::mt19937 rng;
    bool rng_seeded{false};
    std::vector<int32_t> token_history;
  };

  // active_seq_ == -1 means no phased slot is loaded into the flat members.
  int active_seq_{-1};
  std::unordered_map<int, SlotState> slots_;

  // Save the current flat state back to slots_[active_seq_].  No-op when
  // active_seq_ == -1.  Leaves key_cache_/val_cache_ in a moved-from state
  // (re-resized to num_hidden_layers with null entries) so Forward() can be
  // called after LoadSlot().
  void SaveActiveSlot();

  // Load seq_id's slot into the flat members.  Calls SaveActiveSlot() first
  // if a different slot is currently active.  Creates an empty slot if seq_id
  // is not yet in slots_.
  void LoadSlot(int seq_id);
#endif
};

} // namespace inferflux
