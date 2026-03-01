#pragma once

// §2.6 Expert Parallelism (EP) dispatch interface — stub.
//
// This header defines the interface that an EP-aware router would implement to
// partition MoE expert layers across multiple devices or processes.  The full
// implementation requires multi-GPU topology discovery and a mechanism to split
// llama.cpp expert layers by rank (e.g., via NCCL or a shared-memory ring).
//
// Current status: stub.  The MoE metadata (ModelInfo::is_moe, n_experts,
// n_active_experts) is now populated by SingleModelRouter from GGUF metadata,
// and RecordMoERequest() increments the Prometheus counter on each MoE
// dispatch.  Actual expert partitioning is tracked in TechDebt §2.6.

#include <memory>
#include <string>
#include <vector>

namespace inferflux {

class LlamaCPUBackend;

// A rank in an EP group: identifies which experts [expert_start, expert_end)
// this worker is responsible for.
struct EPRank {
  int rank{0};
  int world_size{1};
  int expert_start{0};  // inclusive
  int expert_end{0};    // exclusive
};

// EPDispatch routes MoE forward passes across an EP group.
// Future implementation will shard expert layers and all-reduce activations.
class EPDispatch {
 public:
  virtual ~EPDispatch() = default;

  // Returns the local rank configuration for this worker.
  virtual EPRank LocalRank() const = 0;

  // Returns true if this worker owns expert `expert_id`.
  virtual bool OwnsExpert(int expert_id) const = 0;

  // Stub: returns the name of this dispatch strategy (e.g., "local", "nccl").
  virtual std::string Name() const = 0;
};

// LocalEPDispatch: single-process stub — owns all experts, world_size=1.
// Used when no multi-device EP topology is configured.
class LocalEPDispatch : public EPDispatch {
 public:
  explicit LocalEPDispatch(int n_experts)
      : rank_{0, 1, 0, n_experts} {}

  EPRank LocalRank() const override { return rank_; }
  bool OwnsExpert(int expert_id) const override {
    return rank_.expert_end > 0 &&
           expert_id >= rank_.expert_start &&
           expert_id < rank_.expert_end;
  }
  std::string Name() const override { return "local"; }

 private:
  EPRank rank_;
};

}  // namespace inferflux
