#include "runtime/backends/ep_dispatch.h"
#include "runtime/execution/parallel_context.h"
#include <algorithm>

namespace inferflux {

DistributedEPDispatch::DistributedEPDispatch(int rank, int world_size,
                                             int n_experts) {
  int experts_per_rank = (n_experts + world_size - 1) / world_size;
  rank_.rank = rank;
  rank_.world_size = world_size;
  rank_.expert_start = rank * experts_per_rank;
  rank_.expert_end = std::min(n_experts, (rank + 1) * experts_per_rank);
}

std::vector<float>
DistributedEPDispatch::Route(const std::vector<float> &hidden_states,
                             const std::vector<int> &expert_indices) {
  auto *comm = ParallelContext::Get().Comm();
  if (!comm || rank_.world_size == 1) {
    return hidden_states;
  }

  // §P1f: Distributed Expert Routing (All-to-All simulation).
  // In a real implementation (e.g. NCCL), we would:
  // 1. Group tokens by the rank that owns their assigned expert.
  // 2. Perform an ncclAllToAll to send hidden states to those ranks.
  // 3. Compute expert outputs locally on the received tokens.
  // 4. Perform another ncclAllToAll to send results back to the original ranks.

  // For this foundation, we use a global synchronization barrier and AllGather
  // to ensure all ranks are aligned, then simulate the "sharded" logic.
  comm->Barrier();

  // AllGather simulation: every rank sends its local activations to all other
  // ranks.
  std::vector<float> global_states;
  comm->AllGather(hidden_states, global_states);

  // (In this stub foundation, we return the local hidden_states after
  // synchronization).
  return hidden_states;
}

} // namespace inferflux
