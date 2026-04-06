#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "scheduler/request_batch.h" // For SamplingParams

namespace inferflux {

// ============================================================================
// Unified Batch Types
// ============================================================================

/// Input for one sequence in a unified batch execution.
/// A single ExecuteUnifiedBatch() call can mix prefill (multiple tokens,
/// n_past=0) and decode (one token, n_past>0) sequences in the same forward
/// pass.
struct UnifiedBatchInput {
  int sequence_id{0};
  int n_past{0};
  std::vector<int> tokens;
  bool request_logits{true};
  SamplingParams sampling; // Per-request sampling parameters
  int64_t request_id{-1};
  std::string client_request_id;
  uint64_t sequence_generation{0};
};

/// Output for one sequence in a unified batch execution.
struct UnifiedBatchOutput {
  int token{-1};     // Next sampled token; -1 = EOS or error
  std::string piece; // Text of token; empty when token == -1
  bool ok{false};    // true if token was successfully sampled
};

/// Execution lane hint for async unified-batch submission.
/// kDecode should be favored for lower token latency.
enum class UnifiedBatchLane {
  kAuto,    // Let backend decide (default)
  kDecode,  // Decode lane (lower latency priority)
  kPrefill, // Prefill lane (higher throughput priority)
};

/// Handle for tracking async batch submissions.
/// Used with SubmitUnifiedBatchAsync/TryCollectUnifiedBatchAsync.
using UnifiedBatchHandle = uint64_t;

/// Result of a phased prefill pass (for backends that support phased
/// execution).
struct PrefillResult {
  int n_past{0};           ///< KV position after prompt evaluation
  bool ok{false};          ///< true on success
  int first_token{-1};     ///< First output token sampled from prefill logits
  std::string first_piece; ///< Text of first_token
};

} // namespace inferflux
