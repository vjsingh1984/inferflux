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

// Canonical request->unified-batch metadata projection used by both the
// scheduler-side phased prefill path and executor-side unified batching.
template <typename BatchInput = UnifiedBatchInput>
inline BatchInput
MakeUnifiedBatchInput(const InferenceRequest &request, int sequence_id,
                      uint64_t sequence_generation, int n_past,
                      std::vector<int> tokens, bool request_logits) {
  BatchInput input;
  input.sequence_id = sequence_id;
  input.n_past = n_past;
  input.tokens = std::move(tokens);
  input.request_logits = request_logits;
  input.sampling = request.sampling;
  input.request_id = static_cast<int64_t>(request.id);
  input.client_request_id = request.client_request_id;
  input.sequence_generation = sequence_generation;
  return input;
}

template <typename BatchInput = UnifiedBatchInput>
inline BatchInput
MakeUnifiedBatchInput(const InferenceRequest &request, int n_past,
                      std::vector<int> tokens, bool request_logits) {
  return MakeUnifiedBatchInput<BatchInput>(
      request, request.sequence_id, request.sequence_generation, n_past,
      std::move(tokens), request_logits);
}

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
