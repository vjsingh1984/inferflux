#pragma once

#include "backend_types.h"
#include <algorithm>
#include <vector>

namespace inferflux {

/// Utility class for analyzing and manipulating unified batches.
///
/// This class provides static methods for common batch operations that
/// are shared across multiple backends (CPU, CUDA, ROCm, Metal).
class BatchAnalyzer {
public:
  /// Determine if an input is a prefill operation (multiple tokens).
  /// @param input Batch input to analyze
  /// @return true if input has more than one token
  static bool IsPrefillLikeInput(const UnifiedBatchInput &input) {
    return input.tokens.size() > 1;
  }

  /// Determine if an input is a decode operation (single token).
  /// @param input Batch input to analyze
  /// @return true if input has exactly one token
  static bool IsDecodeLikeInput(const UnifiedBatchInput &input) {
    return input.tokens.size() == 1;
  }

  /// Check if a batch contains only prefill operations.
  /// @param inputs Batch to analyze
  /// @return true if all inputs are prefill (or batch is empty)
  static bool IsPrefillOnlyBatch(const std::vector<UnifiedBatchInput> &inputs) {
    if (inputs.empty()) {
      return false;
    }
    for (const auto &input : inputs) {
      if (!IsPrefillLikeInput(input)) {
        return false;
      }
    }
    return true;
  }

  /// Check if a batch contains only decode operations.
  /// @param inputs Batch to analyze
  /// @return true if all inputs are decode (or batch is empty)
  static bool IsDecodeOnlyBatch(const std::vector<UnifiedBatchInput> &inputs) {
    if (inputs.empty()) {
      return false;
    }
    for (const auto &input : inputs) {
      if (!IsDecodeLikeInput(input)) {
        return false;
      }
    }
    return true;
  }

  /// Check if a batch contains both prefill and decode operations.
  /// @param inputs Batch to analyze
  /// @return true if batch has mixed workload
  static bool HasMixedWorkload(const std::vector<UnifiedBatchInput> &inputs) {
    bool has_prefill = false;
    bool has_decode = false;

    for (const auto &input : inputs) {
      if (IsPrefillLikeInput(input)) {
        has_prefill = true;
      } else {
        has_decode = true;
      }
      // Early exit if we found both
      if (has_prefill && has_decode) {
        return true;
      }
    }
    return false;
  }

  /// Split a batch into prefill and decode indices.
  /// @param inputs Batch to split
  /// @param prefill_indices Output: indices of prefill inputs
  /// @param decode_indices Output: indices of decode inputs
  static void SplitBatchByType(
      const std::vector<UnifiedBatchInput> &inputs,
      std::vector<size_t> &prefill_indices,
      std::vector<size_t> &decode_indices) {

    prefill_indices.clear();
    decode_indices.clear();

    for (size_t i = 0; i < inputs.size(); ++i) {
      if (IsPrefillLikeInput(inputs[i])) {
        prefill_indices.push_back(i);
      } else {
        decode_indices.push_back(i);
      }
    }
  }

  /// Count total tokens in a batch.
  /// @param inputs Batch to analyze
  /// @return Total number of tokens across all inputs
  static size_t CountTotalTokens(const std::vector<UnifiedBatchInput> &inputs) {
    size_t total = 0;
    for (const auto &input : inputs) {
      total += input.tokens.size();
    }
    return total;
  }

  /// Check if a batch exceeds a token capacity limit.
  /// @param inputs Batch to check
  /// @param max_tokens Maximum allowed tokens
  /// @return true if batch would exceed capacity
  static bool ExceedsTokenCapacity(const std::vector<UnifiedBatchInput> &inputs,
                                   size_t max_tokens) {
    return CountTotalTokens(inputs) > max_tokens;
  }
};

} // namespace inferflux
