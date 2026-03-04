#pragma once

#include "backend_types.h"
#include <filesystem>
#include <memory>
#include <vector>

namespace inferflux {

// Forward declarations
struct LlamaBackendConfig;

/// Abstract interface for all inference backends.
///
/// This interface defines the contract that all backends (CPU, CUDA, ROCm,
/// Metal) must implement. It focuses on the unified batching API and async
/// execution, leaving backend-specific methods (Prefill, Decode, etc.) to
/// subclass interfaces.
class BackendInterface {
public:
  virtual ~BackendInterface() = default;

  // ========================================================================
  // Model Loading
  // ========================================================================

  /// Load model from the given path with the specified configuration.
  /// @param model_path Path to model file (GGUF, safetensors, etc.)
  /// @param config Backend-specific configuration
  /// @return true on success, false on failure
  virtual bool LoadModel(const std::filesystem::path &model_path,
                         const LlamaBackendConfig &config) = 0;

  // ========================================================================
  // Unified Batch Execution (Core API)
  // ========================================================================

  /// Execute a mixed batch of prefill and decode sequences.
  /// Returns results for all inputs where request_logits=true.
  /// @param inputs Batch of sequences to execute
  /// @return Vector of outputs (one per input where request_logits=true)
  virtual std::vector<UnifiedBatchOutput>
  ExecuteUnifiedBatch(const std::vector<UnifiedBatchInput> &inputs) = 0;

  /// Maximum number of tokens the backend can safely accept in one unified
  /// batch. Used by scheduler/executor-side chunking guards.
  /// @return Maximum token capacity
  virtual int UnifiedBatchTokenCapacity() const {
    return 2048; // Default safe limit
  }

  // ========================================================================
  // Async Execution
  // ========================================================================

  /// Whether this backend supports async unified batch execution.
  /// @return true if SupportsAsyncSubmit/TryCollect are implemented
  virtual bool SupportsAsyncUnifiedBatch() const {
    return false; // Default: synchronous only
  }

  /// Submit a batch for async execution.
  /// @param inputs Batch to execute
  /// @param lane Execution lane hint (kAuto, kPrefill, or kDecode)
  /// @return Handle for later collection (or 0 if sync-only)
  virtual UnifiedBatchHandle
  SubmitUnifiedBatchAsync(const std::vector<UnifiedBatchInput> &inputs,
                          UnifiedBatchLane lane = UnifiedBatchLane::kAuto) {
    return 0; // Default: not supported
  }

  /// Try to collect results from a previously submitted async batch.
  /// @param handle Handle returned by SubmitUnifiedBatchAsync
  /// @param outputs Output vector to populate if ready
  /// @return true if outputs were collected, false if not ready or handle
  /// invalid
  virtual bool
  TryCollectUnifiedBatchAsync(UnifiedBatchHandle handle,
                              std::vector<UnifiedBatchOutput> *outputs) {
    return false; // Default: not supported
  }

  // ========================================================================
  // Metadata and Diagnostics
  // ========================================================================

  /// Human-readable name of this backend (e.g., "llama_cpp_cpu",
  /// "native_cuda").
  virtual std::string Name() const = 0;

  /// Whether this backend is a fallback (degraded mode).
  virtual bool IsFallback() const { return false; }

  /// Human-readable reason for fallback status.
  virtual const std::string &FallbackReason() const {
    static const std::string empty_reason;
    return empty_reason;
  }
};

} // namespace inferflux
