#pragma once

#include "runtime/backends/common/backend_interface.h"
#include "runtime/backends/cpu/llama_backend.h"
#include "runtime/backends/cuda/cuda_backend.h"
#include "runtime/backends/llama/llama_backend_traits.h"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace inferflux {

class NativeCudaExecutor {
public:
  // When INFERFLUX_USE_COMMON_BACKEND_TYPES is enabled, use common types directly.
  // Otherwise, use type aliases to LlamaCPUBackend for backward compatibility.
#ifdef INFERFLUX_USE_COMMON_BACKEND_TYPES
  using UnifiedBatchHandle = ::inferflux::UnifiedBatchHandle;
  using UnifiedBatchInput = ::inferflux::UnifiedBatchInput;
  using UnifiedBatchLane = ::inferflux::UnifiedBatchLane;
  using UnifiedBatchOutput = ::inferflux::UnifiedBatchOutput;
#else
  using UnifiedBatchHandle = LlamaCPUBackend::UnifiedBatchHandle;
  using UnifiedBatchInput = LlamaCPUBackend::UnifiedBatchInput;
  using UnifiedBatchLane = LlamaCPUBackend::UnifiedBatchLane;
  using UnifiedBatchOutput = LlamaCPUBackend::UnifiedBatchOutput;
#endif

  virtual ~NativeCudaExecutor() = default;

  virtual std::string Name() const = 0;
  virtual bool IsFallback() const = 0;
  virtual const std::string &FallbackReason() const = 0;

  virtual bool LoadModel(const std::filesystem::path &model_path,
                         const LlamaBackendConfig &config) = 0;
  virtual std::vector<UnifiedBatchOutput>
  ExecuteUnifiedBatch(const std::vector<UnifiedBatchInput> &inputs) = 0;
  virtual bool SupportsAsyncUnifiedBatch() const = 0;
  virtual UnifiedBatchHandle
  SubmitUnifiedBatchAsync(const std::vector<UnifiedBatchInput> &inputs,
                          UnifiedBatchLane lane) = 0;
  virtual bool
  TryCollectUnifiedBatchAsync(UnifiedBatchHandle handle,
                              std::vector<UnifiedBatchOutput> *outputs) = 0;
  virtual std::shared_ptr<LlamaCPUBackend> BackendHandle() const = 0;
};

std::unique_ptr<NativeCudaExecutor>
CreateNativeCudaExecutor(const std::string &executor_hint);

} // namespace inferflux
