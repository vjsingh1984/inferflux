#pragma once

#include "runtime/backends/common/backend_interface.h"
#include "runtime/backends/common/backend_types.h"
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
  // When INFERFLUX_USE_COMMON_BACKEND_TYPES is enabled, use common types
  // directly. Otherwise, use type aliases to LlamaCPUBackend for backward
  // compatibility.
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

  // Native perf snapshot for backends without a llama.cpp BackendHandle().
  struct NativePerfSnapshot {
    double prefill_ms{0.0};
    double decode_ms{0.0};
    int32_t prompt_tokens{0};
    int32_t generated_tokens{0};
  };
  virtual NativePerfSnapshot NativeTakePerf() { return {}; }

  // Native* methods: allow NativeKernelExecutor to provide tokenization,
  // readiness, and sequence management without a llama.cpp BackendHandle().
  // Default implementations delegate to BackendHandle() for backward compat.
  virtual std::vector<int> NativeTokenize(const std::string &prompt) const {
    auto bh = BackendHandle();
    return bh ? bh->TokenizeForCache(prompt) : std::vector<int>{};
  }
  virtual int NativeTokenCount(const std::string &text) const {
    auto bh = BackendHandle();
    return bh ? bh->TokenCount(text) : 0;
  }
  virtual bool NativeIsReady() const {
    auto bh = BackendHandle();
    return bh && bh->IsReady();
  }
  virtual void NativeFreeSequence(int sequence_id) {
    auto bh = BackendHandle();
    if (bh)
      bh->FreeSequence(sequence_id);
  }
  virtual void NativeCopySequencePrefix(int src_seq, int dst_seq,
                                        int n_tokens) {
    auto bh = BackendHandle();
    if (bh)
      bh->CopySequencePrefix(src_seq, dst_seq, n_tokens);
  }
};

std::unique_ptr<NativeCudaExecutor>
CreateNativeCudaExecutor(const std::string &executor_hint);

} // namespace inferflux
