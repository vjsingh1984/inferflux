#pragma once

#include "runtime/backends/common/backend_types.h"
#include "runtime/backends/cpu/llama_backend.h"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace inferflux {

class NativeCudaRuntime {
public:
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

  virtual ~NativeCudaRuntime() = default;

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

  struct NativePerfSnapshot {
    double prefill_ms{0.0};
    double decode_ms{0.0};
    int32_t prompt_tokens{0};
    int32_t generated_tokens{0};
  };
  virtual NativePerfSnapshot NativeTakePerf() { return {}; }

  virtual std::vector<int> NativeTokenize(const std::string &prompt) const {
    auto backend = BackendHandle();
    return backend ? backend->TokenizeForCache(prompt) : std::vector<int>{};
  }

  virtual int NativeTokenCount(const std::string &text) const {
    auto backend = BackendHandle();
    return backend ? backend->TokenCount(text) : 0;
  }

  virtual bool NativeIsReady() const {
    auto backend = BackendHandle();
    return backend && backend->IsReady();
  }

  virtual void NativeFreeSequence(int sequence_id) {
    auto backend = BackendHandle();
    if (backend) {
      backend->FreeSequence(sequence_id);
    }
  }

  virtual void NativeCopySequencePrefix(int src_seq, int dst_seq,
                                        int n_tokens) {
    auto backend = BackendHandle();
    if (backend) {
      backend->CopySequencePrefix(src_seq, dst_seq, n_tokens);
    }
  }
};

std::unique_ptr<NativeCudaRuntime> CreateNativeCudaRuntime();

} // namespace inferflux
