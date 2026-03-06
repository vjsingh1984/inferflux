#include "runtime/backends/cuda/native_cuda_backend.h"

#include "runtime/backends/cuda/native_cuda_runtime.h"
#include "server/logging/logger.h"

#ifdef INFERFLUX_HAS_CUDA
#include <cuda_runtime_api.h>
#endif

#include <cctype>
#include <cstdlib>
#include <string>

namespace inferflux {

namespace {

bool ParseBoolValue(const char *raw) {
  if (!raw) {
    return false;
  }
  std::string lowered(raw);
  for (auto &ch : lowered) {
    ch = static_cast<char>(
        std::tolower(static_cast<unsigned char>(ch)));
  }
  return lowered == "1" || lowered == "true" || lowered == "yes" ||
         lowered == "on";
}

bool ParseBoolEnv(const char *name, bool default_value) {
  const char *raw = std::getenv(name);
  if (!raw) {
    return default_value;
  }
  return ParseBoolValue(raw);
}

} // namespace

NativeCudaBackend::NativeCudaBackend() = default;

NativeCudaBackend::~NativeCudaBackend() = default;

bool NativeCudaBackend::LoadModel(const std::filesystem::path &model_path,
                                  const LlamaBackendConfig &config) {
#ifdef INFERFLUX_HAS_CUDA
  const bool strict_native_execution =
      ParseBoolEnv("INFERFLUX_NATIVE_CUDA_STRICT", false);

  if (!NativeKernelsReady()) {
    log::Error("native_cuda_backend",
               "native CUDA backend requested but native kernels are not "
               "ready");
    return false;
  }

  runtime_ = CreateNativeCudaRuntime();
  if (!runtime_) {
    log::Error("native_cuda_backend",
               "failed to create native CUDA runtime");
    return false;
  }

  if (!runtime_->LoadModel(model_path, config)) {
    log::Error("native_cuda_backend",
               "failed to load model using runtime '" + runtime_->Name() + "'");
    return false;
  }

  fallback_mode_ = runtime_->IsFallback();
  fallback_reason_ = runtime_->FallbackReason();
  runtime_kind_ = runtime_->Name();
  if (strict_native_execution && fallback_mode_) {
    std::string strict_reason =
        fallback_reason_.empty()
            ? "native CUDA strict mode rejected runtime '" + runtime_kind_ + "'"
            : fallback_reason_;
    log::Error("native_cuda_backend",
               "native CUDA strict mode rejected model load: " + strict_reason);
    return false;
  }
  if (fallback_mode_ && !fallback_reason_.empty()) {
    log::Warn("native_cuda_backend", fallback_reason_ +
                                      " (runtime=" + runtime_kind_ + ")");
  }
  return true;
#else
  (void)model_path;
  (void)config;
  log::Error("native_cuda_backend",
             "native CUDA backend requested but binary was built without CUDA "
             "support");
  return false;
#endif
}

bool NativeCudaBackend::NativeKernelsReady() {
#ifdef INFERFLUX_NATIVE_KERNELS_READY
#ifdef INFERFLUX_HAS_CUDA
  // Allow explicit opt-out for emergency fallback operation.
  if (ParseBoolValue(std::getenv("INFERFLUX_DISABLE_NATIVE_CUDA"))) {
    return false;
  }

  int device_count = 0;
  const cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess) {
    return false;
  }
  return device_count > 0;
#else
  return false;
#endif
#else
  return false;
#endif
}

std::vector<LlamaCPUBackend::UnifiedBatchOutput>
NativeCudaBackend::ExecuteUnifiedBatch(
    const std::vector<UnifiedBatchInput> &inputs) {
  if (!runtime_) {
    return {};
  }
  return runtime_->ExecuteUnifiedBatch(inputs);
}

bool NativeCudaBackend::SupportsAsyncUnifiedBatch() const {
  if (!runtime_) {
    return false;
  }
  return runtime_->SupportsAsyncUnifiedBatch();
}

LlamaCPUBackend::UnifiedBatchHandle NativeCudaBackend::SubmitUnifiedBatchAsync(
    const std::vector<UnifiedBatchInput> &inputs, UnifiedBatchLane lane) {
  if (!runtime_) {
    return 0;
  }
  return runtime_->SubmitUnifiedBatchAsync(inputs, lane);
}

bool NativeCudaBackend::TryCollectUnifiedBatchAsync(
    UnifiedBatchHandle handle, std::vector<UnifiedBatchOutput> *outputs) {
  if (!runtime_) {
    return false;
  }
  return runtime_->TryCollectUnifiedBatchAsync(handle, outputs);
}

int NativeCudaBackend::UnifiedBatchTokenCapacity() const {
  auto backend = DelegateBackend();
  if (!backend) {
    return LlamaCPUBackend::UnifiedBatchTokenCapacity();
  }
  return backend->UnifiedBatchTokenCapacity();
}

LlamaCPUBackend::PrefillResult
NativeCudaBackend::Prefill(const std::string &prompt, int sequence_id) {
  auto backend = DelegateBackend();
  if (!backend) {
    return {};
  }
  return backend->Prefill(prompt, sequence_id);
}

LlamaCPUBackend::PrefillResult
NativeCudaBackend::PrefillPartial(const std::string &prompt, int sequence_id,
                                  int n_past_start) {
  auto backend = DelegateBackend();
  if (!backend) {
    return {};
  }
  return backend->PrefillPartial(prompt, sequence_id, n_past_start);
}

void NativeCudaBackend::CopySequencePrefix(int src_seq, int dst_seq,
                                           int n_tokens) {
  if (runtime_) {
    runtime_->NativeCopySequencePrefix(src_seq, dst_seq, n_tokens);
  }
}

void NativeCudaBackend::FreeSequence(int sequence_id) {
  if (runtime_) {
    runtime_->NativeFreeSequence(sequence_id);
  }
}

std::vector<uint8_t> NativeCudaBackend::SerializeSequence(int sequence_id) {
  auto backend = DelegateBackend();
  if (!backend) {
    return {};
  }
  return backend->SerializeSequence(sequence_id);
}

bool NativeCudaBackend::HydrateSequence(int dest_sequence_id,
                                        const std::vector<uint8_t> &blob) {
  auto backend = DelegateBackend();
  if (!backend) {
    return false;
  }
  return backend->HydrateSequence(dest_sequence_id, blob);
}

std::string NativeCudaBackend::Decode(
    int n_past, int sequence_id, int max_tokens,
    const std::function<bool(const std::string &, const TokenLogprob *)>
        &on_chunk,
    const std::function<bool()> &should_stop, int logprob_top_n,
    std::vector<TokenLogprob> *out_logprobs, int first_token,
    const std::vector<std::string> &stop_seqs) {
  auto backend = DelegateBackend();
  if (!backend) {
    return {};
  }
  return backend->Decode(n_past, sequence_id, max_tokens, on_chunk, should_stop,
                         logprob_top_n, out_logprobs, first_token, stop_seqs);
}

std::string NativeCudaBackend::Generate(
    const std::string &prompt, int max_tokens,
    const std::function<bool(const std::string &, const TokenLogprob *)>
        &on_chunk,
    const std::function<bool()> &should_stop, int logprob_top_n,
    std::vector<TokenLogprob> *out_logprobs,
    const std::vector<std::string> &stop_seqs) {
  auto backend = DelegateBackend();
  if (!backend) {
    return {};
  }
  return backend->Generate(prompt, max_tokens, on_chunk, should_stop,
                           logprob_top_n, out_logprobs, stop_seqs);
}

void NativeCudaBackend::SetupSampler(const std::string &grammar,
                                     const std::string &root,
                                     const SamplingParams &sp) {
  auto backend = DelegateBackend();
  if (!backend) {
    return;
  }
  backend->SetupSampler(grammar, root, sp);
}

void NativeCudaBackend::TeardownSampler() {
  auto backend = DelegateBackend();
  if (!backend) {
    return;
  }
  backend->TeardownSampler();
}

LlamaCPUBackend::PerfSnapshot NativeCudaBackend::TakePerf() {
  auto backend = DelegateBackend();
  if (backend) {
    return backend->TakePerf();
  }
  // Native path: build PerfSnapshot from runtime accumulator.
  if (runtime_) {
    auto native = runtime_->NativeTakePerf();
    PerfSnapshot snap;
    snap.prefill_ms = native.prefill_ms;
    snap.decode_ms = native.decode_ms;
    snap.prompt_tokens = native.prompt_tokens;
    snap.generated_tokens = native.generated_tokens;
    return snap;
  }
  return {};
}

LlamaCPUBackend::ChatTemplateResult NativeCudaBackend::FormatChatMessages(
    const std::vector<std::pair<std::string, std::string>> &messages,
    bool add_assistant_prefix) {
  auto backend = DelegateBackend();
  if (!backend) {
    return {};
  }
  return backend->FormatChatMessages(messages, add_assistant_prefix);
}

int NativeCudaBackend::TokenCount(const std::string &text) const {
  if (runtime_) {
    return runtime_->NativeTokenCount(text);
  }
  return 0;
}

std::vector<int>
NativeCudaBackend::TokenizeForCache(const std::string &prompt) const {
  if (runtime_) {
    return runtime_->NativeTokenize(prompt);
  }
  return {};
}

bool NativeCudaBackend::IsReady() const {
  if (runtime_) {
    return runtime_->NativeIsReady();
  }
  return false;
}

std::shared_ptr<LlamaCPUBackend> NativeCudaBackend::DelegateBackend() const {
  if (!runtime_) {
    return nullptr;
  }
  return runtime_->BackendHandle();
}

} // namespace inferflux
