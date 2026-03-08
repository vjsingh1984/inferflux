#include "runtime/backends/cuda/native_cuda_backend.h"

#include "model/model_format.h"
#include "model/tokenizer.h"
#include "runtime/backends/backend_utils.h"
#include "runtime/backends/cuda/native_cuda_runtime.h"
#include "runtime/backends/llama/llama_backend_traits.h"
#include "server/logging/logger.h"

#ifdef INFERFLUX_HAS_CUDA
#include <cuda_runtime_api.h>
#endif

#include <algorithm>
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
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
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

std::atomic<int> NativeCudaBackend::next_ephemeral_sequence_id_{1 << 20};

NativeCudaBackend::NativeCudaBackend() = default;

NativeCudaBackend::~NativeCudaBackend() = default;

bool NativeCudaBackend::LoadModel(const std::filesystem::path &model_path,
                                  const LlamaBackendConfig &config) {
#ifdef INFERFLUX_HAS_CUDA
  const bool strict_native_execution =
      ParseBoolEnv("INFERFLUX_NATIVE_CUDA_STRICT", false);
  loaded_model_path_ = model_path;
  loaded_config_ = config;
  parity_delegate_enabled_ =
      !ParseBoolEnv("INFERFLUX_NATIVE_DISABLE_PARITY_DELEGATE", false);
  parity_delegate_available_ = false;
  parity_delegate_init_attempted_ = false;
  parity_backend_.reset();
  parity_load_path_.clear();

  if (!NativeKernelsReady()) {
    log::Error("native_cuda_backend",
               "native CUDA backend requested but native kernels are not "
               "ready");
    return false;
  }

  runtime_ = CreateNativeCudaRuntime();
  if (!runtime_) {
    log::Error("native_cuda_backend", "failed to create native CUDA runtime");
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
    log::Warn("native_cuda_backend",
              fallback_reason_ + " (runtime=" + runtime_kind_ + ")");
  }
  if (parity_delegate_enabled_) {
    const std::string resolved_format =
        ResolveModelFormat(model_path.string(), /*requested_format=*/"auto");
    const std::string parity_path =
        ResolveLlamaLoadPath(model_path.string(), resolved_format);
    if (!parity_path.empty()) {
      parity_load_path_ = parity_path;
      parity_delegate_available_ = true;
    } else {
      log::Info("native_cuda_backend",
                "Native parity delegate unavailable for model path '" +
                    model_path.string() +
                    "' (no GGUF-compatible artifact detected)");
    }
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
  if (runtime_) {
    auto blob = runtime_->NativeSerializeSequence(sequence_id);
    if (!blob.empty()) {
      return blob;
    }
  }
  auto backend = DelegateBackend();
  if (!backend) {
    return {};
  }
  return backend->SerializeSequence(sequence_id);
}

bool NativeCudaBackend::HydrateSequence(int dest_sequence_id,
                                        const std::vector<uint8_t> &blob) {
  if (runtime_ && runtime_->NativeHydrateSequence(dest_sequence_id, blob)) {
    return true;
  }
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
  const bool needs_parity_path = (logprob_top_n > 0) ||
                                 (out_logprobs != nullptr) ||
                                 UsesStructuredConstraintSampler();
  if (needs_parity_path) {
    auto parity_backend = EnsureParityBackend();
    if (parity_backend && parity_backend->IsReady()) {
      return parity_backend->Decode(n_past, sequence_id, max_tokens, on_chunk,
                                    should_stop, logprob_top_n, out_logprobs,
                                    first_token, stop_seqs);
    }
  }

  if (!runtime_ || sequence_id < 0 || n_past < 0 || max_tokens <= 0) {
    return {};
  }

  if (logprob_top_n > 0 || out_logprobs != nullptr) {
    log::Warn("native_cuda_backend",
              "native decode logprobs are not implemented yet");
  }

  const SamplingParams sampling = SnapshotSamplingParams();
  const ITokenizer *tokenizer = runtime_->NativeGetTokenizer();
  if (first_token < 0 || tokenizer == nullptr) {
    log::Warn("native_cuda_backend",
              "native decode requires first_token and tokenizer");
    return {};
  }

  std::string output;
  int tokens_remaining = std::max(max_tokens, 1);
  int current_token = first_token;

  // Emit the pre-sampled token first, matching llama.cpp decode semantics.
  if (tokens_remaining > 0) {
    std::string piece = tokenizer->TokenToString(current_token);
    output += piece;
    --tokens_remaining;

    std::string emit_piece;
    const bool stop_hit = ApplyStop(piece, output, stop_seqs, &emit_piece);
    if (on_chunk && !emit_piece.empty() && !on_chunk(emit_piece, nullptr)) {
      return output;
    }
    if (stop_hit || (should_stop && should_stop())) {
      return output;
    }
  }

  while (tokens_remaining-- > 0) {
    if (should_stop && should_stop()) {
      break;
    }
    UnifiedBatchInput step_input;
    step_input.sequence_id = sequence_id;
    step_input.n_past = n_past;
    step_input.tokens = {current_token};
    step_input.request_logits = true;
    step_input.sampling = sampling;

    auto step = runtime_->ExecuteUnifiedBatch({step_input});
    if (step.empty() || !step.front().ok || step.front().token < 0) {
      break;
    }

    ++n_past;
    current_token = step.front().token;
    output += step.front().piece;

    std::string emit_piece;
    const bool stop_hit =
        ApplyStop(step.front().piece, output, stop_seqs, &emit_piece);
    if (on_chunk && !emit_piece.empty() && !on_chunk(emit_piece, nullptr)) {
      break;
    }
    if (stop_hit) {
      break;
    }
  }

  return output;
}

std::string NativeCudaBackend::Generate(
    const std::string &prompt, int max_tokens,
    const std::function<bool(const std::string &, const TokenLogprob *)>
        &on_chunk,
    const std::function<bool()> &should_stop, int logprob_top_n,
    std::vector<TokenLogprob> *out_logprobs,
    const std::vector<std::string> &stop_seqs) {
  const bool needs_parity_path = (logprob_top_n > 0) ||
                                 (out_logprobs != nullptr) ||
                                 UsesStructuredConstraintSampler();
  if (needs_parity_path) {
    auto parity_backend = EnsureParityBackend();
    if (parity_backend && parity_backend->IsReady()) {
      return parity_backend->Generate(prompt, max_tokens, on_chunk, should_stop,
                                      logprob_top_n, out_logprobs, stop_seqs);
    }
  }

  if (!runtime_ || max_tokens <= 0) {
    return {};
  }

  if (logprob_top_n > 0 || out_logprobs != nullptr) {
    log::Warn("native_cuda_backend",
              "native generate logprobs are not implemented yet");
  }

  const std::vector<int> prompt_tokens = runtime_->NativeTokenize(prompt);
  if (prompt_tokens.empty()) {
    return {};
  }

  const int sequence_id = AcquireEphemeralSequenceId();
  const SamplingParams sampling = SnapshotSamplingParams();
  std::string output;

  // Scope guard: always release the temporary sequence.
  struct SequenceReleaser final {
    NativeCudaRuntime *runtime{nullptr};
    int sequence_id{-1};
    ~SequenceReleaser() {
      if (runtime && sequence_id >= 0) {
        runtime->NativeFreeSequence(sequence_id);
      }
    }
  } release{runtime_.get(), sequence_id};

  UnifiedBatchInput prefill_input;
  prefill_input.sequence_id = sequence_id;
  prefill_input.n_past = 0;
  prefill_input.tokens = prompt_tokens;
  prefill_input.request_logits = true;
  prefill_input.sampling = sampling;

  auto first_step = runtime_->ExecuteUnifiedBatch({prefill_input});
  if (first_step.empty() || !first_step.front().ok ||
      first_step.front().token < 0) {
    return {};
  }

  int current_token = first_step.front().token;
  int n_past = static_cast<int>(prompt_tokens.size());
  int tokens_remaining = std::max(max_tokens, 1);

  if (tokens_remaining > 0) {
    output += first_step.front().piece;
    --tokens_remaining;

    std::string emit_piece;
    const bool stop_hit =
        ApplyStop(first_step.front().piece, output, stop_seqs, &emit_piece);
    if (on_chunk && !emit_piece.empty() && !on_chunk(emit_piece, nullptr)) {
      return output;
    }
    if (stop_hit || (should_stop && should_stop())) {
      return output;
    }
  }

  while (tokens_remaining-- > 0) {
    if (should_stop && should_stop()) {
      break;
    }
    UnifiedBatchInput step_input;
    step_input.sequence_id = sequence_id;
    step_input.n_past = n_past;
    step_input.tokens = {current_token};
    step_input.request_logits = true;
    step_input.sampling = sampling;

    auto step = runtime_->ExecuteUnifiedBatch({step_input});
    if (step.empty() || !step.front().ok || step.front().token < 0) {
      break;
    }

    ++n_past;
    current_token = step.front().token;
    output += step.front().piece;

    std::string emit_piece;
    const bool stop_hit =
        ApplyStop(step.front().piece, output, stop_seqs, &emit_piece);
    if (on_chunk && !emit_piece.empty() && !on_chunk(emit_piece, nullptr)) {
      break;
    }
    if (stop_hit) {
      break;
    }
  }

  return output;
}

void NativeCudaBackend::SetupSampler(const std::string &grammar,
                                     const std::string &root,
                                     const SamplingParams &sp) {
  if (auto backend = DelegateBackend()) {
    backend->SetupSampler(grammar, root, sp);
  }
  if (!grammar.empty()) {
    if (auto parity_backend = EnsureParityBackend()) {
      parity_backend->SetupSampler(grammar, root, sp);
    }
  }
  if (!grammar.empty()) {
    if (!IsParityDelegateAvailable()) {
      log::Warn("native_cuda_backend",
                "native sampler grammar constraints are unavailable (no parity "
                "delegate backend)");
    }
  }
  std::lock_guard<std::mutex> lock(sampling_mutex_);
  active_sampling_ = sp;
  sampling_active_ = true;
  structured_constraint_sampler_active_ = !grammar.empty();
}

void NativeCudaBackend::TeardownSampler() {
  if (auto backend = DelegateBackend()) {
    backend->TeardownSampler();
  }
  {
    std::lock_guard<std::mutex> lock(parity_backend_mutex_);
    if (parity_backend_) {
      parity_backend_->TeardownSampler();
    }
  }
  std::lock_guard<std::mutex> lock(sampling_mutex_);
  active_sampling_ = SamplingParams{};
  sampling_active_ = false;
  structured_constraint_sampler_active_ = false;
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
  // Prefer runtime's native tokenizer (LlamaTokenizer with chat templates)
  if (runtime_) {
    auto native = runtime_->NativeFormatChat(messages, add_assistant_prefix);
    if (native.valid) {
      ChatTemplateResult result;
      result.prompt = std::move(native.prompt);
      result.valid = true;
      return result;
    }
  }
  // Fallback to llama.cpp backend (if available)
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

bool NativeCudaBackend::SupportsSpeculativeDecodingContract() const {
  return runtime_ && runtime_->NativeIsReady() &&
         runtime_->NativeGetTokenizer() != nullptr;
}

bool NativeCudaBackend::SupportsLogprobsContract() const {
  return IsParityDelegateAvailable();
}

bool NativeCudaBackend::SupportsStructuredOutputContract() const {
  return IsParityDelegateAvailable();
}

bool NativeCudaBackend::SupportsEmbeddingsContract() const {
  return IsParityDelegateAvailable();
}

bool NativeCudaBackend::IsParityDelegateAvailable() const {
  return runtime_ && runtime_->NativeIsReady() && parity_delegate_enabled_ &&
         parity_delegate_available_;
}

std::shared_ptr<LlamaCPUBackend>
NativeCudaBackend::EnsureParityBackend() const {
  if (!IsParityDelegateAvailable()) {
    return nullptr;
  }

  std::lock_guard<std::mutex> lock(parity_backend_mutex_);
  if (parity_backend_) {
    return parity_backend_;
  }
  if (parity_delegate_init_attempted_) {
    return nullptr;
  }
  parity_delegate_init_attempted_ = true;

  auto backend = std::make_shared<LlamaCPUBackend>();
  const auto tuned_cfg =
      TuneLlamaBackendConfig(LlamaBackendTarget::kCuda, loaded_config_);
  if (!backend->LoadModel(parity_load_path_, tuned_cfg)) {
    log::Warn("native_cuda_backend",
              "Failed to initialize parity delegate backend from '" +
                  parity_load_path_.string() +
                  "'; logprobs/structured-output/embeddings remain "
                  "unavailable on native provider");
    parity_delegate_available_ = false;
    return nullptr;
  }

  parity_backend_ = std::move(backend);
  log::Info("native_cuda_backend",
            "Initialized native parity delegate backend from '" +
                parity_load_path_.string() + "'");
  return parity_backend_;
}

std::shared_ptr<LlamaCPUBackend> NativeCudaBackend::DelegateBackend() const {
  if (!runtime_) {
    return nullptr;
  }
  return runtime_->BackendHandle();
}

bool NativeCudaBackend::UsesStructuredConstraintSampler() const {
  std::lock_guard<std::mutex> lock(sampling_mutex_);
  return structured_constraint_sampler_active_;
}

std::vector<float> NativeCudaBackend::EmbedForParity(const std::string &text) {
  auto backend = EnsureParityBackend();
  if (!backend || !backend->IsReady()) {
    return {};
  }
  return backend->Embed(text);
}

int NativeCudaBackend::EmbedDimsForParity() const {
  auto backend = EnsureParityBackend();
  if (!backend || !backend->IsReady()) {
    return 0;
  }
  return backend->EmbedDims();
}

SamplingParams NativeCudaBackend::SnapshotSamplingParams() const {
  std::lock_guard<std::mutex> lock(sampling_mutex_);
  if (!sampling_active_) {
    return {};
  }
  return active_sampling_;
}

int NativeCudaBackend::AcquireEphemeralSequenceId() {
  int seq = next_ephemeral_sequence_id_.fetch_add(1, std::memory_order_relaxed);
  if (seq < 0) {
    next_ephemeral_sequence_id_.store(1 << 20, std::memory_order_relaxed);
    seq = next_ephemeral_sequence_id_.fetch_add(1, std::memory_order_relaxed);
  }
  return seq;
}

} // namespace inferflux
