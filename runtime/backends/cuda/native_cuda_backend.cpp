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
#include <string_view>

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

bool IsVisibleNativePiece(std::string_view piece) { return !piece.empty(); }

std::string NormalizeNativeOutputPiece(const ITokenizer *tokenizer,
                                       int token_id,
                                       std::string_view decoded_piece) {
  if (token_id < 0) {
    return {};
  }
  if (tokenizer && tokenizer->IsTerminalGeneratedToken(token_id)) {
    return {};
  }
  if (!decoded_piece.empty()) {
    return std::string(decoded_piece);
  }
  if (!tokenizer) {
    return {};
  }
  return tokenizer->TokenToString(token_id);
}

} // namespace

std::atomic<int> NativeCudaBackend::next_ephemeral_sequence_id_{1 << 20};

NativeCudaBackend::NativeCudaBackend() : LlamaCPUBackend(false) {}

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

  const auto log_probe_failure = [](const char *stage, cudaError_t err) {
    static bool logged = false;
    if (logged) {
      return;
    }
    logged = true;
    log::Warn("native_cuda_backend",
              std::string("Native CUDA readiness probe failed at ") + stage +
                  ": " + cudaGetErrorString(err));
  };

  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err == cudaSuccess && device_count > 0) {
    return true;
  }

  // Some hosts report a transient/runtime-init failure on the first probe even
  // though CUDA execution is available for the process. Force a lightweight
  // runtime init and retry once before declaring native CUDA unavailable.
  const cudaError_t init_err = cudaFree(nullptr);
  if (init_err != cudaSuccess && init_err != cudaErrorCudartUnloading) {
    log_probe_failure("cudaFree(0)", init_err);
    return false;
  }

  err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess) {
    log_probe_failure("cudaGetDeviceCount(retry)", err);
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
  std::lock_guard<std::recursive_mutex> lock(runtime_mutex_);
  if (!runtime_) {
    return {};
  }
  return runtime_->ExecuteUnifiedBatch(inputs);
}

bool NativeCudaBackend::SupportsAsyncUnifiedBatch() const {
  std::lock_guard<std::recursive_mutex> lock(runtime_mutex_);
  if (!runtime_) {
    return false;
  }
  return runtime_->SupportsAsyncUnifiedBatch();
}

bool NativeCudaBackend::SupportsSplitPrefillDecodeHandoff() const {
  return runtime_ != nullptr && !fallback_mode_;
}

bool NativeCudaBackend::SupportsProcessLocalSequenceTransfer() const {
  return runtime_ != nullptr && !fallback_mode_;
}

LlamaCPUBackend::UnifiedBatchHandle NativeCudaBackend::SubmitUnifiedBatchAsync(
    const std::vector<UnifiedBatchInput> &inputs, UnifiedBatchLane lane) {
  std::lock_guard<std::recursive_mutex> lock(runtime_mutex_);
  if (!runtime_) {
    return 0;
  }
  return runtime_->SubmitUnifiedBatchAsync(inputs, lane);
}

bool NativeCudaBackend::TryCollectUnifiedBatchAsync(
    UnifiedBatchHandle handle, std::vector<UnifiedBatchOutput> *outputs) {
  std::lock_guard<std::recursive_mutex> lock(runtime_mutex_);
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
  std::lock_guard<std::recursive_mutex> lock(runtime_mutex_);
  if (runtime_) {
    runtime_->NativeCopySequencePrefix(src_seq, dst_seq, n_tokens);
  }
}

void NativeCudaBackend::FreeSequence(int sequence_id) {
  std::lock_guard<std::recursive_mutex> lock(runtime_mutex_);
  if (runtime_) {
    runtime_->NativeFreeSequence(sequence_id);
  }
}

LlamaCPUBackend::SequenceReleaseFence
NativeCudaBackend::BeginFreeSequence(int sequence_id) {
  std::lock_guard<std::recursive_mutex> lock(runtime_mutex_);
  if (runtime_) {
    return runtime_->NativeBeginFreeSequence(sequence_id);
  }
  auto backend = DelegateBackend();
  return backend ? backend->BeginFreeSequence(sequence_id)
                 : LlamaCPUBackend::SequenceReleaseFence{};
}

bool NativeCudaBackend::PollFreeSequence(const SequenceReleaseFence &fence) {
  std::lock_guard<std::recursive_mutex> lock(runtime_mutex_);
  if (runtime_) {
    return runtime_->NativePollFreeSequence(fence);
  }
  auto backend = DelegateBackend();
  return backend ? backend->PollFreeSequence(fence) : true;
}

std::vector<uint8_t> NativeCudaBackend::SerializeSequence(int sequence_id) {
  std::lock_guard<std::recursive_mutex> lock(runtime_mutex_);
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
  std::lock_guard<std::recursive_mutex> lock(runtime_mutex_);
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
  // Delegate to parity backend only for structured output (not logprobs).
  if (UsesStructuredConstraintSampler()) {
    auto parity_backend = EnsureParityBackend();
    if (parity_backend && parity_backend->IsReady()) {
      return parity_backend->Decode(n_past, sequence_id, max_tokens, on_chunk,
                                    should_stop, logprob_top_n, out_logprobs,
                                    first_token, stop_seqs);
    }
  }

  std::lock_guard<std::recursive_mutex> runtime_lock(runtime_mutex_);

  if (!runtime_ || sequence_id < 0 || n_past < 0 || max_tokens <= 0) {
    return {};
  }

  const bool collect_lp = (logprob_top_n > 0) || (out_logprobs != nullptr);
  const SamplingParams sampling = SnapshotSamplingParams();
  const ITokenizer *tokenizer = runtime_->NativeGetTokenizer();
  if (first_token < 0 || tokenizer == nullptr) {
    log::Warn("native_cuda_backend",
              "native decode requires first_token and tokenizer");
    return {};
  }

  std::string output;
  int current_token = first_token;
  int visible_tokens_generated = 0;
  int non_emitting_steps = 0;
  const int max_non_emitting_steps = std::max(max_tokens * 8, 32);

  // Emit the pre-sampled token first, matching llama.cpp decode semantics.
  // For the first token logprobs are unavailable (logits were consumed by the
  // prefill step that produced this token).
  if (max_tokens > 0) {
    if (tokenizer->IsTerminalGeneratedToken(current_token)) {
      return output;
    }
    std::string piece =
        NormalizeNativeOutputPiece(tokenizer, current_token, std::string_view{});
    if (IsVisibleNativePiece(piece)) {
      output += piece;
      ++visible_tokens_generated;

      std::string emit_piece;
      const bool stop_hit = ApplyStop(piece, output, stop_seqs, &emit_piece);
      if (on_chunk && !emit_piece.empty() && !on_chunk(emit_piece, nullptr)) {
        return output;
      }
      if (stop_hit || (should_stop && should_stop())) {
        return output;
      }
    } else {
      ++non_emitting_steps;
    }
  }

  while (visible_tokens_generated < max_tokens) {
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
    if (tokenizer->IsTerminalGeneratedToken(current_token)) {
      break;
    }
    std::string piece = NormalizeNativeOutputPiece(tokenizer, current_token,
                                                   step.front().piece);
    if (!IsVisibleNativePiece(piece)) {
      if (++non_emitting_steps >= max_non_emitting_steps) {
        log::Warn("native_cuda_backend",
                  "native decode aborted after too many non-emitting tokens");
        break;
      }
      continue;
    }

    non_emitting_steps = 0;
    output += piece;
    ++visible_tokens_generated;

    // Collect logprobs from the logits that produced this token.
    TokenLogprob tlp;
    const TokenLogprob *tlp_ptr = nullptr;
    if (collect_lp) {
      tlp = CollectNativeLogprob(current_token, piece, logprob_top_n);
      tlp_ptr = &tlp;
      if (out_logprobs) {
        out_logprobs->push_back(tlp);
      }
    }

    std::string emit_piece;
    const bool stop_hit = ApplyStop(piece, output, stop_seqs, &emit_piece);
    if (on_chunk && !emit_piece.empty() && !on_chunk(emit_piece, tlp_ptr)) {
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
  // Delegate to parity backend only for structured output (not logprobs).
  if (UsesStructuredConstraintSampler()) {
    auto parity_backend = EnsureParityBackend();
    if (parity_backend && parity_backend->IsReady()) {
      return parity_backend->Generate(prompt, max_tokens, on_chunk, should_stop,
                                      logprob_top_n, out_logprobs, stop_seqs);
    }
  }

  std::lock_guard<std::recursive_mutex> runtime_lock(runtime_mutex_);

  if (!runtime_ || max_tokens <= 0) {
    return {};
  }

  const bool collect_lp = (logprob_top_n > 0) || (out_logprobs != nullptr);
  const std::vector<int> prompt_tokens = runtime_->NativeTokenize(prompt);
  if (prompt_tokens.empty()) {
    return {};
  }

  const int sequence_id = AcquireEphemeralSequenceId();
  const SamplingParams sampling = SnapshotSamplingParams();
  const ITokenizer *tokenizer = runtime_->NativeGetTokenizer();
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
  int visible_tokens_generated = 0;
  int non_emitting_steps = 0;
  const int max_non_emitting_steps = std::max(max_tokens * 8, 32);

  if (max_tokens > 0) {
    if (tokenizer->IsTerminalGeneratedToken(current_token)) {
      return output;
    }
    std::string piece = NormalizeNativeOutputPiece(tokenizer, current_token,
                                                   first_step.front().piece);
    if (IsVisibleNativePiece(piece)) {
      output += piece;
      ++visible_tokens_generated;

      // First token: logits are available from prefill.
      TokenLogprob tlp;
      const TokenLogprob *tlp_ptr = nullptr;
      if (collect_lp) {
        tlp = CollectNativeLogprob(current_token, piece, logprob_top_n);
        tlp_ptr = &tlp;
        if (out_logprobs) {
          out_logprobs->push_back(tlp);
        }
      }

      std::string emit_piece;
      const bool stop_hit = ApplyStop(piece, output, stop_seqs, &emit_piece);
      if (on_chunk && !emit_piece.empty() && !on_chunk(emit_piece, tlp_ptr)) {
        return output;
      }
      if (stop_hit || (should_stop && should_stop())) {
        return output;
      }
    } else {
      ++non_emitting_steps;
    }
  }

  while (visible_tokens_generated < max_tokens) {
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
    if (tokenizer->IsTerminalGeneratedToken(current_token)) {
      break;
    }
    std::string piece = NormalizeNativeOutputPiece(tokenizer, current_token,
                                                   step.front().piece);
    if (!IsVisibleNativePiece(piece)) {
      if (++non_emitting_steps >= max_non_emitting_steps) {
        log::Warn("native_cuda_backend",
                  "native generate aborted after too many non-emitting "
                  "tokens");
        break;
      }
      continue;
    }

    non_emitting_steps = 0;
    output += piece;
    ++visible_tokens_generated;

    TokenLogprob tlp;
    const TokenLogprob *tlp_ptr = nullptr;
    if (collect_lp) {
      tlp = CollectNativeLogprob(current_token, piece, logprob_top_n);
      tlp_ptr = &tlp;
      if (out_logprobs) {
        out_logprobs->push_back(tlp);
      }
    }

    std::string emit_piece;
    const bool stop_hit = ApplyStop(piece, output, stop_seqs, &emit_piece);
    if (on_chunk && !emit_piece.empty() && !on_chunk(emit_piece, tlp_ptr)) {
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
  std::lock_guard<std::recursive_mutex> lock(runtime_mutex_);
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
  std::lock_guard<std::recursive_mutex> lock(runtime_mutex_);
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
  std::lock_guard<std::recursive_mutex> lock(runtime_mutex_);
  if (runtime_) {
    return runtime_->NativeTokenCount(text);
  }
  return 0;
}

std::vector<int>
NativeCudaBackend::TokenizeForCache(const std::string &prompt) const {
  std::lock_guard<std::recursive_mutex> lock(runtime_mutex_);
  if (runtime_) {
    return runtime_->NativeTokenize(prompt);
  }
  return {};
}

bool NativeCudaBackend::IsReady() const {
  std::lock_guard<std::recursive_mutex> lock(runtime_mutex_);
  if (runtime_) {
    return runtime_->NativeIsReady();
  }
  return false;
}

bool NativeCudaBackend::SupportsSpeculativeDecodingContract() const {
  std::lock_guard<std::recursive_mutex> lock(runtime_mutex_);
  return runtime_ && runtime_->NativeIsReady() &&
         runtime_->NativeGetTokenizer() != nullptr;
}

bool NativeCudaBackend::SupportsLogprobsContract() const { return true; }

bool NativeCudaBackend::SupportsStructuredOutputContract() const {
  return IsParityDelegateAvailable();
}

bool NativeCudaBackend::SupportsEmbeddingsContract() const { return true; }

bool NativeCudaBackend::IsParityDelegateAvailable() const {
  std::lock_guard<std::recursive_mutex> lock(runtime_mutex_);
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
                  "'; structured-output/embeddings remain "
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

TokenLogprob NativeCudaBackend::CollectNativeLogprob(int token_id,
                                                     const std::string &piece,
                                                     int top_n) {
  const int vocab = runtime_->NativeVocabSize();
  if (vocab <= 0) {
    return {};
  }
  if (static_cast<int>(host_logits_buf_.size()) < vocab) {
    host_logits_buf_.resize(vocab);
  }
  int copied = runtime_->CopyLastLogitsToHost(host_logits_buf_.data(), vocab);
  if (copied <= 0) {
    return {};
  }
  const ITokenizer *tok = runtime_->NativeGetTokenizer();
  return ComputeLogprob(host_logits_buf_.data(), vocab, token_id, piece, top_n,
                        [tok](int32_t id) -> std::string {
                          return tok ? tok->TokenToString(id) : "";
                        });
}

std::vector<float> NativeCudaBackend::EmbedForParity(const std::string &text) {
  // Try native embedding first.
  std::lock_guard<std::recursive_mutex> runtime_lock(runtime_mutex_);
  if (runtime_ && runtime_->NativeEmbedDims() > 0) {
    auto result = runtime_->NativeEmbed(text);
    if (!result.empty()) {
      return result;
    }
  }
  // Fall back to parity delegate.
  auto backend = EnsureParityBackend();
  if (!backend || !backend->IsReady()) {
    return {};
  }
  return backend->Embed(text);
}

int NativeCudaBackend::EmbedDimsForParity() const {
  // Try native first.
  std::lock_guard<std::recursive_mutex> runtime_lock(runtime_mutex_);
  if (runtime_) {
    int dims = runtime_->NativeEmbedDims();
    if (dims > 0) {
      return dims;
    }
  }
  // Fall back to parity delegate.
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
