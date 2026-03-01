#include "runtime/execution/batch_executor.h"

#include "server/metrics/metrics.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>

using json = nlohmann::json;

namespace inferflux {

namespace {

class GrammarScope {
 public:
  GrammarScope(const StructuredConstraint& constraint,
               const std::shared_ptr<LlamaCPUBackend>& backend)
      : backend_(backend.get() ? backend.get() : nullptr) {
    if (!backend_ || !constraint.has_grammar) {
      backend_ = nullptr;
      return;
    }
    backend_->EnableGrammarConstraint(constraint.grammar, constraint.root);
    active_ = true;
  }

  GrammarScope(const GrammarScope&) = delete;
  GrammarScope& operator=(const GrammarScope&) = delete;

  ~GrammarScope() {
    if (active_ && backend_) {
      backend_->DisableGrammarConstraint();
    }
  }

 private:
  LlamaCPUBackend* backend_{nullptr};
  bool active_{false};
};

}  // namespace

BatchExecutor::BatchExecutor(SimpleTokenizer* tokenizer,
                             std::shared_ptr<CPUDeviceContext> device,
                             std::shared_ptr<PagedKVCache> cache,
                             std::shared_ptr<ModelRouter> router,
                             std::shared_ptr<SpeculativeDecoder> speculative_decoder,
                             std::shared_ptr<RadixPrefixCache> prefix_cache)
    : tokenizer_(tokenizer),
      device_(std::move(device)),
      cache_(std::move(cache)),
      router_(std::move(router)),
      speculative_decoder_(std::move(speculative_decoder)),
      prefix_cache_(std::move(prefix_cache)) {}

std::vector<InferenceResult> BatchExecutor::ExecuteBatch(
    const RequestBatch& batch,
    const std::vector<std::shared_ptr<LlamaCPUBackend>>& backend_overrides) {
  std::vector<InferenceResult> results;
  results.reserve(batch.requests.size());
  double total_prefill_ms = 0.0;
  double total_decode_ms = 0.0;
  for (std::size_t i = 0; i < batch.requests.size(); ++i) {
    auto* request = batch.requests[i];
    std::shared_ptr<LlamaCPUBackend> backend_override;
    if (i < backend_overrides.size()) {
      backend_override = backend_overrides[i];
    }
    auto outcome = ExecuteRequest(*request, backend_override);
    results.push_back(std::move(outcome.result));
    total_prefill_ms += outcome.prefill_ms;
    total_decode_ms += outcome.decode_ms;
  }
  if (total_prefill_ms > 0) {
    GlobalMetrics().RecordPrefillDuration(total_prefill_ms);
  }
  if (total_decode_ms > 0) {
    GlobalMetrics().RecordDecodeDuration(total_decode_ms);
  }
  return results;
}

BatchExecutor::ExecutionOutcome BatchExecutor::ExecuteRequest(
    InferenceRequest& inference,
    std::shared_ptr<LlamaCPUBackend> backend_override) {
  ExecutionOutcome outcome;
  auto& response = outcome.result;
  response.prompt_tokens = static_cast<int>(inference.prompt_tokens.size());
  inference.fairness_yielded = false;

  if (inference.cancellation_flag && inference.cancellation_flag->load()) {
    inference.phase = RequestPhase::kAborted;
    inference.first_token_time = std::chrono::steady_clock::now();
    response.no_backend = true;
    response.completion = "[cancelled]";
    return outcome;
  }

  if (prefix_cache_) {
    std::string cached_completion;
    int cached_tokens = 0;
    int matched_tokens = 0;
    if (prefix_cache_->Lookup(inference.prompt_tokens, &cached_completion, &cached_tokens,
                              &matched_tokens)) {
      GlobalMetrics().RecordPrefixLookup(true);
      response.completion = cached_completion;
      response.completion_tokens = cached_tokens;
      inference.phase = RequestPhase::kFinished;
      inference.first_token_time = std::chrono::steady_clock::now();
      if (inference.on_token && !cached_completion.empty()) {
        if (inference.stream) {
          GlobalMetrics().RecordStreamCacheHit();
        }
        inference.on_token(cached_completion);
      }
      return outcome;
    }
    GlobalMetrics().RecordPrefixLookup(false);
    if (matched_tokens > 0) {
      GlobalMetrics().RecordPartialPrefixHit();
      GlobalMetrics().RecordPrefixMatchedTokens(matched_tokens);
    }
  }

  std::string resolved_model = inference.resolved_model;
  auto backend = std::move(backend_override);
  if (!backend) {
    backend = ResolveBackend(inference.model, &resolved_model);
    inference.resolved_model = resolved_model;
  }
  if (resolved_model.empty()) {
    resolved_model = inference.model;
  }
  response.model_id = resolved_model;

  bool backend_ready = backend && backend->IsReady();
  int slice_limit = inference.timeslice_tokens;
  inference.last_timeslice_tokens = slice_limit;
  inference.timeslice_tokens = 0;  // consumed; refreshed on next fairness pass.
  int decode_limit = inference.max_tokens;
  if (inference.remaining_decode_tokens >= 0) {
    decode_limit = std::min(decode_limit, inference.remaining_decode_tokens);
  }
  if (slice_limit > 0) {
    decode_limit = std::min(decode_limit, slice_limit);
  }
  if (decode_limit <= 0) {
    decode_limit = 1;
  }

  // Prefill phase: prompt token counting (mirrors what the backend will evaluate).
  auto prefill_start = std::chrono::steady_clock::now();
  if (backend_ready) {
    response.prompt_tokens = backend->TokenCount(inference.prompt);
  }
  if (inference.total_completion_tokens == 0) {
    inference.reported_prompt_tokens = response.prompt_tokens;
    if (inference.service_tokens == 0) {
      inference.service_tokens = response.prompt_tokens;
    }
  }
  outcome.prefill_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - prefill_start)
          .count();

  int kv_page = -1;
  if (cache_) {
    try {
      kv_page = cache_->ReservePage();
    } catch (const std::exception& ex) {
      std::cerr << "[BatchExecutor] KV cache reserve failed: " << ex.what() << std::endl;
    }
  }

  // Decode phase: token generation (speculative or direct).
  inference.phase = RequestPhase::kDecode;
  auto decode_start = std::chrono::steady_clock::now();

  GrammarScope grammar_scope(inference.response_constraint, backend);

  if (speculative_decoder_ && speculative_decoder_->Enabled() &&
      !inference.has_response_format &&
      (!inference.cancellation_flag || !inference.cancellation_flag->load())) {
    auto draft = speculative_decoder_->Draft(inference.prompt_tokens, decode_limit);
    auto validated = speculative_decoder_->Validate(
        inference.prompt_tokens, draft, decode_limit, backend);
    response.speculative.total_chunks = static_cast<int>(validated.metrics.total_chunks);
    response.speculative.accepted_chunks = static_cast<int>(validated.metrics.accepted_chunks);
    response.speculative.reused_tokens = static_cast<int>(validated.metrics.reused_tokens);
    response.completion = tokenizer_->Decode(validated.completion_tokens);
    inference.output_tokens = validated.completion_tokens;
  }

  if (response.completion.empty() &&
      (!inference.cancellation_flag || !inference.cancellation_flag->load())) {
    if (backend && backend->IsReady()) {
      std::function<bool(const std::string&)> chunk_cb;
      if (inference.on_token) {
        auto on_token = inference.on_token;
        auto cancel_flag = inference.cancellation_flag;
        chunk_cb = [on_token, cancel_flag](const std::string& token_chunk) {
          GlobalMetrics().RecordStreamTokens(1);
          on_token(token_chunk);
          if (cancel_flag && cancel_flag->load()) {
            return false;
          }
          return true;
        };
      }
      std::function<bool()> should_stop;
      if (inference.cancellation_flag) {
        auto flag = inference.cancellation_flag;
        should_stop = [flag]() { return flag->load(); };
      }
      auto text = backend->Generate(inference.prompt,
                                    decode_limit,
                                    chunk_cb,
                                    should_stop);
      response.completion = text.empty() ? "[backend returned empty response]" : text;
    } else {
      response.no_backend = true;
      response.completion =
          "No model backend is loaded. Set INFERFLUX_MODEL_PATH or configure model.path in server.yaml.";
    }
  }

  outcome.decode_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - decode_start)
          .count();

  if (backend_ready) {
    response.completion_tokens = backend->TokenCount(response.completion);
  } else {
    auto completion_tokens = tokenizer_->Encode(response.completion);
    response.completion_tokens = static_cast<int>(completion_tokens.size());
    inference.output_tokens = std::move(completion_tokens);
  }
  bool first_slice = (inference.total_completion_tokens == 0);
  int fairness_delta = response.completion_tokens;
  if (first_slice) {
    int prompt_component = response.prompt_tokens;
    if (prompt_component <= 0 && inference.reported_prompt_tokens >= 0) {
      prompt_component = inference.reported_prompt_tokens;
    }
    fairness_delta += prompt_component;
  }
  if (fairness_delta > 0) {
    GlobalMetrics().RecordFairnessTokens(inference.priority_level, fairness_delta);
  }
  inference.service_tokens += response.completion_tokens;
  inference.total_completion_tokens += response.completion_tokens;
  bool fairness_active = slice_limit > 0;
  bool exhausted_slice = fairness_active && response.completion_tokens >= decode_limit;
  if (fairness_active && exhausted_slice && inference.remaining_decode_tokens > 0) {
    inference.fairness_yielded = true;
    GlobalMetrics().RecordFairnessYield(inference.priority_level,
                                        response.completion_tokens,
                                        inference.remaining_decode_tokens);
    inference.phase = RequestPhase::kPending;
  } else {
    inference.phase = RequestPhase::kFinished;
  }

  bool needs_json_validation = inference.json_mode || inference.response_constraint.require_json_object;
  if (needs_json_validation) {
    try {
      auto parsed = json::parse(response.completion);
      if (inference.response_constraint.require_json_object && !parsed.is_object()) {
        throw std::runtime_error("model output was not a JSON object");
      }
      response.completion = parsed.dump();
    } catch (const std::exception&) {
      if (inference.response_constraint.require_json_object && !response.no_backend) {
        response.no_backend = true;
        response.completion = "Model output violated response_format constraints";
        response.completion_tokens = 0;
        inference.output_tokens.clear();
        inference.phase = RequestPhase::kFinished;
        inference.fairness_yielded = false;
        if (cache_ && kv_page >= 0) {
          cache_->ReleasePage(kv_page);
          kv_page = -1;
        }
        inference.first_token_time = std::chrono::steady_clock::now();
        return outcome;
      }
      response.completion = json({{"output", response.completion}}).dump();
    }
    auto retokens = tokenizer_->Encode(response.completion);
    response.completion_tokens = static_cast<int>(retokens.size());
    inference.output_tokens = std::move(retokens);
  }

  // Accumulate after json_mode so ProcessBatch sees the final post-processed text.
  if (!response.completion.empty()) {
    inference.accumulated_output.append(response.completion);
    if (inference.remaining_decode_tokens >= 0) {
      inference.remaining_decode_tokens =
          std::max(0, inference.remaining_decode_tokens - response.completion_tokens);
    }
  }

  if (inference.output_tokens.empty()) {
    inference.output_tokens = tokenizer_->Encode(response.completion);
  }
  if (cache_ && kv_page >= 0) {
    cache_->ReleasePage(kv_page);
  }

  if (!inference.fairness_yielded &&
      prefix_cache_ &&
      !response.no_backend &&
      !response.completion.empty()) {
    prefix_cache_->Insert(inference.prompt_tokens, response.completion, response.completion_tokens);
  }

  inference.first_token_time = std::chrono::steady_clock::now();
  inference.service_tokens += response.completion_tokens;
  return outcome;
}

std::shared_ptr<LlamaCPUBackend> BatchExecutor::ResolveBackend(const std::string& requested_model,
                                                               std::string* resolved_id) {
  if (!router_) {
    return nullptr;
  }
  auto* info = router_->Resolve(requested_model);
  if (!info) {
    return nullptr;
  }
  if (resolved_id) {
    *resolved_id = info->id;
  }
  return router_->GetBackend(info->id);
}

}  // namespace inferflux
