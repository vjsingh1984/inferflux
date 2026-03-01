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
  GrammarScope(const StructuredConstraint &constraint,
               const std::shared_ptr<LlamaCPUBackend> &backend)
      : backend_(backend.get() ? backend.get() : nullptr) {
    if (!backend_ || !constraint.has_grammar) {
      backend_ = nullptr;
      return;
    }
    backend_->EnableGrammarConstraint(constraint.grammar, constraint.root);
    active_ = true;
  }

  GrammarScope(const GrammarScope &) = delete;
  GrammarScope &operator=(const GrammarScope &) = delete;

  ~GrammarScope() {
    if (active_ && backend_) {
      backend_->DisableGrammarConstraint();
    }
  }

private:
  LlamaCPUBackend *backend_{nullptr};
  bool active_{false};
};

} // namespace

BatchExecutor::BatchExecutor(
    SimpleTokenizer *tokenizer, std::shared_ptr<CPUDeviceContext> device,
    std::shared_ptr<PagedKVCache> cache, std::shared_ptr<ModelRouter> router,
    std::shared_ptr<SpeculativeDecoder> speculative_decoder,
    std::shared_ptr<RadixPrefixCache> prefix_cache)
    : tokenizer_(tokenizer), device_(std::move(device)),
      cache_(std::move(cache)), router_(std::move(router)),
      speculative_decoder_(std::move(speculative_decoder)),
      prefix_cache_(std::move(prefix_cache)) {}

std::vector<InferenceResult> BatchExecutor::ExecuteBatch(
    const RequestBatch &batch,
    const std::vector<std::shared_ptr<LlamaCPUBackend>> &backend_overrides) {
  std::size_t n = batch.requests.size();
  std::vector<InferenceResult> results(n);
  std::vector<bool> handled(n, false);
  double total_prefill_ms = 0.0;
  double total_decode_ms = 0.0;

  // Identify requests eligible for multi-sequence batch decode.
  // Eligibility: phased decode (n_past >= 0, seq_id >= 0), same backend,
  // no grammar constraints, no logprob collection, no response format.
  // Grammar uses per-backend grammar_sampler_ — not safe to interleave.
  std::vector<std::size_t> eligible_indices;
  std::shared_ptr<LlamaCPUBackend> shared_be;
  bool homogeneous_backend = true;
  for (std::size_t i = 0; i < n; ++i) {
    auto *req = batch.requests[i];
    auto be = (i < backend_overrides.size()) ? backend_overrides[i] : nullptr;
    if (!be) {
      be = ResolveBackend(req->model, nullptr);
    }
    if (req->n_past >= 0 && req->sequence_id >= 0 &&
        !req->response_constraint.has_grammar && !req->collect_logprobs &&
        !req->has_response_format && be && be->IsReady()) {
      if (!shared_be) {
        shared_be = be;
      } else if (be.get() != shared_be.get()) {
        homogeneous_backend = false;
      }
      eligible_indices.push_back(i);
    }
  }

  if (eligible_indices.size() >= 2 && homogeneous_backend && shared_be) {
    std::vector<InferenceRequest *> eligible_reqs;
    eligible_reqs.reserve(eligible_indices.size());
    for (auto idx : eligible_indices) {
      eligible_reqs.push_back(batch.requests[idx]);
    }
    auto outcomes = ExecuteBatchDecodePhased(eligible_reqs, shared_be);
    for (std::size_t j = 0; j < eligible_indices.size(); ++j) {
      results[eligible_indices[j]] = std::move(outcomes[j].result);
      total_decode_ms += outcomes[j].decode_ms;
      handled[eligible_indices[j]] = true;
    }
  }

  // Process remaining requests individually (non-eligible or batch size < 2).
  for (std::size_t i = 0; i < n; ++i) {
    if (handled[i])
      continue;
    auto *request = batch.requests[i];
    std::shared_ptr<LlamaCPUBackend> backend_override;
    if (i < backend_overrides.size()) {
      backend_override = backend_overrides[i];
    }
    auto outcome = ExecuteRequest(*request, backend_override);
    results[i] = std::move(outcome.result);
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
    InferenceRequest &inference,
    std::shared_ptr<LlamaCPUBackend> backend_override) {
  ExecutionOutcome outcome;
  auto &response = outcome.result;
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
    if (prefix_cache_->Lookup(inference.prompt_tokens, &cached_completion,
                              &cached_tokens, &matched_tokens)) {
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
  inference.timeslice_tokens = 0; // consumed; refreshed on next fairness pass.
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

  // Prefill phase: prompt token counting (mirrors what the backend will
  // evaluate).
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
  outcome.prefill_ms = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - prefill_start)
                           .count();

  int kv_page = -1;
  if (cache_) {
    try {
      kv_page = cache_->ReservePage();
      inference.kv_page = kv_page;
    } catch (const std::exception &ex) {
      std::cerr << "[BatchExecutor] KV cache reserve failed: " << ex.what()
                << std::endl;
    }
  }

  // Decode phase: token generation (speculative or direct).
  inference.phase = RequestPhase::kDecode;
  auto decode_start = std::chrono::steady_clock::now();

  GrammarScope grammar_scope(inference.response_constraint, backend);

  if (speculative_decoder_ && speculative_decoder_->Enabled() &&
      !inference.has_response_format &&
      (!inference.cancellation_flag || !inference.cancellation_flag->load())) {
    auto draft =
        speculative_decoder_->Draft(inference.prompt_tokens, decode_limit);
    auto validated = speculative_decoder_->Validate(
        inference.prompt_tokens, draft, decode_limit, backend);
    response.speculative.total_chunks =
        static_cast<int>(validated.metrics.total_chunks);
    response.speculative.accepted_chunks =
        static_cast<int>(validated.metrics.accepted_chunks);
    response.speculative.reused_tokens =
        static_cast<int>(validated.metrics.reused_tokens);
    response.completion = tokenizer_->Decode(validated.completion_tokens);
    inference.output_tokens = validated.completion_tokens;
  }

  if (response.completion.empty() &&
      (!inference.cancellation_flag || !inference.cancellation_flag->load())) {
    if (backend && backend->IsReady()) {
      std::function<bool(const std::string &)> chunk_cb;
      if (inference.on_token) {
        auto on_token = inference.on_token;
        auto cancel_flag = inference.cancellation_flag;
        chunk_cb = [on_token, cancel_flag](const std::string &token_chunk) {
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
      std::string text;
      const int logprob_top_n = inference.logprob_top_n;
      // Use collect_logprobs (not logprob_top_n > 0) so that logprobs=true
      // with top_logprobs=0 still records the selected token's log-probability
      // without the O(V log V) partial-sort for alternatives.
      std::vector<TokenLogprob> *lp_out =
          inference.collect_logprobs ? &response.logprobs : nullptr;
      if (inference.n_past >= 0 && inference.sequence_id >= 0) {
        // Phased path: prompt was already prefilled by the scheduler; run
        // decode only.  Pass first_token so Decode() starts from the
        // pre-sampled token rather than re-sampling from a potentially stale
        // logit buffer.
        text = backend->Decode(inference.n_past, inference.sequence_id,
                               decode_limit, chunk_cb, should_stop,
                               logprob_top_n, lp_out, inference.first_token);
      } else if (inference.has_images && backend->SupportsVision()) {
        text = backend->GenerateWithImages(inference.prompt, inference.images,
                                           decode_limit, chunk_cb, should_stop);
      } else {
        text = backend->Generate(inference.prompt, decode_limit, chunk_cb,
                                 should_stop, logprob_top_n, lp_out);
      }
      response.completion =
          text.empty() ? "[backend returned empty response]" : text;
    } else {
      response.no_backend = true;
      response.completion =
          "No model backend is loaded. Set INFERFLUX_MODEL_PATH or configure "
          "model.path in server.yaml.";
    }
  }

  outcome.decode_ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - decode_start)
                          .count();

  if (backend_ready) {
    response.completion_tokens = backend->TokenCount(response.completion);
  } else {
    auto completion_tokens = tokenizer_->Encode(response.completion);
    response.completion_tokens = static_cast<int>(completion_tokens.size());
    inference.output_tokens = std::move(completion_tokens);
  }
  // Advance n_past so the next fairness slice continues from the right KV
  // position.
  if (inference.n_past >= 0 && response.completion_tokens > 0) {
    inference.n_past += response.completion_tokens;
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
    GlobalMetrics().RecordFairnessTokens(inference.priority_level,
                                         fairness_delta);
  }
  inference.service_tokens += response.completion_tokens;
  inference.total_completion_tokens += response.completion_tokens;
  bool fairness_active = slice_limit > 0;
  bool exhausted_slice =
      fairness_active && response.completion_tokens >= decode_limit;
  if (fairness_active && exhausted_slice &&
      inference.remaining_decode_tokens > 0) {
    inference.fairness_yielded = true;
    GlobalMetrics().RecordFairnessYield(inference.priority_level,
                                        response.completion_tokens,
                                        inference.remaining_decode_tokens);
    inference.phase = RequestPhase::kPending;
  } else {
    inference.phase = RequestPhase::kFinished;
  }

  bool needs_json_validation =
      inference.json_mode || inference.response_constraint.require_json_object;
  if (needs_json_validation) {
    try {
      auto parsed = json::parse(response.completion);
      if (inference.response_constraint.require_json_object &&
          !parsed.is_object()) {
        throw std::runtime_error("model output was not a JSON object");
      }
      response.completion = parsed.dump();
    } catch (const std::exception &) {
      if (inference.response_constraint.require_json_object &&
          !response.no_backend) {
        response.no_backend = true;
        response.completion =
            "Model output violated response_format constraints";
        response.completion_tokens = 0;
        inference.output_tokens.clear();
        inference.phase = RequestPhase::kFinished;
        inference.fairness_yielded = false;
        if (cache_ && kv_page >= 0) {
          cache_->ReleasePage(kv_page);
          kv_page = -1;
          inference.kv_page = -1;
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

  // Accumulate after json_mode so ProcessBatch sees the final post-processed
  // text.
  if (!response.completion.empty()) {
    inference.accumulated_output.append(response.completion);
    if (inference.remaining_decode_tokens >= 0) {
      inference.remaining_decode_tokens = std::max(
          0, inference.remaining_decode_tokens - response.completion_tokens);
    }
  }

  if (inference.output_tokens.empty()) {
    inference.output_tokens = tokenizer_->Encode(response.completion);
  }
  if (cache_ && kv_page >= 0) {
    cache_->ReleasePage(kv_page);
    inference.kv_page = -1;
  }

  if (!inference.fairness_yielded && prefix_cache_ && !response.no_backend &&
      !response.completion.empty()) {
    prefix_cache_->Insert(inference.prompt_tokens, response.completion,
                          response.completion_tokens);
  }

  inference.first_token_time = std::chrono::steady_clock::now();
  inference.service_tokens += response.completion_tokens;

  // KV memory release and slot bookkeeping (FreeSequence + FreeSeqSlot) are
  // intentionally handled by the scheduler (ProcessBatch / DecodeWorkerLoop)
  // so they always happen together.  The executor only manages the compute
  // path; leaving sequence_id set here allows the scheduler to act on it.

  return outcome;
}

std::vector<BatchExecutor::ExecutionOutcome>
BatchExecutor::ExecuteBatchDecodePhased(
    std::vector<InferenceRequest *> &eligible,
    std::shared_ptr<LlamaCPUBackend> backend) {
  std::size_t n = eligible.size();
  std::vector<ExecutionOutcome> outcomes(n);

  // Per-request runtime state for the decode loop.
  struct ReqState {
    bool active{true};
    int tokens_generated{0};
    int decode_limit{0};
    int current_token{-1}; // token to feed in the next BatchDecodeStep
    bool slice_active{false};
  };
  std::vector<ReqState> states(n);

  auto decode_start = std::chrono::steady_clock::now();

  for (std::size_t i = 0; i < n; ++i) {
    auto *req = eligible[i];
    auto &out = outcomes[i].result;
    req->fairness_yielded = false;
    out.model_id =
        req->resolved_model.empty() ? req->model : req->resolved_model;
    out.prompt_tokens = static_cast<int>(req->prompt_tokens.size());

    // Compute per-request decode limit respecting fairness timeslice.
    int limit = req->max_tokens;
    if (req->remaining_decode_tokens >= 0) {
      limit = std::min(limit, req->remaining_decode_tokens);
    }
    int slice = req->timeslice_tokens;
    req->last_timeslice_tokens = slice;
    req->timeslice_tokens = 0;
    if (slice > 0) {
      limit = std::min(limit, slice);
    }
    if (limit <= 0)
      limit = 1;
    states[i].decode_limit = limit;
    states[i].slice_active = (slice > 0);

    // Emit the first token (pre-sampled by Prefill while its logits were
    // fresh).
    if (req->first_token >= 0) {
      states[i].current_token = req->first_token;
      out.completion += req->first_piece;
      states[i].tokens_generated = 1;
      if (req->on_token) {
        GlobalMetrics().RecordStreamTokens(1);
        req->on_token(req->first_piece);
      }
      // Immediately check cancellation after the first token callback.
      if (req->cancellation_flag && req->cancellation_flag->load()) {
        states[i].active = false;
      }
    } else {
      // EOS at prefill time — nothing to generate.
      states[i].active = false;
      out.completion = "[backend returned empty response]";
    }
  }

  // Multi-sequence decode loop: one llama_decode() per token step.
  while (true) {
    std::vector<LlamaCPUBackend::BatchDecodeInput> batch_inputs;
    std::vector<std::size_t> active_idx;

    for (std::size_t i = 0; i < n; ++i) {
      if (!states[i].active)
        continue;
      auto *req = eligible[i];
      if (req->cancellation_flag && req->cancellation_flag->load()) {
        states[i].active = false;
        continue;
      }
      if (states[i].tokens_generated >= states[i].decode_limit) {
        states[i].active = false;
        continue;
      }
      batch_inputs.push_back(
          {req->sequence_id, req->n_past, states[i].current_token});
      active_idx.push_back(i);
    }

    if (active_idx.empty())
      break;

    auto step = backend->BatchDecodeStep(batch_inputs);
    if (step.empty())
      break; // llama_decode failed

    for (std::size_t j = 0; j < active_idx.size(); ++j) {
      std::size_t i = active_idx[j];
      auto *req = eligible[i];
      req->n_past =
          batch_inputs[j].n_past; // updated in-place by BatchDecodeStep

      const auto &sr = step[j];
      if (sr.token < 0) {
        states[i].active = false; // EOS
        continue;
      }
      states[i].current_token = sr.token;
      states[i].tokens_generated++;
      outcomes[i].result.completion += sr.piece;
      if (req->on_token) {
        GlobalMetrics().RecordStreamTokens(1);
        req->on_token(sr.piece);
        // Recheck cancellation after the streaming callback (connection may
        // have closed mid-chunk).
        if (req->cancellation_flag && req->cancellation_flag->load()) {
          states[i].active = false;
        }
      }
    }
  }

  double decode_ms = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - decode_start)
                         .count();

  // Finalise per-request results.
  for (std::size_t i = 0; i < n; ++i) {
    auto *req = eligible[i];
    auto &out = outcomes[i].result;
    int gen = states[i].tokens_generated;
    out.completion_tokens = gen;
    outcomes[i].decode_ms = decode_ms / static_cast<double>(n);

    // Fairness token accounting.
    bool first_slice = (req->total_completion_tokens == 0);
    int fairness_delta = gen;
    if (first_slice) {
      int prompt_comp = out.prompt_tokens;
      if (prompt_comp <= 0 && req->reported_prompt_tokens >= 0) {
        prompt_comp = req->reported_prompt_tokens;
      }
      fairness_delta += prompt_comp;
    }
    if (fairness_delta > 0) {
      GlobalMetrics().RecordFairnessTokens(req->priority_level, fairness_delta);
    }
    req->service_tokens += gen;
    req->total_completion_tokens += gen;

    // Fairness yield decision.
    bool exhausted = states[i].slice_active && gen >= states[i].decode_limit;
    if (states[i].slice_active && exhausted &&
        req->remaining_decode_tokens > 0) {
      req->fairness_yielded = true;
      GlobalMetrics().RecordFairnessYield(req->priority_level, gen,
                                          req->remaining_decode_tokens);
      req->phase = RequestPhase::kPending;
    } else {
      req->phase = RequestPhase::kFinished;
    }

    // Accumulate and track remaining decode budget.
    if (!out.completion.empty() &&
        out.completion != "[backend returned empty response]") {
      req->accumulated_output.append(out.completion);
      if (req->remaining_decode_tokens >= 0) {
        req->remaining_decode_tokens =
            std::max(0, req->remaining_decode_tokens - gen);
      }
    }

    req->first_token_time = std::chrono::steady_clock::now();
    // sequence_id left intact — scheduler handles FreeSequence + FreeSeqSlot.
  }

  GlobalMetrics().RecordDecodeDuration(decode_ms);
  return outcomes;
}

std::shared_ptr<LlamaCPUBackend>
BatchExecutor::ResolveBackend(const std::string &requested_model,
                              std::string *resolved_id) {
  if (!router_) {
    return nullptr;
  }
  auto *info = router_->Resolve(requested_model);
  if (!info) {
    return nullptr;
  }
  if (resolved_id) {
    *resolved_id = info->id;
  }
  return router_->GetBackend(info->id);
}

} // namespace inferflux
