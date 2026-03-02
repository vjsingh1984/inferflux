#include "runtime/execution/batch_executor.h"
#include "runtime/backends/backend_utils.h"
#include "server/logging/logger.h"
#include "server/metrics/metrics.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <unordered_map>

using json = nlohmann::json;

namespace inferflux {

namespace {

// RAII guard that calls SetupSampler before execution and TeardownSampler on
// scope exit.  Always sets up a sampler chain (grammar optional), so sampling
// params (temperature, top_p, …) are always in effect.
class SamplerScope {
public:
  SamplerScope(const InferenceRequest &req,
               const std::shared_ptr<LlamaCPUBackend> &backend)
      : backend_(backend ? backend.get() : nullptr) {
    if (!backend_)
      return;
    const auto &c = req.response_constraint;
    backend_->SetupSampler(c.has_grammar ? c.grammar : "",
                           c.has_grammar ? c.root : "", req.sampling);
    active_ = true;
  }

  // Alternate constructor for unified batch phased execution where grammar and
  // sampling params are provided directly (not via InferenceRequest).
  SamplerScope(LlamaCPUBackend *backend, const std::string &grammar,
               const std::string &root, const SamplingParams &sp)
      : backend_(backend) {
    if (!backend_)
      return;
    backend_->SetupSampler(grammar, root, sp);
    active_ = true;
  }

  SamplerScope(const SamplerScope &) = delete;
  SamplerScope &operator=(const SamplerScope &) = delete;

  ~SamplerScope() {
    if (active_ && backend_) {
      backend_->TeardownSampler();
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
    std::shared_ptr<SpeculativeDecoder> speculative_decoder)
    : tokenizer_(tokenizer), device_(std::move(device)),
      cache_(std::move(cache)), router_(std::move(router)),
      speculative_decoder_(std::move(speculative_decoder)) {}

std::vector<InferenceResult> BatchExecutor::ExecuteBatch(
    const RequestBatch &batch,
    const std::vector<std::shared_ptr<LlamaCPUBackend>> &backend_overrides) {
  std::size_t n = batch.requests.size();
  std::vector<InferenceResult> results(n);
  std::vector<bool> handled(n, false);
  double total_prefill_ms = 0.0;
  double total_decode_ms = 0.0;

  // Identify requests eligible for multi-sequence batch decode and group them
  // by backend instance so mixed-model/multi-hardware batches still benefit
  // from unified phased execution per backend.
  // Eligibility: phased decode (n_past >= 0, seq_id >= 0), no grammar
  // constraints, no logprob collection, no response format.
  // Grammar uses per-backend grammar_sampler_ — not safe to interleave across
  // backends.
  struct EligibleGroup {
    std::shared_ptr<LlamaCPUBackend> backend;
    std::vector<std::size_t> indices;
  };
  std::vector<EligibleGroup> eligible_groups;
  std::unordered_map<LlamaCPUBackend *, std::size_t> group_index_by_backend;

  for (std::size_t i = 0; i < n; ++i) {
    auto *req = batch.requests[i];
    auto be = (i < backend_overrides.size()) ? backend_overrides[i] : nullptr;
    if (!be) {
      be = ResolveBackend(req->model, nullptr);
    }
    if (req->n_past >= 0 && req->sequence_id >= 0 &&
        !req->response_constraint.has_grammar && !req->collect_logprobs &&
        !req->has_response_format && be && be->IsReady()) {
      auto *key = be.get();
      auto it = group_index_by_backend.find(key);
      if (it == group_index_by_backend.end()) {
        group_index_by_backend.emplace(key, eligible_groups.size());
        eligible_groups.push_back({be, {}});
        it = group_index_by_backend.find(key);
      }
      eligible_groups[it->second].indices.push_back(i);
    }
  }

  for (const auto &group : eligible_groups) {
    std::vector<InferenceRequest *> eligible_reqs;
    eligible_reqs.reserve(group.indices.size());
    for (auto idx : group.indices) {
      eligible_reqs.push_back(batch.requests[idx]);
    }
    // Use the unified execution path (§P1b) to interleave prefill and decode
    // tokens.
    auto outcomes = ExecuteUnifiedBatchPhased(eligible_reqs, group.backend);
    for (std::size_t j = 0; j < group.indices.size(); ++j) {
      results[group.indices[j]] = std::move(outcomes[j].result);
      total_decode_ms += outcomes[j].decode_ms;
      handled[group.indices[j]] = true;
    }
  }

  // Process remaining requests individually (non-eligible).
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
      if (kv_page >= 0) {
        inference.block_table = {kv_page};
      }
    } catch (const std::exception &ex) {
      log::Error("batch_executor",
                 std::string("KV cache reserve failed: ") + ex.what());
    }
  }

  // Decode phase: token generation (speculative or direct).
  inference.phase = RequestPhase::kDecode;
  auto decode_start = std::chrono::steady_clock::now();

  SamplerScope sampler_scope(inference, backend);

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
      std::function<bool(const std::string &, const TokenLogprob *)> chunk_cb;
      if (inference.on_token) {
        auto on_token = inference.on_token;
        auto cancel_flag = inference.cancellation_flag;
        chunk_cb = [on_token, cancel_flag](const std::string &token_chunk,
                                           const TokenLogprob *lp) {
          GlobalMetrics().RecordStreamTokens(1);
          on_token(token_chunk, lp);
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
        text =
            backend->Decode(inference.n_past, inference.sequence_id,
                            decode_limit, chunk_cb, should_stop, logprob_top_n,
                            lp_out, inference.first_token, inference.stop);
      } else if (inference.has_images && backend->SupportsVision()) {
        text = backend->GenerateWithImages(inference.prompt, inference.images,
                                           decode_limit, chunk_cb, should_stop,
                                           inference.stop);
      } else {
        text = backend->Generate(inference.prompt, decode_limit, chunk_cb,
                                 should_stop, logprob_top_n, lp_out,
                                 inference.stop);
      }
      // Record GGML-native perf counters (P1b).
      {
        auto perf = backend->TakePerf();
        if (perf.generated_tokens > 0) {
          GlobalMetrics().RecordLlamaPerf(perf.prefill_ms, perf.decode_ms,
                                          perf.prompt_tokens,
                                          perf.generated_tokens);
        }
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
  // finish_reason="length" when the full max_tokens budget is exhausted.
  // inference.total_completion_tokens holds prior-slice counts; adding this
  // slice's count gives the running total.  Fairness mid-slice yields will
  // not trigger this because their partial total stays below max_tokens.
  if (!response.no_backend && inference.max_tokens > 0 &&
      inference.total_completion_tokens + response.completion_tokens >=
          inference.max_tokens) {
    response.finish_reason_length = true;
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
          cache_->ReleaseBlocks(inference.block_table);
          kv_page = -1;
          inference.block_table.clear();
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
    cache_->ReleaseBlocks(inference.block_table);
    inference.block_table.clear();
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

      // Correctness Fix (§ Item 1): decode and check for stop sequences.
      std::string piece = req->first_piece;
      out.completion += piece;
      states[i].tokens_generated = 1;

      std::string emit_piece;
      bool stop_hit = ApplyStop(piece, out.completion, req->stop, &emit_piece);

      if (req->on_token && !emit_piece.empty()) {
        GlobalMetrics().RecordStreamTokens(1);
        req->on_token(emit_piece, nullptr);
      }

      if (stop_hit ||
          (req->cancellation_flag && req->cancellation_flag->load())) {
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
        req->on_token(sr.piece, nullptr);
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

std::vector<BatchExecutor::ExecutionOutcome>
BatchExecutor::ExecuteUnifiedBatchPhased(
    std::vector<InferenceRequest *> &eligible,
    std::shared_ptr<LlamaCPUBackend> backend) {
  std::size_t n = eligible.size();
  std::vector<ExecutionOutcome> outcomes(n);

  struct ReqState {
    bool active{true};
    int tokens_generated{0};
    int decode_limit{0};
    int current_token{-1};
    bool slice_active{false};
    bool in_prefill{false};
  };
  std::vector<ReqState> states(n);

  auto exec_start = std::chrono::steady_clock::now();

  for (std::size_t i = 0; i < n; ++i) {
    auto *req = eligible[i];
    auto &out = outcomes[i].result;
    req->fairness_yielded = false;
    out.model_id =
        req->resolved_model.empty() ? req->model : req->resolved_model;
    out.prompt_tokens = static_cast<int>(req->prompt_tokens.size());

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

    // If n_past is 0 and bpe_prompt_tokens is not empty, this is a fresh
    // prefill. (Note: In the current scheduler, Prefill() is called BEFORE
    // ExecuteBatch, so req->n_past > 0 and req->first_token is already set. We
    // handle both cases for future flexibility where Prefill might be
    // interleaved).
    if (req->n_past >= 0 && req->first_token >= 0) {
      states[i].current_token = req->first_token;

      // Correctness Fix (§ Item 1): decode and check for stop sequences.
      std::string piece = req->first_piece;
      out.completion += piece;
      states[i].tokens_generated = 1;

      std::string emit_piece;
      bool stop_hit = ApplyStop(piece, out.completion, req->stop, &emit_piece);

      if (req->on_token && !emit_piece.empty()) {
        GlobalMetrics().RecordStreamTokens(1);
        req->on_token(emit_piece, nullptr);
      }

      if (stop_hit ||
          (req->cancellation_flag && req->cancellation_flag->load())) {
        states[i].active = false;
      }
    } else if (req->n_past == 0 && !req->bpe_prompt_tokens.empty()) {
      states[i].in_prefill = true;
    } else {
      states[i].active = false;
      out.completion = "[batch state error]";
    }
  }

  const int kMaxPrefillChunkSize = 512; // §P1d
  int loop_count = 0;

  while (true) {
    if (++loop_count > 10000) {
      log::Error("batch_executor", "DEADLOCK GUARD: ExecuteUnifiedBatchPhased "
                                   "exceeded 10000 iterations");
      break;
    }

    std::vector<LlamaCPUBackend::UnifiedBatchInput> batch_inputs;
    std::vector<std::size_t> active_idx;

    for (std::size_t i = 0; i < n; ++i) {
      if (!states[i].active)
        continue;
      auto *req = eligible[i];
      if (req->cancellation_flag && req->cancellation_flag->load()) {
        states[i].active = false;
        continue;
      }

      if (states[i].in_prefill) {
        // Slice the next chunk of the prompt (§P1d).
        int remaining = static_cast<int>(req->bpe_prompt_tokens.size()) -
                        req->prefill_offset;
        int chunk_size = std::min(remaining, kMaxPrefillChunkSize);
        if (chunk_size <= 0) {
          states[i].in_prefill = false; // Safety
          continue;
        }
        bool is_last_chunk = (req->prefill_offset + chunk_size >=
                              static_cast<int>(req->bpe_prompt_tokens.size()));

        std::vector<int> chunk(
            req->bpe_prompt_tokens.begin() + req->prefill_offset,
            req->bpe_prompt_tokens.begin() + req->prefill_offset + chunk_size);

        // Only request logits on the final chunk of the prefill.
        batch_inputs.push_back({req->sequence_id, req->prefill_offset,
                                std::move(chunk), is_last_chunk,
                                req->sampling});
        active_idx.push_back(i);
      } else if (states[i].tokens_generated < states[i].decode_limit) {
        batch_inputs.push_back({req->sequence_id,
                                req->n_past,
                                {states[i].current_token},
                                true,
                                req->sampling});
        active_idx.push_back(i);
      } else {
        states[i].active = false;
      }
    }

    if (active_idx.empty())
      break;

    auto step = backend->ExecuteUnifiedBatch(batch_inputs);
    if (step.empty()) {
      // Backend returned nothing for active inputs — fail them to avoid loop.
      for (auto idx : active_idx)
        states[idx].active = false;
      break;
    }

    for (std::size_t j = 0; j < active_idx.size(); ++j) {
      std::size_t i = active_idx[j];
      auto *req = eligible[i];
      const auto &res = step[j];

      if (states[i].in_prefill) {
        int chunk_processed = static_cast<int>(batch_inputs[j].tokens.size());
        req->prefill_offset += chunk_processed;
        req->n_past = req->prefill_offset;

        if (req->prefill_offset >=
            static_cast<int>(req->bpe_prompt_tokens.size())) {
          states[i].in_prefill = false;
          // After final chunk, the 'res' contains the first generated token.
          if (!res.ok || res.token < 0) {
            states[i].active = false;
            continue;
          }
          states[i].current_token = res.token;
          states[i].tokens_generated++;

          // Correctness Fix (§ Item 1): decode and check for stop sequences.
          std::string piece = res.piece;
          outcomes[i].result.completion += piece;

          std::string emit_piece;
          bool stop_hit = ApplyStop(piece, outcomes[i].result.completion,
                                    req->stop, &emit_piece);

          if (req->on_token && !emit_piece.empty()) {
            GlobalMetrics().RecordStreamTokens(1);
            req->on_token(emit_piece, nullptr);
          }

          if (stop_hit) {
            states[i].active = false;
          }
        }
      } else {
        // Decode step processing.
        req->n_past += 1;
        if (!res.ok || res.token < 0) {
          states[i].active = false;
          continue;
        }

        states[i].current_token = res.token;
        states[i].tokens_generated++;

        // Correctness Fix (§P1d): decode and check for stop sequences.
        std::string piece = res.piece;
        outcomes[i].result.completion += piece;

        std::string emit_piece;
        bool stop_hit = ApplyStop(piece, outcomes[i].result.completion,
                                  req->stop, &emit_piece);

        if (req->on_token && !emit_piece.empty()) {
          GlobalMetrics().RecordStreamTokens(1);
          req->on_token(emit_piece, nullptr);
          if (req->cancellation_flag && req->cancellation_flag->load()) {
            states[i].active = false;
          }
        }

        if (stop_hit) {
          states[i].active = false;
        }
      }
    }
  }

  double total_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - exec_start)
                        .count();

  for (std::size_t i = 0; i < n; ++i) {
    auto *req = eligible[i];
    auto &out = outcomes[i].result;
    int gen = states[i].tokens_generated;
    out.completion_tokens = gen;
    outcomes[i].decode_ms = total_ms / static_cast<double>(n);

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

    if (!out.completion.empty() && out.completion != "[batch state error]") {
      req->accumulated_output.append(out.completion);
      if (req->remaining_decode_tokens >= 0) {
        req->remaining_decode_tokens =
            std::max(0, req->remaining_decode_tokens - gen);
      }
    }
    req->first_token_time = std::chrono::steady_clock::now();
  }

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

void BatchExecutor::ExecuteUnifiedBatchStep(
    RequestBatch &batch, std::shared_ptr<LlamaCPUBackend> backend) {
  if (!backend || batch.empty())
    return;

  const int kMaxPrefillChunkSize = 512;
  std::vector<LlamaCPUBackend::UnifiedBatchInput> batch_inputs;
  std::vector<InferenceRequest *> active_reqs;

  for (auto *req : batch.requests) {
    if (!req->exec_active || req->phase == RequestPhase::kFinished ||
        req->phase == RequestPhase::kAborted) {
      continue;
    }

    if (req->cancellation_flag && req->cancellation_flag->load()) {
      req->exec_active = false;
      req->phase = RequestPhase::kAborted;
      continue;
    }

    if (req->exec_in_prefill) {
      int remaining =
          static_cast<int>(req->bpe_prompt_tokens.size()) - req->prefill_offset;
      int chunk_size = std::min(remaining, kMaxPrefillChunkSize);
      if (chunk_size <= 0) {
        req->exec_in_prefill = false;
        continue;
      }
      bool is_last_chunk = (req->prefill_offset + chunk_size >=
                            static_cast<int>(req->bpe_prompt_tokens.size()));
      std::vector<int> chunk(
          req->bpe_prompt_tokens.begin() + req->prefill_offset,
          req->bpe_prompt_tokens.begin() + req->prefill_offset + chunk_size);
      batch_inputs.push_back({req->sequence_id, req->prefill_offset,
                              std::move(chunk), is_last_chunk, req->sampling});
      active_reqs.push_back(req);
    } else if (req->exec_tokens_generated < req->exec_decode_limit) {
      batch_inputs.push_back({req->sequence_id,
                              req->n_past,
                              {req->exec_current_token},
                              true,
                              req->sampling});
      active_reqs.push_back(req);
    } else {
      req->exec_active = false;
      req->phase = RequestPhase::kFinished;
    }
  }

  if (active_reqs.empty()) {
    return;
  }

  auto step_outputs = backend->ExecuteUnifiedBatch(batch_inputs);
  if (step_outputs.empty()) {
    for (auto *req : active_reqs)
      req->exec_active = false;
    return;
  }

  for (std::size_t j = 0; j < active_reqs.size(); ++j) {
    auto *req = active_reqs[j];
    const auto &res = step_outputs[j];

    if (req->exec_in_prefill) {
      req->prefill_offset += static_cast<int>(batch_inputs[j].tokens.size());
      if (req->prefill_offset >=
          static_cast<int>(req->bpe_prompt_tokens.size())) {
        req->exec_in_prefill = false;
        if (!res.ok || res.token < 0) {
          req->exec_active = false;
          continue;
        }
        req->exec_current_token = res.token;
        req->exec_tokens_generated++;
        req->n_past = req->prefill_offset;

        std::string piece = res.piece;
        req->exec_result.completion += piece;
        std::string emit_piece;
        bool stop_hit = ApplyStop(piece, req->exec_result.completion, req->stop,
                                  &emit_piece);
        if (req->on_token && !emit_piece.empty()) {
          GlobalMetrics().RecordStreamTokens(1);
          req->on_token(emit_piece, nullptr);
        }
        if (stop_hit)
          req->exec_active = false;
      }
    } else {
      req->n_past += 1;
      if (!res.ok || res.token < 0) {
        req->exec_active = false;
        continue;
      }
      req->exec_current_token = res.token;
      req->exec_tokens_generated++;

      std::string piece = res.piece;
      req->exec_result.completion += piece;
      std::string emit_piece;
      bool stop_hit =
          ApplyStop(piece, req->exec_result.completion, req->stop, &emit_piece);
      if (req->on_token && !emit_piece.empty()) {
        GlobalMetrics().RecordStreamTokens(1);
        req->on_token(emit_piece, nullptr);
        if (req->cancellation_flag && req->cancellation_flag->load()) {
          req->exec_active = false;
        }
      }
      if (stop_hit)
        req->exec_active = false;
    }
  }
}

} // namespace inferflux
