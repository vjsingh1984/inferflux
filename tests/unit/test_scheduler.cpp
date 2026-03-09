#include <catch2/catch_amalgamated.hpp>

#include "runtime/backends/cpu/llama_backend.h"
#include "runtime/prefix_cache/radix_prefix_cache.h"
#include "scheduler/fairness_controller.h"
#include "scheduler/scheduler.h"
#include "scheduler/single_model_router.h"
#include "server/metrics/metrics.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

using namespace inferflux;

namespace {

int64_t ReadBatchTokenBudgetSkipsTotal() {
  const std::string output = GlobalMetrics().RenderPrometheus();
  const std::string key =
      "inferflux_scheduler_batch_token_budget_skips_total{backend=\"cpu\"} ";
  auto pos = output.find(key);
  if (pos == std::string::npos) {
    return 0;
  }
  pos += key.size();
  auto line_end = output.find('\n', pos);
  const std::string value = output.substr(
      pos, line_end == std::string::npos ? std::string::npos : line_end - pos);
  try {
    return std::stoll(value);
  } catch (const std::exception &) {
    return 0;
  }
}

class ReadyStubBackend : public LlamaCPUBackend {
public:
  explicit ReadyStubBackend(std::string output) : output_(std::move(output)) {}

  bool LoadModel(const std::filesystem::path &,
                 const LlamaBackendConfig &) override {
    return true;
  }

  bool IsReady() const override { return true; }

  PrefillResult Prefill(const std::string &, int) override {
    PrefillResult out;
    out.n_past = 1;
    out.ok = true;
    return out;
  }

  PrefillResult PrefillPartial(const std::string &, int,
                               int n_past_start) override {
    PrefillResult out;
    out.n_past = n_past_start + 1;
    out.ok = true;
    return out;
  }

  std::string Decode(int, int, int,
                     const std::function<bool(const std::string &,
                                              const TokenLogprob *)> &on_chunk,
                     const std::function<bool()> &, int,
                     std::vector<TokenLogprob> *out_logprobs, int,
                     const std::vector<std::string> &) override {
    if (out_logprobs) {
      TokenLogprob lp;
      lp.token = output_;
      lp.logprob = -0.1f;
      out_logprobs->push_back(lp);
    }
    if (on_chunk && !on_chunk(output_, nullptr)) {
      return {};
    }
    return output_;
  }

  std::string
  Generate(const std::string &, int,
           const std::function<bool(const std::string &, const TokenLogprob *)>
               &on_chunk,
           const std::function<bool()> &, int,
           std::vector<TokenLogprob> *out_logprobs,
           const std::vector<std::string> &) override {
    if (out_logprobs) {
      TokenLogprob lp;
      lp.token = output_;
      lp.logprob = -0.1f;
      out_logprobs->push_back(lp);
    }
    if (on_chunk && !on_chunk(output_, nullptr)) {
      return {};
    }
    return output_;
  }

  void FreeSequence(int) override {}

  int TokenCount(const std::string &text) const override {
    return static_cast<int>(text.size());
  }

  std::vector<int> TokenizeForCache(const std::string &) const override {
    return {1, 2, 3};
  }

private:
  std::string output_;
};

class AsyncLaneStubBackend final : public ReadyStubBackend {
public:
  explicit AsyncLaneStubBackend(std::string output)
      : ReadyStubBackend(std::move(output)) {}

  bool SupportsAsyncUnifiedBatch() const override { return true; }

  UnifiedBatchHandle
  SubmitUnifiedBatchAsync(const std::vector<UnifiedBatchInput> &inputs,
                          UnifiedBatchLane lane) override {
    const int submission_ticket =
        global_submission_ticket_.fetch_add(1, std::memory_order_relaxed) + 1;
    int expected_ticket = 0;
    first_submission_ticket_.compare_exchange_strong(
        expected_ticket, submission_ticket, std::memory_order_relaxed,
        std::memory_order_relaxed);

    if (lane == UnifiedBatchLane::kPrefill) {
      prefill_submissions_.fetch_add(1, std::memory_order_relaxed);
    } else if (lane == UnifiedBatchLane::kDecode) {
      decode_submissions_.fetch_add(1, std::memory_order_relaxed);
    } else {
      auto_submissions_.fetch_add(1, std::memory_order_relaxed);
    }

    const auto handle =
        next_handle_.fetch_add(1, std::memory_order_relaxed) + 1;
    std::lock_guard<std::mutex> lock(async_mutex_);
    async_results_[handle] = BuildOutputs(inputs, lane);
    return handle;
  }

  bool TryCollectUnifiedBatchAsync(
      UnifiedBatchHandle handle,
      std::vector<UnifiedBatchOutput> *outputs) override {
    if (!outputs || handle == 0) {
      return false;
    }
    std::lock_guard<std::mutex> lock(async_mutex_);
    auto it = async_results_.find(handle);
    if (it == async_results_.end()) {
      return false;
    }
    *outputs = std::move(it->second);
    async_results_.erase(it);
    return true;
  }

  std::vector<UnifiedBatchOutput>
  ExecuteUnifiedBatch(const std::vector<UnifiedBatchInput> &inputs) override {
    return BuildOutputs(inputs, UnifiedBatchLane::kAuto);
  }

  PrefillResult Prefill(const std::string &prompt, int sequence_id) override {
    prefill_fallback_calls_.fetch_add(1, std::memory_order_relaxed);
    return ReadyStubBackend::Prefill(prompt, sequence_id);
  }

  PrefillResult PrefillPartial(const std::string &prompt, int sequence_id,
                               int n_past_start) override {
    prefill_partial_fallback_calls_.fetch_add(1, std::memory_order_relaxed);
    return ReadyStubBackend::PrefillPartial(prompt, sequence_id, n_past_start);
  }

  std::vector<int> TokenizeForCache(const std::string &) const override {
    return {1, 2, 3, 4};
  }

  int PrefillSubmissions() const {
    return prefill_submissions_.load(std::memory_order_relaxed);
  }
  int DecodeSubmissions() const {
    return decode_submissions_.load(std::memory_order_relaxed);
  }
  int PrefillFallbackCalls() const {
    return prefill_fallback_calls_.load(std::memory_order_relaxed);
  }
  int PrefillPartialFallbackCalls() const {
    return prefill_partial_fallback_calls_.load(std::memory_order_relaxed);
  }
  int FirstSubmissionTicket() const {
    return first_submission_ticket_.load(std::memory_order_relaxed);
  }

private:
  std::vector<UnifiedBatchOutput>
  BuildOutputs(const std::vector<UnifiedBatchInput> &inputs,
               UnifiedBatchLane lane) const {
    std::vector<UnifiedBatchOutput> outputs(inputs.size());
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      outputs[i].ok = true;
      if (!inputs[i].request_logits) {
        continue;
      }
      const bool prefill_lane =
          lane == UnifiedBatchLane::kPrefill || inputs[i].tokens.size() > 1;
      outputs[i].token = prefill_lane ? 100 : 101;
      outputs[i].piece = prefill_lane ? "p" : "d";
    }
    return outputs;
  }

  std::atomic<uint64_t> next_handle_{0};
  mutable std::mutex async_mutex_;
  std::unordered_map<UnifiedBatchHandle, std::vector<UnifiedBatchOutput>>
      async_results_;
  std::atomic<int> prefill_submissions_{0};
  std::atomic<int> decode_submissions_{0};
  std::atomic<int> auto_submissions_{0};
  std::atomic<int> prefill_fallback_calls_{0};
  std::atomic<int> prefill_partial_fallback_calls_{0};
  std::atomic<int> first_submission_ticket_{0};

  static std::atomic<int> global_submission_ticket_;
};

std::atomic<int> AsyncLaneStubBackend::global_submission_ticket_{0};

class SessionLeaseStubBackend final : public ReadyStubBackend {
public:
  explicit SessionLeaseStubBackend(std::string output)
      : ReadyStubBackend(std::move(output)) {}

  void FreeSequence(int) override {
    free_sequence_calls_.fetch_add(1, std::memory_order_relaxed);
  }

  int FreeSequenceCalls() const {
    return free_sequence_calls_.load(std::memory_order_relaxed);
  }

private:
  std::atomic<int> free_sequence_calls_{0};
};

class RejectingKVTransport final : public disaggregated::IKVTransport {
public:
  bool Enqueue(disaggregated::KVPacket packet) override {
    (void)packet;
    enqueue_calls_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }

  std::optional<disaggregated::KVPacket> TryDequeue() override {
    return std::nullopt;
  }

  std::size_t Size() const override { return 0; }
  std::size_t Capacity() const override { return 0; }

  int EnqueueCalls() const {
    return enqueue_calls_.load(std::memory_order_relaxed);
  }

private:
  std::atomic<int> enqueue_calls_{0};
};

} // namespace

TEST_CASE("Scheduler stub response with no backend", "[scheduler]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      4, 1024, PagedKVCache::EvictionPolicy::kLRU);

  // No router → no backend → should return no_backend flag.
  Scheduler scheduler(tokenizer, device, cache, nullptr);

  InferenceRequest req;
  req.prompt = "Hello world";
  req.max_tokens = 10;
  auto fut = scheduler.Generate(req);
  auto resp = fut.get();

  REQUIRE(resp.no_backend);
  REQUIRE(!resp.completion.empty());
}

TEST_CASE("Scheduler with empty SingleModelRouter returns no_backend",
          "[scheduler]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      4, 1024, PagedKVCache::EvictionPolicy::kLRU);

  auto router = std::make_shared<SingleModelRouter>();
  Scheduler scheduler(tokenizer, device, cache, router);

  InferenceRequest req;
  req.prompt = "Hello";
  req.max_tokens = 5;
  auto fut = scheduler.Generate(req);
  auto resp = fut.get();

  REQUIRE(resp.no_backend);
}

TEST_CASE("on_token callback fires on prefix cache hit", "[scheduler]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      4, 1024, PagedKVCache::EvictionPolicy::kLRU);
  auto prefix_cache = std::make_shared<RadixPrefixCache>(
      cache, [](int) {}, RadixPrefixCacheLimits{1024, 12});

  const std::string prompt = "cached prompt";
  auto prompt_tokens = tokenizer.Encode(prompt);
  prefix_cache->Insert(prompt_tokens, {100}, 0, nullptr);

  Scheduler scheduler(tokenizer, device, cache, nullptr, nullptr, prefix_cache);

  std::vector<std::string> tokens_received;
  InferenceRequest req;
  req.prompt = prompt;
  req.max_tokens = 8;
  req.stream = true;
  req.on_token = [&](const std::string &tok, const TokenLogprob *) {
    tokens_received.push_back(tok);
  };

  auto fut = scheduler.Generate(std::move(req));
  auto resp = fut.get();

  REQUIRE(resp.no_backend);
  REQUIRE(!resp.completion.empty());
}

TEST_CASE("Scheduler clamps max_tokens=0 to 1", "[scheduler]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      4, 1024, PagedKVCache::EvictionPolicy::kLRU);
  Scheduler scheduler(tokenizer, device, cache, nullptr);

  InferenceRequest req;
  req.prompt = "test";
  req.max_tokens = 0;
  auto fut = scheduler.Generate(std::move(req));
  auto resp = fut.get();

  REQUIRE(resp.no_backend);
}

TEST_CASE("Scheduler prefill uses async unified prefill lane when available",
          "[scheduler]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      4, 1024, PagedKVCache::EvictionPolicy::kLRU);
  auto router = std::make_shared<SingleModelRouter>();
  auto backend = std::make_shared<AsyncLaneStubBackend>("ok");

  ModelInfo info;
  info.id = "lane-model";
  info.path = "/tmp/lane.gguf";
  info.backend = "cuda";
  REQUIRE(router->RegisterModel(info, backend));
  REQUIRE(router->SetDefaultModel(info.id));

  Scheduler scheduler(tokenizer, device, cache, router);

  InferenceRequest req;
  req.prompt = "lane activity";
  req.max_tokens = 2;
  auto resp = scheduler.Generate(std::move(req)).get();

  REQUIRE_FALSE(resp.no_backend);
  REQUIRE_FALSE(resp.completion.empty());
  REQUIRE(backend->PrefillSubmissions() > 0);
  REQUIRE(backend->DecodeSubmissions() > 0);
  REQUIRE(backend->PrefillFallbackCalls() == 0);
  REQUIRE(backend->PrefillPartialFallbackCalls() == 0);
}

TEST_CASE("Scheduler token budget accounts decode slices, not full prompts",
          "[scheduler]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      4, 1024, PagedKVCache::EvictionPolicy::kLRU);
  auto router = std::make_shared<SingleModelRouter>();
  auto backend = std::make_shared<ReadyStubBackend>(" next");

  ModelInfo info;
  info.id = "decode-budget-model";
  info.path = "/tmp/decode-budget.gguf";
  info.backend = "cpu";
  REQUIRE(router->RegisterModel(info, backend));
  REQUIRE(router->SetDefaultModel(info.id));

  FairnessConfig fairness_config;
  fairness_config.max_timeslice_tokens = 1;
  fairness_config.high_priority_threshold = 5;

  Scheduler::Config scheduler_config;
  scheduler_config.max_batch_size = 2;
  scheduler_config.max_batch_tokens = 12;
  scheduler_config.min_batch_size = 2;
  scheduler_config.batch_accumulation_ms = 20;

  Scheduler scheduler(
      tokenizer, device, cache, router, nullptr, nullptr, fairness_config,
      DisaggregatedConfig{},
      ModelSelectionOptions{/*allow_capability_fallback_for_default=*/true,
                            /*require_ready_backend=*/true},
      scheduler_config);

  GlobalMetrics().SetBackend("cpu");
  const int64_t skips_before = ReadBatchTokenBudgetSkipsTotal();

  auto make_request = []() {
    InferenceRequest req;
    req.prompt = "alpha beta gamma delta epsilon";
    req.max_tokens = 20;
    req.priority = 0;
    return req;
  };

  auto fut1 = scheduler.Generate(make_request());
  auto fut2 = scheduler.Generate(make_request());
  auto resp1 = fut1.get();
  auto resp2 = fut2.get();

  REQUIRE_FALSE(resp1.no_backend);
  REQUIRE_FALSE(resp2.no_backend);
  REQUIRE_FALSE(resp1.completion.empty());
  REQUIRE_FALSE(resp2.completion.empty());

  const int64_t skips_after = ReadBatchTokenBudgetSkipsTotal();
  REQUIRE(skips_after == skips_before);
}

TEST_CASE("Scheduler session handles preserve sequence until lease release",
          "[scheduler]") {
  auto backend = std::make_shared<SessionLeaseStubBackend>("ok");

  {
    SimpleTokenizer tokenizer;
    auto device = std::make_shared<CPUDeviceContext>();
    auto cache = std::make_shared<PagedKVCache>(
        16, 1024, PagedKVCache::EvictionPolicy::kLRU);
    auto router = std::make_shared<SingleModelRouter>();

    ModelInfo info;
    info.id = "session-model";
    info.path = "/tmp/session.gguf";
    info.backend = "cpu";
    REQUIRE(router->RegisterModel(info, backend));
    REQUIRE(router->SetDefaultModel(info.id));

    Scheduler::Config cfg;
    cfg.session_handles.enabled = true;
    cfg.session_handles.ttl_ms = 60000;
    cfg.session_handles.max_sessions = 64;

    Scheduler scheduler(tokenizer, device, cache, router, nullptr, nullptr,
                        FairnessConfig{}, DisaggregatedConfig{},
                        ModelSelectionOptions{}, cfg);

    InferenceRequest req1;
    req1.model = info.id;
    req1.session_id = "session-a";
    req1.prompt = "hello session";
    req1.max_tokens = 2;
    auto resp1 = scheduler.Generate(std::move(req1)).get();
    REQUIRE_FALSE(resp1.no_backend);

    InferenceRequest req2;
    req2.model = info.id;
    req2.session_id = "session-a";
    req2.prompt = "hello session";
    req2.max_tokens = 2;
    auto resp2 = scheduler.Generate(std::move(req2)).get();
    REQUIRE_FALSE(resp2.no_backend);

    // Session-owned sequences should reduce per-request sequence teardown.
    REQUIRE(backend->FreeSequenceCalls() < 2);
  }

  // Scheduler teardown drains session handles and frees retained sequence
  // state.
  REQUIRE(backend->FreeSequenceCalls() >= 1);
}

TEST_CASE("Scheduler fails fast when distributed enqueue retries are exhausted",
          "[scheduler][distributed]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      16, 1024, PagedKVCache::EvictionPolicy::kLRU);
  auto router = std::make_shared<SingleModelRouter>();
  auto backend = std::make_shared<SessionLeaseStubBackend>("ok");

  ModelInfo info;
  info.id = "dist-retry-model";
  info.path = "/tmp/dist-retry.gguf";
  info.backend = "cpu";
  REQUIRE(router->RegisterModel(info, backend));
  REQUIRE(router->SetDefaultModel(info.id));

  auto rejecting_transport = std::make_shared<RejectingKVTransport>();
  DisaggregatedConfig disagg_config;
  disagg_config.decode_pool_size = 1;
  disagg_config.kv_transport = rejecting_transport;
  disagg_config.kv_enqueue_max_retries = 1;

  Scheduler scheduler(tokenizer, device, cache, router, nullptr, nullptr,
                      FairnessConfig{}, disagg_config);

  InferenceRequest req;
  req.model = info.id;
  req.prompt = "distributed retry test";
  req.max_tokens = 4;

  auto resp = scheduler.Generate(std::move(req)).get();
  REQUIRE(resp.no_backend);
  REQUIRE(resp.completion.find("distributed_overloaded") != std::string::npos);
  REQUIRE(rejecting_transport->EnqueueCalls() >= 2);
  REQUIRE(backend->FreeSequenceCalls() >= 1);
}

TEST_CASE("FairnessController evaluation", "[fairness]") {
  FairnessConfig cfg;
  cfg.enable_preemption = true;
  cfg.high_priority_threshold = 5;
  FairnessController controller(cfg);

  InferenceRequest low, high;
  low.priority_level = 1;
  high.priority_level = 10;

  std::vector<FairnessEntry> batch{{&low, 1, 0}};
  std::vector<FairnessEntry> queue{{&high, 10, 0}};

  auto decision = controller.Evaluate(batch, queue);
  REQUIRE(decision.swap);
}

TEST_CASE(
    "Scheduler falls back to compatible backend for default model routing",
    "[scheduler]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      4, 1024, PagedKVCache::EvictionPolicy::kLRU);
  auto router = std::make_shared<SingleModelRouter>();

  ModelInfo cuda_info;
  cuda_info.id = "shared-cuda";
  cuda_info.path = "/tmp/shared.gguf";
  cuda_info.backend = "cuda";
  REQUIRE(router->RegisterModel(
      cuda_info, std::make_shared<ReadyStubBackend>("from-cuda")));

  ModelInfo cpu_info;
  cpu_info.id = "shared-cpu";
  cpu_info.path = "/tmp/shared.gguf";
  cpu_info.backend = "cpu";
  REQUIRE(router->RegisterModel(
      cpu_info, std::make_shared<ReadyStubBackend>("from-cpu")));
  REQUIRE(router->SetDefaultModel("shared-cuda"));

  auto *primary = router->Resolve("shared-cuda");
  REQUIRE(primary != nullptr);
  primary->capabilities.supports_logprobs = false;

  Scheduler scheduler(tokenizer, device, cache, router);

  InferenceRequest req;
  req.prompt = "hello";
  req.max_tokens = 4;
  req.collect_logprobs = true;

  auto resp = scheduler.Generate(req).get();
  REQUIRE_FALSE(resp.no_backend);
  REQUIRE(resp.model_id == "shared-cpu");
  REQUIRE_FALSE(resp.completion.empty());
}

TEST_CASE("Scheduler routes streaming requests to streaming-capable backend",
          "[scheduler]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      4, 1024, PagedKVCache::EvictionPolicy::kLRU);
  auto router = std::make_shared<SingleModelRouter>();

  ModelInfo cuda_info;
  cuda_info.id = "shared-cuda";
  cuda_info.path = "/tmp/shared.gguf";
  cuda_info.backend = "cuda";
  REQUIRE(router->RegisterModel(
      cuda_info, std::make_shared<ReadyStubBackend>("from-cuda")));

  ModelInfo cpu_info;
  cpu_info.id = "shared-cpu";
  cpu_info.path = "/tmp/other.gguf";
  cpu_info.backend = "cpu";
  REQUIRE(router->RegisterModel(
      cpu_info, std::make_shared<ReadyStubBackend>("from-cpu")));
  REQUIRE(router->SetDefaultModel("shared-cuda"));

  auto *primary = router->Resolve("shared-cuda");
  REQUIRE(primary != nullptr);
  primary->capabilities.supports_streaming = false;

  Scheduler scheduler(tokenizer, device, cache, router);

  InferenceRequest req;
  req.prompt = "hello";
  req.max_tokens = 4;
  req.stream = true;

  auto resp = scheduler.Generate(req).get();
  REQUIRE_FALSE(resp.no_backend);
  REQUIRE(resp.model_id == "shared-cpu");
  REQUIRE_FALSE(resp.completion.empty());
}

TEST_CASE("Scheduler respects same-path fallback routing scope",
          "[scheduler]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      4, 1024, PagedKVCache::EvictionPolicy::kLRU);
  auto router = std::make_shared<SingleModelRouter>();

  ModelInfo primary;
  primary.id = "default-cuda";
  primary.path = "/tmp/shared.gguf";
  primary.backend = "cuda";
  REQUIRE(router->RegisterModel(
      primary, std::make_shared<ReadyStubBackend>("from-cuda")));

  ModelInfo cross_path_fallback;
  cross_path_fallback.id = "other-cpu";
  cross_path_fallback.path = "/tmp/other.gguf";
  cross_path_fallback.backend = "cpu";
  REQUIRE(router->RegisterModel(
      cross_path_fallback, std::make_shared<ReadyStubBackend>("from-cpu")));
  REQUIRE(router->SetDefaultModel(primary.id));

  auto *resolved_primary = router->Resolve(primary.id);
  REQUIRE(resolved_primary != nullptr);
  resolved_primary->capabilities.supports_logprobs = false;

  ModelSelectionOptions selection_options;
  selection_options.allow_capability_fallback_for_default = true;
  selection_options.require_ready_backend = true;
  selection_options.capability_fallback_scope =
      CapabilityFallbackScope::kSamePathOnly;

  Scheduler scheduler(tokenizer, device, cache, router, nullptr, nullptr,
                      FairnessConfig{}, DisaggregatedConfig{},
                      selection_options);

  InferenceRequest req;
  req.prompt = "hello";
  req.max_tokens = 4;
  req.collect_logprobs = true;

  auto resp = scheduler.Generate(req).get();
  REQUIRE(resp.no_backend);
  REQUIRE(resp.model_id.empty());
  REQUIRE(resp.completion.find("logprobs") != std::string::npos);
}

TEST_CASE("Scheduler applies runtime routing policy updates", "[scheduler]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      4, 1024, PagedKVCache::EvictionPolicy::kLRU);
  auto router = std::make_shared<SingleModelRouter>();

  ModelInfo primary;
  primary.id = "default-cuda";
  primary.path = "/tmp/shared.gguf";
  primary.backend = "cuda";
  REQUIRE(router->RegisterModel(
      primary, std::make_shared<ReadyStubBackend>("from-cuda")));

  ModelInfo cross_path_fallback;
  cross_path_fallback.id = "other-cpu";
  cross_path_fallback.path = "/tmp/other.gguf";
  cross_path_fallback.backend = "cpu";
  REQUIRE(router->RegisterModel(
      cross_path_fallback, std::make_shared<ReadyStubBackend>("from-cpu")));
  REQUIRE(router->SetDefaultModel(primary.id));

  auto *resolved_primary = router->Resolve(primary.id);
  REQUIRE(resolved_primary != nullptr);
  resolved_primary->capabilities.supports_logprobs = false;

  ModelSelectionOptions selection_options;
  selection_options.allow_capability_fallback_for_default = true;
  selection_options.require_ready_backend = true;
  selection_options.capability_fallback_scope =
      CapabilityFallbackScope::kAnyCompatible;

  Scheduler scheduler(tokenizer, device, cache, router, nullptr, nullptr,
                      FairnessConfig{}, DisaggregatedConfig{},
                      selection_options);

  InferenceRequest req_before;
  req_before.prompt = "hello";
  req_before.max_tokens = 4;
  req_before.collect_logprobs = true;
  auto resp_before = scheduler.Generate(req_before).get();
  REQUIRE_FALSE(resp_before.no_backend);
  REQUIRE(resp_before.model_id == "other-cpu");

  ModelSelectionOptions tightened = selection_options;
  tightened.capability_fallback_scope = CapabilityFallbackScope::kSamePathOnly;
  scheduler.UpdateModelSelectionOptions(tightened);

  auto snapshot = scheduler.ModelSelectionOptionsSnapshot();
  REQUIRE(snapshot.capability_fallback_scope ==
          CapabilityFallbackScope::kSamePathOnly);

  InferenceRequest req_after;
  req_after.prompt = "hello";
  req_after.max_tokens = 4;
  req_after.collect_logprobs = true;
  auto resp_after = scheduler.Generate(req_after).get();
  REQUIRE(resp_after.no_backend);
  REQUIRE(resp_after.completion.find("logprobs") != std::string::npos);
}

TEST_CASE("Scheduler does not auto-fallback for explicit model requests",
          "[scheduler]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      4, 1024, PagedKVCache::EvictionPolicy::kLRU);
  auto router = std::make_shared<SingleModelRouter>();

  ModelInfo cuda_info;
  cuda_info.id = "shared-cuda";
  cuda_info.path = "/tmp/shared.gguf";
  cuda_info.backend = "cuda";
  REQUIRE(router->RegisterModel(
      cuda_info, std::make_shared<ReadyStubBackend>("from-cuda")));

  ModelInfo cpu_info;
  cpu_info.id = "shared-cpu";
  cpu_info.path = "/tmp/shared.gguf";
  cpu_info.backend = "cpu";
  REQUIRE(router->RegisterModel(
      cpu_info, std::make_shared<ReadyStubBackend>("from-cpu")));
  REQUIRE(router->SetDefaultModel("shared-cuda"));

  auto *primary = router->Resolve("shared-cuda");
  REQUIRE(primary != nullptr);
  primary->capabilities.supports_logprobs = false;

  Scheduler scheduler(tokenizer, device, cache, router);

  InferenceRequest req;
  req.model = "shared-cuda";
  req.prompt = "hello";
  req.max_tokens = 4;
  req.collect_logprobs = true;

  auto resp = scheduler.Generate(req).get();
  REQUIRE(resp.no_backend);
  REQUIRE(resp.model_id.empty());
  REQUIRE(resp.completion.find("logprobs") != std::string::npos);
}

TEST_CASE("Scheduler batch policy parse and stringify are stable",
          "[scheduler]") {
  REQUIRE(SchedulerBatchPolicyToString(SchedulerBatchPolicy::kPriorityAge) ==
          "priority_age");
  REQUIRE(SchedulerBatchPolicyToString(SchedulerBatchPolicy::kLpmPriority) ==
          "lpm_priority");
  REQUIRE(
      SchedulerBatchPolicyToString(SchedulerBatchPolicy::kThroughputBalanced) ==
      "throughput_balanced");

  REQUIRE(IsSchedulerBatchPolicyValue("priority_age"));
  REQUIRE(IsSchedulerBatchPolicyValue("LPM_PRIORITY"));
  REQUIRE(IsSchedulerBatchPolicyValue("throughput_balanced"));
  REQUIRE_FALSE(IsSchedulerBatchPolicyValue("unknown_policy"));

  REQUIRE(ParseSchedulerBatchPolicy("priority_age") ==
          SchedulerBatchPolicy::kPriorityAge);
  REQUIRE(ParseSchedulerBatchPolicy("lpm_priority") ==
          SchedulerBatchPolicy::kLpmPriority);
  REQUIRE(ParseSchedulerBatchPolicy("THROUGHPUT_BALANCED") ==
          SchedulerBatchPolicy::kThroughputBalanced);
  REQUIRE(ParseSchedulerBatchPolicy(
              "invalid", SchedulerBatchPolicy::kThroughputBalanced) ==
          SchedulerBatchPolicy::kThroughputBalanced);
}

TEST_CASE("Scheduler preserves configured batch policy", "[scheduler]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      4, 1024, PagedKVCache::EvictionPolicy::kLRU);

  Scheduler::Config cfg;
  cfg.batch_policy = SchedulerBatchPolicy::kLpmPriority;
  cfg.continuous_decode_steps = 3;
  cfg.chunked_prefill_tokens = 96;
  cfg.mixed_prefill_budget_ratio = 0.4;
  Scheduler scheduler(tokenizer, device, cache, nullptr, nullptr, nullptr,
                      FairnessConfig{}, DisaggregatedConfig{},
                      ModelSelectionOptions{}, cfg);

  REQUIRE(scheduler.BatchPolicy() == SchedulerBatchPolicy::kLpmPriority);
  REQUIRE(scheduler.ContinuousDecodeSteps() == 3);
  REQUIRE(scheduler.ChunkedPrefillTokens() == 96);
  REQUIRE(scheduler.MixedPrefillBudgetRatio() ==
          Catch::Approx(0.4).epsilon(1e-6));
}

TEST_CASE("Scheduler normalizes mixed-step tuning bounds", "[scheduler]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      4, 1024, PagedKVCache::EvictionPolicy::kLRU);

  Scheduler::Config cfg;
  cfg.continuous_decode_steps = -7;
  cfg.chunked_prefill_tokens = 0;
  cfg.mixed_prefill_budget_ratio = 2.5;
  Scheduler scheduler(tokenizer, device, cache, nullptr, nullptr, nullptr,
                      FairnessConfig{}, DisaggregatedConfig{},
                      ModelSelectionOptions{}, cfg);

  REQUIRE(scheduler.ContinuousDecodeSteps() == 0);
  REQUIRE(scheduler.ChunkedPrefillTokens() == 1);
  REQUIRE(scheduler.MixedPrefillBudgetRatio() ==
          Catch::Approx(1.0).epsilon(1e-6));
}

TEST_CASE("Scheduler lpm policy prioritizes prefix-affinity request",
          "[scheduler]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      8, 1024, PagedKVCache::EvictionPolicy::kLRU);
  auto router = std::make_shared<SingleModelRouter>();
  auto prefix_cache = std::make_shared<RadixPrefixCache>(
      cache, [](int) {}, RadixPrefixCacheLimits{1024, 32});

  auto cold_backend = std::make_shared<AsyncLaneStubBackend>("cold");
  auto hot_backend = std::make_shared<AsyncLaneStubBackend>("hot");

  ModelInfo cold_info;
  cold_info.id = "cold-model";
  cold_info.path = "/tmp/cold.gguf";
  cold_info.backend = "cpu";
  REQUIRE(router->RegisterModel(cold_info, cold_backend));

  ModelInfo hot_info;
  hot_info.id = "hot-model";
  hot_info.path = "/tmp/hot.gguf";
  hot_info.backend = "cpu";
  REQUIRE(router->RegisterModel(hot_info, hot_backend));
  REQUIRE(router->SetDefaultModel(cold_info.id));

  const std::string hot_prompt = "prefix hot prompt";
  auto hot_tokens = tokenizer.Encode(hot_prompt);
  prefix_cache->Insert(hot_tokens, {101}, 7, hot_backend);

  Scheduler::Config cfg;
  cfg.max_batch_size = 1;
  cfg.min_batch_size = 2;
  cfg.batch_accumulation_ms = 20;
  cfg.batch_policy = SchedulerBatchPolicy::kLpmPriority;

  Scheduler scheduler(tokenizer, device, cache, router, nullptr, prefix_cache,
                      FairnessConfig{}, DisaggregatedConfig{},
                      ModelSelectionOptions{}, cfg);

  InferenceRequest cold_req;
  cold_req.model = cold_info.id;
  cold_req.prompt = "cold request prompt";
  cold_req.max_tokens = 2;

  InferenceRequest hot_req;
  hot_req.model = hot_info.id;
  hot_req.prompt = hot_prompt;
  hot_req.max_tokens = 2;

  auto cold_future = scheduler.Generate(std::move(cold_req));
  auto hot_future = scheduler.Generate(std::move(hot_req));

  auto cold_resp = cold_future.get();
  auto hot_resp = hot_future.get();
  REQUIRE_FALSE(cold_resp.no_backend);
  REQUIRE_FALSE(hot_resp.no_backend);

  REQUIRE(hot_backend->FirstSubmissionTicket() > 0);
  REQUIRE(cold_backend->FirstSubmissionTicket() > 0);
  REQUIRE(hot_backend->FirstSubmissionTicket() <
          cold_backend->FirstSubmissionTicket());
}
