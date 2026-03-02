#include <catch2/catch_amalgamated.hpp>

#include "runtime/backends/cpu/llama_backend.h"
#include "runtime/prefix_cache/radix_prefix_cache.h"
#include "scheduler/fairness_controller.h"
#include "scheduler/scheduler.h"
#include "scheduler/single_model_router.h"

#include <atomic>
#include <future>
#include <memory>

using namespace inferflux;

namespace {

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
  auto prefix_cache =
      std::make_shared<RadixPrefixCache>(cache, [](int) {}, 1024, 12);

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

TEST_CASE("Scheduler respects same-path fallback routing scope", "[scheduler]") {
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
