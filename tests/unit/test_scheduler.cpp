#include <catch2/catch_amalgamated.hpp>

#include "runtime/backends/cpu/llama_backend.h"
#include "runtime/prefix_cache/prefix_cache.h"
#include "scheduler/scheduler.h"
#include "scheduler/single_model_router.h"
#include "scheduler/fairness_controller.h"

#include <atomic>
#include <future>
#include <memory>

TEST_CASE("Scheduler stub response with no backend", "[scheduler]") {
  inferflux::SimpleTokenizer tokenizer;
  auto device = std::make_shared<inferflux::CPUDeviceContext>();
  auto cache = std::make_shared<inferflux::PagedKVCache>(
      4, sizeof(float) * 4, inferflux::PagedKVCache::EvictionPolicy::kLRU);

  // No router → no backend → should return no_backend flag.
  inferflux::Scheduler scheduler(tokenizer, device, cache);

  inferflux::InferenceRequest req;
  req.prompt = "Hello world";
  req.max_tokens = 10;
  auto resp = scheduler.Generate(req);

  REQUIRE(resp.no_backend);
  REQUIRE(!resp.completion.empty());
  REQUIRE(resp.prompt_tokens > 0);
}

TEST_CASE("Scheduler with empty SingleModelRouter returns no_backend", "[scheduler]") {
  inferflux::SimpleTokenizer tokenizer;
  auto device = std::make_shared<inferflux::CPUDeviceContext>();
  auto cache = std::make_shared<inferflux::PagedKVCache>(
      4, sizeof(float) * 4, inferflux::PagedKVCache::EvictionPolicy::kLRU);

  // Router with no model loaded.
  auto router = std::make_shared<inferflux::SingleModelRouter>();
  REQUIRE(router->ListModels().empty());
  REQUIRE(router->Resolve("") == nullptr);

  inferflux::Scheduler scheduler(tokenizer, device, cache, router);

  inferflux::InferenceRequest req;
  req.prompt = "Hello";
  req.max_tokens = 5;
  auto resp = scheduler.Generate(req);

  REQUIRE(resp.no_backend);
}

TEST_CASE("Scheduler with speculative decoder override", "[scheduler]") {
  inferflux::SimpleTokenizer tokenizer;
  auto device = std::make_shared<inferflux::CPUDeviceContext>();
  auto cache = std::make_shared<inferflux::PagedKVCache>(
      4, sizeof(float) * 4, inferflux::PagedKVCache::EvictionPolicy::kLRU);

  inferflux::SpeculativeConfig cfg;
  cfg.enabled = true;
  cfg.chunk_size = 2;
  auto decoder = std::make_shared<inferflux::SpeculativeDecoder>(cfg, device, &tokenizer, nullptr);
  // Set a validation override that produces known output.
  decoder->SetValidationOverride([](const std::vector<int>&, int) {
    return std::vector<int>{3, 4, 5};
  });

  inferflux::Scheduler scheduler(tokenizer, device, cache, nullptr, decoder);

  inferflux::InferenceRequest req;
  req.prompt = "Hello";
  req.max_tokens = 10;
  auto resp = scheduler.Generate(req);

  // Speculative path should have produced output.
  REQUIRE(!resp.no_backend);
  REQUIRE(!resp.completion.empty());
  REQUIRE(resp.speculative.total_chunks > 0);

  decoder->ClearValidationOverride();
}

TEST_CASE("LlamaCPUBackend TokenCount returns 0 without loaded model", "[scheduler]") {
  // No model loaded — Tokenize() returns empty, so TokenCount() must return 0.
  inferflux::LlamaCPUBackend backend;
  REQUIRE(!backend.IsReady());
  REQUIRE(backend.TokenCount("hello world") == 0);
  REQUIRE(backend.TokenCount("") == 0);
}

TEST_CASE("Scheduler falls back to SimpleTokenizer when no backend", "[scheduler]") {
  // Verify prompt_tokens > 0 when SimpleTokenizer is the only tokenizer available.
  inferflux::SimpleTokenizer tokenizer;
  auto device = std::make_shared<inferflux::CPUDeviceContext>();
  auto cache = std::make_shared<inferflux::PagedKVCache>(
      4, sizeof(float) * 4, inferflux::PagedKVCache::EvictionPolicy::kLRU);
  inferflux::Scheduler scheduler(tokenizer, device, cache);

  inferflux::InferenceRequest req;
  req.prompt = "the quick brown fox";
  req.max_tokens = 5;
  auto resp = scheduler.Generate(req);

  REQUIRE(resp.no_backend);
  // SimpleTokenizer prepends <bos> then splits on alphanumeric runs:
  // "the quick brown fox" → <bos> + 4 words = 5 tokens.
  REQUIRE(resp.prompt_tokens == 5);
  // completion_tokens is also counted via SimpleTokenizer (no backend).
  REQUIRE(resp.completion_tokens > 0);
}

TEST_CASE("SingleModelRouter ListModels and Resolve", "[scheduler]") {
  auto router = std::make_shared<inferflux::SingleModelRouter>();
  REQUIRE(router->Name() == "single");
  REQUIRE(router->ListModels().empty());
  REQUIRE(router->Resolve("any") == nullptr);
  REQUIRE(router->Resolve("") == nullptr);
  REQUIRE(router->Backend() == nullptr);

  // UnloadModel on empty router returns false.
  REQUIRE_FALSE(router->UnloadModel("nonexistent"));
}

TEST_CASE("SingleModelRouter registers and resolves preloaded backends", "[scheduler]") {
  inferflux::SingleModelRouter router;
  auto backend = std::make_shared<inferflux::LlamaCPUBackend>();
  backend->ForceReadyForTests();
  inferflux::ModelInfo info{"primary", "/tmp/model.gguf", "cpu", true};
  REQUIRE(router.RegisterModel(info, backend));

  auto models = router.ListModels();
  REQUIRE(models.size() == 1);
  REQUIRE(models[0].id == "primary");
  REQUIRE(models[0].ready);

  auto resolved = router.Resolve("");
  REQUIRE(resolved != nullptr);
  REQUIRE(resolved->id == "primary");
  REQUIRE(router.GetBackend("primary") == backend);

  // Unknown model id falls back to the default entry.
  auto fallback = router.Resolve("unknown");
  REQUIRE(fallback != nullptr);
  REQUIRE(fallback->id == "primary");

  REQUIRE(router.UnloadModel("primary"));
  REQUIRE(router.ListModels().empty());
}

TEST_CASE("SingleModelRouter tracks multiple models", "[scheduler]") {
  inferflux::SingleModelRouter router;
  auto backend_a = std::make_shared<inferflux::LlamaCPUBackend>();
  backend_a->ForceReadyForTests();
  auto backend_b = std::make_shared<inferflux::LlamaCPUBackend>();
  backend_b->ForceReadyForTests();

  inferflux::ModelInfo info_a{"alpha", "/tmp/a.gguf", "cpu", true};
  inferflux::ModelInfo info_b{"beta", "/tmp/b.gguf", "mps", true};
  REQUIRE(router.RegisterModel(info_a, backend_a));
  REQUIRE(router.RegisterModel(info_b, backend_b));

  auto resolved_beta = router.Resolve("beta");
  REQUIRE(resolved_beta != nullptr);
  REQUIRE(resolved_beta->id == "beta");
  REQUIRE(router.GetBackend("beta") == backend_b);
  REQUIRE(router.UnloadModel("beta"));

  auto resolved_alpha = router.Resolve("beta");
  REQUIRE(resolved_alpha != nullptr);
  REQUIRE(resolved_alpha->id == "alpha");
}

TEST_CASE("Scheduler processes multiple concurrent requests", "[scheduler]") {
  inferflux::SimpleTokenizer tokenizer;
  auto device = std::make_shared<inferflux::CPUDeviceContext>();
  auto cache = std::make_shared<inferflux::PagedKVCache>(
      4, sizeof(float) * 4, inferflux::PagedKVCache::EvictionPolicy::kLRU);
  inferflux::Scheduler scheduler(tokenizer, device, cache);

  auto submit = [&](const std::string& prompt) {
    inferflux::InferenceRequest req;
    req.prompt = prompt;
    req.max_tokens = 4;
    return scheduler.Generate(std::move(req));
  };

  auto fut1 = std::async(std::launch::async, submit, "first");
  auto fut2 = std::async(std::launch::async, submit, "second");

  auto resp1 = fut1.get();
  auto resp2 = fut2.get();
  REQUIRE(resp1.no_backend);
  REQUIRE(resp2.no_backend);
}

TEST_CASE("Scheduler returns cancelled when cancellation flag is pre-set", "[scheduler]") {
  inferflux::SimpleTokenizer tokenizer;
  auto device = std::make_shared<inferflux::CPUDeviceContext>();
  auto cache = std::make_shared<inferflux::PagedKVCache>(
      4, sizeof(float) * 4, inferflux::PagedKVCache::EvictionPolicy::kLRU);
  inferflux::Scheduler scheduler(tokenizer, device, cache);

  auto cancel_flag = std::make_shared<std::atomic<bool>>(true);  // pre-set

  inferflux::InferenceRequest req;
  req.prompt = "This should be cancelled";
  req.max_tokens = 10;
  req.cancellation_flag = cancel_flag;
  auto resp = scheduler.Generate(std::move(req));

  REQUIRE(resp.no_backend);
  REQUIRE(resp.completion == "[cancelled]");
}

TEST_CASE("on_token callback fires on prefix cache hit", "[scheduler]") {
  inferflux::SimpleTokenizer tokenizer;
  auto device = std::make_shared<inferflux::CPUDeviceContext>();
  auto cache = std::make_shared<inferflux::PagedKVCache>(
      4, sizeof(float) * 4, inferflux::PagedKVCache::EvictionPolicy::kLRU);
  auto prefix_cache = std::make_shared<inferflux::PrefixCache>(64);

  // Pre-populate the cache with a known completion for the prompt we'll submit.
  // Use the tokenizer to derive the key as the scheduler will.
  const std::string prompt = "cached prompt hello";
  const std::string cached_text = "stub cached answer";
  auto prompt_tokens = tokenizer.Encode(prompt);
  prefix_cache->Insert(prompt_tokens, cached_text, 3);

  inferflux::Scheduler scheduler(tokenizer, device, cache, nullptr, nullptr, prefix_cache);

  // Request with on_token set — should hit the cache and fire the callback.
  std::vector<std::string> tokens_received;
  inferflux::InferenceRequest req;
  req.prompt = prompt;
  req.max_tokens = 8;
  req.stream = true;
  req.on_token = [&tokens_received](const std::string& tok) {
    tokens_received.push_back(tok);
  };
  auto resp = scheduler.Generate(std::move(req));

  REQUIRE(!resp.completion.empty());
  REQUIRE(!tokens_received.empty());
  REQUIRE(tokens_received[0] == cached_text);
}

TEST_CASE("Scheduler clamps max_tokens=0 to 1 without hanging", "[scheduler]") {
  inferflux::SimpleTokenizer tokenizer;
  auto device = std::make_shared<inferflux::CPUDeviceContext>();
  auto cache = std::make_shared<inferflux::PagedKVCache>(
      4, sizeof(float) * 4, inferflux::PagedKVCache::EvictionPolicy::kLRU);
  inferflux::Scheduler scheduler(tokenizer, device, cache);

  inferflux::InferenceRequest req;
  req.prompt = "test";
  req.max_tokens = 0;
  auto resp = scheduler.Generate(std::move(req));

  REQUIRE(resp.no_backend);
  REQUIRE(!resp.completion.empty());
}

TEST_CASE("FairnessController swaps in high priority entries", "[fairness]") {
  inferflux::FairnessConfig cfg;
  cfg.enable_preemption = true;
  cfg.high_priority_threshold = 5;
  inferflux::FairnessController controller(cfg);

  inferflux::InferenceRequest low;
  low.priority_level = 1;
  inferflux::InferenceRequest high;
  high.priority_level = 10;

  std::vector<inferflux::FairnessEntry> batch{{&low, low.priority_level, 0}};
  std::vector<inferflux::FairnessEntry> queue{{&high, high.priority_level, 0}};

  auto decision = controller.Evaluate(batch, queue);
  REQUIRE(decision.swap);
  REQUIRE(decision.batch_index == 0);
  REQUIRE(decision.queue_index == 0);
}

TEST_CASE("FairnessController applies timeslice to low-priority entries", "[fairness]") {
  inferflux::FairnessConfig cfg;
  cfg.max_timeslice_tokens = 4;
  cfg.high_priority_threshold = 5;
  inferflux::FairnessController controller(cfg);

  inferflux::InferenceRequest low;
  low.priority_level = 0;
  low.max_tokens = 32;

  inferflux::InferenceRequest high;
  high.priority_level = 6;
  high.max_tokens = 32;

  std::vector<inferflux::FairnessEntry> batch{
      {&low, low.priority_level, 0},
      {&high, high.priority_level, 1},
  };
  controller.ApplyTimeslice(&batch);
  REQUIRE(low.timeslice_tokens == 4);
  REQUIRE(low.max_tokens == 32);
  REQUIRE(high.timeslice_tokens == 0);
  low.remaining_decode_tokens = 2;
  controller.ApplyTimeslice(&batch);
  REQUIRE(low.timeslice_tokens == 2);
}

TEST_CASE("FairnessController skips timeslice for high-priority entries", "[fairness]") {
  inferflux::FairnessConfig cfg;
  cfg.max_timeslice_tokens = 3;
  cfg.high_priority_threshold = 5;
  inferflux::FairnessController controller(cfg);

  inferflux::InferenceRequest high;
  high.priority_level = 6;
  high.max_tokens = 64;

  std::vector<inferflux::FairnessEntry> batch{
      {&high, high.priority_level, 0},
  };
  controller.ApplyTimeslice(&batch);
  REQUIRE(high.timeslice_tokens == 0);
  REQUIRE(high.max_tokens == 64);
}
