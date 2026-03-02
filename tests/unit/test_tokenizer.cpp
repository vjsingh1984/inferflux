#include <catch2/catch_amalgamated.hpp>

#include "model/tokenizer/simple_tokenizer.h"
#include "runtime/kv_cache/paged_kv_cache.h"
#include "runtime/speculative/speculative_decoder.h"
#include "server/policy/guardrail.h"

#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>

TEST_CASE("SimpleTokenizer encode/decode round-trip", "[tokenizer]") {
  inferflux::SimpleTokenizer tokenizer;
  auto tokens = tokenizer.Encode("hello world");
  REQUIRE(!tokens.empty());
  auto text = tokenizer.Decode(tokens);
  REQUIRE(!text.empty());
}

TEST_CASE("SimpleTokenizer handles empty input", "[tokenizer]") {
  inferflux::SimpleTokenizer tokenizer;
  auto tokens = tokenizer.Encode("");
  REQUIRE(tokens.empty());
}

TEST_CASE("PagedKVCache LRU and Clock eviction policies", "[kv_cache]") {
  inferflux::PagedKVCache cache(2, sizeof(float) * 4,
                                inferflux::PagedKVCache::EvictionPolicy::kLRU);
  cache.ConfigureAsyncWriter(2, 8);

  int first = cache.ReservePage();
  REQUIRE(first >= 0);
  cache.Write(first, std::vector<float>{1.f, 2.f, 3.f, 4.f});
  cache.ReleasePage(first);

  cache.SetEvictionPolicy(inferflux::PagedKVCache::EvictionPolicy::kClock);
  int second = cache.ReservePage();
  REQUIRE(second >= 0);
  cache.ReleasePage(second);
}

TEST_CASE("SpeculativeDecoder chunk validation", "[speculative]") {
  inferflux::SpeculativeConfig cfg;
  cfg.enabled = true;
  cfg.chunk_size = 2;
  auto device = std::make_shared<inferflux::CPUDeviceContext>();
  inferflux::SimpleTokenizer tokenizer;
  inferflux::SpeculativeDecoder decoder(cfg, device, &tokenizer, nullptr);

  inferflux::SpeculativeDraft draft;
  draft.completion_tokens = {1, 2, 3, 4};
  draft.chunks.push_back({0, 2});
  draft.chunks.push_back({2, 4});

  decoder.SetValidationOverride([](const std::vector<int> &, int) {
    return std::vector<int>{1, 2, 9, 9};
  });

  auto result = decoder.Validate({}, draft, 4, nullptr);
  REQUIRE(result.metrics.total_chunks == 2);
  REQUIRE(result.metrics.accepted_chunks == 1);
  REQUIRE(result.metrics.reused_tokens == 2);
  REQUIRE(result.completion_tokens.size() == 4);
  REQUIRE(result.completion_tokens[0] == 1);
  REQUIRE(result.completion_tokens[1] == 2);
  REQUIRE(result.completion_tokens[2] == 9);
  REQUIRE(result.completion_tokens[3] == 9);

  decoder.ClearValidationOverride();
}

TEST_CASE("OPA guardrail file-based policy denial", "[guardrail]") {
  inferflux::Guardrail guardrail;
  REQUIRE(!guardrail.Enabled());

  auto tmp_path =
      std::filesystem::temp_directory_path() / "inferflux_opa_test.json";
  {
    std::ofstream out(tmp_path);
    out << R"({"result":{"allow":false,"reason":"deny"}})";
  }
  guardrail.SetOPAEndpoint(std::string("file://") + tmp_path.string());
  REQUIRE(guardrail.Enabled());

  std::string reason;
  bool allowed = guardrail.Check("hello world", &reason);
  REQUIRE(!allowed);
  REQUIRE(!reason.empty());

  std::filesystem::remove(tmp_path);
}
