#include "runtime/prefix_cache/radix_prefix_cache.h"

#include <catch2/catch_amalgamated.hpp>

#include <atomic>
#include <thread>
#include <vector>

using namespace inferflux;

TEST_CASE("RadixPrefixCache: miss on empty cache", "[radix_cache]") {
  RadixPrefixCache cache(16);
  std::string out;
  int tok = 0;
  int matched = -1;
  REQUIRE_FALSE(cache.Lookup({1, 2, 3}, &out, &tok, &matched));
  REQUIRE(matched == 0);
}

TEST_CASE("RadixPrefixCache: exact hit after insert", "[radix_cache]") {
  RadixPrefixCache cache(16);
  cache.Insert({1, 2, 3}, "hello", 3);
  std::string out;
  int tok = 0;
  REQUIRE(cache.Lookup({1, 2, 3}, &out, &tok));
  REQUIRE(out == "hello");
  REQUIRE(tok == 3);
}

TEST_CASE("RadixPrefixCache: partial prefix match reports matched_tokens", "[radix_cache]") {
  RadixPrefixCache cache(16);
  // Insert [1, 2, 3] — creates a node at depth 3.
  cache.Insert({1, 2, 3}, "abc", 3);

  std::string out;
  int tok = 0;
  int matched = -1;
  // Query [1, 2, 4] — shares first 2 tokens with the stored path.
  bool hit = cache.Lookup({1, 2, 4}, &out, &tok, &matched);
  REQUIRE_FALSE(hit);
  // matched should be 2 (the shared edge portion [1,2] up to divergence).
  // Actually depends on radix structure: [1,2,3] stored as one edge;
  // [1,2,4] diverges after 2 tokens into the edge.
  REQUIRE(matched == 2);
}

TEST_CASE("RadixPrefixCache: capacity=0 is no-op", "[radix_cache]") {
  RadixPrefixCache cache(0);
  cache.Insert({1, 2, 3}, "hello", 3);
  std::string out;
  int tok = 0;
  REQUIRE_FALSE(cache.Lookup({1, 2, 3}, &out, &tok));
  REQUIRE(cache.Size() == 0);
}

TEST_CASE("RadixPrefixCache: LRU eviction on full cache", "[radix_cache]") {
  RadixPrefixCache cache(2);
  cache.Insert({1}, "a", 1);
  cache.Insert({2}, "b", 1);
  REQUIRE(cache.Size() == 2);
  // Insert a third entry — should evict the LRU one (key {1} was inserted first).
  cache.Insert({3}, "c", 1);
  REQUIRE(cache.Size() == 2);

  // {1} should have been evicted (lowest last_used).
  std::string out;
  int tok = 0;
  REQUIRE_FALSE(cache.Lookup({1}, &out, &tok));
  // {2} and {3} should be present.
  REQUIRE(cache.Lookup({2}, &out, &tok));
  REQUIRE(cache.Lookup({3}, &out, &tok));
}

TEST_CASE("RadixPrefixCache: re-insert same tokens updates completion", "[radix_cache]") {
  RadixPrefixCache cache(16);
  cache.Insert({1, 2, 3}, "first", 5);
  cache.Insert({1, 2, 3}, "second", 6);
  REQUIRE(cache.Size() == 1);
  std::string out;
  int tok = 0;
  REQUIRE(cache.Lookup({1, 2, 3}, &out, &tok));
  REQUIRE(out == "second");
  REQUIRE(tok == 6);
}

TEST_CASE("RadixPrefixCache: shared prefix, both entries retrievable", "[radix_cache]") {
  RadixPrefixCache cache(16);
  cache.Insert({1, 2, 3}, "abc", 3);
  cache.Insert({1, 2, 4}, "abd", 3);
  REQUIRE(cache.Size() == 2);

  std::string out;
  int tok = 0;
  REQUIRE(cache.Lookup({1, 2, 3}, &out, &tok));
  REQUIRE(out == "abc");
  REQUIRE(cache.Lookup({1, 2, 4}, &out, &tok));
  REQUIRE(out == "abd");
}

TEST_CASE("RadixPrefixCache: empty token vector", "[radix_cache]") {
  RadixPrefixCache cache(16);
  // Insert/lookup empty tokens should not crash.
  cache.Insert({}, "empty", 0);
  std::string out;
  int tok = 0;
  // Root itself is not a completion node in our implementation,
  // so this is a miss (or hit depending on design — we only store on leaf insertion).
  // Just verify no crash.
  cache.Lookup({}, &out, &tok);
  REQUIRE(cache.Size() <= 1);
}

TEST_CASE("RadixPrefixCache: single token partial match", "[radix_cache]") {
  RadixPrefixCache cache(16);
  cache.Insert({1, 2}, "x", 1);

  std::string out;
  int tok = 0;
  int matched = -1;
  // Query [1, 3] — shares first token with the stored edge [1,2].
  bool hit = cache.Lookup({1, 3}, &out, &tok, &matched);
  REQUIRE_FALSE(hit);
  REQUIRE(matched == 1);
}

TEST_CASE("RadixPrefixCache: deep radix tree with progressive prefixes", "[radix_cache]") {
  RadixPrefixCache cache(32);
  cache.Insert({1}, "c1", 1);
  cache.Insert({1, 2}, "c12", 1);
  cache.Insert({1, 2, 3}, "c123", 1);
  cache.Insert({1, 2, 3, 4}, "c1234", 1);
  cache.Insert({1, 2, 3, 4, 5}, "c12345", 1);

  std::string out;
  int tok = 0;
  REQUIRE(cache.Lookup({1}, &out, &tok));
  REQUIRE(out == "c1");
  REQUIRE(cache.Lookup({1, 2}, &out, &tok));
  REQUIRE(out == "c12");
  REQUIRE(cache.Lookup({1, 2, 3, 4, 5}, &out, &tok));
  REQUIRE(out == "c12345");
}

TEST_CASE("RadixPrefixCache: concurrent inserts thread safety", "[radix_cache]") {
  RadixPrefixCache cache(8);
  std::atomic<int> errors{0};
  std::vector<std::thread> threads;
  threads.reserve(8);
  for (int i = 0; i < 8; ++i) {
    threads.emplace_back([&cache, &errors, i]() {
      try {
        cache.Insert({i, i + 1}, std::string("val") + std::to_string(i), 1);
      } catch (...) {
        errors.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }
  REQUIRE(errors.load() == 0);
  REQUIRE(cache.Size() <= cache.Capacity());
}

TEST_CASE("RadixPrefixCache: Size() respects evictions", "[radix_cache]") {
  constexpr std::size_t cap = 4;
  RadixPrefixCache cache(cap);
  for (int i = 0; i < 10; ++i) {
    cache.Insert({i}, std::string("v") + std::to_string(i), 1);
  }
  REQUIRE(cache.Size() <= cap);
  REQUIRE(cache.Size() == cap);
}
