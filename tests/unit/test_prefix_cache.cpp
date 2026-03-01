#include <catch2/catch_amalgamated.hpp>

#include "runtime/prefix_cache/prefix_cache.h"

#include <string>
#include <thread>
#include <vector>

TEST_CASE("PrefixCache miss on empty cache", "[prefix_cache]") {
  inferflux::PrefixCache cache(16);
  std::string completion;
  int tokens = 0;
  REQUIRE(!cache.Lookup({1, 2, 3}, &completion, &tokens));
  REQUIRE(completion.empty());
  REQUIRE(tokens == 0);
}

TEST_CASE("PrefixCache insert then hit", "[prefix_cache]") {
  inferflux::PrefixCache cache(16);
  cache.Insert({10, 20, 30}, "hello world", 3);

  std::string completion;
  int tokens = 0;
  REQUIRE(cache.Lookup({10, 20, 30}, &completion, &tokens));
  REQUIRE(completion == "hello world");
  REQUIRE(tokens == 3);
}

TEST_CASE("PrefixCache different token sequences are independent", "[prefix_cache]") {
  inferflux::PrefixCache cache(16);
  cache.Insert({1, 2}, "response A", 2);
  cache.Insert({3, 4}, "response B", 2);

  std::string out;
  int tok = 0;
  REQUIRE(cache.Lookup({1, 2}, &out, &tok));
  REQUIRE(out == "response A");
  REQUIRE(!cache.Lookup({9, 9}, &out, &tok));
}

TEST_CASE("PrefixCache duplicate insert updates entry", "[prefix_cache]") {
  inferflux::PrefixCache cache(16);
  cache.Insert({1, 2, 3}, "first", 1);
  cache.Insert({1, 2, 3}, "second", 2);

  std::string out;
  int tok = 0;
  REQUIRE(cache.Lookup({1, 2, 3}, &out, &tok));
  REQUIRE(out == "second");
  REQUIRE(tok == 2);
}

TEST_CASE("PrefixCache evicts LRU when full", "[prefix_cache]") {
  // Capacity = 2; insert A, B, then C — A should be evicted (LRU).
  inferflux::PrefixCache cache(2);
  cache.Insert({1}, "A", 1);
  cache.Insert({2}, "B", 1);

  // Access A so B is now LRU.
  std::string out;
  int tok = 0;
  cache.Lookup({1}, &out, &tok);

  // Insert C — B should be evicted.
  cache.Insert({3}, "C", 1);

  REQUIRE(cache.Lookup({1}, &out, &tok));   // A still present
  REQUIRE(cache.Lookup({3}, &out, &tok));   // C present
  REQUIRE(!cache.Lookup({2}, &out, &tok));  // B evicted
}

TEST_CASE("PrefixCache capacity=0 is a no-op", "[prefix_cache]") {
  inferflux::PrefixCache cache(0);
  cache.Insert({1, 2}, "should not be stored", 3);
  std::string out;
  int tok = 0;
  REQUIRE(!cache.Lookup({1, 2}, &out, &tok));
  REQUIRE(out.empty());
}

TEST_CASE("PrefixCache handles empty token vector", "[prefix_cache]") {
  inferflux::PrefixCache cache(8);
  cache.Insert({}, "empty key", 1);
  std::string out;
  int tok = 0;
  REQUIRE(cache.Lookup({}, &out, &tok));
  REQUIRE(out == "empty key");
}

TEST_CASE("PrefixCache is safe under concurrent inserts", "[prefix_cache]") {
  inferflux::PrefixCache cache(64);
  std::vector<std::thread> threads;
  for (int i = 0; i < 8; ++i) {
    threads.emplace_back([&cache, i] {
      std::vector<int> key = {i, i * 2};
      cache.Insert(key, "val" + std::to_string(i), i);
    });
  }
  for (auto& t : threads) {
    t.join();
  }
  // All 8 entries fit within capacity=64; at least some must be present.
  int hits = 0;
  for (int i = 0; i < 8; ++i) {
    std::string out;
    int tok = 0;
    if (cache.Lookup({i, i * 2}, &out, &tok)) {
      ++hits;
    }
  }
  REQUIRE(hits > 0);
}
