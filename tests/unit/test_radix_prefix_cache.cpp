#include "runtime/prefix_cache/radix_prefix_cache.h"

#include <catch2/catch_amalgamated.hpp>

#include <atomic>
#include <thread>
#include <vector>

using namespace inferflux;

TEST_CASE("RadixPrefixCache: miss on empty cache", "[radix_cache]") {
  RadixPrefixCache cache(nullptr, [](int) {}, 16, 12);
  std::vector<int> out;
  int out_seq = -1;
  int matched = -1;
  REQUIRE_FALSE(cache.Lookup({1, 2, 3}, nullptr, &out, &out_seq, &matched));
  REQUIRE(matched == 0);
}

TEST_CASE("RadixPrefixCache: exact hit after insert", "[radix_cache]") {
  RadixPrefixCache cache(nullptr, [](int) {}, 16, 12);
  // Assume [1, 2, 3] covered by block 100, computed in seq 5.
  cache.Insert({1, 2, 3}, {100}, 5, nullptr);
  std::vector<int> out;
  int out_seq = -1;
  int matched = 0;
  bool hit = cache.Lookup({1, 2, 3}, nullptr, &out, &out_seq, &matched);
  UNSCOPED_INFO("hit=" << hit << " matched=" << matched
                       << " out_seq=" << out_seq);
  REQUIRE(hit);
  REQUIRE(out == std::vector<int>{100});
  REQUIRE(out_seq == 5);
  REQUIRE(matched == 3);
}

TEST_CASE("RadixPrefixCache: partial prefix match reports matched_tokens",
          "[radix_cache]") {
  RadixPrefixCache cache(nullptr, [](int) {}, 16, 12);
  cache.Insert({1, 2, 3}, {100}, 5, nullptr);

  std::vector<int> out;
  int out_seq = -1;
  int matched = -1;
  // Query [1, 2, 4] — [1, 2] shares a prefix but divergences mid-edge [1, 2,
  // 3]. Correctness Fix (§ Item 3): matched_tokens should be 2, but hit is
  // false (no full node).
  bool hit = cache.Lookup({1, 2, 4}, nullptr, &out, &out_seq, &matched);
  REQUIRE_FALSE(hit);
  REQUIRE(matched == 2);
}

TEST_CASE("RadixPrefixCache: deep radix tree with progressive prefixes",
          "[radix_cache]") {
  RadixPrefixCache cache(nullptr, [](int) {}, 32, 12);
  // Blocks of 16 tokens each.
  cache.Insert({1, 2}, {10}, 1, nullptr);
  cache.Insert({1, 2, 3, 4}, {10, 11}, 2, nullptr);

  std::vector<int> out;
  int out_seq = -1;
  int matched = 0;
  bool hit1 = cache.Lookup({1, 2}, nullptr, &out, &out_seq, &matched);
  UNSCOPED_INFO("hit1=" << hit1 << " matched=" << matched
                        << " out_seq=" << out_seq);
  REQUIRE(hit1);
  REQUIRE(matched == 2);
  REQUIRE(out_seq == 1);

  bool hit2 = cache.Lookup({1, 2, 3, 4}, nullptr, &out, &out_seq, &matched);
  UNSCOPED_INFO("hit2=" << hit2 << " matched=" << matched
                        << " out_seq=" << out_seq);
  REQUIRE(hit2);
  REQUIRE(matched == 4);
  REQUIRE(out_seq == 2);
}

TEST_CASE("RadixPrefixCache: sequence slot capping and eviction",
          "[radix_cache]") {
  int evicted_seq = -1;
  RadixPrefixCache cache(nullptr, [&](int seq) { evicted_seq = seq; }, 100, 2);

  cache.Insert({1}, {10}, 101, nullptr);
  cache.Insert({2}, {11}, 102, nullptr);
  REQUIRE(cache.LiveSequences() == 2);

  // Inserting 3rd sequence should trigger eviction of the oldest (101).
  cache.Insert({3}, {12}, 103, nullptr);
  REQUIRE(cache.LiveSequences() == 2);
  REQUIRE(evicted_seq == 101);
}
