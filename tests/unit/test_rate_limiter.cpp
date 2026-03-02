#include <catch2/catch_amalgamated.hpp>

#include "server/auth/rate_limiter.h"

TEST_CASE("RateLimiter disabled when limit is 0", "[ratelimit]") {
  inferflux::RateLimiter limiter(0);
  REQUIRE(!limiter.Enabled());
  REQUIRE(limiter.CurrentLimit() == 0);
  // Always allowed when disabled.
  REQUIRE(limiter.Allow("any-key"));
}

TEST_CASE("RateLimiter allows up to limit", "[ratelimit]") {
  inferflux::RateLimiter limiter(3);
  REQUIRE(limiter.Enabled());
  REQUIRE(limiter.CurrentLimit() == 3);

  // First 3 requests should succeed (initial bucket = 3).
  REQUIRE(limiter.Allow("user1"));
  REQUIRE(limiter.Allow("user1"));
  REQUIRE(limiter.Allow("user1"));

  // 4th should be denied (bucket exhausted, no time passed).
  REQUIRE(!limiter.Allow("user1"));
}

TEST_CASE("RateLimiter per-key isolation", "[ratelimit]") {
  inferflux::RateLimiter limiter(1);

  REQUIRE(limiter.Allow("user-a"));
  REQUIRE(!limiter.Allow("user-a")); // exhausted for user-a

  // user-b has its own bucket.
  REQUIRE(limiter.Allow("user-b"));
}

TEST_CASE("RateLimiter UpdateLimit resets state", "[ratelimit]") {
  inferflux::RateLimiter limiter(2);
  REQUIRE(limiter.Allow("x"));
  REQUIRE(limiter.Allow("x"));
  REQUIRE(!limiter.Allow("x"));

  limiter.UpdateLimit(5);
  REQUIRE(limiter.CurrentLimit() == 5);
  // After reset, user gets a fresh bucket.
  REQUIRE(limiter.Allow("x"));
}
