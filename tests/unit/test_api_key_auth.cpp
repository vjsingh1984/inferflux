#include <catch2/catch_amalgamated.hpp>

#include "server/auth/api_key_auth.h"

#include <algorithm>
#include <string>
#include <vector>

TEST_CASE("ApiKeyAuth add and check keys", "[auth]") {
  inferflux::ApiKeyAuth auth;
  REQUIRE(!auth.HasKeys());

  auth.AddKey("secret-key-1");
  REQUIRE(auth.HasKeys());
  REQUIRE(auth.IsAllowed("secret-key-1"));
  REQUIRE(!auth.IsAllowed("wrong-key"));
}

TEST_CASE("ApiKeyAuth stores keys by SHA-256 hash", "[auth]") {
  inferflux::ApiKeyAuth auth;
  auto hash = inferflux::ApiKeyAuth::HashKey("test-key");
  REQUIRE(hash.size() == 64);  // SHA-256 hex = 64 chars

  // Same input always produces same hash.
  REQUIRE(inferflux::ApiKeyAuth::HashKey("test-key") == hash);
  // Different input produces different hash.
  REQUIRE(inferflux::ApiKeyAuth::HashKey("other-key") != hash);
}

TEST_CASE("ApiKeyAuth default scopes", "[auth]") {
  inferflux::ApiKeyAuth auth;
  auth.AddKey("key1");

  auto scopes = auth.Scopes("key1");
  REQUIRE(scopes.size() >= 2);
  REQUIRE(std::find(scopes.begin(), scopes.end(), "generate") != scopes.end());
  REQUIRE(std::find(scopes.begin(), scopes.end(), "read") != scopes.end());
}

TEST_CASE("ApiKeyAuth custom scopes include read", "[auth]") {
  inferflux::ApiKeyAuth auth;
  auth.AddKey("key2", {"admin"});

  auto scopes = auth.Scopes("key2");
  REQUIRE(std::find(scopes.begin(), scopes.end(), "admin") != scopes.end());
  // read is always added.
  REQUIRE(std::find(scopes.begin(), scopes.end(), "read") != scopes.end());
}

TEST_CASE("ApiKeyAuth remove key", "[auth]") {
  inferflux::ApiKeyAuth auth;
  auth.AddKey("to-remove");
  REQUIRE(auth.IsAllowed("to-remove"));

  auth.RemoveKey("to-remove");
  REQUIRE(!auth.IsAllowed("to-remove"));
  REQUIRE(!auth.HasKeys());
}

TEST_CASE("ApiKeyAuth scopes for unknown key returns empty", "[auth]") {
  inferflux::ApiKeyAuth auth;
  auto scopes = auth.Scopes("nonexistent");
  REQUIRE(scopes.empty());
}
