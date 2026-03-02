#include <catch2/catch_amalgamated.hpp>

#include "policy/policy_store.h"
#include "server/auth/api_key_auth.h"

#include <filesystem>
#include <string>
#include <vector>

TEST_CASE("PolicyStore round-trip without encryption", "[policy]") {
  auto tmp_path =
      std::filesystem::temp_directory_path() / "inferflux_policy_test.conf";

  {
    inferflux::PolicyStore store(tmp_path.string());
    store.SetApiKey("key1", {"generate", "admin"});
    store.SetApiKey("key2", {"read"});
    store.SetGuardrailBlocklist({"bad", "evil"});
    store.SetRateLimitPerMinute(100);
    REQUIRE(store.Save());
  }

  {
    inferflux::PolicyStore store(tmp_path.string());
    REQUIRE(store.Load());

    auto keys = store.ApiKeys();
    REQUIRE(keys.size() == 2);

    auto bl = store.GuardrailBlocklist();
    REQUIRE(bl.size() == 2);

    REQUIRE(store.RateLimitPerMinute() == 100);
  }

  std::filesystem::remove(tmp_path);
}

TEST_CASE("PolicyStore round-trip with AES-GCM encryption", "[policy]") {
  auto tmp_path =
      std::filesystem::temp_directory_path() / "inferflux_policy_enc.conf";

  {
    inferflux::PolicyStore store(tmp_path.string(), "my-secret-pass");
    store.SetApiKey("encrypted-key", {"generate"});
    store.SetGuardrailBlocklist({"secret"});
    store.SetRateLimitPerMinute(50);
    REQUIRE(store.Save());
  }

  // Re-open with same passphrase: should decrypt.
  {
    inferflux::PolicyStore store(tmp_path.string(), "my-secret-pass");
    REQUIRE(store.Load());
    REQUIRE(store.ApiKeys().size() == 1);
    REQUIRE(store.GuardrailBlocklist().size() == 1);
    REQUIRE(store.RateLimitPerMinute() == 50);
  }

  // Wrong passphrase: should fail.
  {
    inferflux::PolicyStore store(tmp_path.string(), "wrong-pass");
    REQUIRE(!store.Load());
  }

  // No passphrase on encrypted file: should fail.
  {
    inferflux::PolicyStore store(tmp_path.string());
    REQUIRE(!store.Load());
  }

  std::filesystem::remove(tmp_path);
}

TEST_CASE("PolicyStore RemoveApiKey", "[policy]") {
  auto tmp_path =
      std::filesystem::temp_directory_path() / "inferflux_policy_rm.conf";
  inferflux::PolicyStore store(tmp_path.string());
  store.SetApiKey("to-remove", {"read"});
  REQUIRE(store.ApiKeys().size() == 1);

  REQUIRE(store.RemoveApiKey("to-remove"));
  REQUIRE(store.ApiKeys().empty());

  // Removing non-existent key returns false.
  REQUIRE(!store.RemoveApiKey("nope"));

  std::filesystem::remove(tmp_path);
}

TEST_CASE("PolicyStore Load from nonexistent file", "[policy]") {
  inferflux::PolicyStore store("/tmp/inferflux_no_such_file_xyz.conf");
  REQUIRE(!store.Load());
}

TEST_CASE(
    "PolicyStore stores hashed keys; AddKeyHashed round-trip with ApiKeyAuth",
    "[policy][auth]") {
  // Keys stored in PolicyStore are SHA-256 hashed — never raw plaintext.
  inferflux::PolicyStore store("/tmp/inferflux_hash_test_xyz.conf");
  store.SetApiKey("plaintext-key", {"generate"});

  auto keys = store.ApiKeys();
  REQUIRE(keys.size() == 1);
  // The stored value should be a 64-character hex string (SHA-256), not the
  // original.
  REQUIRE(keys[0].key != "plaintext-key");
  REQUIRE(keys[0].key.size() == 64); // SHA-256 hex = 32 bytes × 2

  // AddKeyHashed must let a pre-hashed key be authenticated.
  inferflux::ApiKeyAuth auth;
  auth.AddKeyHashed(keys[0].key, keys[0].scopes);

  // The original plaintext is authenticated via normal IsAllowed (which also
  // hashes).
  REQUIRE(auth.IsAllowed("plaintext-key"));

  // A different plaintext is rejected.
  REQUIRE(!auth.IsAllowed("wrong-key"));
}
