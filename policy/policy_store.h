#pragma once

#include "policy/policy_backend.h"

#include <array>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace inferflux {

struct ApiKeyPolicy {
  std::string key;
  std::vector<std::string> scopes;
};

class PolicyStore : public PolicyBackend {
 public:
  PolicyStore(std::string path, std::string passphrase = "");

  bool Load() override;
  bool Save() const override;

  std::vector<PolicyKeyEntry> ApiKeys() const override;
  void SetApiKey(const std::string& key, const std::vector<std::string>& scopes) override;
  bool RemoveApiKey(const std::string& key) override;

  std::vector<std::string> GuardrailBlocklist() const override;
  void SetGuardrailBlocklist(const std::vector<std::string>& blocklist) override;

  int RateLimitPerMinute() const override;
  void SetRateLimitPerMinute(int limit) override;

  std::string Name() const override { return "ini"; }

 private:
  std::string path_;
  mutable std::mutex mutex_;
  std::unordered_map<std::string, std::vector<std::string>> api_keys_;
  std::vector<std::string> guardrail_blocklist_;
  int rate_limit_per_minute_{0};
  bool encryption_enabled_{false};
  std::array<unsigned char, 32> key_{};

  void EnsureParentDir() const;
  static std::vector<std::string> SplitCSV(const std::string& line);
  static std::string JoinCSV(const std::vector<std::string>& values);
  bool Encrypt(const std::string& plaintext, std::string* output) const;
  bool Decrypt(const std::string& encrypted, std::string* plaintext) const;
};

}  // namespace inferflux
