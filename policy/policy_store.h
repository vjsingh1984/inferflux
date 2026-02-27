#pragma once

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

class PolicyStore {
 public:
  PolicyStore(std::string path, std::string passphrase = "");

  bool Load();
  bool Save() const;

  std::vector<ApiKeyPolicy> ApiKeys() const;
  void SetApiKey(const std::string& key, const std::vector<std::string>& scopes);
  bool RemoveApiKey(const std::string& key);

  std::vector<std::string> GuardrailBlocklist() const;
  void SetGuardrailBlocklist(const std::vector<std::string>& blocklist);

  int RateLimitPerMinute() const;
  void SetRateLimitPerMinute(int limit);

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
