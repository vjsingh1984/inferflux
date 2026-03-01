#pragma once

#include <chrono>
#include <functional>
#include <map>
#include <mutex>
#include <string>

#include <nlohmann/json.hpp>

#include "net/http_client.h"

namespace inferflux {

class OIDCValidator {
 public:
  struct JwkKey {
    std::string kid;
    std::string n;
    std::string e;
  };

  OIDCValidator() = default;
  OIDCValidator(std::string issuer, std::string audience);

  bool Enabled() const { return !issuer_.empty() && !audience_.empty(); }
  bool Validate(const std::string& token, std::string* subject_out) const;

  // Testing helper: inject JWKS JSON directly without network fetch.
  void LoadJwksForTesting(const std::string& jwks_json) const;
  bool HasCachedKeyForTesting(const std::string& kid) const;
  bool EnsureJwksForTesting(const std::string& kid) const;
  bool VerifySignatureForTesting(const std::string& header_payload,
                                 const std::string& signature_b64url,
                                 const std::string& kid) const;
  void SetSignatureVerifierForTesting(
      std::function<bool(const std::string&, const std::string&, const JwkKey&)> verifier) const;

 private:
  std::string issuer_;
  std::string audience_;
  mutable std::mutex jwks_mutex_;
  mutable std::map<std::string, JwkKey> jwks_;
  mutable std::chrono::steady_clock::time_point jwks_expiry_;
  mutable bool jwks_loaded_{false};
  mutable HttpClient http_client_;
  mutable std::function<bool(const std::string&, const std::string&, const JwkKey&)> signature_verifier_override_;

  static std::string Base64UrlDecode(const std::string& input);
  static bool VerifyRS256(const std::string& header_payload,
                          const std::string& signature_b64url,
                          const std::string& n_b64url,
                          const std::string& e_b64url);

  bool AudienceMatches(const nlohmann::json& payload) const;
  bool EnsureJwks(const std::string& desired_kid) const;
  bool FetchJwksFromNetwork(std::map<std::string, JwkKey>* out) const;
  bool LoadJwksJson(const std::string& body, std::map<std::string, JwkKey>* out) const;
  bool ResolveKey(const std::string& kid, JwkKey* key) const;
  std::string JwksUrl() const;
};

}  // namespace inferflux
