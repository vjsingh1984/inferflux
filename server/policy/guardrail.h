#pragma once

#include "policy/opa_client.h"

#include <mutex>
#include <string>
#include <vector>

namespace inferflux {

class Guardrail {
 public:
  Guardrail() = default;

  void SetBlocklist(const std::vector<std::string>& words);
  void UpdateBlocklist(const std::vector<std::string>& words);
  std::vector<std::string> Blocklist() const;
  bool Check(const std::string& text, std::string* reason) const;
  bool Enabled() const;
  void SetOPAEndpoint(const std::string& endpoint);
  std::string OPAEndpoint() const;

 private:
  std::vector<std::string> blocklist_;
  std::string opa_endpoint_;
  mutable std::mutex mutex_;
  OPAClient opa_client_;
  static std::vector<std::string> Normalize(const std::vector<std::string>& words);
};

}  // namespace inferflux
