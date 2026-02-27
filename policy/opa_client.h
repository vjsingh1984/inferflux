#pragma once

#include <string>

namespace inferflux {

struct OPAResult {
  bool allow{true};
  std::string reason;
};

class OPAClient {
 public:
  OPAClient() = default;
  explicit OPAClient(std::string endpoint);

  bool Evaluate(const std::string& prompt, OPAResult* result) const;

 private:
  std::string endpoint_;
};

}  // namespace inferflux
