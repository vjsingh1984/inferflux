#pragma once

#include <string>
#include <vector>

namespace inferflux {

class Guardrail {
 public:
  Guardrail() = default;

  void SetBlocklist(const std::vector<std::string>& words);
  bool Check(const std::string& text, std::string* reason) const;
  bool Enabled() const { return !blocklist_.empty(); }

 private:
  std::vector<std::string> blocklist_;
};

}  // namespace inferflux
