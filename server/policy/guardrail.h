#pragma once

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

 private:
  std::vector<std::string> blocklist_;
  mutable std::mutex mutex_;
  static std::vector<std::string> Normalize(const std::vector<std::string>& words);
};

}  // namespace inferflux
