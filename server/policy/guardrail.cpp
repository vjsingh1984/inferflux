#include "server/policy/guardrail.h"

#include <algorithm>
#include <cctype>

namespace inferflux {

void Guardrail::SetBlocklist(const std::vector<std::string>& words) {
  blocklist_.clear();
  for (auto word : words) {
    std::transform(word.begin(), word.end(), word.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (!word.empty()) {
      blocklist_.push_back(word);
    }
  }
}

bool Guardrail::Check(const std::string& text, std::string* reason) const {
  if (blocklist_.empty()) {
    return true;
  }
  std::string lower = text;
  std::transform(lower.begin(), lower.end(), lower.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  for (const auto& word : blocklist_) {
    if (word.empty()) {
      continue;
    }
    if (lower.find(word) != std::string::npos) {
      if (reason) {
        *reason = "Blocked content keyword: " + word;
      }
      return false;
    }
  }
  return true;
}

}  // namespace inferflux
