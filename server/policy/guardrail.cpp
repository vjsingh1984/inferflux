#include "server/policy/guardrail.h"

#include <algorithm>
#include <cctype>

namespace inferflux {
namespace {
std::vector<std::string> NormalizeWords(const std::vector<std::string>& words) {
  std::vector<std::string> normalized;
  normalized.reserve(words.size());
  for (auto word : words) {
    std::transform(word.begin(), word.end(), word.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (!word.empty()) {
      normalized.push_back(word);
    }
  }
  return normalized;
}
}

void Guardrail::SetBlocklist(const std::vector<std::string>& words) {
  std::lock_guard<std::mutex> lock(mutex_);
  blocklist_ = NormalizeWords(words);
}

void Guardrail::UpdateBlocklist(const std::vector<std::string>& words) {
  std::lock_guard<std::mutex> lock(mutex_);
  blocklist_ = NormalizeWords(words);
}

std::vector<std::string> Guardrail::Blocklist() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return blocklist_;
}

bool Guardrail::Check(const std::string& text, std::string* reason) const {
  std::lock_guard<std::mutex> lock(mutex_);
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
  if (!opa_endpoint_.empty()) {
    // Placeholder for future OPA integration. For now we simply log intent via reason when requested.
    if (reason) {
      *reason = "OPA endpoint configured (" + opa_endpoint_ + ") but not yet enforced";
    }
  }
  return true;
}

bool Guardrail::Enabled() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return !blocklist_.empty();
}

void Guardrail::SetOPAEndpoint(const std::string& endpoint) {
  std::lock_guard<std::mutex> lock(mutex_);
  opa_endpoint_ = endpoint;
}

std::string Guardrail::OPAEndpoint() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return opa_endpoint_;
}

}  // namespace inferflux
