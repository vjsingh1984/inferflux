#include "server/auth/api_key_auth.h"

namespace inferflux {

namespace {
std::unordered_set<std::string> MakeScopes(const std::vector<std::string>& scopes) {
  std::unordered_set<std::string> set;
  if (scopes.empty()) {
    set.insert("generate");
    set.insert("read");
    return set;
  }
  for (const auto& scope : scopes) {
    if (!scope.empty()) {
      set.insert(scope);
    }
  }
  return set;
}
}  // namespace

void ApiKeyAuth::AddKey(const std::string& key, const std::vector<std::string>& scopes) {
  keys_[key] = MakeScopes(scopes);
  if (keys_[key].find("read") == keys_[key].end()) {
    keys_[key].insert("read");
  }
}

bool ApiKeyAuth::IsAllowed(const std::string& key) const {
  return keys_.find(key) != keys_.end();
}

bool ApiKeyAuth::HasKeys() const { return !keys_.empty(); }

std::vector<std::string> ApiKeyAuth::Scopes(const std::string& key) const {
  auto it = keys_.find(key);
  if (it == keys_.end()) {
    return {};
  }
  return std::vector<std::string>(it->second.begin(), it->second.end());
}

}  // namespace inferflux
