#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace inferflux {

class ApiKeyAuth {
 public:
  void AddKey(const std::string& key, const std::vector<std::string>& scopes = {});
  bool IsAllowed(const std::string& key) const;
  bool HasKeys() const;
  std::vector<std::string> Scopes(const std::string& key) const;
  void RemoveKey(const std::string& key);

 private:
  std::unordered_map<std::string, std::unordered_set<std::string>> keys_;
};

}  // namespace inferflux
