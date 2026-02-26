#pragma once

#include <string>
#include <unordered_set>

namespace inferflux {

 class ApiKeyAuth {
 public:
  void AddKey(const std::string& key);
  bool IsAllowed(const std::string& key) const;
  bool HasKeys() const;

 private:
  std::unordered_set<std::string> keys_;
 };

}  // namespace inferflux
