#include "server/auth/api_key_auth.h"

namespace inferflux {

void ApiKeyAuth::AddKey(const std::string& key) { keys_.insert(key); }

bool ApiKeyAuth::IsAllowed(const std::string& key) const {
  return keys_.find(key) != keys_.end();
}

bool ApiKeyAuth::HasKeys() const { return !keys_.empty(); }

}  // namespace inferflux
