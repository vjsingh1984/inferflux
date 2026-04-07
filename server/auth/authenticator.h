#pragma once

#include <string>

namespace inferflux {

// Unified authentication interface.
// Implementations: ApiKeyAuth (SHA-256 API keys), OIDCValidator (RS256 JWTs).
class IAuthenticator {
public:
  virtual ~IAuthenticator() = default;

  // Returns true if the token/key is valid.
  // On success, sets *subject_out to the authenticated identity
  // (e.g., the key hash or OIDC subject claim).
  virtual bool Authenticate(const std::string &token,
                            std::string *subject_out) const = 0;

  // Returns true if this authenticator has been configured with credentials.
  virtual bool Enabled() const = 0;
};

} // namespace inferflux
