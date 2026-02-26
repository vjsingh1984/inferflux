#pragma once

#include <string>

namespace inferflux {

class OIDCValidator {
 public:
  OIDCValidator() = default;
  OIDCValidator(std::string issuer, std::string audience);

  bool Enabled() const { return !issuer_.empty() && !audience_.empty(); }
  bool Validate(const std::string& token, std::string* subject_out) const;

 private:
  std::string issuer_;
  std::string audience_;

  static std::string Base64UrlDecode(const std::string& input);
  static std::string ExtractField(const std::string& payload, const std::string& field);
};

}  // namespace inferflux
