#include "server/auth/oidc_validator.h"

#include <cctype>
#include <stdexcept>

namespace inferflux {

OIDCValidator::OIDCValidator(std::string issuer, std::string audience)
    : issuer_(std::move(issuer)), audience_(std::move(audience)) {}

std::string OIDCValidator::Base64UrlDecode(const std::string& input) {
  std::string normalized = input;
  for (char& c : normalized) {
    if (c == '-') c = '+';
    if (c == '_') c = '/';
  }
  while (normalized.size() % 4 != 0) {
    normalized.push_back('=');
  }
  std::string output;
  output.reserve(normalized.size() * 3 / 4);
  auto decode_char = [](char c) -> int {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
  };
  for (std::size_t i = 0; i < normalized.size(); i += 4) {
    int b0 = decode_char(normalized[i]);
    int b1 = decode_char(normalized[i + 1]);
    int b2 = normalized[i + 2] == '=' ? -1 : decode_char(normalized[i + 2]);
    int b3 = normalized[i + 3] == '=' ? -1 : decode_char(normalized[i + 3]);
    if (b0 < 0 || b1 < 0 || (normalized[i + 2] != '=' && b2 < 0) ||
        (normalized[i + 3] != '=' && b3 < 0)) {
      return {};
    }
    output.push_back(static_cast<char>((b0 << 2) | (b1 >> 4)));
    if (b2 >= 0) {
      output.push_back(static_cast<char>(((b1 & 0xF) << 4) | (b2 >> 2)));
    }
    if (b3 >= 0) {
      output.push_back(static_cast<char>(((b2 & 0x3) << 6) | b3));
    }
  }
  return output;
}

std::string OIDCValidator::ExtractField(const std::string& payload, const std::string& field) {
  std::string needle = "\"" + field + "\"";
  auto key_pos = payload.find(needle);
  if (key_pos == std::string::npos) return {};
  auto colon = payload.find(':', key_pos + needle.size());
  if (colon == std::string::npos) return {};
  auto start = payload.find('"', colon);
  if (start == std::string::npos) return {};
  ++start;
  std::string value;
  bool escape = false;
  for (std::size_t i = start; i < payload.size(); ++i) {
    char c = payload[i];
    if (!escape && c == '\\') {
      escape = true;
      continue;
    }
    if (!escape && c == '"') {
      break;
    }
    value.push_back(c);
    escape = false;
  }
  return value;
}

bool OIDCValidator::Validate(const std::string& token, std::string* subject_out) const {
  if (!Enabled()) {
    return false;
  }
  auto first_dot = token.find('.');
  auto second_dot = token.find('.', first_dot == std::string::npos ? 0 : first_dot + 1);
  if (first_dot == std::string::npos || second_dot == std::string::npos) {
    return false;
  }
  std::string payload = Base64UrlDecode(token.substr(first_dot + 1, second_dot - first_dot - 1));
  if (payload.empty()) {
    return false;
  }
  auto iss = ExtractField(payload, "iss");
  auto aud = ExtractField(payload, "aud");
  auto sub = ExtractField(payload, "sub");
  if (iss != issuer_ || aud != audience_) {
    return false;
  }
  if (subject_out) {
    *subject_out = sub.empty() ? "oidc-user" : sub;
  }
  return true;
}

}  // namespace inferflux
