#include <catch2/catch_amalgamated.hpp>

#include "server/auth/oidc_validator.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <string>

using json = nlohmann::json;

namespace {

std::string Base64UrlEncode(const std::string &input) {
  static const char table[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  int val = 0;
  int valb = -6;
  for (unsigned char c : input) {
    val = (val << 8) + c;
    valb += 8;
    while (valb >= 0) {
      out.push_back(table[(val >> valb) & 0x3F]);
      valb -= 6;
    }
  }
  if (valb > -6) {
    out.push_back(table[((val << 8) >> (valb + 8)) & 0x3F]);
  }
  for (char &c : out) {
    if (c == '+')
      c = '-';
    if (c == '/')
      c = '_';
  }
  while (!out.empty() && out.back() == '=') {
    out.pop_back();
  }
  return out;
}

std::string Base64UrlDecode(const std::string &input) {
  std::string normalized = input;
  for (char &c : normalized) {
    if (c == '-')
      c = '+';
    if (c == '_')
      c = '/';
  }
  while (normalized.size() % 4 != 0) {
    normalized.push_back('=');
  }
  std::string output;
  output.reserve(normalized.size() * 3 / 4);
  auto decode_char = [](char c) -> int {
    if (c >= 'A' && c <= 'Z')
      return c - 'A';
    if (c >= 'a' && c <= 'z')
      return c - 'a' + 26;
    if (c >= '0' && c <= '9')
      return c - '0' + 52;
    if (c == '+')
      return 62;
    if (c == '/')
      return 63;
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

const std::string &TestJwksJson() {
  static std::string jwks = R"({
    "keys": [
      {
        "alg": "RS256",
        "e": "AQAB",
        "kid": "test-key",
        "kty": "RSA",
        "n": "qnhNDv34eShaxVcVLlkr2sM4dAYNuPvaDf_ZFZ6crTv1QjXfyzdYIHElCjJ4OD6lYrlCWiMmvX-kKznBl4A2YZIMn1spdIJJFeXJjNykSxM5w-m6Fq4ikSrQYNPf4hQHyfSOTC5MD5-C9z6U4XWC7bAW4D31AYB1E8H9HcMWSL4n8FWM4jDaxq0294iux131cWjKECA4oyO41a9Y8BiXpt9S8BBKzz4eNHU15hdKN50i2OBtmQgm8x36ywMQ2QuRUXdHgZcM8t8oftJ5e0IzUQoNQk67WpuUOr2K-pDawDo0GBmWIRxeXAxKKlsgAhiwF-Z1w9He3SSZGGpGCdGANw",
        "use": "sig"
      }
    ]
  })";
  return jwks;
}

const std::string &TestSignatureB64() {
  static std::string sig = Base64UrlEncode("signature-ok");
  return sig;
}

std::string
MakeSignedJWT(const json &payload,
              const std::string &signature_b64 = TestSignatureB64()) {
  json header = {{"alg", "RS256"}, {"kid", "test-key"}, {"typ", "JWT"}};
  return Base64UrlEncode(header.dump()) + "." +
         Base64UrlEncode(payload.dump()) + "." + signature_b64;
}

void ConfigureValidator(inferflux::OIDCValidator *validator) {
  validator->LoadJwksForTesting(TestJwksJson());
  validator->SetSignatureVerifierForTesting(
      [](const std::string &header_payload, const std::string &signature,
         const inferflux::OIDCValidator::JwkKey &jwk) {
        (void)header_payload;
        return jwk.kid == "test-key" && signature == TestSignatureB64();
      });
}

} // namespace

TEST_CASE("OIDCValidator disabled without config", "[oidc]") {
  inferflux::OIDCValidator validator;
  REQUIRE(!validator.Enabled());
  REQUIRE(!validator.Validate("any-token", nullptr));
}

TEST_CASE("OIDCValidator validates issuer and audience", "[oidc]") {
  inferflux::OIDCValidator validator("https://issuer.example.com",
                                     "my-audience");
  ConfigureValidator(&validator);

  auto now = std::chrono::duration_cast<std::chrono::seconds>(
                 std::chrono::system_clock::now().time_since_epoch())
                 .count();
  json payload = {
      {"iss", "https://issuer.example.com"},
      {"aud", "my-audience"},
      {"sub", "user-123"},
      {"exp", now + 3600},
  };
  std::string subject;
  REQUIRE(validator.Validate(MakeSignedJWT(payload), &subject));
  REQUIRE(subject == "user-123");
}

TEST_CASE("OIDCValidator rejects wrong issuer", "[oidc]") {
  inferflux::OIDCValidator validator("https://correct.example.com", "aud");
  ConfigureValidator(&validator);
  auto now = std::chrono::duration_cast<std::chrono::seconds>(
                 std::chrono::system_clock::now().time_since_epoch())
                 .count();
  json payload = {
      {"iss", "https://wrong.example.com"},
      {"aud", "aud"},
      {"exp", now + 3600},
  };
  REQUIRE(!validator.Validate(MakeSignedJWT(payload), nullptr));
}

TEST_CASE("OIDCValidator rejects wrong audience", "[oidc]") {
  inferflux::OIDCValidator validator("https://iss.example.com", "correct-aud");
  ConfigureValidator(&validator);
  auto now = std::chrono::duration_cast<std::chrono::seconds>(
                 std::chrono::system_clock::now().time_since_epoch())
                 .count();
  json payload = {
      {"iss", "https://iss.example.com"},
      {"aud", "wrong-aud"},
      {"exp", now + 3600},
  };
  REQUIRE(!validator.Validate(MakeSignedJWT(payload), nullptr));
}

TEST_CASE("OIDCValidator rejects expired token", "[oidc]") {
  inferflux::OIDCValidator validator("https://iss.example.com", "aud");
  ConfigureValidator(&validator);
  json payload = {
      {"iss", "https://iss.example.com"},
      {"aud", "aud"},
      {"exp", 1000},
  };
  REQUIRE(!validator.Validate(MakeSignedJWT(payload), nullptr));
}

TEST_CASE("OIDCValidator rejects not-yet-valid token", "[oidc]") {
  inferflux::OIDCValidator validator("https://iss.example.com", "aud");
  ConfigureValidator(&validator);
  auto now = std::chrono::duration_cast<std::chrono::seconds>(
                 std::chrono::system_clock::now().time_since_epoch())
                 .count();
  json payload = {
      {"iss", "https://iss.example.com"},
      {"aud", "aud"},
      {"exp", now + 3600},
      {"nbf", now + 7200},
  };
  REQUIRE(!validator.Validate(MakeSignedJWT(payload), nullptr));
}

TEST_CASE("OIDCValidator defaults subject to oidc-user", "[oidc]") {
  inferflux::OIDCValidator validator("https://iss.example.com", "aud");
  ConfigureValidator(&validator);
  auto now = std::chrono::duration_cast<std::chrono::seconds>(
                 std::chrono::system_clock::now().time_since_epoch())
                 .count();
  json payload = {
      {"iss", "https://iss.example.com"},
      {"aud", "aud"},
      {"exp", now + 3600},
  };
  std::string subject;
  REQUIRE(validator.Validate(MakeSignedJWT(payload), &subject));
  REQUIRE(subject == "oidc-user");
}

TEST_CASE("OIDCValidator rejects invalid signature via override", "[oidc]") {
  inferflux::OIDCValidator validator("https://iss.example.com", "aud");
  ConfigureValidator(&validator);
  auto now = std::chrono::duration_cast<std::chrono::seconds>(
                 std::chrono::system_clock::now().time_since_epoch())
                 .count();
  json payload = {
      {"iss", "https://iss.example.com"},
      {"aud", "aud"},
      {"exp", now + 3600},
  };
  auto bad_signature = Base64UrlEncode("different-signature");
  REQUIRE(!validator.Validate(MakeSignedJWT(payload, bad_signature), nullptr));
}

TEST_CASE("OIDCValidator rejects malformed token", "[oidc]") {
  inferflux::OIDCValidator validator("https://iss.example.com", "aud");
  ConfigureValidator(&validator);
  REQUIRE(!validator.Validate("not-a-jwt", nullptr));
  REQUIRE(!validator.Validate("only.one.dot", nullptr));
}
