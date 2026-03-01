#include "runtime/multimodal/image_preprocessor.h"

#include "net/http_client.h"

#include <nlohmann/json.hpp>
#include <openssl/evp.h>

#include <algorithm>
#include <iomanip>
#include <sstream>

using json = nlohmann::json;

namespace inferflux {

namespace {

// Standard base64 alphabet including URL-safe variant.
constexpr char kBase64Table[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

int Base64CharValue(unsigned char c) {
  if (c >= 'A' && c <= 'Z') return c - 'A';
  if (c >= 'a' && c <= 'z') return 26 + (c - 'a');
  if (c >= '0' && c <= '9') return 52 + (c - '0');
  if (c == '+' || c == '-') return 62;   // + (standard) or - (URL-safe)
  if (c == '/' || c == '_') return 63;   // / (standard) or _ (URL-safe)
  return -1;
}

std::vector<uint8_t> Base64Decode(const std::string& b64) {
  std::vector<uint8_t> out;
  out.reserve((b64.size() * 3) / 4 + 1);
  int val = 0;
  int valb = -8;
  for (unsigned char c : b64) {
    if (c == '=' || c == '\n' || c == '\r') {
      break;
    }
    int v = Base64CharValue(c);
    if (v < 0) continue;
    val = (val << 6) + v;
    valb += 6;
    if (valb >= 0) {
      out.push_back(static_cast<uint8_t>((val >> valb) & 0xFF));
      valb -= 8;
    }
  }
  return out;
}

// The default media marker used by libmtmd for image injection.
constexpr char kMediaMarker[] = "<__media__>";

}  // namespace

std::vector<uint8_t> ImagePreprocessor::DecodeBase64DataUri(const std::string& data_uri,
                                                             std::string* error) {
  // Format: data:<mediatype>;base64,<data>
  if (data_uri.substr(0, 5) != "data:") {
    if (error) *error = "not a data URI";
    return {};
  }
  auto comma = data_uri.find(',');
  if (comma == std::string::npos) {
    if (error) *error = "malformed data URI: missing comma";
    return {};
  }
  auto header = data_uri.substr(0, comma);
  if (header.find("base64") == std::string::npos) {
    if (error) *error = "data URI is not base64 encoded";
    return {};
  }
  auto encoded = data_uri.substr(comma + 1);
  auto bytes = Base64Decode(encoded);
  if (bytes.empty()) {
    if (error) *error = "base64 decode produced empty result";
    return {};
  }
  return bytes;
}

std::vector<uint8_t> ImagePreprocessor::FetchUrl(const std::string& url,
                                                  std::string* error) {
  if (url.empty()) {
    if (error) *error = "empty URL";
    return {};
  }
  try {
    HttpClient client;
    auto resp = client.Get(url);
    if (resp.status < 200 || resp.status >= 300) {
      if (error) {
        *error = "HTTP " + std::to_string(resp.status) + " fetching image";
      }
      return {};
    }
    return std::vector<uint8_t>(resp.body.begin(), resp.body.end());
  } catch (const std::exception& ex) {
    if (error) *error = std::string("HTTP fetch error: ") + ex.what();
    return {};
  }
}

std::string ImagePreprocessor::ComputeSha256Hex(const std::vector<uint8_t>& data) {
  unsigned char hash[EVP_MAX_MD_SIZE];
  unsigned int hash_len = 0;
  auto* ctx = EVP_MD_CTX_new();
  if (!ctx) return {};
  if (EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) != 1 ||
      EVP_DigestUpdate(ctx, data.data(), data.size()) != 1 ||
      EVP_DigestFinal_ex(ctx, hash, &hash_len) != 1) {
    EVP_MD_CTX_free(ctx);
    return {};
  }
  EVP_MD_CTX_free(ctx);
  std::ostringstream oss;
  for (unsigned int i = 0; i < hash_len; ++i) {
    oss << std::hex << std::setw(2) << std::setfill('0')
        << static_cast<int>(hash[i]);
  }
  return oss.str();
}

ContentArrayResult ImagePreprocessor::ProcessContentArray(const std::string& content_array_json) {
  ContentArrayResult result;
  if (content_array_json.empty()) {
    return result;
  }
  try {
    auto arr = json::parse(content_array_json);
    if (!arr.is_array()) {
      result.error = "content is not a JSON array";
      return result;
    }
    for (const auto& part : arr) {
      if (!part.is_object()) continue;
      std::string type;
      if (part.contains("type") && part["type"].is_string()) {
        type = part["type"].get<std::string>();
      }
      if (type == "text") {
        if (part.contains("text") && part["text"].is_string()) {
          result.text += part["text"].get<std::string>();
        }
      } else if (type == "image_url") {
        // Extract URL from {"image_url": {"url": "..."}} or {"url": "..."}.
        std::string url;
        if (part.contains("image_url") && part["image_url"].is_object()) {
          const auto& img = part["image_url"];
          if (img.contains("url") && img["url"].is_string()) {
            url = img["url"].get<std::string>();
          }
        } else if (part.contains("url") && part["url"].is_string()) {
          url = part["url"].get<std::string>();
        }
        if (url.empty()) {
          result.error = "image_url part missing url field";
          result.text += kMediaMarker;  // Keep marker count consistent.
          continue;
        }
        // Inject media marker in place of the image.
        result.text += kMediaMarker;
        DecodedImage img;
        img.source_uri = url;
        std::string decode_error;
        if (url.substr(0, 5) == "data:") {
          img.raw_bytes = DecodeBase64DataUri(url, &decode_error);
        } else {
          img.raw_bytes = FetchUrl(url, &decode_error);
        }
        if (!decode_error.empty()) {
          result.error = "image decode error: " + decode_error;
        }
        if (!img.raw_bytes.empty()) {
          img.image_id = ComputeSha256Hex(img.raw_bytes);
        }
        result.images.push_back(std::move(img));
      }
      // Other types (e.g., "audio") are silently skipped.
    }
  } catch (const json::exception& ex) {
    result.error = std::string("JSON parse error: ") + ex.what();
  }
  return result;
}

}  // namespace inferflux
