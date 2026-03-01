#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace inferflux {

// A decoded image ready for multimodal inference.
// raw_bytes holds the original compressed image data (JPEG, PNG, etc.);
// mtmd_helper_bitmap_init_from_buf decodes them internally via stb_image.
struct DecodedImage {
  std::vector<uint8_t> raw_bytes;  // Compressed image bytes (JPEG/PNG/etc.).
  std::string source_uri;          // Original URL or data URI.
  std::string image_id;            // SHA-256 hex of raw_bytes (KV cache tracking).
};

// Result of processing a JSON content array that may contain image_url parts.
struct ContentArrayResult {
  std::string text;                   // Text with <__media__> markers where images appear.
  std::vector<DecodedImage> images;   // Decoded images in order of appearance.
  std::string error;                  // Non-empty if any image failed to decode.
};

// Stateless utilities for extracting and decoding images from OpenAI-style
// multipart content arrays. Designed for use in the HTTP server layer.
class ImagePreprocessor {
 public:
  // Parse an OpenAI content array JSON string (already serialized), extracting
  // text (with <__media__> markers) and decoded images in order.
  // Example input: [{"type":"text","text":"describe: "},
  //                 {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}}]
  static ContentArrayResult ProcessContentArray(const std::string& content_array_json);

  // Decode a base64 data URI (data:image/...;base64,...) to raw bytes.
  // Returns empty vector on error; sets *error if non-null.
  static std::vector<uint8_t> DecodeBase64DataUri(const std::string& data_uri,
                                                   std::string* error = nullptr);

  // Fetch image bytes from an HTTP/HTTPS URL using HttpClient.
  // Returns empty vector on error; sets *error if non-null.
  static std::vector<uint8_t> FetchUrl(const std::string& url,
                                       std::string* error = nullptr);

  // Compute SHA-256 hex digest of raw bytes using OpenSSL EVP.
  static std::string ComputeSha256Hex(const std::vector<uint8_t>& data);
};

}  // namespace inferflux
