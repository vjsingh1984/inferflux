#include <catch2/catch_amalgamated.hpp>

#include "runtime/multimodal/image_preprocessor.h"

#include <string>
#include <vector>

namespace {

// 1x1 red pixel JPEG encoded as base64 (for offline testing).
// Generated via: echo -n ... | base64
// This is a minimal valid JPEG binary (SOI + minimal segments).
// We use a tiny 1x1 red pixel generated offline.
// Hex: FF D8 FF E0 ... FF D9
// Here we just use a hardcoded valid base64 JPEG for testing purposes.
constexpr const char *kSmallJpegBase64 =
    "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8U"
    "HRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgN"
    "DRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy"
    "MjL/wAARCAABAAEDASIAAhEBAxEB/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAA"
    "AAAAAAAAAAAAAAD/xAAUAQEAAAAAAAAAAAAAAAAAAAAA/"
    "8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/"
    "aAAwDAQACEQMRAD8AJQAB/9k=";

// Minimal 1x1 white PNG as base64 (RFC 2397 data URI).
constexpr const char *kSmallPngDataUri =
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/"
    "PchI6QAAAABJRU5ErkJggg==";

} // namespace

namespace inferflux {

// ---------------------------------------------------------------------------
// DecodeBase64DataUri tests
// ---------------------------------------------------------------------------

TEST_CASE("DecodeBase64DataUri returns bytes for valid JPEG data URI",
          "[multimodal]") {
  std::string data_uri =
      std::string("data:image/jpeg;base64,") + kSmallJpegBase64;
  std::string err;
  auto bytes = ImagePreprocessor::DecodeBase64DataUri(data_uri, &err);
  CHECK(err.empty());
  CHECK(!bytes.empty());
  // JPEG magic bytes: FF D8
  REQUIRE(bytes.size() >= 2);
  CHECK(static_cast<unsigned char>(bytes[0]) == 0xFF);
  CHECK(static_cast<unsigned char>(bytes[1]) == 0xD8);
}

TEST_CASE("DecodeBase64DataUri returns error for non-data-URI",
          "[multimodal]") {
  std::string err;
  auto bytes = ImagePreprocessor::DecodeBase64DataUri(
      "https://example.com/img.jpg", &err);
  CHECK(bytes.empty());
  CHECK(!err.empty());
}

TEST_CASE("DecodeBase64DataUri returns error for malformed data URI",
          "[multimodal]") {
  std::string err;
  auto bytes = ImagePreprocessor::DecodeBase64DataUri(
      "data:image/jpeg;no-comma-here", &err);
  CHECK(bytes.empty());
  CHECK(!err.empty());
}

// ---------------------------------------------------------------------------
// ComputeSha256Hex tests
// ---------------------------------------------------------------------------

TEST_CASE("ComputeSha256Hex of empty data returns known value",
          "[multimodal]") {
  std::vector<uint8_t> empty;
  auto hex = ImagePreprocessor::ComputeSha256Hex(empty);
  // SHA-256 of empty string =
  // e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
  CHECK(hex ==
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
}

TEST_CASE("ComputeSha256Hex of known bytes returns correct digest",
          "[multimodal]") {
  // SHA-256("abc") verified via: echo -n "abc" | openssl dgst -sha256
  std::vector<uint8_t> abc = {'a', 'b', 'c'};
  auto hex = ImagePreprocessor::ComputeSha256Hex(abc);
  CHECK(hex ==
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

// ---------------------------------------------------------------------------
// ProcessContentArray tests
// ---------------------------------------------------------------------------

TEST_CASE("ProcessContentArray returns empty result for empty input",
          "[multimodal]") {
  auto result = ImagePreprocessor::ProcessContentArray("");
  CHECK(result.text.empty());
  CHECK(result.images.empty());
}

TEST_CASE("ProcessContentArray handles text-only content array",
          "[multimodal]") {
  std::string json = R"([{"type":"text","text":"Hello world"}])";
  auto result = ImagePreprocessor::ProcessContentArray(json);
  CHECK(result.text == "Hello world");
  CHECK(result.images.empty());
  CHECK(result.error.empty());
}

TEST_CASE("ProcessContentArray inserts media marker for image_url parts",
          "[multimodal]") {
  // Use a data URI so no network is required.
  std::string json = R"([
    {"type":"text","text":"Describe: "},
    {"type":"image_url","image_url":{"url":")" +
                     std::string(kSmallPngDataUri) + R"("}}
  ])";
  auto result = ImagePreprocessor::ProcessContentArray(json);
  CHECK(result.text == "Describe: <__media__>");
  REQUIRE(result.images.size() == 1);
  CHECK(!result.images[0].raw_bytes.empty());
  CHECK(!result.images[0].image_id.empty());
  CHECK(result.images[0].source_uri == std::string(kSmallPngDataUri));
}

TEST_CASE("ProcessContentArray handles multiple text and image parts",
          "[multimodal]") {
  std::string json = R"([
    {"type":"text","text":"First: "},
    {"type":"image_url","image_url":{"url":")" +
                     std::string(kSmallPngDataUri) + R"("}},
    {"type":"text","text":" Second: "},
    {"type":"image_url","image_url":{"url":")" +
                     std::string(kSmallPngDataUri) + R"("}}
  ])";
  auto result = ImagePreprocessor::ProcessContentArray(json);
  CHECK(result.text == "First: <__media__> Second: <__media__>");
  CHECK(result.images.size() == 2);
}

TEST_CASE("ProcessContentArray handles non-array JSON gracefully",
          "[multimodal]") {
  auto result = ImagePreprocessor::ProcessContentArray(R"({"type":"text"})");
  CHECK(!result.error.empty());
  CHECK(result.images.empty());
}

TEST_CASE(
    "ProcessContentArray handles image_url with HTTP URL without crashing",
    "[multimodal]") {
  // Network is unavailable in test environments; expect graceful error, no
  // crash.
  std::string json = R"([
    {"type":"image_url","image_url":{"url":"http://localhost:0/nonexistent.jpg"}}
  ])";
  auto result = ImagePreprocessor::ProcessContentArray(json);
  // Should have one (failed) image entry with empty raw_bytes and one marker.
  CHECK(result.text == "<__media__>");
  REQUIRE(result.images.size() == 1);
  CHECK(result.images[0].raw_bytes.empty());
  CHECK(!result.error.empty());
}

} // namespace inferflux
