#pragma once

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>

namespace inferflux {

/// Convert a string to lowercase (ASCII only).
inline std::string ToLower(const std::string &s) {
  std::string result = s;
  std::transform(result.begin(), result.end(), result.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return result;
}

/// Trim leading and trailing whitespace.
inline std::string Trim(const std::string &s) {
  auto start = s.find_first_not_of(" \t\r\n");
  if (start == std::string::npos)
    return "";
  auto end = s.find_last_not_of(" \t\r\n");
  return s.substr(start, end - start + 1);
}

/// Parse a boolean from string ("true"/"1"/"yes" → true, else false).
inline bool ParseBool(const std::string &val) {
  std::string lower = ToLower(val);
  return lower == "true" || lower == "1" || lower == "yes";
}

/// Parse a boolean from environment variable with fallback.
inline bool ParseBoolEnv(const char *env_name, bool fallback) {
  const char *val = std::getenv(env_name);
  if (!val)
    return fallback;
  return ParseBool(val);
}

// Common size constants
constexpr std::size_t kKiB = 1024;
constexpr std::size_t kMiB = 1024 * kKiB;
constexpr std::size_t kGiB = 1024 * kMiB;

} // namespace inferflux
