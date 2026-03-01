#include <sstream>
#include <string>
#include <vector>

// The upstream json-schema-to-grammar helper (pulled from llama.cpp) expects a
// handful of utility symbols from common/common.cpp. Re-implement the minimal
// subset here so InferFlux can consume the converter without linking against
// llama.cpp's entire CLI support stack.

int LLAMA_BUILD_NUMBER = 0;
const char* LLAMA_COMMIT = "inferflux";
const char* LLAMA_COMPILER = "clang";
const char* LLAMA_BUILD_TARGET = "unknown";

std::string string_join(const std::vector<std::string>& values,
                        const std::string& separator) {
  std::ostringstream result;
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      result << separator;
    }
    result << values[i];
  }
  return result.str();
}

std::vector<std::string> string_split(const std::string& str,
                                      const std::string& delimiter) {
  std::vector<std::string> parts;
  std::size_t start = 0;
  std::size_t end = str.find(delimiter);
  while (end != std::string::npos) {
    parts.push_back(str.substr(start, end - start));
    start = end + delimiter.length();
    end = str.find(delimiter, start);
  }
  parts.push_back(str.substr(start));
  return parts;
}

std::string string_repeat(const std::string& str, std::size_t n) {
  if (n == 0) {
    return "";
  }
  std::string result;
  result.reserve(str.size() * n);
  for (std::size_t i = 0; i < n; ++i) {
    result += str;
  }
  return result;
}
