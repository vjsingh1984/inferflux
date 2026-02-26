#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace inferflux {

class SimpleTokenizer {
 public:
  SimpleTokenizer();

  std::vector<int> Encode(const std::string& text);
  std::string Decode(const std::vector<int>& tokens) const;

 private:
  int AddToken(const std::string& token);
  std::vector<std::string> Tokenize(const std::string& text) const;

  std::unordered_map<std::string, int> vocab_;
  std::vector<std::string> reverse_;
};

}  // namespace inferflux
