#include "model/tokenizer/simple_tokenizer.h"

#include <cassert>
#include <iostream>

int main() {
  inferflux::SimpleTokenizer tokenizer;
  auto tokens = tokenizer.Encode("hello world");
  assert(!tokens.empty());
  auto text = tokenizer.Decode(tokens);
  assert(!text.empty());
  std::cout << "Tokenizer encode/decode OK" << std::endl;
  return 0;
}
