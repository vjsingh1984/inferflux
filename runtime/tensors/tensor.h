#pragma once

#include <string>
#include <vector>

namespace inferflux {

struct Tensor {
  std::string name;
  std::vector<float> data;
  std::vector<int64_t> shape;
};

}  // namespace inferflux
