#pragma once

#include "runtime/tensors/tensor.h"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace inferflux {

struct ModelWeights {
  std::string format;
  std::vector<Tensor> tensors;
};

class GGUFLoader {
 public:
  std::optional<ModelWeights> Load(const std::filesystem::path& path) const;
};

}  // namespace inferflux
