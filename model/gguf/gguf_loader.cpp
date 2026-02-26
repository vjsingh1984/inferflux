#include "model/gguf/gguf_loader.h"

#include <fstream>

namespace inferflux {

std::optional<ModelWeights> GGUFLoader::Load(const std::filesystem::path& path) const {
  if (!std::filesystem::exists(path)) {
    return std::nullopt;
  }
  std::ifstream input(path, std::ios::binary);
  if (!input.good()) {
    return std::nullopt;
  }
  // The full GGUF parsing logic will live here; for the MVP we just capture file size.
  input.seekg(0, std::ios::end);
  std::size_t bytes = static_cast<std::size_t>(input.tellg());
  Tensor meta_tensor;
  meta_tensor.name = "__gguf_stub__";
  meta_tensor.shape = {static_cast<int64_t>(bytes)};
  meta_tensor.data = {static_cast<float>(bytes)};
  ModelWeights weights;
  weights.format = "gguf";
  weights.tensors.push_back(std::move(meta_tensor));
  return weights;
}

}  // namespace inferflux
