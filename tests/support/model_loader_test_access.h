#pragma once

#include "runtime/backends/cuda/native/gguf_model_loader.h"

namespace inferflux {
namespace runtime {
namespace cuda {
namespace native {

/// Friend-based accessor for GGUFModelLoader private members.
class ModelLoaderTestAccess {
public:
  explicit ModelLoaderTestAccess(GGUFModelLoader &l) : l_(l) {}

  auto &tensors() { return l_.tensors_; }
  auto &gguf_to_internal_name_map() { return l_.gguf_to_internal_name_map_; }
  auto &internal_to_gguf_name_map() { return l_.internal_to_gguf_name_map_; }
  auto &d_quantized_buffer() { return l_.d_quantized_buffer_; }
  auto &quantized_buffer_size() { return l_.quantized_buffer_size_; }
  auto &has_dequantized_entries() { return l_.has_dequantized_entries_; }

private:
  GGUFModelLoader &l_;
};

} // namespace native
} // namespace cuda
} // namespace runtime
} // namespace inferflux
