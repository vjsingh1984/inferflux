#pragma once

#include "runtime/backends/gpu/gpu_accelerated_backend.h"

namespace inferflux {

class VulkanBackend : public GpuAcceleratedBackend {
public:
  VulkanBackend();

#ifdef INFERFLUX_USE_COMMON_BACKEND_TYPES
  std::string Name() const override { return "llama_cpp_vulkan"; }
#endif
};

} // namespace inferflux
