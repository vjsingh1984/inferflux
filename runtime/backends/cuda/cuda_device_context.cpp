#include "runtime/backends/cuda/cuda_device_context.h"
#include "server/logging/logger.h"

#include <cstdlib>

namespace inferflux {

CudaDeviceContext::CudaDeviceContext() {
#ifdef INFERFLUX_HAS_CUDA
  available_ = true;
#else
  available_ = false;
#endif
  if (!available_) {
    log::Error("cuda_backend",
               "CUDA runtime not linked; falling back to host allocations.");
  }
}

CudaDeviceContext::~CudaDeviceContext() = default;

std::unique_ptr<DeviceBuffer> CudaDeviceContext::Allocate(std::size_t bytes) {
  void *ptr = std::malloc(bytes);
  if (!ptr) {
    throw std::bad_alloc();
  }
  return std::make_unique<DeviceBuffer>(ptr, bytes);
}

void CudaDeviceContext::Free(std::unique_ptr<DeviceBuffer> buffer) {
  if (!buffer) {
    return;
  }
  std::free(buffer->data());
}

} // namespace inferflux
