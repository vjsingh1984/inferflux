#include "runtime/backends/cuda/cuda_device_context.h"
#include "server/logging/logger.h"

#ifdef INFERFLUX_HAS_CUDA
#include <cuda_runtime_api.h>
#endif

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
#ifdef INFERFLUX_HAS_CUDA
  if (available_) {
    void *ptr = nullptr;
    auto err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) {
      log::Error("cuda_backend", std::string("cudaMalloc failed: ") +
                                     cudaGetErrorString(err) +
                                     "; falling back to host");
    } else {
      return std::make_unique<DeviceBuffer>(ptr, bytes);
    }
  }
#endif
  // CPU fallback when CUDA is not linked or cudaMalloc fails.
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
#ifdef INFERFLUX_HAS_CUDA
  if (available_) {
    // Check if this pointer is on device memory.  cudaPointerGetAttributes
    // succeeds for both device and host pointers under UVA, so we inspect
    // the memory type to choose the correct free path.
    cudaPointerAttributes attrs{};
    auto err = cudaPointerGetAttributes(&attrs, buffer->data());
    if (err == cudaSuccess && attrs.type == cudaMemoryTypeDevice) {
      cudaFree(buffer->data());
      return;
    }
    // Not a device pointer — fall through to host free.
    // Clear any CUDA error from the failed query.
    cudaGetLastError();
  }
#endif
  std::free(buffer->data());
}

} // namespace inferflux
