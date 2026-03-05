#include "runtime/backends/cuda/native/q8_0_handler.h"
#include "runtime/backends/cuda/native/kernels/dequantization.cuh"
#include "server/logging/logger.h"

namespace inferflux {
namespace runtime {
namespace cuda {
namespace native {

void Q8_0_Handler::DequantizeGpuToGpu(const void *quantized, half *dequantized,
                                      size_t num_elements,
                                      cudaStream_t stream) {
  if (!quantized || !dequantized) {
    log::Error("q8_0_handler", "Null pointer passed to DequantizeGpuToGpu");
    return;
  }

  if (num_elements == 0) {
    return;
  }

  // Use CUDA kernel for dequantization
  cudaError_t err = dequantize_q8_0(quantized, dequantized, num_elements, stream);

  if (err != cudaSuccess) {
    log::Error("q8_0_handler",
               "CUDA dequantization failed: " + std::string(cudaGetErrorString(err)));
  }
}

// Register Q8_0 handler in the registry
namespace {
QuantizationHandlerRegistrar<Q8_0_Handler> registrar("q8_0");
}

} // namespace native
} // namespace cuda
} // namespace runtime
} // namespace inferflux
