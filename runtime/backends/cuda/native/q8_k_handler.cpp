#include "runtime/backends/cuda/native/q8_k_handler.h"
#include "server/logging/logger.h"

namespace inferflux {
namespace runtime {
namespace cuda {
namespace native {

void Q8_K_Handler::DequantizeGpuToGpu(const void *quantized, half *dequantized,
                                      size_t num_elements,
                                      cudaStream_t stream) {
  if (!quantized || !dequantized) {
    log::Error("q8_k_handler", "Null pointer passed to DequantizeGpuToGpu");
    return;
  }

  if (num_elements == 0) {
    return;
  }

  cudaError_t err =
      dequantize_q8_k(quantized, dequantized, num_elements, stream);

  if (err != cudaSuccess) {
    log::Error("q8_k_handler", "CUDA dequantization failed: " +
                                   std::string(cudaGetErrorString(err)));
  }
}

// Register Q8_K handler in the registry
namespace {
QuantizationHandlerRegistrar<Q8_K_Handler> registrar("q8_k");
}

} // namespace native
} // namespace cuda
} // namespace runtime
} // namespace inferflux
