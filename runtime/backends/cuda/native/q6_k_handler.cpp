#include "runtime/backends/cuda/native/q6_k_handler.h"
#include "server/logging/logger.h"

namespace inferflux {
namespace runtime {
namespace cuda {
namespace native {

void Q6_K_Handler::DequantizeGpuToGpu(const void *quantized,
                                       half *dequantized,
                                       size_t num_elements,
                                       cudaStream_t stream) {
  cudaError_t err =
      dequantize_q6_k(quantized, dequantized, num_elements, stream);

  if (err != cudaSuccess) {
    log::Error("q6_k_handler", "Dequantization failed: " +
                                     std::string(cudaGetErrorString(err)));
  }
}

size_t Q6_K_Handler::GetDequantizedSize(size_t quantized_size) const {
  // Q6_K: 6.5625 bits per value → 16 bits (FP16) per value
  size_t num_blocks = quantized_size / sizeof(block_q6_k);
  size_t num_elements = num_blocks * QK_K;
  return num_elements * sizeof(half);
}

// Register handler
namespace {
QuantizationHandlerRegistrar<Q6_K_Handler> registrar("q6_k");
}

} // namespace native
} // namespace cuda
} // namespace runtime
} // namespace inferflux
