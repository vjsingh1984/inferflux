#include "runtime/backends/cuda/native/q5_k_m_handler.h"
#include "server/logging/logger.h"

namespace inferflux {
namespace runtime {
namespace cuda {
namespace native {

void Q5_K_M_Handler::DequantizeGpuToGpu(const void *quantized,
                                        half *dequantized, size_t num_elements,
                                        cudaStream_t stream) {
  cudaError_t err =
      dequantize_q5_k(quantized, dequantized, num_elements, stream);

  if (err != cudaSuccess) {
    log::Error("q5_k_m_handler", "Dequantization failed: " +
                                     std::string(cudaGetErrorString(err)));
  }
}

size_t Q5_K_M_Handler::GetDequantizedSize(size_t quantized_size) const {
  // Q5_K_M: 5.5 bits per value → 16 bits (FP16) per value
  size_t num_blocks = quantized_size / sizeof(block_q5_k);
  size_t num_elements = num_blocks * QK_K;
  return num_elements * sizeof(half);
}

// Register handler
namespace {
QuantizationHandlerRegistrar<Q5_K_M_Handler> registrar("q5_k_m");
QuantizationHandlerRegistrar<Q5_K_M_Handler> registrar_short("q5_k");
} // namespace

} // namespace native
} // namespace cuda
} // namespace runtime
} // namespace inferflux
