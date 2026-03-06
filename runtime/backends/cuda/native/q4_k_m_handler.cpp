#include "runtime/backends/cuda/native/q4_k_m_handler.h"
#include "server/logging/logger.h"

namespace inferflux {
namespace runtime {
namespace cuda {
namespace native {

void Q4_K_M_Handler::DequantizeGpuToGpu(const void *quantized,
                                        half *dequantized, size_t num_elements,
                                        cudaStream_t stream) {
  cudaError_t err =
      dequantize_q4_k(quantized, dequantized, num_elements, stream);

  if (err != cudaSuccess) {
    log::Error("q4_k_m_handler", "Dequantization failed: " +
                                     std::string(cudaGetErrorString(err)));
  }
}

size_t Q4_K_M_Handler::GetDequantizedSize(size_t quantized_size) const {
  // Q4_K_M: 4.5 bits per value → 16 bits (FP16) per value
  // Ratio: 16 / 4.5 ≈ 3.56
  size_t num_blocks = quantized_size / sizeof(block_q4_k);
  size_t num_elements = num_blocks * QK_K;
  return num_elements * sizeof(half);
}

// Register handler
namespace {
QuantizationHandlerRegistrar<Q4_K_M_Handler> registrar("q4_k_m");
QuantizationHandlerRegistrar<Q4_K_M_Handler> registrar_short("q4_k");
} // namespace

} // namespace native
} // namespace cuda
} // namespace runtime
} // namespace inferflux
