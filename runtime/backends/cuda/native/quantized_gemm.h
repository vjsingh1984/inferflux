#pragma once

#include "runtime/backends/cuda/native/model_loader.h"
#include "runtime/backends/cuda/native/quantization_handler.h"
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <memory>

namespace inferflux {

// Bring in the types from the nested namespace
using runtime::cuda::native::IModelLoader;
using runtime::cuda::native::IQuantizationHandler;
using runtime::cuda::native::IWeightAccessor;
using runtime::cuda::native::ModelInfo;

/**
 * @brief QuantizedGemm - GEMM dispatcher for quantized weights
 *
 * Handles matrix multiplication with quantized weights efficiently.
 * Can either:
 * 1. Dequantize first, then use cuBLAS (for large repeated ops)
 * 2. Use custom quantized GEMM kernels (for memory efficiency)
 *
 * This is the key component that enables efficient inference with
 * quantized GGUF models.
 */
class QuantizedGemm {
public:
  QuantizedGemm() = default;
  ~QuantizedGemm();

  /**
   * @brief Initialize GEMM dispatcher
   */
  bool Initialize(cudaStream_t stream);

  /**
   * @brief Switch to different stream
   */
  void SetStream(cudaStream_t stream);

  /**
   * @brief Matrix multiplication with quantized weights
   *
   * Computes: C = A * W^T where W is quantized
   *
   * @param M Batch size (number of vectors in A)
   * @param N Output size (number of vectors in W)
   * @param K Inner dimension (vector size)
   * @param A Input matrix [M, K] in row-major (FP16)
   * @param weight_accessor Quantized weight accessor for W
   * @param C Output matrix [M, N] in row-major (FP16)
   * @return true on success
   */
  bool Gemm(int M, int N, int K, const half *A,
            std::shared_ptr<IWeightAccessor> weight_accessor, half *C);

  /**
   * @brief Batched GEMM with quantized weights
   *
   * For GQA attention where different heads have different K/V projections
   */
  bool GemmBatched(int M, int N, int K, const half *A,
                   std::shared_ptr<IWeightAccessor> weight_accessor, half *C,
                   int batch_count, long long stride_A, long long stride_C);

  /**
   * @brief Check if we should use cached dequantized weights
   *
   * For frequently accessed weights, dequantize once and cache
   */
  bool ShouldUseCache(std::shared_ptr<IWeightAccessor> accessor) const;

private:
  cudaStream_t stream_{nullptr};

  // Cache for frequently accessed dequantized weights
  struct DequantizedCache {
    std::shared_ptr<IWeightAccessor> accessor;
    half *d_weights{nullptr};
    size_t num_elements{0};
    bool is_valid{false};
  };

  static const int MAX_CACHE_SIZE = 32;
  DequantizedCache cache_[MAX_CACHE_SIZE];
  int cache_size_{0};

  // Find or create cache entry
  DequantizedCache *
  FindOrCreateCache(std::shared_ptr<IWeightAccessor> accessor);

  // Direct GEMM (dequantize then cuBLAS)
  bool GemmDirect(int M, int N, int K, const half *A, half *W, half *C);
};

/**
 * @brief Factory function
 */
std::unique_ptr<QuantizedGemm> CreateQuantizedGemm();

} // namespace inferflux
