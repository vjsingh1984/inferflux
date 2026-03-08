#pragma once

#include "runtime/backends/cuda/native/weight_map.h"

#ifdef INFERFLUX_HAS_CUDA
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif

namespace inferflux {

/**
 * Adaptive dispatch for fused dequant-GEMV/GEMM kernels.
 *
 * For small M (decode and short prefill), launches fused kernels that read
 * raw quantized blocks and dequantize in registers — avoiding full FP16
 * weight materialization.
 *
 * For large M, returns false so the caller falls back to cuBLAS with tensor
 * cores. The crossover threshold is adaptive: computed from GPU compute
 * capability (tensor core generation) and quantization bits per weight.
 */
class FusedQuantGemm {
public:
  /**
   * Attempt a fused dequant-GEMV/GEMM.
   *
   * @param weight  Raw quantized weight info (GPU pointer + quant type)
   * @param activation  Input activation matrix [M, K] (half, device)
   * @param output  Output matrix [M, N] (half, device)
   * @param M  Number of rows in activation (1 = decode, M>1 = prefill)
   * @param N  Number of output columns (weight rows)
   * @param K  Inner dimension (weight cols / activation cols)
   * @param stream  CUDA stream
   * @return true if fused kernel was launched, false to fall back to cuBLAS
   */
  static bool Gemv(const QuantizedWeightInfo &weight, const half *activation,
                   half *output, int M, int N, int K, cudaStream_t stream);

  /**
   * Fused RmsNorm+GEMV: computes normalization inside the GEMV kernel's shared
   * memory loading phase, eliminating standalone RmsNorm kernel launches and
   * the intermediate d_norm_out_ buffer round-trip.
   *
   * @param weight  Raw quantized weight info (GPU pointer + quant type)
   * @param residual  Raw residual activation [M, K] (half, device, not
   * normalized)
   * @param norm_weight  RmsNorm weight vector [K] (half, device)
   * @param output  Output matrix [M, N] (half, device)
   * @param M  Number of rows (1 = decode, M>1 = batched decode)
   * @param N  Number of output columns
   * @param K  Inner dimension
   * @param rms_norm_eps  RmsNorm epsilon
   * @param stream  CUDA stream
   * @return true if fused kernel was launched, false to fall back to separate
   *         RmsNorm + GEMV/cuBLAS
   */
  static bool RmsNormGemv(const QuantizedWeightInfo &weight,
                          const half *residual, const half *norm_weight,
                          half *output, int M, int N, int K, float rms_norm_eps,
                          cudaStream_t stream);

  /**
   * Get the adaptive M threshold for a given quant type.
   * Above this M, cuBLAS with tensor cores is expected to be faster.
   * Queries GPU properties once on first call (thread-safe).
   */
  static int GetAdaptiveThreshold(int quant_type);
};

} // namespace inferflux
