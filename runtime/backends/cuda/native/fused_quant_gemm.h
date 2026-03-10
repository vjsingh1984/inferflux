#pragma once

#include <array>

#include "runtime/backends/cuda/native/native_execution_policy.h"
#include "runtime/backends/cuda/native/weight_map.h"

#ifdef INFERFLUX_HAS_CUDA
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif

namespace inferflux {

struct PackedActivationInfo {
  const int8_t *data{nullptr};
  const float *row_scales{nullptr};
};

struct PackedProjectionSpec {
  QuantizedWeightInfo weight;
  half *output{nullptr};
  int output_cols{0};
};

struct FusedDispatchGeometry {
  int M{0};
  int N{0};
  int K{0};
  int grouped_outputs{1};
  bool packed_activation{false};
  bool includes_rmsnorm{false};
};

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
  static constexpr int kDownProjMmqTileCols = 8;

  // V2 cooperative-warp GEMV constants (visible to callers for graph capture).
  static constexpr int kGemvWarpsPerBlockV2 = 4;
  static constexpr int kGemvThreadsPerBlockV2 = kGemvWarpsPerBlockV2 * 32;

  enum class FfnProjOperator {
    kFallback = 0,
    kQ81Group,
    kQ81GroupHotQ4K,
    kPackedGroup,
  };

  enum class DownProjOperator {
    kFallback = 0,
    kQ81Gemv,
    kQ81GemvHotFixed,
    kQ81GemvRowPair,
    kQ81GemvRowQuad,
    kPackedGemv,
    kMmq,
  };

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
                   half *output, int M, int N, int K, cudaStream_t stream,
                   const NativeExecutionPolicy *policy = nullptr);

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
                          cudaStream_t stream,
                          const NativeExecutionPolicy *policy = nullptr);

  /**
   * Attempt a fused dequant-GEMV using pre-quantized int8 activations packed
   * once per row and shared across sibling projections.
   *
   * This is the memory-first/native-throughput path for dp4a-capable quant
   * types where repeated activation preparation dominates decode cost.
   */
  static bool GemvPacked(const QuantizedWeightInfo &weight,
                         const PackedActivationInfo &activation, half *output,
                         int M, int N, int K, cudaStream_t stream,
                         const NativeExecutionPolicy *policy = nullptr);

  /**
   * Attempt a single packed-activation launch for two sibling projections.
   *
   * All projections must share the same quant type. Returns false when the
   * grouped path is unsupported so the caller can fall back to per-projection
   * launches without losing the shared activation pack.
   */
  static bool GemvPackedPair(
      const std::array<PackedProjectionSpec, 2> &projections,
      const PackedActivationInfo &activation, int M, int K,
      cudaStream_t stream);

  /**
   * Attempt a single packed-activation launch for three sibling projections.
   *
   * All projections must share the same quant type. Returns false when the
   * grouped path is unsupported so the caller can fall back to per-projection
   * launches without losing the shared activation pack.
   */
  static bool GemvPackedTriple(
      const std::array<PackedProjectionSpec, 3> &projections,
      const PackedActivationInfo &activation, int M, int K,
      cudaStream_t stream);

  /**
   * Quantize FP16 activation rows to Q8_1 block format in global memory.
   * Each 32-element block gets its own scale and pre-computed sum.
   *
   * @param input  FP16 activations [M, K] (half, device)
   * @param output_q8_1  Output Q8_1 blocks [M, K/32] (block_q8_1, device)
   * @param M  Number of rows
   * @param K  Number of columns (must be multiple of 32)
   * @param stream  CUDA stream
   */
  static void QuantizeRowQ8_1(const half *input, void *output_q8_1, int M,
                              int K, cudaStream_t stream);

  /**
   * Fuse SwiGLU activation and Q8_1 quantization for down-projection input.
   *
   * Avoids materializing the post-SwiGLU activation in FP16 before the
   * immediate quantized GEMV that consumes it.
   */
  static void SiluMulQuantizeQ8_1(const half *gate, const half *up,
                                  void *output_q8_1, int M, int K,
                                  cudaStream_t stream);

  /**
   * Fused RmsNorm + Q8_1 quantization: normalizes and quantizes in one pass.
   * Produces Q8_1 blocks for reuse across sibling projections.
   *
   * @param residual  Raw residual [M, K] (half, device)
   * @param norm_weight  RmsNorm weight vector [K] (half, device)
   * @param output_q8_1  Output Q8_1 blocks [M, K/32] (block_q8_1, device)
   * @param M  Number of rows
   * @param K  Number of columns (must be multiple of 32)
   * @param rms_norm_eps  RmsNorm epsilon
   * @param stream  CUDA stream
   */
  static void FusedRmsNormQuantizeQ8_1(const half *residual,
                                       const half *norm_weight,
                                       void *output_q8_1, int M, int K,
                                       float rms_norm_eps, cudaStream_t stream);

  /**
   * GEMV using pre-quantized Q8_1 activations.
   * Per-32-element block scales provide better precision than per-row.
   * No shared memory needed for activations (reads from L2 cache).
   *
   * @param weight  Raw quantized weight info (GPU pointer + quant type)
   * @param act_q8_1  Pre-quantized Q8_1 activations (block_q8_1, device)
   * @param output  Output matrix [M, N] (half, device)
   * @param M  Number of rows
   * @param N  Number of output columns
   * @param K  Inner dimension (must be multiple of 32)
   * @param stream  CUDA stream
   * @return true if Q8_1 kernel was launched, false if unsupported
  */
  static bool GemvQ8_1(const QuantizedWeightInfo &weight,
                       const void *act_q8_1, half *output, int M, int N, int K,
                       cudaStream_t stream,
                       const NativeExecutionPolicy *policy = nullptr);

  /**
   * Explicit fused FFN prototype for Q4_K weights.
   *
   * This is currently a bring-up/testing entry point only. It uses an
   * output-tiled design that reuses activated intermediate values across a
   * hidden-output tile, but it is not yet selected by the runtime hot path.
   * Exposing it here allows the kernel to be compiled and parity-tested
   * through the main native CUDA test suite before any rollout decision.
   */
  static bool FusedFfnQ4K(const QuantizedWeightInfo &gate_weight,
                          const QuantizedWeightInfo &up_weight,
                          const QuantizedWeightInfo &down_weight,
                          const half *activation, half *output, int M,
                          int N_inter, int N_hidden, int K,
                          cudaStream_t stream);

  /**
   * MMQ-style tiled down-projection path for transformed GGUF weights.
   *
   * This is an additive migration path used only for native GGUF decode on
   * projections that have an MMQ-ready transformed layout.
   */
  static bool DownProjMmq(const MmqWeightInfo &weight, const void *act_q8_1,
                          half *output, int M, int N, int K,
                          cudaStream_t stream,
                          const NativeExecutionPolicy *policy = nullptr);

  static bool IsDownProjMmqEnabled(const NativeExecutionPolicy *policy = nullptr);

  /**
   * Build a tile-major MMQ layout for a quantized tensor.
   *
   * The current incremental slice supports native GGUF down-proj weights for
   * the quant types with explicit MMQ dispatch entries.
   */
  static bool BuildDownProjMmqLayout(const QuantizedWeightInfo &weight,
                                     int rows, int cols,
                                     MmqWeightInfo *layout,
                                     cudaStream_t stream);

  static void DestroyDownProjMmqLayout(const MmqWeightInfo &layout);

  static bool SupportsDownProjMmq(int quant_type);

  /**
   * Hybrid FFN gate+up grouped operator selector.
   *
   * Starts with the live native decode hot path: small-batch Q8_1 grouped
   * kernels for Q4_K, then falls back to the generic Q8_1 shared-quantization
   * path, then packed activations, then the compatibility path.
   */
  static FfnProjOperator
  SelectFfnProjOperator(int quant_type0, int quant_type1,
                        const FusedDispatchGeometry &geometry, bool allow_q81,
                        bool allow_packed,
                        const NativeExecutionPolicy *policy = nullptr);

  static const char *FfnProjOperatorName(FfnProjOperator op);
  static const char *FfnProjOperatorMetricName(FfnProjOperator op,
                                               int quant_type, int k);

  /**
   * Hybrid down-proj operator selector.
   *
   * Mirrors the vendored llama.cpp split conceptually: keep the MMVQ-like
   * Q8_1 path for smaller decode envelopes and promote MMQ only when the
   * caller allows it and geometry crosses the tiled-kernel threshold.
   */
  static DownProjOperator
  SelectDownProjOperator(int quant_type, const FusedDispatchGeometry &geometry,
                         bool allow_q81, bool allow_packed, bool allow_mmq,
                         const NativeExecutionPolicy *policy = nullptr);

  static const char *DownProjOperatorName(DownProjOperator op);
  static const char *DownProjOperatorMetricName(DownProjOperator op,
                                                int quant_type, int m, int k);

  /**
   * Grouped Q8_1 GEMV for two sibling projections (single kernel launch).
   */
  static bool GemvQ8_1Pair(
      const std::array<PackedProjectionSpec, 2> &projections,
      const void *act_q8_1, int M, int K, cudaStream_t stream,
      const NativeExecutionPolicy *policy = nullptr);

  /**
   * Grouped Q8_1 GEMV for three sibling projections (single kernel launch).
   */
  static bool GemvQ8_1Triple(
      const std::array<PackedProjectionSpec, 3> &projections,
      const void *act_q8_1, int M, int K, cudaStream_t stream,
      const NativeExecutionPolicy *policy = nullptr);

  /**
   * Returns true when a quant type supports Q8_1 activation GEMV.
   */
  static bool SupportsQ8_1Activations(int quant_type);

  /**
   * Small-batch grouped Q8_1 fast-path selector.
   *
   * This is an additive hot-shape optimization for native decode kernels.
   * The current staged rollout only promotes the grouped Q4_K FFN path for
   * the measured K=2048, M<=2 envelope and falls back to the generic grouped
   * kernels for every other geometry.
   */
  static bool
  ShouldUseSpecializedQ8_1GroupedFastPath(int quant_type,
                                          const FusedDispatchGeometry &geometry,
                                          const NativeExecutionPolicy *policy =
                                              nullptr);

  /**
   * Get the adaptive M threshold for a given quant type.
   * Above this M, cuBLAS with tensor cores is expected to be faster.
   * Queries GPU properties once on first call (thread-safe).
   */
  static int GetAdaptiveThreshold(int quant_type);

  /**
   * Geometry-aware fused dispatch ceiling.
   *
   * The base threshold is quant/GPU dependent. Packed activations, grouped
   * sibling projections, and very large output surfaces shift the crossover.
   */
  static int GetGeometryAwareThreshold(int quant_type,
                                       const FusedDispatchGeometry &geometry);

  /**
   * Deterministic dispatch policy helper used by runtime paths and tests.
   *
   * Returns true when a quant type is fused-kernel supported and batch size M
   * is within the adaptive threshold for fused execution.
   */
  static bool ShouldUseFusedPath(int quant_type, int M);

  static bool ShouldUseFusedPath(int quant_type,
                                 const FusedDispatchGeometry &geometry);

  /**
   * Returns true when a quant type can consume pre-quantized int8 activation
   * rows directly from global memory.
   */
  static bool SupportsPackedActivations(int quant_type);
};

} // namespace inferflux
