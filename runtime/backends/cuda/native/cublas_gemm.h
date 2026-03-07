#pragma once

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <initializer_list>
#include <unordered_map>

namespace inferflux {

/**
 * CublasGemm: cuBLAS handle manager + typed GEMM with FP32 accumulation.
 *
 * Wraps cublasGemmEx for y = x * W^T (safetensors row-major convention).
 */
class CublasGemm {
public:
  CublasGemm() = default;
  ~CublasGemm();

  CublasGemm(const CublasGemm &) = delete;
  CublasGemm &operator=(const CublasGemm &) = delete;

  /**
   * Create cuBLAS handle and bind to stream.
   */
  bool Initialize(cudaStream_t stream);

  /**
   * Switch cuBLAS handle to a different stream.
   */
  void SetStream(cudaStream_t stream);

  /**
   * FP16 GEMM with FP32 accumulation: C = A * B^T
   *
   * A: [M, K] row-major (half)
   * B: [N, K] row-major (half) — transposed during multiply
   * C: [M, N] row-major (half)
   */
  bool Gemm(int M, int N, int K, const half *A, const half *B, half *C);

  /**
   * Typed GEMM with FP32 accumulation: C = A * B^T
   * Supports half (FP16) and __nv_bfloat16 (BF16) via DtypeTraits.
   */
  template <typename T>
  bool GemmTyped(int M, int N, int K, const T *A, const T *B, T *C);

  /**
   * Typed GEMM with FP32 output: C(float) = A(T) * B(T)^T
   * Uses cublasGemmEx with CUDA_R_32F for C matrix.
   * Eliminates the need for a separate HalfToFloat conversion kernel.
   */
  template <typename T>
  bool GemmTypedFP32Out(int M, int N, int K, const T *A, const T *B, float *C);

  /**
   * Strided batched GEMM for GQA attention.
   * Each batch: C_i = A_i * B_i^T
   */
  bool GemmBatched(int M, int N, int K, const half *A, const half *B, half *C,
                   long long stride_A, long long stride_B, long long stride_C,
                   int batch_count);

  /**
   * Typed strided batched GEMM.
   */
  template <typename T>
  bool GemmBatchedTyped(int M, int N, int K, const T *A, const T *B, T *C,
                        long long stride_A, long long stride_B,
                        long long stride_C, int batch_count);

  /**
   * Create additional cuBLAS handles permanently bound to specific streams.
   * Avoids serialization from cublasSetStream() when switching between
   * decode/prefill streams during phase overlap.
   */
  void InitializeMultiStream(std::initializer_list<cudaStream_t> streams);

private:
  cublasHandle_t handle_{nullptr};
  std::unordered_map<cudaStream_t, cublasHandle_t> stream_handles_;

  cublasHandle_t GetHandleForStream(cudaStream_t stream) const;
};

} // namespace inferflux
