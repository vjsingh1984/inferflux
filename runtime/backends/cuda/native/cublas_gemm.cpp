#include "runtime/backends/cuda/native/cublas_gemm.h"
#include "runtime/backends/cuda/common/dtype_traits.cuh"
#include "server/logging/logger.h"

namespace inferflux {

CublasGemm::~CublasGemm() {
  for (auto &[stream, h] : stream_handles_) {
    if (h)
      cublasDestroy(h);
  }
  stream_handles_.clear();
  if (handle_) {
    cublasDestroy(handle_);
  }
}

bool CublasGemm::Initialize(cudaStream_t stream) {
  cublasStatus_t st = cublasCreate(&handle_);
  if (st != CUBLAS_STATUS_SUCCESS) {
    log::Error("cublas_gemm", "cublasCreate failed: " + std::to_string(st));
    return false;
  }

  st = cublasSetStream(handle_, stream);
  if (st != CUBLAS_STATUS_SUCCESS) {
    log::Error("cublas_gemm", "cublasSetStream failed: " + std::to_string(st));
    return false;
  }

  st = cublasSetMathMode(handle_, CUBLAS_DEFAULT_MATH);
  if (st != CUBLAS_STATUS_SUCCESS) {
    log::Warn("cublas_gemm", "cublasSetMathMode failed, using default");
  }

  log::Info("cublas_gemm", "cuBLAS initialized");
  return true;
}

void CublasGemm::SetStream(cudaStream_t stream) {
  // If we have a dedicated handle for this stream, no-op (already bound)
  if (stream_handles_.count(stream))
    return;
  if (handle_) {
    cublasSetStream(handle_, stream);
  }
}

void CublasGemm::InitializeMultiStream(
    std::initializer_list<cudaStream_t> streams) {
  for (auto stream : streams) {
    if (stream_handles_.count(stream))
      continue;
    cublasHandle_t h;
    cublasStatus_t st = cublasCreate(&h);
    if (st != CUBLAS_STATUS_SUCCESS) {
      log::Error("cublas_gemm",
                 "Failed to create multi-stream handle: " + std::to_string(st));
      continue;
    }
    cublasSetStream(h, stream);
    cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);
    stream_handles_[stream] = h;
  }
  log::Info("cublas_gemm", "Multi-stream initialized: " +
                               std::to_string(stream_handles_.size()) +
                               " dedicated handles");
}

cublasHandle_t CublasGemm::GetHandleForStream(cudaStream_t stream) const {
  auto it = stream_handles_.find(stream);
  if (it != stream_handles_.end())
    return it->second;
  return handle_;
}

bool CublasGemm::Gemm(int M, int N, int K, const half *A, const half *B,
                      half *C) {
  return GemmTyped<half>(M, N, K, A, B, C);
}

template <typename T>
bool CublasGemm::GemmTyped(int M, int N, int K, const T *A, const T *B, T *C) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  constexpr cudaDataType_t dtype = DtypeTraits<T>::cublas_type;

  // Use stream-specific handle if available (avoids SetStream serialization)
  cudaStream_t current_stream = nullptr;
  cublasGetStream(handle_, &current_stream);
  cublasHandle_t h = GetHandleForStream(current_stream);

  cublasStatus_t st = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha,
                                   B, dtype, K, A, dtype, K, &beta, C, dtype, N,
                                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

  if (st != CUBLAS_STATUS_SUCCESS) {
    log::Error("cublas_gemm", "GemmTyped failed: " + std::to_string(st) +
                                  " (M=" + std::to_string(M) +
                                  " N=" + std::to_string(N) +
                                  " K=" + std::to_string(K) + ")");
    return false;
  }
  return true;
}

template <typename T>
bool CublasGemm::GemmTypedFP32Out(int M, int N, int K, const T *A, const T *B,
                                  float *C) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  constexpr cudaDataType_t dtype = DtypeTraits<T>::cublas_type;

  cudaStream_t current_stream = nullptr;
  cublasGetStream(handle_, &current_stream);
  cublasHandle_t h = GetHandleForStream(current_stream);

  cublasStatus_t st = cublasGemmEx(
      h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, B, dtype, K, A, dtype, K,
      &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

  if (st != CUBLAS_STATUS_SUCCESS) {
    log::Error("cublas_gemm", "GemmTypedFP32Out failed: " + std::to_string(st) +
                                  " (M=" + std::to_string(M) +
                                  " N=" + std::to_string(N) +
                                  " K=" + std::to_string(K) + ")");
    return false;
  }
  return true;
}

bool CublasGemm::GemmBatched(int M, int N, int K, const half *A, const half *B,
                             half *C, long long stride_A, long long stride_B,
                             long long stride_C, int batch_count) {
  return GemmBatchedTyped<half>(M, N, K, A, B, C, stride_A, stride_B, stride_C,
                                batch_count);
}

template <typename T>
bool CublasGemm::GemmBatchedTyped(int M, int N, int K, const T *A, const T *B,
                                  T *C, long long stride_A, long long stride_B,
                                  long long stride_C, int batch_count) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  constexpr cudaDataType_t dtype = DtypeTraits<T>::cublas_type;

  cudaStream_t current_stream = nullptr;
  cublasGetStream(handle_, &current_stream);
  cublasHandle_t h = GetHandleForStream(current_stream);

  cublasStatus_t st = cublasGemmStridedBatchedEx(
      h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, B, dtype, K, stride_B, A,
      dtype, K, stride_A, &beta, C, dtype, N, stride_C, batch_count,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

  if (st != CUBLAS_STATUS_SUCCESS) {
    log::Error("cublas_gemm", "GemmBatchedTyped failed: " + std::to_string(st) +
                                  " (batch=" + std::to_string(batch_count) +
                                  ")");
    return false;
  }
  return true;
}

// Explicit template instantiations
template bool CublasGemm::GemmTyped<half>(int, int, int, const half *,
                                          const half *, half *);
template bool CublasGemm::GemmTyped<__nv_bfloat16>(int, int, int,
                                                   const __nv_bfloat16 *,
                                                   const __nv_bfloat16 *,
                                                   __nv_bfloat16 *);

template bool CublasGemm::GemmTypedFP32Out<half>(int, int, int, const half *,
                                                 const half *, float *);
template bool CublasGemm::GemmTypedFP32Out<__nv_bfloat16>(int, int, int,
                                                          const __nv_bfloat16 *,
                                                          const __nv_bfloat16 *,
                                                          float *);

template bool CublasGemm::GemmBatchedTyped<half>(int, int, int, const half *,
                                                 const half *, half *,
                                                 long long, long long,
                                                 long long, int);
template bool CublasGemm::GemmBatchedTyped<__nv_bfloat16>(
    int, int, int, const __nv_bfloat16 *, const __nv_bfloat16 *,
    __nv_bfloat16 *, long long, long long, long long, int);

} // namespace inferflux
