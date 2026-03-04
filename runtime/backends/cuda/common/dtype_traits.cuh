#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

namespace inferflux {

template <typename T>
struct DtypeTraits;

template <>
struct DtypeTraits<half> {
  static constexpr cudaDataType_t cublas_type = CUDA_R_16F;
  static constexpr const char* name = "fp16";
  __device__ static float to_float(half x) { return __half2float(x); }
  __device__ static half from_float(float x) { return __float2half(x); }
};

template <>
struct DtypeTraits<__nv_bfloat16> {
  static constexpr cudaDataType_t cublas_type = CUDA_R_16BF;
  static constexpr const char* name = "bf16";
  __device__ static float to_float(__nv_bfloat16 x) {
    return __bfloat162float(x);
  }
  __device__ static __nv_bfloat16 from_float(float x) {
    return __float2bfloat16(x);
  }
};

}  // namespace inferflux
