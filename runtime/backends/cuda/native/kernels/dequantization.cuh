#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

namespace inferflux {
namespace runtime {
namespace cuda {
namespace native {

// GGUF block structures (from ggml-common.h)
#define QK_K 256
#define K_SCALE_SIZE 12

// Q4_K block: 4.5 bits per weight
typedef struct {
    unsigned short d;      // super-block scale for quantized scales (as uint16)
    unsigned short dmin;   // super-block scale for quantized mins (as uint16)
    unsigned char scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    unsigned char qs[QK_K/2];           // 4-bit quants
} block_q4_k;

// Q5_K block: 5.5 bits per weight
typedef struct {
    unsigned short d;      // super-block scale for quantized scales (as uint16)
    unsigned short dmin;   // super-block scale for quantized mins (as uint16)
    unsigned char scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    unsigned char qh[QK_K/8];           // quants, high bit
    unsigned char qs[QK_K/2];           // quants, low 4 bits
} block_q5_k;

// Q6_K block: 6.5625 bits per weight
typedef struct {
    unsigned short d;       // super-block scale (as uint16)
    unsigned char ql[QK_K/2];      // quants, lower 4 bits
    unsigned char qh[QK_K/4];      // quants, upper 2 bits
    char  scales[QK_K/16];        // scales, quantized with 8 bits
} block_q6_k;

//==============================================================================
// Q4_K Dequantization
//==============================================================================

/**
 * @brief CUDA kernel for Q4_K dequantization
 *
 * Dequantizes Q4_K_M format (4.5 bits per weight) to FP16.
 * Reference: llama.cpp ggml_cuda_dequantize_q4_k_m
 *
 * @param quantized Input quantized weights
 * @param dequantized Output FP16 weights
 * @param num_elements Number of elements to dequantize
 */
__global__ void dequantize_q4_k_kernel(
    const block_q4_k *quantized,
    half *dequantized,
    size_t num_elements);

/**
 * @brief Host wrapper for Q4_K dequantization
 */
cudaError_t dequantize_q4_k(
    const void *quantized,
    half *dequantized,
    size_t num_elements,
    cudaStream_t stream = 0);

//==============================================================================
// Q5_K Dequantization
//==============================================================================

/**
 * @brief CUDA kernel for Q5_K dequantization
 *
 * Dequantizes Q5_K_M format (5.5 bits per weight) to FP16.
 * Reference: llama.cpp ggml_cuda_dequantize_q5_k_m
 */
__global__ void dequantize_q5_k_kernel(
    const block_q5_k *quantized,
    half *dequantized,
    size_t num_elements);

/**
 * @brief Host wrapper for Q5_K dequantization
 */
cudaError_t dequantize_q5_k(
    const void *quantized,
    half *dequantized,
    size_t num_elements,
    cudaStream_t stream = 0);

//==============================================================================
// Q6_K Dequantization
//==============================================================================

/**
 * @brief CUDA kernel for Q6_K dequantization
 *
 * Dequantizes Q6_K format (6.5625 bits per weight) to FP16.
 * Reference: llama.cpp ggml_cuda_dequantize_q6_k
 */
__global__ void dequantize_q6_k_kernel(
    const block_q6_k *quantized,
    half *dequantized,
    size_t num_elements);

/**
 * @brief Host wrapper for Q6_K dequantization
 */
cudaError_t dequantize_q6_k(
    const void *quantized,
    half *dequantized,
    size_t num_elements,
    cudaStream_t stream = 0);

//==============================================================================
// Q8_0 Dequantization
//==============================================================================

// Q8_0 block constants
#define QK8_0 32

// Q8_0 block: 8 bits per weight
typedef struct {
    unsigned short d;       // scale (delta), 2 bytes
    signed char qs[QK8_0];  // 8-bit quants, 32 bytes
} block_q8_0;

/**
 * @brief CUDA kernel for Q8_0 dequantization
 *
 * Dequantizes Q8_0 format (8.5 bits per weight) to FP16.
 * Reference: llama.cpp ggml_cuda_dequantize_q8_0
 *
 * Formula: dequantized[i] = qs[i] * d
 *
 * @param quantized Input quantized weights
 * @param dequantized Output FP16 weights
 * @param num_elements Number of elements to dequantize
 */
__global__ void dequantize_q8_0_kernel(
    const block_q8_0 *quantized,
    half *dequantized,
    size_t num_elements);

/**
 * @brief Host wrapper for Q8_0 dequantization
 */
cudaError_t dequantize_q8_0(
    const void *quantized,
    half *dequantized,
    size_t num_elements,
    cudaStream_t stream = 0);

//==============================================================================
// Utility Functions
//==============================================================================

/**
 * @brief Calculate grid dimensions for dequantization
 */
inline dim3 calc_dequant_grid(size_t num_elements) {
  int num_blocks = (num_elements + QK_K - 1) / QK_K;
  return dim3((num_blocks + 255) / 256);
}

/**
 * @brief Calculate block dimensions for dequantization
 */
inline dim3 calc_dequant_block() {
  return dim3(256);
}

} // namespace native
} // namespace cuda
} // namespace runtime
} // namespace inferflux
