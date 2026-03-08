#pragma once

#include "runtime/backends/cuda/native/kernels/dequantization.cuh"

#include <cuda_fp16.h>

namespace inferflux {
namespace runtime {
namespace cuda {
namespace native {

// Single source of truth for quantized element extraction.
// Used by both standalone dequantization kernels and fused GEMV/GEMM kernels
// to guarantee identical dequantization math across all code paths.

//==============================================================================
// Q4_K helpers
//==============================================================================

// Extract 6-bit scale and min from Q4_K/Q5_K packed scales array.
// Matches llama.cpp get_scale_min_k4 exactly.
__device__ __forceinline__ void get_scale_min_k4(int j,
                                                 const unsigned char *q,
                                                 unsigned char *d,
                                                 unsigned char *m) {
  if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

// Dequantize one element from a Q4_K super-block.
//   sb:  sub-block index (0-7), where each sub-block has 32 elements
//   e:   element within sub-block (0-31)
// Returns the dequantized FP32 weight value.
__device__ __forceinline__ float dequant_q4k_element(const block_q4_k &b,
                                                     float d, float dmin,
                                                     int sb, int e) {
  unsigned char sc, m;
  get_scale_min_k4(sb, b.scales, &sc, &m);

  int qs_byte_idx = (sb / 2) * 32 + e;
  unsigned char qbyte = b.qs[qs_byte_idx];
  int q = (sb & 1) ? (qbyte >> 4) : (qbyte & 0x0F);

  return d * static_cast<float>(sc) * static_cast<float>(q) -
         dmin * static_cast<float>(m);
}

//==============================================================================
// Q5_K helpers
//==============================================================================

// Dequantize one element from a Q5_K super-block.
//   sb:  sub-block index (0-7)
//   e:   element within sub-block (0-31)
// Returns the dequantized FP32 weight value.
//
// Q5_K qh layout: qh[e] holds the high bits for element `e` across all
// sub-blocks. Bit `sb` of qh[e] is the high bit for sub-block `sb`.
// (Matches llama.cpp dequantize_row_q5_K which uses shifting u1/u2 masks.)
__device__ __forceinline__ float dequant_q5k_element(const block_q5_k &b,
                                                     float d, float dmin,
                                                     int sb, int e) {
  unsigned char sc, m;
  get_scale_min_k4(sb, b.scales, &sc, &m);

  int group = sb / 2;              // 0-3
  int ql_base = group * 32;        // qs byte for this pair
  unsigned char ql_byte = b.qs[ql_base + e];
  int q_low = (sb & 1) ? (ql_byte >> 4) : (ql_byte & 0x0F);

  // High bit: qh[e] bit sb (same byte for all sub-blocks, different bit per sb)
  int q_high = ((b.qh[e] >> sb) & 1) ? 16 : 0;

  int q = q_low + q_high;
  return d * static_cast<float>(sc) * static_cast<float>(q) -
         dmin * static_cast<float>(m);
}

//==============================================================================
// Q6_K helpers
//==============================================================================

// Dequantize one element from a Q6_K super-block.
//   g:    group index (0-1), each group covers 128 elements
//   sub:  sub-pattern within group (0-3), each covers 32 elements
//   e:    element within sub-pattern (0-31)
// Returns the dequantized FP32 weight value.
//
// Q6_K layout (256 elements = 2 groups of 128):
//   Sub 0: low nibble of ql[base..+31],     qh bits 0-1
//   Sub 1: low nibble of ql[base+32..+63],  qh bits 2-3
//   Sub 2: high nibble of ql[base..+31],    qh bits 4-5
//   Sub 3: high nibble of ql[base+32..+63], qh bits 6-7
// Matches llama.cpp dequantize_row_q6_K exactly.
__device__ __forceinline__ float dequant_q6k_element(const block_q6_k &b,
                                                     float d, int g, int sub,
                                                     int e) {
  int ql_idx = g * 64 + ((sub & 1) ? 32 : 0) + e;
  unsigned char ql_byte = b.ql[ql_idx];
  int ql_val = (sub >= 2) ? (ql_byte >> 4) : (ql_byte & 0x0F);

  int qh_idx = g * 32 + e;
  int qh_val = (b.qh[qh_idx] >> (sub * 2)) & 0x03;

  int q = (ql_val | (qh_val << 4)) - 32;

  int scale_idx = g * 8 + sub * 2 + e / 16;
  float scale = static_cast<float>(b.scales[scale_idx]);

  return d * scale * static_cast<float>(q);
}

//==============================================================================
// Q8_0 helpers
//==============================================================================

// Dequantize one element from a Q8_0 block.
//   e: element index (0-31)
__device__ __forceinline__ float dequant_q8_0_element(const block_q8_0 &b,
                                                      float d, int e) {
  return d * static_cast<float>(b.qs[e]);
}

//==============================================================================
// Q8_K helpers
//==============================================================================

// Dequantize one element from a Q8_K block.
//   e: element index (0-255)
__device__ __forceinline__ float dequant_q8k_element(const block_q8_k &b,
                                                     float d, int e) {
  return d * static_cast<float>(b.qs[e]);
}

} // namespace native
} // namespace cuda
} // namespace runtime
} // namespace inferflux
