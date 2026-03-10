#include "runtime/backends/cuda/native/fused_quant_gemm.h"
#include "runtime/backends/cuda/native/gguf_util.h"
// Deprecated reference-only fused GEMM kernels. The current dispatcher never
// launches these directly: large-M work falls back to cuBLAS before dispatch,
// and the active fused path uses 2D GEMV-style kernels. Keep the include so
// the table below documents the legacy mapping and the kernels remain available
// for bring-up or re-evaluation.
#include "runtime/backends/cuda/native/kernels/fused_dequant_gemm.cuh"
#include "runtime/backends/cuda/native/kernels/fused_dequant_gemv.cuh"
#include "runtime/backends/cuda/native/kernels/fused_dequant_gemv_v2.cuh"
#include "server/logging/logger.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <string>

namespace inferflux {

using namespace runtime::cuda::native;

namespace {

thread_local const NativeExecutionPolicy *g_execution_policy_override = nullptr;

const NativeExecutionPolicy &
ResolveExecutionPolicy(const NativeExecutionPolicy *policy) {
  if (policy) {
    return *policy;
  }
  if (g_execution_policy_override) {
    return *g_execution_policy_override;
  }
  static thread_local NativeExecutionPolicy env_policy;
  env_policy = NativeExecutionPolicy::FromEnv();
  return env_policy;
}

class ScopedExecutionPolicyOverride {
public:
  explicit ScopedExecutionPolicyOverride(const NativeExecutionPolicy *policy)
      : prev_(g_execution_policy_override) {
    if (policy) {
      g_execution_policy_override = policy;
    }
  }

  ~ScopedExecutionPolicyOverride() { g_execution_policy_override = prev_; }

private:
  const NativeExecutionPolicy *prev_{nullptr};
};

// ============================================================================
// GPU-adaptive threshold computation
// ============================================================================

struct GpuProfile {
  int sm_major{0};
  int sm_minor{0};
  float memory_bandwidth_gb_s{0}; // GB/s
  int sm_count{0};
  bool initialized{false};
  bool has_dp4a{false}; // SM 6.1+ supports __dp4a int8x4 dot product

  // Base M threshold before scaling by bits-per-weight.
  // This reflects how much faster cuBLAS tensor cores are than our scalar
  // fused kernels — lower means cuBLAS wins sooner.
  int base_threshold() const {
    if (sm_major < 7) {
      // Pascal and older: no tensor cores. cuBLAS uses scalar FP16 GEMM,
      // no significant compute advantage over fused kernels.
      // Fused wins on memory bandwidth savings up to high M.
      return 16;
    }
    if (sm_major == 7 && sm_minor < 5) {
      // Volta (V100): first-gen FP16 tensor cores.
      // cuBLAS gets ~8x compute throughput boost over scalar.
      return 8;
    }
    if (sm_major == 7) {
      // Turing (RTX 20xx): FP16 tensor cores, lower bandwidth than V100.
      return 6;
    }
    if (sm_major == 8 && sm_minor == 0) {
      // Ampere A100: enhanced tensor cores, 2 TB/s HBM2e.
      // Very fast cuBLAS — lower threshold.
      return 4;
    }
    if (sm_major == 8) {
      // Ampere consumer (RTX 30xx) / Ada Lovelace (RTX 40xx, SM 8.9).
      // Good tensor cores but lower memory bandwidth than A100.
      return 5;
    }
    if (sm_major >= 9) {
      // Hopper (H100) and beyond: FP8 tensor cores, 3+ TB/s HBM3.
      // cuBLAS extremely fast — fused advantage shrinks quickly.
      return 3;
    }
    return 4; // Conservative default for unknown future GPUs
  }
};

GpuProfile &GetGpuProfile() {
  static GpuProfile profile;
  static std::once_flag flag;
  std::call_once(flag, [&] {
    cudaDeviceProp prop{};
    int device = 0;
    cudaGetDevice(&device);
    if (cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
      profile.sm_major = prop.major;
      profile.sm_minor = prop.minor;
      profile.memory_bandwidth_gb_s =
          static_cast<float>(prop.memoryClockRate) * 2.0f *
          static_cast<float>(prop.memoryBusWidth) / 8.0f / 1e6f;
      profile.sm_count = prop.multiProcessorCount;
      profile.has_dp4a =
          (prop.major > 6) || (prop.major == 6 && prop.minor >= 1);
      profile.initialized = true;

      log::Info(
          "fused_quant_gemm",
          "GPU profile: SM " + std::to_string(prop.major) + "." +
              std::to_string(prop.minor) + ", " +
              std::to_string(profile.sm_count) + " SMs, " +
              std::to_string(static_cast<int>(profile.memory_bandwidth_gb_s)) +
              " GB/s bandwidth, base_threshold=" +
              std::to_string(profile.base_threshold()) +
              ", dp4a=" + (profile.has_dp4a ? "yes" : "no"));
    }
  });
  return profile;
}

// Effective bits per weight for each quant type.
// Used to scale the M threshold: lower bits = fused reads less memory =
// fused advantage extends to higher M.
float BitsPerWeight(GGUF::TensorType qtype) {
  switch (qtype) {
  case GGUF::TensorType::Q4_K:
    return 4.5f; // 4-bit + scales/mins overhead
  case GGUF::TensorType::Q6_K:
    return 6.5625f; // 6-bit + scales overhead
  case GGUF::TensorType::Q8_0:
    return 8.5f; // 8-bit + FP16 scale per 32 elements
  case GGUF::TensorType::Q8_K:
    return 8.5f; // 8-bit + FP32 scale per 256 elements
  case GGUF::TensorType::F16:
    return 16.0f;
  case GGUF::TensorType::F32:
    return 32.0f;
  default:
    return 16.0f; // Conservative: assume FP16-like
  }
}

bool ExperimentalQ8_1TripleRowPairEnabled() {
  return ResolveExecutionPolicy(nullptr).enable_experimental_q81_triple_rowpair;
}

// Compute the adaptive M threshold for a given quant type.
//
// Formula: threshold = base_threshold * (16.0 / bpw)
//
// Intuition: cuBLAS reads dequantized FP16 weights (16 bits each).
// Fused reads raw quantized weights (bpw bits each).
// The memory bandwidth ratio is 16/bpw.
// For Q4_K (4.5 bpw): fused reads 3.6x less → threshold = base * 3.6
// For Q8_0 (8.5 bpw): fused reads 1.9x less → threshold = base * 1.9
//
// Clamped to [4, kFusedGemmMaxM] — lower bound ensures fused GEMV (M=1)
// always runs, upper bound is the kernel's shared memory capacity.
int ComputeThreshold(int base, float bpw) {
  float raw = static_cast<float>(base) * (16.0f / bpw);
  return std::max(4, std::min(static_cast<int>(kFusedGemmMaxM),
                              static_cast<int>(std::round(raw))));
}

// ============================================================================
// Table-driven kernel dispatch
// ============================================================================

using FusedDispatchFn = bool (*)(const void *data, const half *activation,
                                 half *output, int M, int N, int K,
                                 cudaStream_t stream);
using PackedDispatchFn = bool (*)(const void *data, const int8_t *activation,
                                  const float *row_scales, half *output, int M,
                                  int N, int K, cudaStream_t stream);
using PackedDispatchPairFn = bool (*)(
    const void *data0, const void *data1, const int8_t *activation,
    const float *row_scales, half *output0, int N0, half *output1, int N1,
    int M, int K, cudaStream_t stream);
using PackedDispatchTripleFn = bool (*)(
    const void *data0, const void *data1, const void *data2,
    const int8_t *activation, const float *row_scales, half *output0, int N0,
    half *output1, int N1, half *output2, int N2, int M, int K,
    cudaStream_t stream);

template <typename BlockType,
          void (*GemvKernel)(const BlockType *, const half *, half *, int, int),
          void (*GemmKernel)(const BlockType *, const half *, half *, int, int,
                             int)>
bool DispatchFused(const void *data, const half *activation, half *output,
                   int M, int N, int K, cudaStream_t stream) {
  // Deprecated note: GemmKernel is intentionally unused in the current
  // dispatcher. We keep the template parameter and dispatch-table wiring as a
  // breadcrumb that these reference fused GEMM kernels exist, but runtime
  // policy never reaches them today because large-M falls back to cuBLAS.
  (void)GemmKernel;
  auto *w = static_cast<const BlockType *>(data);
  // GEMV kernel with 2D grid: blockIdx.x covers output columns, blockIdx.y
  // covers input rows. Each block loads one row of X into shared memory and
  // computes output elements for that row. Single launch for all M rows.
  int grid_x = (N + kGemvWarpsPerBlock - 1) / kGemvWarpsPerBlock;
  dim3 grid(grid_x, M);
  size_t smem = static_cast<size_t>(K) * sizeof(float);
  GemvKernel<<<grid, kGemvThreadsPerBlock, smem, stream>>>(w, activation,
                                                           output, N, K);
  return true;
}

// dp4a variant: uses K bytes (int8 activations) + workspace after the int8 data.
// Q4_K uses per-super-block scales (K/QK_K floats); Q8_0/Q8_K use warp
// reduction workspace (kGemvWarpsPerBlock+1 floats). Allocate max of both.
template <typename BlockType,
          void (*Dp4aKernel)(const BlockType *, const half *, half *, int, int)>
bool DispatchFusedDp4a(const void *data, const half *activation, half *output,
                       int M, int N, int K, cudaStream_t stream) {
  auto *w = static_cast<const BlockType *>(data);
  int grid_x = (N + kGemvWarpsPerBlock - 1) / kGemvWarpsPerBlock;
  dim3 grid(grid_x, M);
  // smem: K bytes (int8 activations) + (kGemvWarpsPerBlock + 1) floats workspace
  size_t smem = static_cast<size_t>(K) +
                (kGemvWarpsPerBlock + 1) * sizeof(float);
  Dp4aKernel<<<grid, kGemvThreadsPerBlock, smem, stream>>>(w, activation,
                                                           output, N, K);
  return true;
}

// FP16 smem variant: uses K*sizeof(half) for activations (2x less than FP32).
// Used for Q6_K where int8 dp4a causes precision loss but FP16 is sufficient.
template <typename BlockType,
          void (*Fp16Kernel)(const BlockType *, const half *, half *, int, int)>
bool DispatchFusedFp16(const void *data, const half *activation, half *output,
                       int M, int N, int K, cudaStream_t stream) {
  auto *w = static_cast<const BlockType *>(data);
  int grid_x = (N + kGemvWarpsPerBlock - 1) / kGemvWarpsPerBlock;
  dim3 grid(grid_x, M);
  size_t smem = static_cast<size_t>(K) * sizeof(half);
  Fp16Kernel<<<grid, kGemvThreadsPerBlock, smem, stream>>>(w, activation,
                                                            output, N, K);
  return true;
}

template <typename BlockType,
          void (*PackedKernel)(const BlockType *, const int8_t *,
                               const float *, half *, int, int)>
bool DispatchPackedDp4a(const void *data, const int8_t *activation,
                        const float *row_scales, half *output, int M, int N,
                        int K, cudaStream_t stream) {
  auto *w = static_cast<const BlockType *>(data);
  int grid_x = (N + kGemvWarpsPerBlock - 1) / kGemvWarpsPerBlock;
  dim3 grid(grid_x, M);
  PackedKernel<<<grid, kGemvThreadsPerBlock, 0, stream>>>(
      w, activation, row_scales, output, N, K);
  return true;
}

template <typename BlockType, int Outputs,
          void (*PackedKernel)(
              PackedProjectionGroupParams<BlockType, Outputs>, const int8_t *,
              const float *, int)>
bool DispatchPackedDp4aGroup(
    const std::array<const void *, Outputs> &weights, const int8_t *activation,
    const float *row_scales, const std::array<half *, Outputs> &outputs,
    const std::array<int, Outputs> &output_cols, int M, int K,
    cudaStream_t stream) {
  PackedProjectionGroupParams<BlockType, Outputs> params{};
  int max_output_cols = 0;
  for (int i = 0; i < Outputs; ++i) {
    params.weights[i] = static_cast<const BlockType *>(weights[i]);
    params.outputs[i] = outputs[i];
    params.output_cols[i] = output_cols[i];
    max_output_cols = std::max(max_output_cols, output_cols[i]);
  }

  int grid_x = (max_output_cols + kGemvWarpsPerBlock - 1) / kGemvWarpsPerBlock;
  dim3 grid(grid_x, M);
  PackedKernel<<<grid, kGemvThreadsPerBlock, 0, stream>>>(
      params, activation, row_scales, K);
  return true;
}

template <typename BlockType,
          void (*PackedKernel)(PackedProjectionGroupParams<BlockType, 2>,
                               const int8_t *, const float *, int)>
bool DispatchPackedDp4aPair(const void *data0, const void *data1,
                            const int8_t *activation,
                            const float *row_scales, half *output0, int N0,
                            half *output1, int N1, int M, int K,
                            cudaStream_t stream) {
  return DispatchPackedDp4aGroup<BlockType, 2, PackedKernel>(
      {data0, data1}, activation, row_scales, {output0, output1}, {N0, N1}, M,
      K, stream);
}

template <typename BlockType,
          void (*PackedKernel)(PackedProjectionGroupParams<BlockType, 3>,
                               const int8_t *, const float *, int)>
bool DispatchPackedDp4aTriple(const void *data0, const void *data1,
                              const void *data2, const int8_t *activation,
                              const float *row_scales, half *output0, int N0,
                              half *output1, int N1, half *output2, int N2,
                              int M, int K, cudaStream_t stream) {
  return DispatchPackedDp4aGroup<BlockType, 3, PackedKernel>(
      {data0, data1, data2}, activation, row_scales,
      {output0, output1, output2}, {N0, N1, N2}, M, K, stream);
}

// ============================================================================
// Fused RmsNorm+GEMV dispatch
// ============================================================================

using RmsNormDispatchFn = bool (*)(const void *data, const half *residual,
                                   const half *norm_weight, half *output,
                                   int M, int N, int K, float eps,
                                   cudaStream_t stream);

template <typename BlockType,
          void (*RmsNormGemvKernel)(const BlockType *, const half *,
                                   const half *, half *, int, int, float)>
bool DispatchRmsNormGemv(const void *data, const half *residual,
                         const half *norm_weight, half *output, int M, int N,
                         int K, float eps, cudaStream_t stream) {
  auto *w = static_cast<const BlockType *>(data);
  int grid_x = (N + kGemvWarpsPerBlock - 1) / kGemvWarpsPerBlock;
  dim3 grid(grid_x, M);
  // smem: K floats (activations) + kGemvWarpsPerBlock floats (warp reduction)
  size_t smem =
      static_cast<size_t>(K) * sizeof(float) + kGemvWarpsPerBlock * sizeof(float);
  RmsNormGemvKernel<<<grid, kGemvThreadsPerBlock, smem, stream>>>(
      w, residual, norm_weight, output, N, K, eps);
  return true;
}

// dp4a + RmsNorm variant: needs K*sizeof(float) during norm phase, then
// reuses memory as int8 activations + per-block scales for dp4a phase.
template <typename BlockType,
          void (*RmsNormDp4aKernel)(const BlockType *, const half *,
                                   const half *, half *, int, int, float)>
bool DispatchRmsNormGemvDp4a(const void *data, const half *residual,
                             const half *norm_weight, half *output, int M,
                             int N, int K, float eps, cudaStream_t stream) {
  auto *w = static_cast<const BlockType *>(data);
  int grid_x = (N + kGemvWarpsPerBlock - 1) / kGemvWarpsPerBlock;
  dim3 grid(grid_x, M);
  // smem: K floats (norm phase, reused as int8 in dp4a phase) + workspace
  size_t smem = static_cast<size_t>(K) * sizeof(float) +
                (kGemvWarpsPerBlock + 1) * sizeof(float);
  RmsNormDp4aKernel<<<grid, kGemvThreadsPerBlock, smem, stream>>>(
      w, residual, norm_weight, output, N, K, eps);
  return true;
}

// FP16 smem + RmsNorm variant: norm phase uses K*sizeof(float) (FP32),
// then converts to FP16 for the dot product phase.
template <typename BlockType,
          void (*RmsNormFp16Kernel)(const BlockType *, const half *,
                                   const half *, half *, int, int, float)>
bool DispatchRmsNormGemvFp16(const void *data, const half *residual,
                              const half *norm_weight, half *output, int M,
                              int N, int K, float eps, cudaStream_t stream) {
  auto *w = static_cast<const BlockType *>(data);
  int grid_x = (N + kGemvWarpsPerBlock - 1) / kGemvWarpsPerBlock;
  dim3 grid(grid_x, M);
  // smem: K floats (norm phase) + kGemvWarpsPerBlock floats (warp reduction)
  // After norm, FP32 is reused as FP16 (K*2 < K*4, fits in same allocation)
  size_t smem = static_cast<size_t>(K) * sizeof(float) +
                kGemvWarpsPerBlock * sizeof(float);
  RmsNormFp16Kernel<<<grid, kGemvThreadsPerBlock, smem, stream>>>(
      w, residual, norm_weight, output, N, K, eps);
  return true;
}

struct RmsNormDispatchEntry {
  RmsNormDispatchFn fn;
  const char *name;
};

struct DispatchEntry {
  FusedDispatchFn fn;
  const char *name;
};

struct PackedDispatchEntry {
  PackedDispatchFn fn;
  const char *name;
};

struct PackedDispatchPairEntry {
  PackedDispatchPairFn fn;
  const char *name;
};

struct PackedDispatchTripleEntry {
  PackedDispatchTripleFn fn;
  const char *name;
};

constexpr int kMaxTensorType = 16;
constexpr int kQ4KGroupedHotPathK = 2048;
constexpr int kQ4KGroupedHotPathBlocks = kQ4KGroupedHotPathK / QK_K;
constexpr int kDownProjHotPathK = 11008;
constexpr int kDownProjHotPathN = 2048;
constexpr int kDownProjHotPathBlocks = kDownProjHotPathK / QK_K;

int ClampThreshold(int threshold) {
  return std::max(4, std::min(static_cast<int>(kFusedGemmMaxM), threshold));
}

bool OperatorSelectionDebugEnabled() {
  return ResolveExecutionPolicy(nullptr).debug_operator_selection;
}

int OperatorSelectionDebugLimit() {
  return ResolveExecutionPolicy(nullptr).debug_operator_selection_limit;
}

bool ConsumeOperatorSelectionBudget() {
  static thread_local int last_limit = -1;
  static thread_local int budget = 0;
  const int limit = OperatorSelectionDebugLimit();
  if (limit != last_limit) {
    last_limit = limit;
    budget = limit;
  }
  if (budget <= 0) {
    return false;
  }
  --budget;
  return true;
}

std::string QuantTypeToString(int quant_type) {
  switch (static_cast<GGUF::TensorType>(quant_type)) {
  case GGUF::TensorType::Q4_K:
    return "q4_k";
  case GGUF::TensorType::Q6_K:
    return "q6_k";
  case GGUF::TensorType::Q8_0:
    return "q8_0";
  case GGUF::TensorType::Q8_K:
    return "q8_k";
  default:
    return std::to_string(quant_type);
  }
}

void LogOperatorSelection(std::string_view op_kind, std::string_view selection,
                          const FusedDispatchGeometry &geometry,
                          std::string_view detail) {
  if (!OperatorSelectionDebugEnabled() || !ConsumeOperatorSelectionBudget()) {
    return;
  }
  inferflux::log::Info(
      "fused_quant_gemm",
      std::string("operator_select[") + std::string(op_kind) + "]: chosen=" +
          std::string(selection) + ", M=" + std::to_string(geometry.M) +
          ", N=" + std::to_string(geometry.N) +
          ", K=" + std::to_string(geometry.K) +
          ", grouped_outputs=" +
          std::to_string(geometry.grouped_outputs) +
          ", packed_activation=" +
          std::string(geometry.packed_activation ? "true" : "false") +
          ", includes_rmsnorm=" +
          std::string(geometry.includes_rmsnorm ? "true" : "false") +
          (detail.empty() ? std::string() : ", " + std::string(detail)));
}

bool ShouldUseSpecializedQ8_1GroupedFastPathImpl(
    int quant_type, const FusedDispatchGeometry &geometry) {
  return quant_type ==
             static_cast<int>(GGUF::TensorType::Q4_K) &&
         geometry.grouped_outputs == 2 && geometry.M > 0 && geometry.M <= 2 &&
         geometry.N >= 8192 &&
         geometry.K == kQ4KGroupedHotPathK;
}

bool ShouldUseSpecializedQ8_1DownProjHotPathImpl(
    int quant_type, const FusedDispatchGeometry &geometry) {
  const bool hot_quant =
      quant_type == static_cast<int>(GGUF::TensorType::Q4_K) ||
      quant_type == static_cast<int>(GGUF::TensorType::Q6_K);
  if (!ResolveExecutionPolicy(nullptr)
           .enable_experimental_q81_downproj_hot_fixed) {
    return false;
  }
  return hot_quant && geometry.grouped_outputs == 1 &&
         geometry.M == 1 && geometry.N == kDownProjHotPathN &&
         geometry.K == kDownProjHotPathK;
}

bool ExperimentalQ8_1GroupedHotQ4KEnabled() {
  return ResolveExecutionPolicy(nullptr).enable_experimental_q81_grouped_hot_q4k;
}

const DispatchEntry &GetDispatchEntry(GGUF::TensorType qtype) {
  static const bool dp4a = GetGpuProfile().has_dp4a;
  static const DispatchEntry table[kMaxTensorType] = {
      {nullptr, nullptr}, // 0: F32
      {nullptr, nullptr}, // 1: F16
      {nullptr, nullptr}, // 2: Q4_0
      {nullptr, nullptr}, // 3: Q4_1
      {nullptr, nullptr}, // 4: (unused)
      {nullptr, nullptr}, // 5: (unused)
      {nullptr, nullptr}, // 6: Q5_0
      {nullptr, nullptr}, // 7: Q5_1
      // Q8_0: use dp4a variant on SM 6.1+
      {dp4a ? DispatchFusedDp4a<block_q8_0, fused_dequant_gemv_q8_0_dp4a>
            : DispatchFused<block_q8_0, fused_dequant_gemv_q8_0,
                            fused_dequant_gemm_q8_0>,
       "Q8_0"},           // 8
      {nullptr, nullptr}, // 9: Q8_1
      {nullptr, nullptr}, // 10: Q2_K
      {nullptr, nullptr}, // 11: Q3_K
      // Q4_K: dp4a variant on SM 6.1+ (matches llama.cpp vec_dot_q4_K_q8_1)
      {dp4a ? DispatchFusedDp4a<block_q4_k, fused_dequant_gemv_q4k_dp4a>
            : DispatchFused<block_q4_k, fused_dequant_gemv_q4k,
                            fused_dequant_gemm_q4k>,
       "Q4_K"},           // 12
      {nullptr, nullptr}, // 13: Q5_K
      // Q6_K: use dp4a on SM 6.1+ so the standard path matches the packed path
      // and avoids the older FP16 shared-memory fallback on modern NVIDIA GPUs.
      {dp4a ? DispatchFusedDp4a<block_q6_k, fused_dequant_gemv_q6k_dp4a>
            : DispatchFusedFp16<block_q6_k, fused_dequant_gemv_q6k_fp16>,
       "Q6_K"}, // 14
      // Q8_K: use dp4a variant on SM 6.1+
      {dp4a ? DispatchFusedDp4a<block_q8_k, fused_dequant_gemv_q8k_dp4a>
            : DispatchFused<block_q8_k, fused_dequant_gemv_q8k,
                            fused_dequant_gemm_q8k>,
       "Q8_K"}, // 15
  };

  static const DispatchEntry empty = {nullptr, nullptr};
  auto idx = static_cast<uint32_t>(qtype);
  if (idx >= kMaxTensorType)
    return empty;
  return table[idx];
}

const PackedDispatchEntry &GetPackedDispatchEntry(GGUF::TensorType qtype) {
  static const bool dp4a = GetGpuProfile().has_dp4a;
  static const PackedDispatchEntry table[kMaxTensorType] = {
      {nullptr, nullptr}, // 0: F32
      {nullptr, nullptr}, // 1: F16
      {nullptr, nullptr}, // 2: Q4_0
      {nullptr, nullptr}, // 3: Q4_1
      {nullptr, nullptr}, // 4: (unused)
      {nullptr, nullptr}, // 5: (unused)
      {nullptr, nullptr}, // 6: Q5_0
      {nullptr, nullptr}, // 7: Q5_1
      {dp4a ? DispatchPackedDp4a<block_q8_0, fused_dequant_gemv_q8_0_dp4a_packed>
            : nullptr,
       "Q8_0"},           // 8
      {nullptr, nullptr}, // 9: Q8_1
      {nullptr, nullptr}, // 10: Q2_K
      {nullptr, nullptr}, // 11: Q3_K
      {dp4a ? DispatchPackedDp4a<block_q4_k, fused_dequant_gemv_q4k_dp4a_packed>
            : nullptr,
       "Q4_K"},           // 12
      {nullptr, nullptr}, // 13: Q5_K
      {dp4a ? DispatchPackedDp4a<block_q6_k, fused_dequant_gemv_q6k_dp4a_packed>
            : nullptr,
       "Q6_K"}, // 14
      {dp4a ? DispatchPackedDp4a<block_q8_k, fused_dequant_gemv_q8k_dp4a_packed>
            : nullptr,
       "Q8_K"}, // 15
  };

  static const PackedDispatchEntry empty = {nullptr, nullptr};
  auto idx = static_cast<uint32_t>(qtype);
  if (idx >= kMaxTensorType)
    return empty;
  return table[idx];
}

const PackedDispatchPairEntry &GetPackedDispatchPairEntry(GGUF::TensorType qtype) {
  static const bool dp4a = GetGpuProfile().has_dp4a;
  static const PackedDispatchPairEntry table[kMaxTensorType] = {
      {nullptr, nullptr}, // 0: F32
      {nullptr, nullptr}, // 1: F16
      {nullptr, nullptr}, // 2: Q4_0
      {nullptr, nullptr}, // 3: Q4_1
      {nullptr, nullptr}, // 4: (unused)
      {nullptr, nullptr}, // 5: (unused)
      {nullptr, nullptr}, // 6: Q5_0
      {nullptr, nullptr}, // 7: Q5_1
      {dp4a ? DispatchPackedDp4aPair<block_q8_0,
                                     fused_dequant_gemv_q8_0_dp4a_packed_group<2>>
            : nullptr,
       "Q8_0"},           // 8
      {nullptr, nullptr}, // 9: Q8_1
      {nullptr, nullptr}, // 10: Q2_K
      {nullptr, nullptr}, // 11: Q3_K
      {dp4a ? DispatchPackedDp4aPair<block_q4_k,
                                     fused_dequant_gemv_q4k_dp4a_packed_group<2>>
            : nullptr,
       "Q4_K"},           // 12
      {nullptr, nullptr}, // 13: Q5_K
      {dp4a ? DispatchPackedDp4aPair<block_q6_k,
                                     fused_dequant_gemv_q6k_dp4a_packed_group<2>>
            : nullptr,
       "Q6_K"},           // 14
      {dp4a ? DispatchPackedDp4aPair<block_q8_k,
                                     fused_dequant_gemv_q8k_dp4a_packed_group<2>>
            : nullptr,
       "Q8_K"}, // 15
  };

  static const PackedDispatchPairEntry empty = {nullptr, nullptr};
  auto idx = static_cast<uint32_t>(qtype);
  if (idx >= kMaxTensorType)
    return empty;
  return table[idx];
}

const PackedDispatchTripleEntry &
GetPackedDispatchTripleEntry(GGUF::TensorType qtype) {
  static const bool dp4a = GetGpuProfile().has_dp4a;
  static const PackedDispatchTripleEntry table[kMaxTensorType] = {
      {nullptr, nullptr}, // 0: F32
      {nullptr, nullptr}, // 1: F16
      {nullptr, nullptr}, // 2: Q4_0
      {nullptr, nullptr}, // 3: Q4_1
      {nullptr, nullptr}, // 4: (unused)
      {nullptr, nullptr}, // 5: (unused)
      {nullptr, nullptr}, // 6: Q5_0
      {nullptr, nullptr}, // 7: Q5_1
      {dp4a ? DispatchPackedDp4aTriple<block_q8_0,
                                       fused_dequant_gemv_q8_0_dp4a_packed_group<3>>
            : nullptr,
       "Q8_0"},           // 8
      {nullptr, nullptr}, // 9: Q8_1
      {nullptr, nullptr}, // 10: Q2_K
      {nullptr, nullptr}, // 11: Q3_K
      {dp4a ? DispatchPackedDp4aTriple<block_q4_k,
                                       fused_dequant_gemv_q4k_dp4a_packed_group<3>>
            : nullptr,
       "Q4_K"},           // 12
      {nullptr, nullptr}, // 13: Q5_K
      {dp4a ? DispatchPackedDp4aTriple<block_q6_k,
                                       fused_dequant_gemv_q6k_dp4a_packed_group<3>>
            : nullptr,
       "Q6_K"},           // 14
      {dp4a ? DispatchPackedDp4aTriple<block_q8_k,
                                       fused_dequant_gemv_q8k_dp4a_packed_group<3>>
            : nullptr,
       "Q8_K"}, // 15
  };

  static const PackedDispatchTripleEntry empty = {nullptr, nullptr};
  auto idx = static_cast<uint32_t>(qtype);
  if (idx >= kMaxTensorType)
    return empty;
  return table[idx];
}

const RmsNormDispatchEntry &GetRmsNormDispatchEntry(GGUF::TensorType qtype) {
  static const bool dp4a = GetGpuProfile().has_dp4a;
  static const RmsNormDispatchEntry table[kMaxTensorType] = {
      {nullptr, nullptr}, // 0: F32
      {nullptr, nullptr}, // 1: F16
      {nullptr, nullptr}, // 2: Q4_0
      {nullptr, nullptr}, // 3: Q4_1
      {nullptr, nullptr}, // 4: (unused)
      {nullptr, nullptr}, // 5: (unused)
      {nullptr, nullptr}, // 6: Q5_0
      {nullptr, nullptr}, // 7: Q5_1
      // Q8_0: combined RmsNorm+dp4a on SM 6.1+, standard RmsNorm+GEMV otherwise
      {dp4a ? DispatchRmsNormGemvDp4a<block_q8_0,
                                      fused_rmsnorm_gemv_q8_0_dp4a>
            : DispatchRmsNormGemv<block_q8_0, fused_rmsnorm_gemv_q8_0>,
       "Q8_0"},           // 8
      {nullptr, nullptr}, // 9: Q8_1
      {nullptr, nullptr}, // 10: Q2_K
      {nullptr, nullptr}, // 11: Q3_K
      // Q4_K: dp4a + RmsNorm variant on SM 6.1+
      {dp4a ? DispatchRmsNormGemvDp4a<block_q4_k, fused_rmsnorm_gemv_q4k_dp4a>
            : DispatchRmsNormGemv<block_q4_k, fused_rmsnorm_gemv_q4k>,
       "Q4_K"},           // 12
      {nullptr, nullptr}, // 13: Q5_K
      // Q6_K: use the dp4a fused-RMS path on modern NVIDIA GPUs; retain the
      // FP16 shared-memory kernel as the compatibility fallback.
      {dp4a ? DispatchRmsNormGemvDp4a<block_q6_k,
                                      fused_rmsnorm_gemv_q6k_dp4a>
            : DispatchRmsNormGemvFp16<block_q6_k, fused_rmsnorm_gemv_q6k_fp16>,
       "Q6_K"}, // 14
      // Q8_K: combined RmsNorm+dp4a on SM 6.1+, standard RmsNorm+GEMV otherwise
      {dp4a ? DispatchRmsNormGemvDp4a<block_q8_k,
                                      fused_rmsnorm_gemv_q8k_dp4a>
            : DispatchRmsNormGemv<block_q8_k, fused_rmsnorm_gemv_q8k>,
       "Q8_K"}, // 15
  };

  static const RmsNormDispatchEntry empty = {nullptr, nullptr};
  auto idx = static_cast<uint32_t>(qtype);
  if (idx >= kMaxTensorType)
    return empty;
  return table[idx];
}

template <size_t Count>
bool ValidatePackedProjectionSpecs(
    const std::array<PackedProjectionSpec, Count> &projections,
    const PackedActivationInfo &activation, int M, int K) {
  if (!activation.data || !activation.row_scales || M <= 0) {
    return false;
  }
  const int quant_type = projections[0].weight.quant_type;
  int max_output_cols = 0;
  for (const auto &projection : projections) {
    max_output_cols = std::max(max_output_cols, projection.output_cols);
  }
  const FusedDispatchGeometry geometry{
      M,
      max_output_cols,
      K,
      static_cast<int>(Count),
      true,
      false,
  };
  if (quant_type < 0 || !FusedQuantGemm::SupportsPackedActivations(quant_type) ||
      !FusedQuantGemm::ShouldUseFusedPath(quant_type, geometry)) {
    return false;
  }

  for (const auto &projection : projections) {
    if (!projection.weight.data || projection.output == nullptr ||
        projection.output_cols <= 0 ||
        projection.weight.quant_type != quant_type) {
      return false;
    }
  }
  return true;
}

bool DownProjMmqEnabled() {
  return ResolveExecutionPolicy(nullptr).enable_downproj_mmq;
}

int GetDownProjMmqThresholdOverride() {
  return ResolveExecutionPolicy(nullptr).downproj_mmq_min_batch_override;
}

int GetDownProjMmqThreshold(int quant_type, int M, int N, int K) {
  const int override = GetDownProjMmqThresholdOverride();
  if (override >= 0) {
    return override;
  }
  const FusedDispatchGeometry geometry{M, N, K, 1, true, false};
  return ClampThreshold(
      FusedQuantGemm::GetGeometryAwareThreshold(quant_type, geometry) + 12);
}

using DownProjMmqDispatchFn = bool (*)(const void *weight, const void *act_q8_1,
                                       half *output, int M, int N, int K,
                                       int tile_cols, cudaStream_t stream);

template <typename BlockT>
__global__ void transform_downproj_mmq_layout(const BlockT *__restrict__ src,
                                              BlockT *__restrict__ dst,
                                              int rows, int num_super_blocks,
                                              int tile_cols) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int tile_count = (rows + tile_cols - 1) / tile_cols;
  const int total_blocks = tile_count * num_super_blocks * tile_cols;
  if (idx >= total_blocks) {
    return;
  }

  const int tile_stride = num_super_blocks * tile_cols;
  const int tile = idx / tile_stride;
  const int rem = idx % tile_stride;
  const int super_block = rem / tile_cols;
  const int tile_col = rem % tile_cols;
  const int src_row = tile * tile_cols + tile_col;
  if (src_row < rows) {
    dst[idx] = src[src_row * num_super_blocks + super_block];
  } else {
    dst[idx] = BlockT{};
  }
}

__global__ void fused_downproj_mmq_q4k_q8_1_tile(
    const block_q4_k *__restrict__ tiled_weight,
    const block_q8_1 *__restrict__ act_q8_1, half *__restrict__ output, int N,
    int K, int tile_cols) {
  extern __shared__ unsigned char shared[];
  auto *shared_act = reinterpret_cast<block_q8_1 *>(shared);

  const int lane = threadIdx.x;
  const int out_local = threadIdx.y;
  const int out_idx = blockIdx.x * tile_cols + out_local;
  const int row = blockIdx.y;

  const int num_super_blocks = K / QK_K;
  const int num_q8_blocks = K / QK8_1;
  const int linear_tid = out_local * 32 + lane;
  for (int idx = linear_tid; idx < num_q8_blocks; idx += blockDim.x * blockDim.y) {
    shared_act[idx] = act_q8_1[row * num_q8_blocks + idx];
  }
  __syncthreads();

  if (out_idx >= N) {
    return;
  }

  const block_q4_k *tile_weights =
      tiled_weight + (blockIdx.x * num_super_blocks * tile_cols);
  const int pair = lane >> 3;
  const int offs = (lane & 7) * 4;
  float acc = 0.0f;

  for (int blk = 0; blk < num_super_blocks; ++blk) {
    const block_q4_k &b = tile_weights[blk * tile_cols + out_local];
    const float d = __half2float(*reinterpret_cast<const half *>(&b.d));
    const float dmin = __half2float(*reinterpret_cast<const half *>(&b.dmin));

    unsigned char sc_lo, m_lo, sc_hi, m_hi;
    get_scale_min_k4(pair * 2, b.scales, &sc_lo, &m_lo);
    get_scale_min_k4(pair * 2 + 1, b.scales, &sc_hi, &m_hi);

    const float d_sc_lo = d * static_cast<float>(sc_lo);
    const float dm_m_lo = dmin * static_cast<float>(m_lo);
    const float d_sc_hi = d * static_cast<float>(sc_hi);
    const float dm_m_hi = dmin * static_cast<float>(m_hi);

    const int qs4 = *reinterpret_cast<const int *>(&b.qs[pair * 32 + offs]);
    const int q_lo4 = qs4 & 0x0F0F0F0F;
    const int q_hi4 = (qs4 >> 4) & 0x0F0F0F0F;

    const block_q8_1 &a_lo = shared_act[blk * 8 + pair * 2];
    const block_q8_1 &a_hi = shared_act[blk * 8 + pair * 2 + 1];
    int x_lo4 = 0;
    int x_hi4 = 0;
    memcpy(&x_lo4, &a_lo.qs[offs], sizeof(x_lo4));
    memcpy(&x_hi4, &a_hi.qs[offs], sizeof(x_hi4));

    const float d8_lo = __half2float(__low2half(a_lo.ds));
    const float d8_hi = __half2float(__low2half(a_hi.ds));
    const int dot_lo = Dp4aS8(q_lo4, x_lo4, 0);
    const int dot_hi = Dp4aS8(q_hi4, x_hi4, 0);
    acc += d_sc_lo * d8_lo * static_cast<float>(dot_lo) +
           d_sc_hi * d8_hi * static_cast<float>(dot_hi);

    if ((lane & 7) == 0) {
      const float s_lo = __half2float(__high2half(a_lo.ds));
      const float s_hi = __half2float(__high2half(a_hi.ds));
      acc -= dm_m_lo * s_lo + dm_m_hi * s_hi;
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
  }
  if (lane == 0) {
    output[row * N + out_idx] = __float2half(acc);
  }
}

bool DispatchDownProjMmqQ4K(const void *weight, const void *act_q8_1,
                            half *output, int M, int N, int K, int tile_cols,
                            cudaStream_t stream) {
  if (!weight || !act_q8_1 || !output || M <= 0 || N <= 0 || K <= 0 ||
      tile_cols <= 0 || K % QK_K != 0 || K % QK8_1 != 0) {
    return false;
  }

  const auto *w = static_cast<const block_q4_k *>(weight);
  const auto *a = static_cast<const block_q8_1 *>(act_q8_1);
  const dim3 block(32, tile_cols);
  const dim3 grid((N + tile_cols - 1) / tile_cols, M);
  const size_t shared_bytes = static_cast<size_t>(K / QK8_1) *
                              sizeof(block_q8_1);
  fused_downproj_mmq_q4k_q8_1_tile<<<grid, block, shared_bytes, stream>>>(
      w, a, output, N, K, tile_cols);
  return true;
}

__global__ void fused_downproj_mmq_q6k_q8_1_tile(
    const block_q6_k *__restrict__ tiled_weight,
    const block_q8_1 *__restrict__ act_q8_1, half *__restrict__ output, int N,
    int K, int tile_cols) {
  extern __shared__ unsigned char shared[];
  auto *shared_act = reinterpret_cast<block_q8_1 *>(shared);

  const int lane = threadIdx.x;
  const int out_local = threadIdx.y;
  const int out_idx = blockIdx.x * tile_cols + out_local;
  const int row = blockIdx.y;

  const int num_super_blocks = K / QK_K;
  const int num_q8_blocks = K / QK8_1;
  const int linear_tid = out_local * 32 + lane;
  for (int idx = linear_tid; idx < num_q8_blocks; idx += blockDim.x * blockDim.y) {
    shared_act[idx] = act_q8_1[row * num_q8_blocks + idx];
  }
  __syncthreads();

  if (out_idx >= N) {
    return;
  }

  const block_q6_k *tile_weights =
      tiled_weight + (blockIdx.x * num_super_blocks * tile_cols);
  const int sub_pair = lane >> 3;
  const int e_base = (lane & 7) << 2;
  const int g = sub_pair >> 1;
  const int sub_base = sub_pair & 1;
  const int qh_shift_lo = sub_base * 2;
  const int qh_shift_hi = qh_shift_lo + 4;
  const int sc_lo = g * 8 + sub_base * 2 + e_base / 16;
  const int sc_hi = g * 8 + (sub_base + 2) * 2 + e_base / 16;

  float acc = 0.0f;

  for (int blk = 0; blk < num_super_blocks; ++blk) {
    const block_q6_k &b = tile_weights[blk * tile_cols + out_local];
    const float d = __half2float(*reinterpret_cast<const half *>(&b.d));

    const int ql4 =
        LoadPackedInt32Unaligned(&b.ql[g * 64 + sub_base * 32 + e_base]);
    const int qh4 = LoadPackedInt32Unaligned(&b.qh[g * 32 + e_base]);

    const block_q8_1 &a_lo = shared_act[blk * 8 + g * 4 + sub_base];
    const block_q8_1 &a_hi = shared_act[blk * 8 + g * 4 + sub_base + 2];

    const int x_lo = LoadPackedInt32Unaligned(&a_lo.qs[e_base]);
    const int x_hi = LoadPackedInt32Unaligned(&a_hi.qs[e_base]);
    const float d8_lo = __half2float(__low2half(a_lo.ds));
    const float d8_hi = __half2float(__low2half(a_hi.ds));

    const int vl_lo = ql4 & 0x0F0F0F0F;
    const int vh_lo = ((qh4 >> qh_shift_lo) << 4) & 0x30303030;
    const int vi_lo = Vsubss4(vl_lo | vh_lo, 0x20202020);
    const int dot_lo = Dp4aS8(vi_lo, x_lo, 0);

    const int vl_hi = (ql4 >> 4) & 0x0F0F0F0F;
    const int vh_hi = ((qh4 >> qh_shift_hi) << 4) & 0x30303030;
    const int vi_hi = Vsubss4(vl_hi | vh_hi, 0x20202020);
    const int dot_hi = Dp4aS8(vi_hi, x_hi, 0);

    acc += d * (static_cast<float>(b.scales[sc_lo]) * d8_lo *
                    static_cast<float>(dot_lo) +
                static_cast<float>(b.scales[sc_hi]) * d8_hi *
                    static_cast<float>(dot_hi));
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
  }
  if (lane == 0) {
    output[row * N + out_idx] = __float2half(acc);
  }
}

bool DispatchDownProjMmqQ6K(const void *weight, const void *act_q8_1,
                            half *output, int M, int N, int K, int tile_cols,
                            cudaStream_t stream) {
  if (!weight || !act_q8_1 || !output || M <= 0 || N <= 0 || K <= 0 ||
      tile_cols <= 0 || K % QK_K != 0 || K % QK8_1 != 0) {
    return false;
  }

  const auto *w = static_cast<const block_q6_k *>(weight);
  const auto *a = static_cast<const block_q8_1 *>(act_q8_1);
  const dim3 block(32, tile_cols);
  const dim3 grid((N + tile_cols - 1) / tile_cols, M);
  const size_t shared_bytes = static_cast<size_t>(K / QK8_1) *
                              sizeof(block_q8_1);
  fused_downproj_mmq_q6k_q8_1_tile<<<grid, block, shared_bytes, stream>>>(
      w, a, output, N, K, tile_cols);
  return true;
}

struct DownProjMmqDispatchEntry {
  DownProjMmqDispatchFn fn;
  const char *name;
};

const DownProjMmqDispatchEntry &
GetDownProjMmqDispatchEntry(GGUF::TensorType qtype) {
  static const DownProjMmqDispatchEntry table[kMaxTensorType] = {
      {nullptr, nullptr}, // 0: F32
      {nullptr, nullptr}, // 1: F16
      {nullptr, nullptr}, // 2: Q4_0
      {nullptr, nullptr}, // 3: Q4_1
      {nullptr, nullptr}, // 4: (unused)
      {nullptr, nullptr}, // 5: (unused)
      {nullptr, nullptr}, // 6: Q5_0
      {nullptr, nullptr}, // 7: Q5_1
      {nullptr, nullptr}, // 8: Q8_0
      {nullptr, nullptr}, // 9: Q8_1
      {nullptr, nullptr}, // 10: Q2_K
      {nullptr, nullptr}, // 11: Q3_K
      {DispatchDownProjMmqQ4K, "Q4_K"}, // 12
      {nullptr, nullptr},               // 13: Q5_K
      {DispatchDownProjMmqQ6K, "Q6_K"}, // 14
      {nullptr, nullptr},               // 15: Q8_K
  };

  static const DownProjMmqDispatchEntry empty = {nullptr, nullptr};
  auto idx = static_cast<uint32_t>(qtype);
  if (idx >= kMaxTensorType) {
    return empty;
  }
  return table[idx];
}

} // namespace

int FusedQuantGemm::GetAdaptiveThreshold(int quant_type) {
  auto qtype = static_cast<GGUF::TensorType>(quant_type);
  auto &gpu = GetGpuProfile();
  float bpw = BitsPerWeight(qtype);
  int base = gpu.initialized ? gpu.base_threshold() : 8;
  // dp4a kernels use 4x less shared memory (int8 vs FP32 activations) and
  // hardware integer dot products, so the fused advantage extends to higher M.
  // This matches llama.cpp's MMQ_DP4A_MAX_BATCH_SIZE=64 crossover point.
  if (gpu.has_dp4a && (qtype == GGUF::TensorType::Q4_K ||
                       qtype == GGUF::TensorType::Q6_K ||
                       qtype == GGUF::TensorType::Q8_0 ||
                       qtype == GGUF::TensorType::Q8_K)) {
    base = std::max(base * 2, 12);
  }
  return ComputeThreshold(base, bpw);
}

int FusedQuantGemm::GetGeometryAwareThreshold(
    int quant_type, const FusedDispatchGeometry &geometry) {
  int threshold = GetAdaptiveThreshold(quant_type);
  if (geometry.M <= 0) {
    return 0;
  }

  const bool packed = geometry.packed_activation;
  const int grouped_outputs = std::max(1, geometry.grouped_outputs);
  const int n = std::max(0, geometry.N);
  const int k = std::max(0, geometry.K);

  if (packed) {
    // Packed activations remove repeated half->int8 preparation from each GEMV
    // launch. Grouped sibling projections amortize that cost further.
    threshold += 8;
    threshold += 4 * (grouped_outputs - 1);
    if (k >= 2048) {
      threshold += 4;
    }
    if (k >= 4096) {
      threshold += 4;
    }
    // Very large single-output surfaces (notably lm_head) still favor the
    // compatibility path earlier than sibling projection groups do.
    if (grouped_outputs == 1 && n >= 32768) {
      threshold -= 12;
    } else if (grouped_outputs == 1 && n >= 8192) {
      threshold -= 4;
    }
  } else {
    // Legacy shared-memory activation paths lose relative advantage as output
    // surface grows.
    if (n >= 8192) {
      threshold -= 4;
    }
    if (geometry.includes_rmsnorm) {
      threshold -= 2;
    }
  }

  return ClampThreshold(threshold);
}

bool FusedQuantGemm::ShouldUseFusedPath(int quant_type, int M) {
  const FusedDispatchGeometry geometry{M, 0, 0, 1, false, false};
  return ShouldUseFusedPath(quant_type, geometry);
}

bool FusedQuantGemm::ShouldUseFusedPath(
    int quant_type, const FusedDispatchGeometry &geometry) {
  if (geometry.M <= 0) {
    return false;
  }
  auto qtype = static_cast<GGUF::TensorType>(quant_type);
  if (geometry.grouped_outputs > 1) {
    if (geometry.grouped_outputs == 2) {
      return GetPackedDispatchPairEntry(qtype).fn != nullptr &&
             geometry.M <= GetGeometryAwareThreshold(quant_type, geometry);
    }
    if (geometry.grouped_outputs == 3) {
      return GetPackedDispatchTripleEntry(qtype).fn != nullptr &&
             geometry.M <= GetGeometryAwareThreshold(quant_type, geometry);
    }
    return false;
  }
  if (geometry.packed_activation) {
    if (GetPackedDispatchEntry(qtype).fn == nullptr) {
      return false;
    }
  } else if (GetDispatchEntry(qtype).fn == nullptr) {
    return false;
  }
  return geometry.M <= GetGeometryAwareThreshold(quant_type, geometry);
}

bool FusedQuantGemm::SupportsPackedActivations(int quant_type) {
  auto qtype = static_cast<GGUF::TensorType>(quant_type);
  return GetPackedDispatchEntry(qtype).fn != nullptr;
}

bool FusedQuantGemm::SupportsDownProjMmq(int quant_type) {
  auto qtype = static_cast<GGUF::TensorType>(quant_type);
  return GetDownProjMmqDispatchEntry(qtype).fn != nullptr;
}

bool FusedQuantGemm::IsDownProjMmqEnabled(
    const NativeExecutionPolicy *policy) {
  ScopedExecutionPolicyOverride scoped(policy);
  return DownProjMmqEnabled();
}

bool FusedQuantGemm::ShouldUseSpecializedQ8_1GroupedFastPath(
    int quant_type, const FusedDispatchGeometry &geometry,
    const NativeExecutionPolicy *policy) {
  ScopedExecutionPolicyOverride scoped(policy);
  return ExperimentalQ8_1GroupedHotQ4KEnabled() &&
         ShouldUseSpecializedQ8_1GroupedFastPathImpl(quant_type, geometry);
}

FusedQuantGemm::FfnProjOperator FusedQuantGemm::SelectFfnProjOperator(
    int quant_type0, int quant_type1, const FusedDispatchGeometry &geometry,
    bool allow_q81, bool allow_packed, const NativeExecutionPolicy *policy) {
  ScopedExecutionPolicyOverride scoped(policy);
  FfnProjOperator selected = FusedQuantGemm::FfnProjOperator::kFallback;
  if (geometry.M <= 0 || geometry.N <= 0 || geometry.K <= 0 ||
      geometry.grouped_outputs != 2) {
    LogOperatorSelection("ffn_proj", FfnProjOperatorName(selected), geometry,
                         "invalid_geometry");
    return selected;
  }

  const FusedDispatchGeometry single_geometry{geometry.M, geometry.N,
                                              geometry.K, 1,
                                              geometry.packed_activation,
                                              geometry.includes_rmsnorm};
  const bool q81_ready =
      allow_q81 && SupportsQ8_1Activations(quant_type0) &&
      SupportsQ8_1Activations(quant_type1) &&
      ShouldUseFusedPath(quant_type0, single_geometry) &&
      ShouldUseFusedPath(quant_type1, single_geometry);

  if (q81_ready) {
    if (quant_type0 == quant_type1 &&
        ShouldUseSpecializedQ8_1GroupedFastPath(quant_type0, geometry)) {
      selected = FusedQuantGemm::FfnProjOperator::kQ81GroupHotQ4K;
      LogOperatorSelection(
          "ffn_proj", FfnProjOperatorName(selected), geometry,
          "quant0=" + QuantTypeToString(quant_type0) +
              ", quant1=" + QuantTypeToString(quant_type1) + ", q81_ready=true");
      return selected;
    }
    selected = FusedQuantGemm::FfnProjOperator::kQ81Group;
    LogOperatorSelection(
        "ffn_proj", FfnProjOperatorName(selected), geometry,
        "quant0=" + QuantTypeToString(quant_type0) +
            ", quant1=" + QuantTypeToString(quant_type1) + ", q81_ready=true");
    return selected;
  }

  const bool packed_ready =
      allow_packed && SupportsPackedActivations(quant_type0) &&
      SupportsPackedActivations(quant_type1) &&
      ShouldUseFusedPath(quant_type0, single_geometry) &&
      ShouldUseFusedPath(quant_type1, single_geometry);
  if (packed_ready) {
    selected = FusedQuantGemm::FfnProjOperator::kPackedGroup;
    LogOperatorSelection(
        "ffn_proj", FfnProjOperatorName(selected), geometry,
        "quant0=" + QuantTypeToString(quant_type0) +
            ", quant1=" + QuantTypeToString(quant_type1) +
            ", packed_ready=true");
    return selected;
  }

  LogOperatorSelection(
      "ffn_proj", FfnProjOperatorName(selected), geometry,
      "quant0=" + QuantTypeToString(quant_type0) +
          ", quant1=" + QuantTypeToString(quant_type1) +
          ", q81_ready=false, packed_ready=false");
  return selected;
}

const char *FusedQuantGemm::FfnProjOperatorName(
    FusedQuantGemm::FfnProjOperator op) {
  switch (op) {
  case FusedQuantGemm::FfnProjOperator::kQ81Group:
    return "q8_1_group";
  case FusedQuantGemm::FfnProjOperator::kQ81GroupHotQ4K:
    return "q8_1_group_hot_q4k";
  case FusedQuantGemm::FfnProjOperator::kPackedGroup:
    return "packed_group";
  case FusedQuantGemm::FfnProjOperator::kFallback:
  default:
    return "fallback";
  }
}

FusedQuantGemm::DownProjOperator FusedQuantGemm::SelectDownProjOperator(
    int quant_type, const FusedDispatchGeometry &geometry, bool allow_q81,
    bool allow_packed, bool allow_mmq, const NativeExecutionPolicy *policy) {
  ScopedExecutionPolicyOverride scoped(policy);
  DownProjOperator selected = FusedQuantGemm::DownProjOperator::kFallback;
  if (geometry.M <= 0 || geometry.N <= 0 || geometry.K <= 0) {
    LogOperatorSelection("down_proj", DownProjOperatorName(selected), geometry,
                         "invalid_geometry");
    return selected;
  }

  const auto qtype = static_cast<GGUF::TensorType>(quant_type);
  const bool q81_ready =
      allow_q81 && SupportsQ8_1Activations(quant_type) &&
      ShouldUseFusedPath(quant_type,
                         FusedDispatchGeometry{geometry.M, geometry.N,
                                               geometry.K, 1, true, false});

  const bool packed_ready =
      allow_packed && SupportsPackedActivations(quant_type) &&
      ShouldUseFusedPath(quant_type,
                         FusedDispatchGeometry{geometry.M, geometry.N,
                                               geometry.K, 1, true, false});

  const bool mmq_ready =
      allow_mmq && SupportsDownProjMmq(quant_type) &&
      geometry.M >=
          GetDownProjMmqThreshold(quant_type, geometry.M, geometry.N,
                                  geometry.K);

  if (mmq_ready) {
    selected = FusedQuantGemm::DownProjOperator::kMmq;
    LogOperatorSelection("down_proj", DownProjOperatorName(selected), geometry,
                         "quant=" + QuantTypeToString(quant_type) +
                             ", q81_ready=" + (q81_ready ? std::string("true")
                                                          : std::string("false")) +
                             ", mmq_ready=true");
    return selected;
  }
  if (q81_ready) {
    if (ShouldUseSpecializedQ8_1DownProjHotPathImpl(quant_type, geometry)) {
      selected = FusedQuantGemm::DownProjOperator::kQ81GemvHotFixed;
      LogOperatorSelection("down_proj", DownProjOperatorName(selected),
                           geometry, "quant=" + QuantTypeToString(quant_type) +
                                         ", q81_ready=true");
      return selected;
    }
    if ((qtype == GGUF::TensorType::Q4_K || qtype == GGUF::TensorType::Q6_K) &&
        geometry.M >= 4) {
      selected = FusedQuantGemm::DownProjOperator::kQ81GemvRowQuad;
      LogOperatorSelection("down_proj", DownProjOperatorName(selected),
                           geometry, "quant=" + QuantTypeToString(quant_type) +
                                         ", q81_ready=true");
      return selected;
    }
    if ((qtype == GGUF::TensorType::Q4_K || qtype == GGUF::TensorType::Q6_K) &&
        geometry.M > 1) {
      selected = FusedQuantGemm::DownProjOperator::kQ81GemvRowPair;
      LogOperatorSelection("down_proj", DownProjOperatorName(selected),
                           geometry, "quant=" + QuantTypeToString(quant_type) +
                                         ", q81_ready=true");
      return selected;
    }
    selected = FusedQuantGemm::DownProjOperator::kQ81Gemv;
    LogOperatorSelection("down_proj", DownProjOperatorName(selected), geometry,
                         "quant=" + QuantTypeToString(quant_type) +
                             ", q81_ready=true");
    return selected;
  }
  if (packed_ready) {
    selected = FusedQuantGemm::DownProjOperator::kPackedGemv;
    LogOperatorSelection("down_proj", DownProjOperatorName(selected), geometry,
                         "quant=" + QuantTypeToString(quant_type) +
                             ", packed_ready=true");
    return selected;
  }
  LogOperatorSelection(
      "down_proj", DownProjOperatorName(selected), geometry,
      "quant=" + QuantTypeToString(quant_type) +
          ", q81_ready=false, packed_ready=false, mmq_ready=false");
  return selected;
}

const char *FusedQuantGemm::DownProjOperatorName(
    FusedQuantGemm::DownProjOperator op) {
  switch (op) {
  case FusedQuantGemm::DownProjOperator::kQ81Gemv:
    return "q8_1_gemv";
  case FusedQuantGemm::DownProjOperator::kQ81GemvHotFixed:
    return "q8_1_gemv_hot_fixed";
  case FusedQuantGemm::DownProjOperator::kQ81GemvRowPair:
    return "q8_1_gemv_row_pair";
  case FusedQuantGemm::DownProjOperator::kQ81GemvRowQuad:
    return "q8_1_gemv_row_quad";
  case FusedQuantGemm::DownProjOperator::kPackedGemv:
    return "packed_gemv";
  case FusedQuantGemm::DownProjOperator::kMmq:
    return "mmq";
  case FusedQuantGemm::DownProjOperator::kFallback:
  default:
    return "fallback";
  }
}

bool FusedQuantGemm::Gemv(const QuantizedWeightInfo &weight,
                          const half *activation, half *output, int M, int N,
                          int K, cudaStream_t stream,
                          const NativeExecutionPolicy *policy) {
  ScopedExecutionPolicyOverride scoped(policy);
  if (!weight.data || weight.quant_type < 0)
    return false;

  if (ResolveExecutionPolicy(policy).disable_fused_gemv)
    return false;

  auto qtype = static_cast<GGUF::TensorType>(weight.quant_type);
  const auto &entry = GetDispatchEntry(qtype);

  if (!entry.fn)
    return false; // Unsupported quant type

  // Adaptive threshold: fused vs cuBLAS crossover depends on GPU and quant
  // type.
  const FusedDispatchGeometry geometry{M, N, K, 1, false, false};
  const int threshold = GetGeometryAwareThreshold(weight.quant_type, geometry);
  if (!ShouldUseFusedPath(weight.quant_type, geometry))
    return false; // cuBLAS with tensor cores expected to be faster

  // Log once per quant type on first use
  static bool logged[kMaxTensorType] = {};
  auto idx = static_cast<uint32_t>(qtype);
  if (idx < kMaxTensorType && !logged[idx] && entry.name) {
    logged[idx] = true;
    log::Info("fused_quant_gemm",
              std::string("Using fused dequant kernel for ") + entry.name +
                  " (M=" + std::to_string(M) + ", N=" + std::to_string(N) +
                  ", K=" + std::to_string(K) +
                  ", adaptive_threshold=" + std::to_string(threshold) + ")");
  }

  return entry.fn(weight.data, activation, output, M, N, K, stream);
}

bool FusedQuantGemm::GemvPacked(const QuantizedWeightInfo &weight,
                                const PackedActivationInfo &activation,
                                half *output, int M, int N, int K,
                                cudaStream_t stream,
                                const NativeExecutionPolicy *policy) {
  ScopedExecutionPolicyOverride scoped(policy);
  if (!weight.data || weight.quant_type < 0 || !activation.data ||
      !activation.row_scales) {
    return false;
  }
  if (ResolveExecutionPolicy(policy).disable_fused_gemv) {
    return false;
  }

  auto qtype = static_cast<GGUF::TensorType>(weight.quant_type);
  const auto &entry = GetPackedDispatchEntry(qtype);
  if (!entry.fn) {
    return false;
  }
  const FusedDispatchGeometry geometry{M, N, K, 1, true, false};
  if (!ShouldUseFusedPath(weight.quant_type, geometry)) {
    return false;
  }

  static bool logged[kMaxTensorType] = {};
  auto idx = static_cast<uint32_t>(qtype);
  if (idx < kMaxTensorType && !logged[idx] && entry.name) {
    logged[idx] = true;
    log::Info("fused_quant_gemm",
              std::string("Using packed-activation fused kernel for ") +
                  entry.name + " (M=" + std::to_string(M) +
                  ", N=" + std::to_string(N) +
                  ", K=" + std::to_string(K) + ")");
  }

  return entry.fn(weight.data, activation.data, activation.row_scales, output,
                  M, N, K, stream);
}

bool FusedQuantGemm::GemvPackedPair(
    const std::array<PackedProjectionSpec, 2> &projections,
    const PackedActivationInfo &activation, int M, int K,
    cudaStream_t stream) {
  if (!ValidatePackedProjectionSpecs(projections, activation, M, K)) {
    return false;
  }

  auto qtype = static_cast<GGUF::TensorType>(projections[0].weight.quant_type);
  const auto &entry = GetPackedDispatchPairEntry(qtype);
  if (!entry.fn) {
    return false;
  }

  static bool logged[kMaxTensorType] = {};
  auto idx = static_cast<uint32_t>(qtype);
  if (idx < kMaxTensorType && !logged[idx] && entry.name) {
    logged[idx] = true;
    log::Info("fused_quant_gemm",
              std::string("Using packed grouped pair kernel for ") +
                  entry.name + " (M=" + std::to_string(M) + ", N=" +
                  std::to_string(projections[0].output_cols) + "/" +
                  std::to_string(projections[1].output_cols) +
                  ", K=" + std::to_string(K) + ")");
  }

  return entry.fn(projections[0].weight.data, projections[1].weight.data,
                  activation.data, activation.row_scales,
                  projections[0].output, projections[0].output_cols,
                  projections[1].output, projections[1].output_cols, M, K,
                  stream);
}

bool FusedQuantGemm::GemvPackedTriple(
    const std::array<PackedProjectionSpec, 3> &projections,
    const PackedActivationInfo &activation, int M, int K,
    cudaStream_t stream) {
  if (!ValidatePackedProjectionSpecs(projections, activation, M, K)) {
    return false;
  }

  auto qtype = static_cast<GGUF::TensorType>(projections[0].weight.quant_type);
  const auto &entry = GetPackedDispatchTripleEntry(qtype);
  if (!entry.fn) {
    return false;
  }

  static bool logged[kMaxTensorType] = {};
  auto idx = static_cast<uint32_t>(qtype);
  if (idx < kMaxTensorType && !logged[idx] && entry.name) {
    logged[idx] = true;
    log::Info("fused_quant_gemm",
              std::string("Using packed grouped triple kernel for ") +
                  entry.name + " (M=" + std::to_string(M) + ", N=" +
                  std::to_string(projections[0].output_cols) + "/" +
                  std::to_string(projections[1].output_cols) + "/" +
                  std::to_string(projections[2].output_cols) +
                  ", K=" + std::to_string(K) + ")");
  }

  return entry.fn(projections[0].weight.data, projections[1].weight.data,
                  projections[2].weight.data, activation.data,
                  activation.row_scales, projections[0].output,
                  projections[0].output_cols, projections[1].output,
                  projections[1].output_cols, projections[2].output,
                  projections[2].output_cols, M, K, stream);
}

bool FusedQuantGemm::RmsNormGemv(const QuantizedWeightInfo &weight,
                                 const half *residual, const half *norm_weight,
                                 half *output, int M, int N, int K,
                                 float rms_norm_eps, cudaStream_t stream,
                                 const NativeExecutionPolicy *policy) {
  ScopedExecutionPolicyOverride scoped(policy);
  if (!weight.data || weight.quant_type < 0)
    return false;
  if (ResolveExecutionPolicy(policy).disable_fused_gemv)
    return false;

  auto qtype = static_cast<GGUF::TensorType>(weight.quant_type);
  const auto &entry = GetRmsNormDispatchEntry(qtype);

  if (!entry.fn)
    return false;

  // Same adaptive threshold policy as regular GEMV.
  const FusedDispatchGeometry geometry{M, N, K, 1, false, true};
  if (!ShouldUseFusedPath(weight.quant_type, geometry))
    return false;

  // Log once per quant type on first use
  static bool logged[kMaxTensorType] = {};
  auto idx = static_cast<uint32_t>(qtype);
  if (idx < kMaxTensorType && !logged[idx] && entry.name) {
    logged[idx] = true;
    log::Info("fused_quant_gemm",
              std::string("Using fused RmsNorm+GEMV kernel for ") + entry.name +
                  " (M=" + std::to_string(M) + ", N=" + std::to_string(N) +
                  ", K=" + std::to_string(K) + ")");
  }

  return entry.fn(weight.data, residual, norm_weight, output, M, N, K,
                  rms_norm_eps, stream);
}

// ============================================================================
// Q8_1 activation quantization and dispatch
// ============================================================================

namespace {

using Q8_1DispatchFn = bool (*)(const void *data, const void *act_q8_1,
                                 half *output, int M, int N, int K,
                                 cudaStream_t stream);
using Q8_1DispatchPairFn = bool (*)(
    const void *data0, const void *data1, const void *act_q8_1, half *output0,
    int N0, half *output1, int N1, int M, int K, cudaStream_t stream);
using Q8_1DispatchTripleFn = bool (*)(
    const void *data0, const void *data1, const void *data2,
    const void *act_q8_1, half *output0, int N0, half *output1, int N1,
    half *output2, int N2, int M, int K, cudaStream_t stream);

// Column-pair dispatch: each warp computes 2 output columns, halving grid.x.
// Used for M=1 decode to improve ILP via interleaved weight row processing.
template <typename BlockType,
          void (*Q8_1ColPairKernel)(const BlockType *,
                                    const block_q8_1 *, half *, int, int),
          void (*Q8_1Kernel)(const BlockType *,
                              const block_q8_1 *, half *, int, int),
          void (*Q8_1RowPairKernel)(const BlockType *, const block_q8_1 *,
                                    half *, int, int, int),
          void (*Q8_1RowQuadKernel)(const BlockType *, const block_q8_1 *,
                                    half *, int, int, int)>
bool DispatchQ8_1GemvColPairWithFallback(const void *data,
                                         const void *act_q8_1, half *output,
                                         int M, int N, int K,
                                         cudaStream_t stream) {
  auto *w = static_cast<const BlockType *>(data);
  auto *a = static_cast<const block_q8_1 *>(act_q8_1);
  if (M >= 4) {
    const int grid_x = (N + kGemvWarpsPerBlock - 1) / kGemvWarpsPerBlock;
    dim3 grid(grid_x, (M + 3) / 4);
    Q8_1RowQuadKernel<<<grid, kGemvThreadsPerBlock, 0, stream>>>(w, a, output,
                                                                 N, K, M);
    return true;
  }
  if (M > 1) {
    const int grid_x = (N + kGemvWarpsPerBlock - 1) / kGemvWarpsPerBlock;
    dim3 grid(grid_x, (M + 1) / 2);
    Q8_1RowPairKernel<<<grid, kGemvThreadsPerBlock, 0, stream>>>(w, a, output,
                                                                 N, K, M);
    return true;
  }
  // M == 1: use column-pair kernel (2 outputs per warp)
  // Grid covers ceil(N / (warps_per_block * 2)) blocks in x
  const int warps_x2 = kGemvWarpsPerBlock * 2;
  const int grid_x = (N + warps_x2 - 1) / warps_x2;
  dim3 grid(grid_x, M);
  Q8_1ColPairKernel<<<grid, kGemvThreadsPerBlock, 0, stream>>>(w, a, output, N,
                                                                K);
  return true;
}

template <typename BlockType,
          void (*Q8_1Kernel)(const BlockType *,
                              const block_q8_1 *, half *, int, int)>
bool DispatchQ8_1Gemv(const void *data, const void *act_q8_1, half *output,
                       int M, int N, int K, cudaStream_t stream) {
  auto *w = static_cast<const BlockType *>(data);
  auto *a = static_cast<const block_q8_1 *>(act_q8_1);
  int grid_x = (N + kGemvWarpsPerBlock - 1) / kGemvWarpsPerBlock;
  dim3 grid(grid_x, M);
  Q8_1Kernel<<<grid, kGemvThreadsPerBlock, 0, stream>>>(w, a, output, N, K);
  return true;
}

template <typename BlockType,
          void (*Q8_1Kernel)(const BlockType *,
                              const block_q8_1 *, half *, int, int),
          void (*Q8_1RowPairKernel)(const BlockType *, const block_q8_1 *,
                                    half *, int, int, int)>
bool DispatchQ8_1GemvRowPair(const void *data, const void *act_q8_1,
                             half *output, int M, int N, int K,
                             cudaStream_t stream) {
  auto *w = static_cast<const BlockType *>(data);
  auto *a = static_cast<const block_q8_1 *>(act_q8_1);
  const int grid_x = (N + kGemvWarpsPerBlock - 1) / kGemvWarpsPerBlock;
  if (M > 1) {
    dim3 grid(grid_x, (M + 1) / 2);
    Q8_1RowPairKernel<<<grid, kGemvThreadsPerBlock, 0, stream>>>(w, a, output,
                                                                 N, K, M);
    return true;
  }

  dim3 grid(grid_x, M);
  Q8_1Kernel<<<grid, kGemvThreadsPerBlock, 0, stream>>>(w, a, output, N, K);
  return true;
}

template <typename BlockType,
          void (*Q8_1Kernel)(const BlockType *,
                              const block_q8_1 *, half *, int, int),
          void (*Q8_1RowPairKernel)(const BlockType *, const block_q8_1 *,
                                    half *, int, int, int),
          void (*Q8_1RowQuadKernel)(const BlockType *, const block_q8_1 *,
                                    half *, int, int, int)>
bool DispatchQ8_1GemvRowPairQuad(const void *data, const void *act_q8_1,
                                 half *output, int M, int N, int K,
                                 cudaStream_t stream) {
  auto *w = static_cast<const BlockType *>(data);
  auto *a = static_cast<const block_q8_1 *>(act_q8_1);
  const int grid_x = (N + kGemvWarpsPerBlock - 1) / kGemvWarpsPerBlock;
  if (M >= 4) {
    dim3 grid(grid_x, (M + 3) / 4);
    Q8_1RowQuadKernel<<<grid, kGemvThreadsPerBlock, 0, stream>>>(w, a, output,
                                                                 N, K, M);
    return true;
  }
  if (M > 1) {
    dim3 grid(grid_x, (M + 1) / 2);
    Q8_1RowPairKernel<<<grid, kGemvThreadsPerBlock, 0, stream>>>(w, a, output,
                                                                 N, K, M);
    return true;
  }

  dim3 grid(grid_x, M);
  Q8_1Kernel<<<grid, kGemvThreadsPerBlock, 0, stream>>>(w, a, output, N, K);
  return true;
}

bool DispatchQ8_1GemvQ4KHotDownProj(const void *data, const void *act_q8_1,
                                    half *output, int M, int N, int K,
                                    cudaStream_t stream) {
  if (ShouldUseSpecializedQ8_1DownProjHotPathImpl(
          static_cast<int>(GGUF::TensorType::Q4_K),
          FusedDispatchGeometry{M, N, K, 1, true, false})) {
    auto *w = static_cast<const block_q4_k *>(data);
    auto *a = static_cast<const block_q8_1 *>(act_q8_1);
    const int grid_x = (N + kGemvWarpsPerBlock - 1) / kGemvWarpsPerBlock;
    dim3 grid(grid_x, M);
    fused_dequant_gemv_q4k_q8_1_fixed_blocks<kDownProjHotPathBlocks>
        <<<grid, kGemvThreadsPerBlock, 0, stream>>>(w, a, output, N);
    return true;
  }
  return DispatchQ8_1GemvColPairWithFallback<
      block_q4_k, fused_dequant_gemv_q4k_q8_1_colpair,
      fused_dequant_gemv_q4k_q8_1, fused_dequant_gemv_q4k_q8_1_rowpair,
      fused_dequant_gemv_q4k_q8_1_rowquad>(data, act_q8_1, output, M, N, K,
                                           stream);
}

bool DispatchQ8_1GemvQ6KHotDownProj(const void *data, const void *act_q8_1,
                                    half *output, int M, int N, int K,
                                    cudaStream_t stream) {
  if (ShouldUseSpecializedQ8_1DownProjHotPathImpl(
          static_cast<int>(GGUF::TensorType::Q6_K),
          FusedDispatchGeometry{M, N, K, 1, true, false})) {
    auto *w = static_cast<const block_q6_k *>(data);
    auto *a = static_cast<const block_q8_1 *>(act_q8_1);
    const int grid_x = (N + kGemvWarpsPerBlock - 1) / kGemvWarpsPerBlock;
    dim3 grid(grid_x, M);
    fused_dequant_gemv_q6k_q8_1_fixed_blocks<kDownProjHotPathBlocks>
        <<<grid, kGemvThreadsPerBlock, 0, stream>>>(w, a, output, N);
    return true;
  }
  return DispatchQ8_1GemvColPairWithFallback<
      block_q6_k, fused_dequant_gemv_q6k_q8_1_colpair,
      fused_dequant_gemv_q6k_q8_1, fused_dequant_gemv_q6k_q8_1_rowpair,
      fused_dequant_gemv_q6k_q8_1_rowquad>(data, act_q8_1, output, M, N, K,
                                           stream);
}

template <typename BlockType, int Outputs,
          void (*Q8_1GroupKernel)(
              PackedProjectionGroupParams<BlockType, Outputs>,
              const block_q8_1 *, int)>
bool DispatchQ8_1GemvGroup(
    const std::array<const void *, Outputs> &weights,
    const void *act_q8_1, const std::array<half *, Outputs> &outputs,
    const std::array<int, Outputs> &output_cols, int M, int K,
    cudaStream_t stream) {
  auto *a = static_cast<const block_q8_1 *>(act_q8_1);
  PackedProjectionGroupParams<BlockType, Outputs> params{};
  int max_output_cols = 0;
  for (int i = 0; i < Outputs; ++i) {
    params.weights[i] = static_cast<const BlockType *>(weights[i]);
    params.outputs[i] = outputs[i];
    params.output_cols[i] = output_cols[i];
    max_output_cols = std::max(max_output_cols, output_cols[i]);
  }

  int grid_x = (max_output_cols + kGemvWarpsPerBlock - 1) / kGemvWarpsPerBlock;
  dim3 grid(grid_x, M);
  Q8_1GroupKernel<<<grid, kGemvThreadsPerBlock, 0, stream>>>(params, a, K);
  return true;
}

template <typename BlockType,
          void (*Q8_1GroupKernel)(
              PackedProjectionGroupParams<BlockType, 2>,
              const block_q8_1 *, int)>
bool DispatchQ8_1GemvPair(const void *data0, const void *data1,
                            const void *act_q8_1, half *output0, int N0,
                            half *output1, int N1, int M, int K,
                            cudaStream_t stream) {
  return DispatchQ8_1GemvGroup<BlockType, 2, Q8_1GroupKernel>(
      {data0, data1}, act_q8_1, {output0, output1}, {N0, N1}, M, K, stream);
}

template <typename BlockType,
          void (*Q8_1GroupKernel)(
              PackedProjectionGroupParams<BlockType, 2>,
              const block_q8_1 *, int),
          void (*Q8_1RowPairKernel)(
              PackedProjectionGroupParams<BlockType, 2>,
              const block_q8_1 *, int, int)>
bool DispatchQ8_1GemvPairRowPair(const void *data0, const void *data1,
                                 const void *act_q8_1, half *output0, int N0,
                                 half *output1, int N1, int M, int K,
                                 cudaStream_t stream) {
  auto *a = static_cast<const block_q8_1 *>(act_q8_1);
  PackedProjectionGroupParams<BlockType, 2> params{};
  params.weights[0] = static_cast<const BlockType *>(data0);
  params.weights[1] = static_cast<const BlockType *>(data1);
  params.outputs[0] = output0;
  params.outputs[1] = output1;
  params.output_cols[0] = N0;
  params.output_cols[1] = N1;
  const int max_output_cols = std::max(N0, N1);
  const int grid_x =
      (max_output_cols + kGemvWarpsPerBlock - 1) / kGemvWarpsPerBlock;
  if (M > 1) {
    dim3 grid(grid_x, (M + 1) / 2);
    Q8_1RowPairKernel<<<grid, kGemvThreadsPerBlock, 0, stream>>>(params, a, K,
                                                                 M);
    return true;
  }
  dim3 grid(grid_x, M);
  Q8_1GroupKernel<<<grid, kGemvThreadsPerBlock, 0, stream>>>(params, a, K);
  return true;
}

template <typename BlockType,
          void (*Q8_1GroupKernel)(
              PackedProjectionGroupParams<BlockType, 2>,
              const block_q8_1 *, int),
          void (*Q8_1RowPairKernel)(
              PackedProjectionGroupParams<BlockType, 2>,
              const block_q8_1 *, int, int),
          void (*Q8_1RowQuadKernel)(
              PackedProjectionGroupParams<BlockType, 2>,
              const block_q8_1 *, int, int)>
bool DispatchQ8_1GemvPairRowPairQuad(const void *data0, const void *data1,
                                     const void *act_q8_1, half *output0,
                                     int N0, half *output1, int N1, int M,
                                     int K, cudaStream_t stream) {
  auto *a = static_cast<const block_q8_1 *>(act_q8_1);
  PackedProjectionGroupParams<BlockType, 2> params{};
  params.weights[0] = static_cast<const BlockType *>(data0);
  params.weights[1] = static_cast<const BlockType *>(data1);
  params.outputs[0] = output0;
  params.outputs[1] = output1;
  params.output_cols[0] = N0;
  params.output_cols[1] = N1;
  const int max_output_cols = std::max(N0, N1);
  const int grid_x =
      (max_output_cols + kGemvWarpsPerBlock - 1) / kGemvWarpsPerBlock;
  if (M >= 4) {
    dim3 grid(grid_x, (M + 3) / 4);
    Q8_1RowQuadKernel<<<grid, kGemvThreadsPerBlock, 0, stream>>>(params, a, K,
                                                                 M);
    return true;
  }
  if (M > 1) {
    dim3 grid(grid_x, (M + 1) / 2);
    Q8_1RowPairKernel<<<grid, kGemvThreadsPerBlock, 0, stream>>>(params, a, K,
                                                                 M);
    return true;
  }
  dim3 grid(grid_x, M);
  Q8_1GroupKernel<<<grid, kGemvThreadsPerBlock, 0, stream>>>(params, a, K);
  return true;
}

bool DispatchQ8_1GemvPairQ4KSpecialized(const void *data0, const void *data1,
                                        const void *act_q8_1, half *output0,
                                        int N0, half *output1, int N1, int M,
                                        int K, cudaStream_t stream) {
  const FusedDispatchGeometry geometry{M, std::max(N0, N1), K, 2, true,
                                       false};
  if (FusedQuantGemm::ShouldUseSpecializedQ8_1GroupedFastPath(
          static_cast<int>(GGUF::TensorType::Q4_K), geometry)) {
    auto *a = static_cast<const block_q8_1 *>(act_q8_1);
    PackedProjectionGroupParams<block_q4_k, 2> params{};
    params.weights[0] = static_cast<const block_q4_k *>(data0);
    params.weights[1] = static_cast<const block_q4_k *>(data1);
    params.outputs[0] = output0;
    params.outputs[1] = output1;
    params.output_cols[0] = N0;
    params.output_cols[1] = N1;
    const int max_output_cols = std::max(N0, N1);
    const int grid_x =
        (max_output_cols + kGemvWarpsPerBlock - 1) / kGemvWarpsPerBlock;
    if (M > 1) {
      dim3 grid(grid_x, (M + 1) / 2);
      fused_dequant_gemv_q4k_q8_1_group_rowpair_fixed_blocks<
          2, kQ4KGroupedHotPathBlocks><<<grid, kGemvThreadsPerBlock, 0,
                                         stream>>>(params, a, K, M);
      return true;
    }
    dim3 grid(grid_x, M);
    fused_dequant_gemv_q4k_q8_1_group_fixed_blocks<
        2, kQ4KGroupedHotPathBlocks><<<grid, kGemvThreadsPerBlock, 0, stream>>>(
        params, a, K);
    return true;
  }

  return DispatchQ8_1GemvPairRowPairQuad<
      block_q4_k, fused_dequant_gemv_q4k_q8_1_group<2>,
      fused_dequant_gemv_q4k_q8_1_group_rowpair<2>,
      fused_dequant_gemv_q4k_q8_1_group_rowquad<2>>(
      data0, data1, act_q8_1, output0, N0, output1, N1, M, K, stream);
}

template <typename BlockType,
          void (*Q8_1GroupKernel)(
              PackedProjectionGroupParams<BlockType, 3>,
              const block_q8_1 *, int)>
bool DispatchQ8_1GemvTriple(const void *data0, const void *data1,
                              const void *data2, const void *act_q8_1,
                              half *output0, int N0, half *output1, int N1,
                              half *output2, int N2, int M, int K,
                              cudaStream_t stream) {
  return DispatchQ8_1GemvGroup<BlockType, 3, Q8_1GroupKernel>(
      {data0, data1, data2}, act_q8_1, {output0, output1, output2},
      {N0, N1, N2}, M, K, stream);
}

template <typename BlockType,
          void (*Q8_1GroupKernel)(
              PackedProjectionGroupParams<BlockType, 3>,
              const block_q8_1 *, int),
          void (*Q8_1RowPairKernel)(
              PackedProjectionGroupParams<BlockType, 3>,
              const block_q8_1 *, int, int)>
bool DispatchQ8_1GemvTripleRowPair(const void *data0, const void *data1,
                                   const void *data2, const void *act_q8_1,
                                   half *output0, int N0, half *output1,
                                   int N1, half *output2, int N2, int M,
                                   int K, cudaStream_t stream) {
  auto *a = static_cast<const block_q8_1 *>(act_q8_1);
  PackedProjectionGroupParams<BlockType, 3> params{};
  params.weights[0] = static_cast<const BlockType *>(data0);
  params.weights[1] = static_cast<const BlockType *>(data1);
  params.weights[2] = static_cast<const BlockType *>(data2);
  params.outputs[0] = output0;
  params.outputs[1] = output1;
  params.outputs[2] = output2;
  params.output_cols[0] = N0;
  params.output_cols[1] = N1;
  params.output_cols[2] = N2;
  const int max_output_cols = std::max(N0, std::max(N1, N2));
  const int grid_x =
      (max_output_cols + kGemvWarpsPerBlock - 1) / kGemvWarpsPerBlock;
  if (M > 1) {
    dim3 grid(grid_x, (M + 1) / 2);
    Q8_1RowPairKernel<<<grid, kGemvThreadsPerBlock, 0, stream>>>(params, a, K,
                                                                 M);
    return true;
  }
  dim3 grid(grid_x, M);
  Q8_1GroupKernel<<<grid, kGemvThreadsPerBlock, 0, stream>>>(params, a, K);
  return true;
}

// ============================================================================
// V2 cooperative-warp dispatch helpers
// ============================================================================

// Returns true only when v2 kernels are explicitly requested.
// Auto mode defaults to v1 (faster on RTX 4000 Ada: 0.83x vs 0.79x).
// V2 cooperative-warp kernels are retained for experimentation on GPUs
// where L2 cache pressure from v1's 8-row-per-block pattern dominates.
bool UseV2(int K) {
  static const bool forced = [] {
    return std::getenv("INFERFLUX_GEMV_V2") != nullptr;
  }();
  return forced;
}

// Single-output v2 dispatch for Q4_K and Q6_K.
// Falls back to v1 ColPairWithFallback when v2 is disabled or K is small.
bool DispatchQ8_1GemvQ4KV2(const void *data, const void *act_q8_1,
                            half *output, int M, int N, int K,
                            cudaStream_t stream) {
  if (UseV2(K)) {
    auto *w = static_cast<const block_q4_k *>(data);
    auto *a = static_cast<const block_q8_1 *>(act_q8_1);
    if (M > 1) {
      dim3 grid(N, 1, (M + 1) / 2);
      fused_dequant_gemv_q4k_q8_1_v2_rowpair
          <<<grid, kGemvThreadsPerBlockV2,
             sizeof(float) * kGemvWarpsPerBlockV2 * 2, stream>>>(w, a, output,
                                                                  N, K, M);
      return true;
    }
    dim3 grid(N, 1, M);
    fused_dequant_gemv_q4k_q8_1_v2
        <<<grid, kGemvThreadsPerBlockV2,
           sizeof(float) * kGemvWarpsPerBlockV2, stream>>>(w, a, output, N, K);
    return true;
  }
  // Check hot-path first, then fall back to v1
  if (ShouldUseSpecializedQ8_1DownProjHotPathImpl(
          static_cast<int>(GGUF::TensorType::Q4_K),
          FusedDispatchGeometry{M, N, K, 1, true, false})) {
    auto *w = static_cast<const block_q4_k *>(data);
    auto *a = static_cast<const block_q8_1 *>(act_q8_1);
    const int grid_x = (N + kGemvWarpsPerBlock - 1) / kGemvWarpsPerBlock;
    dim3 grid(grid_x, M);
    fused_dequant_gemv_q4k_q8_1_fixed_blocks<kDownProjHotPathBlocks>
        <<<grid, kGemvThreadsPerBlock, 0, stream>>>(w, a, output, N);
    return true;
  }
  return DispatchQ8_1GemvColPairWithFallback<
      block_q4_k, fused_dequant_gemv_q4k_q8_1_colpair,
      fused_dequant_gemv_q4k_q8_1, fused_dequant_gemv_q4k_q8_1_rowpair,
      fused_dequant_gemv_q4k_q8_1_rowquad>(data, act_q8_1, output, M, N, K,
                                           stream);
}

bool DispatchQ8_1GemvQ6KV2(const void *data, const void *act_q8_1,
                            half *output, int M, int N, int K,
                            cudaStream_t stream) {
  if (UseV2(K)) {
    auto *w = static_cast<const block_q6_k *>(data);
    auto *a = static_cast<const block_q8_1 *>(act_q8_1);
    if (M > 1) {
      dim3 grid(N, 1, (M + 1) / 2);
      fused_dequant_gemv_q6k_q8_1_v2_rowpair
          <<<grid, kGemvThreadsPerBlockV2,
             sizeof(float) * kGemvWarpsPerBlockV2 * 2, stream>>>(w, a, output,
                                                                  N, K, M);
      return true;
    }
    dim3 grid(N, 1, M);
    fused_dequant_gemv_q6k_q8_1_v2
        <<<grid, kGemvThreadsPerBlockV2,
           sizeof(float) * kGemvWarpsPerBlockV2, stream>>>(w, a, output, N, K);
    return true;
  }
  if (ShouldUseSpecializedQ8_1DownProjHotPathImpl(
          static_cast<int>(GGUF::TensorType::Q6_K),
          FusedDispatchGeometry{M, N, K, 1, true, false})) {
    auto *w = static_cast<const block_q6_k *>(data);
    auto *a = static_cast<const block_q8_1 *>(act_q8_1);
    const int grid_x = (N + kGemvWarpsPerBlock - 1) / kGemvWarpsPerBlock;
    dim3 grid(grid_x, M);
    fused_dequant_gemv_q6k_q8_1_fixed_blocks<kDownProjHotPathBlocks>
        <<<grid, kGemvThreadsPerBlock, 0, stream>>>(w, a, output, N);
    return true;
  }
  return DispatchQ8_1GemvColPairWithFallback<
      block_q6_k, fused_dequant_gemv_q6k_q8_1_colpair,
      fused_dequant_gemv_q6k_q8_1, fused_dequant_gemv_q6k_q8_1_rowpair,
      fused_dequant_gemv_q6k_q8_1_rowquad>(data, act_q8_1, output, M, N, K,
                                           stream);
}

// V2 grouped dispatch (pair/triple)
template <typename BlockType, int Outputs,
          void (*V2GroupKernel)(PackedProjectionGroupParams<BlockType, Outputs>,
                                const block_q8_1 *, int),
          void (*V1GroupKernel)(PackedProjectionGroupParams<BlockType, Outputs>,
                                const block_q8_1 *, int)>
bool DispatchQ8_1GemvGroupV2(
    const std::array<const void *, Outputs> &weights,
    const void *act_q8_1, const std::array<half *, Outputs> &outputs,
    const std::array<int, Outputs> &output_cols, int M, int K,
    cudaStream_t stream) {
  auto *a = static_cast<const block_q8_1 *>(act_q8_1);
  PackedProjectionGroupParams<BlockType, Outputs> params{};
  int max_output_cols = 0;
  for (int i = 0; i < Outputs; ++i) {
    params.weights[i] = static_cast<const BlockType *>(weights[i]);
    params.outputs[i] = outputs[i];
    params.output_cols[i] = output_cols[i];
    max_output_cols = std::max(max_output_cols, output_cols[i]);
  }

  if (UseV2(K)) {
    dim3 grid(max_output_cols, 1, M);
    size_t smem = sizeof(float) * kGemvWarpsPerBlockV2 * Outputs;
    V2GroupKernel<<<grid, kGemvThreadsPerBlockV2, smem, stream>>>(params, a, K);
    return true;
  }

  int grid_x = (max_output_cols + kGemvWarpsPerBlock - 1) / kGemvWarpsPerBlock;
  dim3 grid(grid_x, M);
  V1GroupKernel<<<grid, kGemvThreadsPerBlock, 0, stream>>>(params, a, K);
  return true;
}

template <typename BlockType,
          void (*V2GroupKernel)(PackedProjectionGroupParams<BlockType, 2>,
                                const block_q8_1 *, int),
          void (*V1GroupKernel)(PackedProjectionGroupParams<BlockType, 2>,
                                const block_q8_1 *, int)>
bool DispatchQ8_1GemvPairV2(const void *data0, const void *data1,
                             const void *act_q8_1, half *output0, int N0,
                             half *output1, int N1, int M, int K,
                             cudaStream_t stream) {
  return DispatchQ8_1GemvGroupV2<BlockType, 2, V2GroupKernel, V1GroupKernel>(
      {data0, data1}, act_q8_1, {output0, output1}, {N0, N1}, M, K, stream);
}

template <typename BlockType,
          void (*V2GroupKernel)(PackedProjectionGroupParams<BlockType, 3>,
                                const block_q8_1 *, int),
          void (*V1GroupKernel)(PackedProjectionGroupParams<BlockType, 3>,
                                const block_q8_1 *, int)>
bool DispatchQ8_1GemvTripleV2(const void *data0, const void *data1,
                               const void *data2, const void *act_q8_1,
                               half *output0, int N0, half *output1, int N1,
                               half *output2, int N2, int M, int K,
                               cudaStream_t stream) {
  return DispatchQ8_1GemvGroupV2<BlockType, 3, V2GroupKernel, V1GroupKernel>(
      {data0, data1, data2}, act_q8_1, {output0, output1, output2},
      {N0, N1, N2}, M, K, stream);
}

// V2-aware pair dispatch for Q4_K that selects v2 grouped or falls back to
// the existing specialized Q4_K pair dispatch.
bool DispatchQ8_1GemvPairQ4KV2(const void *data0, const void *data1,
                                const void *act_q8_1, half *output0, int N0,
                                half *output1, int N1, int M, int K,
                                cudaStream_t stream) {
  if (UseV2(K)) {
    return DispatchQ8_1GemvPairV2<block_q4_k,
                                   fused_dequant_gemv_q4k_q8_1_v2_group<2>,
                                   fused_dequant_gemv_q4k_q8_1_group<2>>(
        data0, data1, act_q8_1, output0, N0, output1, N1, M, K, stream);
  }
  return DispatchQ8_1GemvPairQ4KSpecialized(data0, data1, act_q8_1, output0,
                                            N0, output1, N1, M, K, stream);
}

struct Q8_1DispatchEntry {
  Q8_1DispatchFn fn;
  const char *name;
};

struct Q8_1DispatchPairEntry {
  Q8_1DispatchPairFn fn;
  const char *name;
};

struct Q8_1DispatchTripleEntry {
  Q8_1DispatchTripleFn fn;
  const char *name;
};

const Q8_1DispatchEntry &GetQ8_1DispatchEntry(GGUF::TensorType qtype) {
  static const bool dp4a = GetGpuProfile().has_dp4a;
  static const Q8_1DispatchEntry table[kMaxTensorType] = {
      {nullptr, nullptr}, // 0: F32
      {nullptr, nullptr}, // 1: F16
      {nullptr, nullptr}, // 2: Q4_0
      {nullptr, nullptr}, // 3: Q4_1
      {nullptr, nullptr}, // 4: (unused)
      {nullptr, nullptr}, // 5: (unused)
      {nullptr, nullptr}, // 6: Q5_0
      {nullptr, nullptr}, // 7: Q5_1
      {dp4a ? DispatchQ8_1Gemv<block_q8_0, fused_dequant_gemv_q8_0_q8_1>
            : nullptr,
       "Q8_0"},           // 8
      {nullptr, nullptr}, // 9: Q8_1
      {nullptr, nullptr}, // 10: Q2_K
      {nullptr, nullptr}, // 11: Q3_K
      {dp4a ? DispatchQ8_1GemvQ4KV2 : nullptr,
       "Q4_K"},           // 12
      {nullptr, nullptr}, // 13: Q5_K
      {dp4a ? DispatchQ8_1GemvQ6KV2 : nullptr,
       "Q6_K"},           // 14
      {dp4a ? DispatchQ8_1Gemv<block_q8_k, fused_dequant_gemv_q8k_q8_1>
            : nullptr,
       "Q8_K"},           // 15
  };

  static const Q8_1DispatchEntry empty = {nullptr, nullptr};
  auto idx = static_cast<uint32_t>(qtype);
  if (idx >= kMaxTensorType)
    return empty;
  return table[idx];
}

const Q8_1DispatchPairEntry &
GetQ8_1DispatchPairEntry(GGUF::TensorType qtype) {
  static const bool dp4a = GetGpuProfile().has_dp4a;
  static const Q8_1DispatchPairEntry table[kMaxTensorType] = {
      {nullptr, nullptr}, // 0: F32
      {nullptr, nullptr}, // 1: F16
      {nullptr, nullptr}, // 2: Q4_0
      {nullptr, nullptr}, // 3: Q4_1
      {nullptr, nullptr}, // 4: (unused)
      {nullptr, nullptr}, // 5: (unused)
      {nullptr, nullptr}, // 6: Q5_0
      {nullptr, nullptr}, // 7: Q5_1
      {dp4a ? DispatchQ8_1GemvPair<block_q8_0,
                                     fused_dequant_gemv_q8_0_q8_1_group<2>>
            : nullptr,
       "Q8_0"},           // 8
      {nullptr, nullptr}, // 9: Q8_1
      {nullptr, nullptr}, // 10: Q2_K
      {nullptr, nullptr}, // 11: Q3_K
      {dp4a ? DispatchQ8_1GemvPairQ4KV2
            : nullptr,
       "Q4_K"},           // 12
      {nullptr, nullptr}, // 13: Q5_K
      {dp4a ? DispatchQ8_1GemvPairV2<
                    block_q6_k, fused_dequant_gemv_q6k_q8_1_v2_group<2>,
                    fused_dequant_gemv_q6k_q8_1_group<2>>
            : nullptr,
       "Q6_K"},           // 14
      {dp4a ? DispatchQ8_1GemvPair<block_q8_k,
                                     fused_dequant_gemv_q8k_q8_1_group<2>>
            : nullptr,
       "Q8_K"},           // 15
  };

  static const Q8_1DispatchPairEntry empty = {nullptr, nullptr};
  auto idx = static_cast<uint32_t>(qtype);
  if (idx >= kMaxTensorType)
    return empty;
  return table[idx];
}

const Q8_1DispatchTripleEntry &
GetQ8_1DispatchTripleEntry(GGUF::TensorType qtype) {
  static const bool dp4a = GetGpuProfile().has_dp4a;
  static const Q8_1DispatchTripleEntry table[kMaxTensorType] = {
      {nullptr, nullptr}, // 0: F32
      {nullptr, nullptr}, // 1: F16
      {nullptr, nullptr}, // 2: Q4_0
      {nullptr, nullptr}, // 3: Q4_1
      {nullptr, nullptr}, // 4: (unused)
      {nullptr, nullptr}, // 5: (unused)
      {nullptr, nullptr}, // 6: Q5_0
      {nullptr, nullptr}, // 7: Q5_1
      {dp4a ? DispatchQ8_1GemvTriple<block_q8_0,
                                       fused_dequant_gemv_q8_0_q8_1_group<3>>
            : nullptr,
       "Q8_0"},           // 8
      {nullptr, nullptr}, // 9: Q8_1
      {nullptr, nullptr}, // 10: Q2_K
      {nullptr, nullptr}, // 11: Q3_K
      {dp4a ? DispatchQ8_1GemvTripleV2<
                    block_q4_k, fused_dequant_gemv_q4k_q8_1_v2_group<3>,
                    fused_dequant_gemv_q4k_q8_1_group<3>>
            : nullptr,
       "Q4_K"},           // 12
      {nullptr, nullptr}, // 13: Q5_K
      {dp4a ? DispatchQ8_1GemvTripleV2<
                    block_q6_k, fused_dequant_gemv_q6k_q8_1_v2_group<3>,
                    fused_dequant_gemv_q6k_q8_1_group<3>>
            : nullptr,
       "Q6_K"},           // 14
      {dp4a ? DispatchQ8_1GemvTriple<block_q8_k,
                                       fused_dequant_gemv_q8k_q8_1_group<3>>
            : nullptr,
       "Q8_K"},           // 15
  };

  static const Q8_1DispatchTripleEntry empty = {nullptr, nullptr};
  auto idx = static_cast<uint32_t>(qtype);
  if (idx >= kMaxTensorType)
    return empty;
  return table[idx];
}

} // namespace

void FusedQuantGemm::QuantizeRowQ8_1(const half *input, void *output_q8_1,
                                      int M, int K, cudaStream_t stream) {
  using namespace runtime::cuda::native;
  auto *output = static_cast<block_q8_1 *>(output_q8_1);
  quantize_row_q8_1_kernel<<<M, kGemvThreadsPerBlock, 0, stream>>>(
      input, output, K);
}

void FusedQuantGemm::SiluMulQuantizeQ8_1(const half *gate, const half *up,
                                         void *output_q8_1, int M, int K,
                                         cudaStream_t stream) {
  using namespace runtime::cuda::native;
  auto *output = static_cast<block_q8_1 *>(output_q8_1);
  silu_mul_quantize_q8_1_kernel<<<M, kGemvThreadsPerBlock, 0, stream>>>(
      gate, up, output, K);
}

void FusedQuantGemm::FusedRmsNormQuantizeQ8_1(
    const half *residual, const half *norm_weight, void *output_q8_1, int M,
    int K, float rms_norm_eps, cudaStream_t stream) {
  using namespace runtime::cuda::native;
  auto *output = static_cast<block_q8_1 *>(output_q8_1);
  size_t smem = static_cast<size_t>(K) * sizeof(float) +
                kGemvWarpsPerBlock * sizeof(float);
  fused_rmsnorm_quantize_q8_1_kernel<<<M, kGemvThreadsPerBlock, smem,
                                        stream>>>(residual, norm_weight,
                                                   output, K, rms_norm_eps);
}

bool FusedQuantGemm::SupportsQ8_1Activations(int quant_type) {
  auto qtype = static_cast<GGUF::TensorType>(quant_type);
  return GetQ8_1DispatchEntry(qtype).fn != nullptr;
}

bool FusedQuantGemm::BuildDownProjMmqLayout(const QuantizedWeightInfo &weight,
                                            int rows, int cols,
                                            MmqWeightInfo *layout,
                                            cudaStream_t stream) {
  if (!layout || !weight.data || rows <= 0 || cols <= 0 || cols % QK_K != 0) {
    return false;
  }
  if (!SupportsDownProjMmq(weight.quant_type)) {
    return false;
  }

  const int tile_cols = FusedQuantGemm::kDownProjMmqTileCols;
  const int num_super_blocks = cols / QK_K;
  const int tile_count = (rows + tile_cols - 1) / tile_cols;
  const size_t transformed_blocks =
      static_cast<size_t>(tile_count) * num_super_blocks * tile_cols;

  constexpr int kThreads = 256;
  const int blocks =
      static_cast<int>((transformed_blocks + kThreads - 1) / kThreads);

  void *transformed = nullptr;
  const auto qtype = static_cast<GGUF::TensorType>(weight.quant_type);
  switch (qtype) {
  case GGUF::TensorType::Q4_K: {
    block_q4_k *typed = nullptr;
    if (cudaMalloc(reinterpret_cast<void **>(&typed),
                   transformed_blocks * sizeof(block_q4_k)) != cudaSuccess) {
      return false;
    }
    transform_downproj_mmq_layout<<<blocks, kThreads, 0, stream>>>(
        static_cast<const block_q4_k *>(weight.data), typed, rows,
        num_super_blocks, tile_cols);
    transformed = typed;
    break;
  }
  case GGUF::TensorType::Q6_K: {
    block_q6_k *typed = nullptr;
    if (cudaMalloc(reinterpret_cast<void **>(&typed),
                   transformed_blocks * sizeof(block_q6_k)) != cudaSuccess) {
      return false;
    }
    transform_downproj_mmq_layout<<<blocks, kThreads, 0, stream>>>(
        static_cast<const block_q6_k *>(weight.data), typed, rows,
        num_super_blocks, tile_cols);
    transformed = typed;
    break;
  }
  default:
    return false;
  }

  if (cudaPeekAtLastError() != cudaSuccess ||
      cudaStreamSynchronize(stream) != cudaSuccess) {
    cudaFree(transformed);
    return false;
  }

  *layout = {transformed, weight.quant_type, rows, cols, tile_cols};
  return true;
}

void FusedQuantGemm::DestroyDownProjMmqLayout(const MmqWeightInfo &layout) {
  if (layout.data) {
    cudaFree(const_cast<void *>(layout.data));
  }
}

bool FusedQuantGemm::DownProjMmq(const MmqWeightInfo &weight,
                                 const void *act_q8_1, half *output, int M,
                                 int N, int K, cudaStream_t stream,
                                 const NativeExecutionPolicy *policy) {
  ScopedExecutionPolicyOverride scoped(policy);
  if (!DownProjMmqEnabled() || !weight.data || !act_q8_1 || !output || M <= 0 ||
      N <= 0 || K <= 0 || N != weight.rows || K != weight.cols ||
      weight.tile_cols <= 0 || !SupportsDownProjMmq(weight.quant_type)) {
    return false;
  }

  const int threshold = GetDownProjMmqThreshold(weight.quant_type, M, N, K);
  if (M > threshold || N < weight.tile_cols || K % QK8_1 != 0) {
    return false;
  }

  cudaDeviceProp prop{};
  int device = 0;
  cudaGetDevice(&device);
  const size_t shared_bytes =
      static_cast<size_t>(K / QK8_1) * sizeof(block_q8_1);
  if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
    return false;
  }
  const size_t shared_limit = static_cast<size_t>(
      std::max(prop.sharedMemPerBlock, prop.sharedMemPerBlockOptin));
  if (
      shared_bytes > shared_limit) {
    return false;
  }

  auto qtype = static_cast<GGUF::TensorType>(weight.quant_type);
  const auto &entry = GetDownProjMmqDispatchEntry(qtype);
  if (!entry.fn) {
    return false;
  }

  static bool logged[kMaxTensorType] = {};
  auto idx = static_cast<uint32_t>(qtype);
  if (idx < kMaxTensorType && !logged[idx] && entry.name) {
    logged[idx] = true;
    log::Info("fused_quant_gemm",
              std::string("Using MMQ-style down-proj kernel for ") +
                  entry.name + " (M=" + std::to_string(M) +
                  ", N=" + std::to_string(N) +
                  ", K=" + std::to_string(K) +
                  ", tile_cols=" + std::to_string(weight.tile_cols) + ")");
  }

  return entry.fn(weight.data, act_q8_1, output, M, N, K, weight.tile_cols,
                  stream);
}

bool FusedQuantGemm::GemvQ8_1(const QuantizedWeightInfo &weight,
                              const void *act_q8_1, half *output, int M, int N,
                              int K, cudaStream_t stream,
                              const NativeExecutionPolicy *policy) {
  ScopedExecutionPolicyOverride scoped(policy);
  if (!weight.data || weight.quant_type < 0 || !act_q8_1)
    return false;
  if (ResolveExecutionPolicy(policy).disable_fused_gemv)
    return false;

  auto qtype = static_cast<GGUF::TensorType>(weight.quant_type);
  const auto &entry = GetQ8_1DispatchEntry(qtype);
  if (!entry.fn)
    return false;

  const FusedDispatchGeometry geometry{M, N, K, 1, true, false};
  if (!ShouldUseFusedPath(weight.quant_type, geometry))
    return false;

  static bool logged[kMaxTensorType] = {};
  auto idx = static_cast<uint32_t>(qtype);
  if (idx < kMaxTensorType && !logged[idx] && entry.name) {
    logged[idx] = true;
    const bool v2 = UseV2(K) && (qtype == GGUF::TensorType::Q4_K ||
                                 qtype == GGUF::TensorType::Q6_K);
    if (v2) {
      log::Info("fused_quant_gemm",
                std::string("Using cooperative-warp v2 Q8_1 GEMV for ") +
                    entry.name + " (M=" + std::to_string(M) + ", N=" +
                    std::to_string(N) + ", K=" + std::to_string(K) + ")");
    } else if (ShouldUseSpecializedQ8_1DownProjHotPathImpl(weight.quant_type,
                                                           geometry)) {
      log::Info("fused_quant_gemm",
                std::string("Using fixed-block Q8_1 down-proj kernel for ") +
                    entry.name + " (M=" + std::to_string(M) + ", N=" +
                    std::to_string(N) + ", K=" + std::to_string(K) + ")");
    } else {
      log::Info("fused_quant_gemm",
                std::string("Using Q8_1 activation GEMV for ") + entry.name +
                    " (M=" + std::to_string(M) + ", N=" + std::to_string(N) +
                    ", K=" + std::to_string(K) + ")");
    }
  }

  return entry.fn(weight.data, act_q8_1, output, M, N, K, stream);
}

bool FusedQuantGemm::GemvQ8_1Pair(
    const std::array<PackedProjectionSpec, 2> &projections,
    const void *act_q8_1, int M, int K, cudaStream_t stream,
    const NativeExecutionPolicy *policy) {
  ScopedExecutionPolicyOverride scoped(policy);
  if (!act_q8_1 || M <= 0 || K <= 0)
    return false;

  const int quant_type = projections[0].weight.quant_type;
  for (const auto &p : projections) {
    if (!p.weight.data || p.output == nullptr || p.output_cols <= 0 ||
        p.weight.quant_type != quant_type)
      return false;
  }

  auto qtype = static_cast<GGUF::TensorType>(quant_type);
  const auto &entry = GetQ8_1DispatchPairEntry(qtype);
  if (!entry.fn)
    return false;

  auto idx = static_cast<uint32_t>(qtype);
  const FusedDispatchGeometry geometry{M,
                                       std::max(projections[0].output_cols,
                                                projections[1].output_cols),
                                       K, 2, true, false};
  if (ShouldUseSpecializedQ8_1GroupedFastPath(quant_type, geometry)) {
    static bool specialized_logged[kMaxTensorType] = {};
    if (idx < kMaxTensorType && !specialized_logged[idx] && entry.name) {
      specialized_logged[idx] = true;
      log::Info("fused_quant_gemm",
                std::string("Using fixed-block grouped Q8_1 pair kernel for ") +
                    entry.name + " (M=" + std::to_string(M) + ", N=" +
                    std::to_string(projections[0].output_cols) + "/" +
                    std::to_string(projections[1].output_cols) + ", K=" +
                    std::to_string(K) + ")");
    }
  } else if (M >= 4 && (qtype == GGUF::TensorType::Q4_K ||
                 qtype == GGUF::TensorType::Q6_K)) {
    static bool rowquad_logged[kMaxTensorType] = {};
    if (idx < kMaxTensorType && !rowquad_logged[idx] && entry.name) {
      rowquad_logged[idx] = true;
      log::Info("fused_quant_gemm",
                std::string("Using batched row-quad grouped Q8_1 pair kernel "
                            "for ") +
                    entry.name + " (M=" + std::to_string(M) + ", N=" +
                    std::to_string(projections[0].output_cols) + "/" +
                    std::to_string(projections[1].output_cols) + ", K=" +
                    std::to_string(K) + ")");
    }
  } else if (M > 1 && (qtype == GGUF::TensorType::Q4_K ||
                       qtype == GGUF::TensorType::Q6_K)) {
    static bool rowpair_logged[kMaxTensorType] = {};
    if (idx < kMaxTensorType && !rowpair_logged[idx] && entry.name) {
      rowpair_logged[idx] = true;
      log::Info("fused_quant_gemm",
                std::string("Using batched row-pair grouped Q8_1 pair kernel "
                            "for ") +
                    entry.name + " (M=" + std::to_string(M) + ", N=" +
                    std::to_string(projections[0].output_cols) + "/" +
                    std::to_string(projections[1].output_cols) + ", K=" +
                    std::to_string(K) + ")");
    }
  }

  return entry.fn(projections[0].weight.data, projections[1].weight.data,
                  act_q8_1, projections[0].output, projections[0].output_cols,
                  projections[1].output, projections[1].output_cols, M, K,
                  stream);
}

bool FusedQuantGemm::GemvQ8_1Triple(
    const std::array<PackedProjectionSpec, 3> &projections,
    const void *act_q8_1, int M, int K, cudaStream_t stream,
    const NativeExecutionPolicy *policy) {
  ScopedExecutionPolicyOverride scoped(policy);
  if (!act_q8_1 || M <= 0 || K <= 0)
    return false;

  const int quant_type = projections[0].weight.quant_type;
  for (const auto &p : projections) {
    if (!p.weight.data || p.output == nullptr || p.output_cols <= 0 ||
        p.weight.quant_type != quant_type)
      return false;
  }

  auto qtype = static_cast<GGUF::TensorType>(quant_type);
  const auto &entry = GetQ8_1DispatchTripleEntry(qtype);
  if (!entry.fn)
    return false;

  auto idx = static_cast<uint32_t>(qtype);
  if (UseV2(K) &&
      (qtype == GGUF::TensorType::Q4_K || qtype == GGUF::TensorType::Q6_K)) {
    static bool v2_logged[kMaxTensorType] = {};
    if (idx < kMaxTensorType && !v2_logged[idx] && entry.name) {
      v2_logged[idx] = true;
      log::Info("fused_quant_gemm",
                std::string(
                    "Using cooperative-warp v2 grouped Q8_1 triple for ") +
                    entry.name + " (M=" + std::to_string(M) + ", N=" +
                    std::to_string(projections[0].output_cols) + "/" +
                    std::to_string(projections[1].output_cols) + "/" +
                    std::to_string(projections[2].output_cols) + ", K=" +
                    std::to_string(K) + ")");
    }
  }

  return entry.fn(
      projections[0].weight.data, projections[1].weight.data,
      projections[2].weight.data, act_q8_1, projections[0].output,
      projections[0].output_cols, projections[1].output,
      projections[1].output_cols, projections[2].output,
      projections[2].output_cols, M, K, stream);
}

} // namespace inferflux
