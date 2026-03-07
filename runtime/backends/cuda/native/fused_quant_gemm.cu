#include "runtime/backends/cuda/native/fused_quant_gemm.h"
#include "runtime/backends/cuda/native/gguf_util.h"
#include "runtime/backends/cuda/native/kernels/fused_dequant_gemm.cuh"
#include "runtime/backends/cuda/native/kernels/fused_dequant_gemv.cuh"
#include "server/logging/logger.h"

#include <algorithm>
#include <cmath>
#include <mutex>

namespace inferflux {

using namespace runtime::cuda::native;

namespace {

// ============================================================================
// GPU-adaptive threshold computation
// ============================================================================

struct GpuProfile {
  int sm_major{0};
  int sm_minor{0};
  float memory_bandwidth_gb_s{0}; // GB/s
  int sm_count{0};
  bool initialized{false};

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
      profile.initialized = true;

      log::Info(
          "fused_quant_gemm",
          "GPU profile: SM " + std::to_string(prop.major) + "." +
              std::to_string(prop.minor) + ", " +
              std::to_string(profile.sm_count) + " SMs, " +
              std::to_string(static_cast<int>(profile.memory_bandwidth_gb_s)) +
              " GB/s bandwidth, base_threshold=" +
              std::to_string(profile.base_threshold()));
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

template <typename BlockType,
          void (*GemvKernel)(const BlockType *, const half *, half *, int, int),
          void (*GemmKernel)(const BlockType *, const half *, half *, int, int,
                             int)>
bool DispatchFused(const void *data, const half *activation, half *output,
                   int M, int N, int K, cudaStream_t stream) {
  auto *w = static_cast<const BlockType *>(data);
  if (M == 1) {
    int grid = (N + kGemvWarpsPerBlock - 1) / kGemvWarpsPerBlock;
    GemvKernel<<<grid, kGemvThreadsPerBlock, 0, stream>>>(w, activation, output,
                                                          N, K);
  } else {
    int grid = (N + kTileN - 1) / kTileN;
    GemmKernel<<<grid, kSmallBatchThreads, 0, stream>>>(w, activation, output,
                                                        M, N, K);
  }
  return true;
}

struct DispatchEntry {
  FusedDispatchFn fn;
  const char *name;
};

constexpr int kMaxTensorType = 16;

const DispatchEntry &GetDispatchEntry(GGUF::TensorType qtype) {
  static const DispatchEntry table[kMaxTensorType] = {
      {nullptr, nullptr}, // 0: F32
      {nullptr, nullptr}, // 1: F16
      {nullptr, nullptr}, // 2: Q4_0
      {nullptr, nullptr}, // 3: Q4_1
      {nullptr, nullptr}, // 4: (unused)
      {nullptr, nullptr}, // 5: (unused)
      {nullptr, nullptr}, // 6: Q5_0
      {nullptr, nullptr}, // 7: Q5_1
      {DispatchFused<block_q8_0, fused_dequant_gemv_q8_0,
                     fused_dequant_gemm_q8_0>,
       "Q8_0"},           // 8
      {nullptr, nullptr}, // 9: Q8_1
      {nullptr, nullptr}, // 10: Q2_K
      {nullptr, nullptr}, // 11: Q3_K
      {DispatchFused<block_q4_k, fused_dequant_gemv_q4k,
                     fused_dequant_gemm_q4k>,
       "Q4_K"},           // 12
      {nullptr, nullptr}, // 13: Q5_K
      {DispatchFused<block_q6_k, fused_dequant_gemv_q6k,
                     fused_dequant_gemm_q6k>,
       "Q6_K"}, // 14
      {DispatchFused<block_q8_k, fused_dequant_gemv_q8k,
                     fused_dequant_gemm_q8k>,
       "Q8_K"}, // 15
  };

  static const DispatchEntry empty = {nullptr, nullptr};
  auto idx = static_cast<uint32_t>(qtype);
  if (idx >= kMaxTensorType)
    return empty;
  return table[idx];
}

} // namespace

int FusedQuantGemm::GetAdaptiveThreshold(int quant_type) {
  auto qtype = static_cast<GGUF::TensorType>(quant_type);
  auto &gpu = GetGpuProfile();
  float bpw = BitsPerWeight(qtype);
  int base = gpu.initialized ? gpu.base_threshold() : 8;
  return ComputeThreshold(base, bpw);
}

bool FusedQuantGemm::Gemv(const QuantizedWeightInfo &weight,
                          const half *activation, half *output, int M, int N,
                          int K, cudaStream_t stream) {
  if (!weight.data || weight.quant_type < 0)
    return false;

  auto qtype = static_cast<GGUF::TensorType>(weight.quant_type);
  const auto &entry = GetDispatchEntry(qtype);

  if (!entry.fn)
    return false; // Unsupported quant type

  // Adaptive threshold: fused vs cuBLAS crossover depends on GPU and quant type
  int threshold = GetAdaptiveThreshold(weight.quant_type);
  if (M > threshold)
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

} // namespace inferflux
