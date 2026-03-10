#include "runtime/backends/cuda/common/dtype_traits.cuh"
#include "runtime/backends/cuda/kernels/flash_attention.cuh"
#include "runtime/backends/cuda/native/cuda_kernels.cuh"
#include "runtime/backends/cuda/native/fused_quant_gemm.h"
#include "runtime/backends/cuda/native/gguf_util.h"
#include "runtime/backends/cuda/native/llama_forward.h"
#include "runtime/backends/cuda/native/native_dispatch_policy.h"
#include "runtime/backends/cuda/native/model_loader.h"
#include "runtime/backends/cuda/native/nvtx_scoped.h"

#include "server/logging/logger.h"
#include "server/metrics/metrics.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <unordered_map>
#include <vector>

namespace inferflux {

namespace {

// Phase timing: sync-based per-phase breakdown when
// INFERFLUX_NATIVE_PHASE_TIMING=1 Serializes GPU pipeline — for
// debugging/profiling only, not production.
struct PhaseTiming {
  bool enabled{false};
  cudaStream_t stream{nullptr};
  std::chrono::steady_clock::time_point last;
  double embed_ms{0}, qkv_ms{0}, rope_ms{0}, kv_ms{0}, attn_ms{0};
  double o_proj_ms{0}, ffn_proj_ms{0}, ffn_silu_ms{0}, ffn_down_ms{0},
      lm_head_ms{0};
  int forward_count{0};

  void Begin(cudaStream_t s, bool should_enable) {
    if (!should_enable)
      return;
    enabled = true;
    stream = s;
    embed_ms = qkv_ms = rope_ms = kv_ms = attn_ms = 0;
    o_proj_ms = ffn_proj_ms = ffn_silu_ms = ffn_down_ms = lm_head_ms = 0;
    cudaStreamSynchronize(stream);
    last = std::chrono::steady_clock::now();
  }

  double Mark() {
    if (!enabled)
      return 0;
    cudaStreamSynchronize(stream);
    auto now = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(now - last).count();
    last = now;
    return ms;
  }

  void Report(int num_layers, int token_count) {
    if (!enabled)
      return;
    ++forward_count;
    const double ffn_ms = ffn_proj_ms + ffn_silu_ms + ffn_down_ms;
    double total = embed_ms + qkv_ms + rope_ms + kv_ms + attn_ms + o_proj_ms +
                   ffn_ms + lm_head_ms;
    // Print every forward pass for first 5, then every 10th
    if (forward_count <= 5 || forward_count % 10 == 0) {
      fprintf(stderr,
              "[phase_timing] #%d L=%d tokens=%d embed=%.2f qkv=%.2f rope=%.2f "
              "kv=%.2f attn=%.2f o_proj=%.2f ffn_proj=%.2f ffn_silu=%.2f "
              "ffn_down=%.2f ffn=%.2f lm_head=%.2f total=%.2f ms\n",
              forward_count, num_layers, token_count, embed_ms, qkv_ms, rope_ms,
              kv_ms, attn_ms, o_proj_ms, ffn_proj_ms, ffn_silu_ms, ffn_down_ms,
              ffn_ms, lm_head_ms, total);
    }
  }
};

// Debug: dump top-K logits to stderr when INFERFLUX_DEBUG_LOGITS=1
void DebugDumpLogits(const float *d_logits, int vocab_size,
                     const std::vector<int> &token_ids, int n_past,
                     cudaStream_t stream,
                     const NativeExecutionPolicy *policy = nullptr) {
  if (!ResolveNativeExecutionPolicy(policy).debug_logits)
    return;
  constexpr int TOP_N = 10;
  std::vector<float> h_logits(vocab_size);
  cudaMemcpyAsync(h_logits.data(), d_logits, vocab_size * sizeof(float),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  // Find top-N by value
  std::vector<std::pair<float, int>> scored(vocab_size);
  for (int i = 0; i < vocab_size; ++i)
    scored[i] = {h_logits[i], i};
  std::partial_sort(scored.begin(), scored.begin() + TOP_N, scored.end(),
                    [](auto &a, auto &b) { return a.first > b.first; });

  fprintf(stderr, "[DEBUG_LOGITS] tokens=[");
  for (size_t i = 0; i < token_ids.size(); ++i)
    fprintf(stderr, "%s%d", i ? "," : "", token_ids[i]);
  fprintf(stderr, "] n_past=%d top-%d:", n_past, TOP_N);
  for (int i = 0; i < TOP_N; ++i)
    fprintf(stderr, " [%d]=%.4f", scored[i].second, scored[i].first);

  // Also check for NaN/Inf
  int nan_count = 0, inf_count = 0, zero_count = 0;
  for (int i = 0; i < vocab_size; ++i) {
    if (std::isnan(h_logits[i]))
      nan_count++;
    if (std::isinf(h_logits[i]))
      inf_count++;
    if (h_logits[i] == 0.0f)
      zero_count++;
  }
  fprintf(stderr, " (nan=%d inf=%d zero=%d/%d)\n", nan_count, inf_count,
          zero_count, vocab_size);
}

// Debug: dump hidden state stats
void DebugDumpHidden(const char *label, const void *d_data, int count,
                     cudaStream_t stream,
                     const NativeExecutionPolicy *policy = nullptr) {
  if (!ResolveNativeExecutionPolicy(policy).debug_logits)
    return;
  // Read as half, convert to float
  std::vector<half> h_data(count);
  cudaMemcpyAsync(h_data.data(), d_data, count * sizeof(half),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  float min_v = 1e30f, max_v = -1e30f, sum = 0.0f;
  int nan_count = 0;
  for (int i = 0; i < count; ++i) {
    float v = __half2float(h_data[i]);
    if (std::isnan(v)) {
      nan_count++;
      continue;
    }
    min_v = std::min(min_v, v);
    max_v = std::max(max_v, v);
    sum += v;
  }
  fprintf(stderr,
          "[DEBUG_HIDDEN] %s: count=%d min=%.6f max=%.6f mean=%.6f nan=%d\n",
          label, count, min_v, max_v, sum / count, nan_count);
}

// One-shot per-projection path logger. Logs which GEMM path (fused vs cuBLAS)
// is taken for each projection name on first invocation only.
void LogGemmPath(const char *proj_name, bool fused) {
  // Hash projection name pointer (string literals have fixed addresses)
  static std::unordered_map<const char *, bool> logged;
  if (logged.count(proj_name))
    return;
  logged[proj_name] = true;
  log::Info("llama_forward", std::string(proj_name) +
                                 (fused ? ": using fused dequant-GEMV"
                                        : ": using cuBLAS (dequantized FP16)"));
}

void LogPackedGemmPath(const char *proj_name, const char *label) {
  static std::unordered_map<std::string, bool> logged;
  const std::string key =
      std::string(proj_name ? proj_name : "") + "|" + (label ? label : "");
  if (logged.count(key)) {
    return;
  }
  logged[key] = true;
  log::Info("llama_forward", std::string(proj_name) + ": " + label);
}

thread_local bool g_allow_fused_quantized_matmul = true;

class ScopedFusedMatmulPolicy {
public:
  explicit ScopedFusedMatmulPolicy(bool allow)
      : prev_(g_allow_fused_quantized_matmul) {
    g_allow_fused_quantized_matmul = allow;
  }
  ~ScopedFusedMatmulPolicy() { g_allow_fused_quantized_matmul = prev_; }

private:
  bool prev_{true};
};

// Fused dequant-GEMV dispatch: only valid for half (FP16) type.
// Returns true if the fused kernel was launched.
template <typename T>
bool TryFusedGemv(const QuantizedWeightInfo &, const T *, T *, int, int, int,
                  cudaStream_t, const char * = nullptr,
                  const NativeExecutionPolicy * = nullptr) {
  return false; // BF16 and other types: no fused path
}

template <>
bool TryFusedGemv<half>(const QuantizedWeightInfo &raw, const half *input,
                        half *output, int M, int N, int K, cudaStream_t stream,
                        const char *proj_name,
                        const NativeExecutionPolicy *policy) {
  const auto &policy_ref = ResolveNativeExecutionPolicy(policy);
  if (ForceCublasRequested(policy_ref) || !g_allow_fused_quantized_matmul)
    return false;
  bool ok =
      raw.data &&
      FusedQuantGemm::Gemv(raw, input, output, M, N, K, stream, policy);
  if (proj_name)
    LogGemmPath(proj_name, ok);
  return ok;
}

// Fused RmsNorm+GEMV dispatch: computes normalization inside the GEMV kernel,
// eliminating the standalone RmsNorm kernel launch and d_norm_out_ round-trip.
// Only valid for half (FP16) type.
template <typename T>
bool TryFusedRmsNormGemv(const QuantizedWeightInfo &, const T *, const T *, T *,
                         int, int, int, float, cudaStream_t,
                         const char * = nullptr,
                         const NativeExecutionPolicy * = nullptr) {
  return false; // BF16 and other types: no fused path
}

template <>
bool TryFusedRmsNormGemv<half>(const QuantizedWeightInfo &raw,
                               const half *residual, const half *norm_weight,
                               half *output, int M, int N, int K, float eps,
                               cudaStream_t stream, const char *proj_name,
                               const NativeExecutionPolicy *policy) {
  const auto &policy_ref = ResolveNativeExecutionPolicy(policy);
  if (ForceCublasRequested(policy_ref) || !g_allow_fused_quantized_matmul)
    return false;
  bool ok =
      raw.data && FusedQuantGemm::RmsNormGemv(raw, residual, norm_weight,
                                              output, M, N, K, eps, stream,
                                              policy);
  if (proj_name) {
    static std::unordered_map<const char *, bool> logged;
    if (!logged.count(proj_name)) {
      logged[proj_name] = true;
      log::Info("llama_forward",
                std::string(proj_name) +
                    (ok ? ": using fused RmsNorm+GEMV"
                        : ": using separate RmsNorm + GEMV/cuBLAS"));
    }
  }
  return ok;
}

template <typename T>
bool TryPackedGemv(const QuantizedWeightInfo &, const T *, T *, int8_t *,
                   float *, int, int, int, cudaStream_t,
                   const char * = nullptr,
                   const NativeExecutionPolicy * = nullptr) {
  return false;
}

template <>
bool TryPackedGemv<half>(const QuantizedWeightInfo &raw, const half *input,
                         half *output, int8_t *packed_activation,
                         float *packed_scales, int M, int N, int K,
                         cudaStream_t stream, const char *proj_name,
                         const NativeExecutionPolicy *policy) {
  const auto &policy_ref = ResolveNativeExecutionPolicy(policy);
  if (PackedActivationsDisabled(policy_ref) || !raw.data || !input || !output ||
      !packed_activation ||
      !packed_scales || M <= 0 || N <= 0 || K <= 0 ||
      !FusedQuantGemm::SupportsPackedActivations(raw.quant_type) ||
      !FusedQuantGemm::ShouldUseFusedPath(
          raw.quant_type, FusedDispatchGeometry{M, N, K, 1, true, false})) {
    return false;
  }

  if (cuda_kernel::QuantizeRowsSymmetric(input, packed_activation,
                                         packed_scales, M, K,
                                         stream) != cudaSuccess) {
    return false;
  }

  PackedActivationInfo packed{packed_activation, packed_scales};
  const bool ok =
      FusedQuantGemm::GemvPacked(raw, packed, output, M, N, K, stream, policy);
  if (ok && proj_name) {
    LogPackedGemmPath(proj_name, "using packed-activation fused dequant-GEMV");
  }
  return ok;
}

template <typename T>
bool TryPackedRmsNormGemv(const QuantizedWeightInfo &, const T *, const T *,
                          T *, int8_t *, float *, T *, int, int, int, float,
                          cudaStream_t, const char * = nullptr,
                          const NativeExecutionPolicy * = nullptr) {
  return false;
}

template <>
bool TryPackedRmsNormGemv<half>(const QuantizedWeightInfo &raw,
                                const half *residual, const half *norm_weight,
                                half *normalized, int8_t *packed_activation,
                                float *packed_scales, half *output, int M,
                                int N, int K, float eps, cudaStream_t stream,
                                const char *proj_name,
                                const NativeExecutionPolicy *policy) {
  const auto &policy_ref = ResolveNativeExecutionPolicy(policy);
  if (PackedActivationsDisabled(policy_ref) || !raw.data || !residual ||
      !norm_weight || !normalized ||
      !packed_activation || !packed_scales || !output || M <= 0 || N <= 0 ||
      K <= 0 || !FusedQuantGemm::SupportsPackedActivations(raw.quant_type) ||
      !FusedQuantGemm::ShouldUseFusedPath(
          raw.quant_type, FusedDispatchGeometry{M, N, K, 1, true, true})) {
    return false;
  }

  if (cuda_kernel::RmsNorm<half>(residual, norm_weight, normalized, M, K, eps,
                                 stream) != cudaSuccess) {
    return false;
  }

  if (cuda_kernel::QuantizeRowsSymmetric(normalized, packed_activation,
                                         packed_scales, M, K,
                                         stream) != cudaSuccess) {
    return false;
  }

  PackedActivationInfo packed{packed_activation, packed_scales};
  const bool ok =
      FusedQuantGemm::GemvPacked(raw, packed, output, M, N, K, stream, policy);
  if (ok && proj_name) {
    LogPackedGemmPath(proj_name,
                      "using packed-activation fused RmsNorm+dequant-GEMV");
  }
  return ok;
}

template <typename T> struct PackedProjectionPlan {
  QuantizedWeightInfo raw;
  T *output{nullptr};
  int output_cols{0};
};

template <size_t GroupSize, typename T>
bool TryPackedProjectionGroup(
    const std::array<PackedProjectionPlan<T>, GroupSize> &, const T *,
    const T *, T *, int8_t *, float *, int, int, float, cudaStream_t,
    const NativeExecutionPolicy * = nullptr) {
  return false;
}

template <size_t GroupSize>
bool TryPackedProjectionGroup(
    const std::array<PackedProjectionPlan<half>, GroupSize> &plans,
    const half *residual, const half *norm_weight, half *normalized,
    int8_t *packed_activation, float *packed_scales, int M, int K, float eps,
    cudaStream_t stream, const NativeExecutionPolicy *policy) {
  const auto &policy_ref = ResolveNativeExecutionPolicy(policy);
  if (PackedActivationsDisabled(policy_ref) || !residual || !norm_weight ||
      !normalized ||
      !packed_activation || !packed_scales || M <= 0 || K <= 0) {
    return false;
  }

  std::array<int, GroupSize> quant_types{};
  std::array<int, GroupSize> output_cols{};
  std::array<bool, GroupSize> pair_ready{};
  std::array<bool, GroupSize> triple_ready{};
  for (size_t i = 0; i < GroupSize; ++i) {
    const auto &plan = plans[i];
    if (!plan.raw.data || plan.output == nullptr || plan.output_cols <= 0 ||
        !FusedQuantGemm::SupportsPackedActivations(plan.raw.quant_type)) {
      return false;
    }
    quant_types[i] = plan.raw.quant_type;
    output_cols[i] = plan.output_cols;
    if (!FusedQuantGemm::ShouldUseFusedPath(
            plan.raw.quant_type,
            FusedDispatchGeometry{M, plan.output_cols, K, 1, true, true})) {
      return false;
    }
    pair_ready[i] = FusedQuantGemm::ShouldUseFusedPath(
        plan.raw.quant_type,
        FusedDispatchGeometry{M, plan.output_cols, K, 2, true, true});
    if constexpr (GroupSize == 3) {
      triple_ready[i] = FusedQuantGemm::ShouldUseFusedPath(
          plan.raw.quant_type,
          FusedDispatchGeometry{M, plan.output_cols, K, 3, true, true});
    }
  }

  if (cuda_kernel::RmsNorm<half>(residual, norm_weight, normalized, M, K, eps,
                                 stream) != cudaSuccess) {
    return false;
  }
  if (cuda_kernel::QuantizeRowsSymmetric(normalized, packed_activation,
                                         packed_scales, M, K,
                                         stream) != cudaSuccess) {
    return false;
  }

  PackedActivationInfo packed;
  packed.data = packed_activation;
  packed.row_scales = packed_scales;

  const auto grouping = SelectSharedActivationGrouping(
      quant_types, output_cols, pair_ready, triple_ready);
  std::array<bool, GroupSize> completed{};
  bool used_grouped = false;

  if constexpr (GroupSize == 3) {
    if (grouping.grouped_count == 3) {
      const std::array<PackedProjectionSpec, 3> grouped = {{
          {plans[0].raw, plans[0].output, plans[0].output_cols},
          {plans[1].raw, plans[1].output, plans[1].output_cols},
          {plans[2].raw, plans[2].output, plans[2].output_cols},
      }};
      if (FusedQuantGemm::GemvPackedTriple(grouped, packed, M, K, stream)) {
        LogPackedGemmPath("packed_triple", "using packed grouped triple GEMV");
        return true;
      }
    }
  }

  if (grouping.grouped_count == 2) {
    const int i = grouping.indices[0];
    const int j = grouping.indices[1];
    const std::array<PackedProjectionSpec, 2> grouped = {{
        {plans[i].raw, plans[i].output, plans[i].output_cols},
        {plans[j].raw, plans[j].output, plans[j].output_cols},
    }};
    if (FusedQuantGemm::GemvPackedPair(grouped, packed, M, K, stream)) {
      completed[i] = true;
      completed[j] = true;
      used_grouped = true;
    }
  }

  for (size_t i = 0; i < GroupSize; ++i) {
    if (completed[i]) {
      continue;
    }
    const auto &plan = plans[i];
    if (!FusedQuantGemm::GemvPacked(plan.raw, packed, plan.output, M,
                                    plan.output_cols, K, stream, policy)) {
      return false;
    }
  }
  if constexpr (GroupSize > 1) {
    if (used_grouped && grouping.grouped_count < static_cast<int>(GroupSize)) {
      LogPackedGemmPath("packed_mixed", "using packed grouped+individual GEMV");
    } else if (used_grouped) {
      LogPackedGemmPath("packed_pair", "using packed grouped pair GEMV");
    } else {
      LogPackedGemmPath("packed_individual",
                        "using packed individual GEMV (shared quantization)");
    }
  }
  return true;
}

// Q8_1 pre-quantized projection group dispatch.
// Quantizes activations once (fused with RmsNorm when norm_weight != nullptr),
// then dispatches grouped or individual Q8_1 GEMV kernels.
template <size_t GroupSize, typename T>
bool TryQ8_1ProjectionGroup(
    const std::array<PackedProjectionPlan<T>, GroupSize> &, const T *,
    const T *, void *, int, int, float, cudaStream_t,
    const NativeExecutionPolicy * = nullptr) {
  return false;
}

template <size_t GroupSize>
bool TryQ8_1ProjectionGroup(
    const std::array<PackedProjectionPlan<half>, GroupSize> &plans,
    const half *input, const half *norm_weight, void *act_q8_1, int M, int K,
    float eps, cudaStream_t stream, const NativeExecutionPolicy *policy) {
  const auto &policy_ref = ResolveNativeExecutionPolicy(policy);
  if (Q81ActivationsDisabled(policy_ref) || !input || !act_q8_1 || M <= 0 ||
      K <= 0) {
    return false;
  }

  std::array<int, GroupSize> quant_types{};
  std::array<int, GroupSize> output_cols{};
  std::array<bool, GroupSize> pair_ready{};
  std::array<bool, GroupSize> triple_ready{};
  for (size_t i = 0; i < GroupSize; ++i) {
    const auto &plan = plans[i];
    if (!plan.raw.data || plan.output == nullptr || plan.output_cols <= 0 ||
        !FusedQuantGemm::SupportsQ8_1Activations(plan.raw.quant_type)) {
      return false;
    }
    quant_types[i] = plan.raw.quant_type;
    output_cols[i] = plan.output_cols;
    if (!FusedQuantGemm::ShouldUseFusedPath(
            plan.raw.quant_type,
            FusedDispatchGeometry{M, plan.output_cols, K, 1, true,
                                  norm_weight != nullptr})) {
      return false;
    }
    pair_ready[i] = FusedQuantGemm::ShouldUseFusedPath(
        plan.raw.quant_type,
        FusedDispatchGeometry{M, plan.output_cols, K, 2, true,
                              norm_weight != nullptr});
    if constexpr (GroupSize == 3) {
      triple_ready[i] = FusedQuantGemm::ShouldUseFusedPath(
          plan.raw.quant_type,
          FusedDispatchGeometry{M, plan.output_cols, K, 3, true,
                                norm_weight != nullptr});
    }
  }

  // Quantize activations: fused RmsNorm+Quantize or standalone quantize
  if (norm_weight) {
    FusedQuantGemm::FusedRmsNormQuantizeQ8_1(input, norm_weight, act_q8_1, M, K,
                                             eps, stream);
  } else {
    FusedQuantGemm::QuantizeRowQ8_1(input, act_q8_1, M, K, stream);
  }

  const auto grouping = SelectSharedActivationGrouping(
      quant_types, output_cols, pair_ready, triple_ready);
  std::array<bool, GroupSize> completed{};
  bool used_grouped = false;

  if constexpr (GroupSize == 3) {
    if (grouping.grouped_count == 3) {
      const std::array<PackedProjectionSpec, 3> grouped = {{
          {plans[0].raw, plans[0].output, plans[0].output_cols},
          {plans[1].raw, plans[1].output, plans[1].output_cols},
          {plans[2].raw, plans[2].output, plans[2].output_cols},
      }};
      if (FusedQuantGemm::GemvQ8_1Triple(grouped, act_q8_1, M, K, stream,
                                         policy)) {
        LogPackedGemmPath("q8_1_triple", "using Q8_1 grouped triple GEMV");
        return true;
      }
    }
  }

  if (grouping.grouped_count == 2) {
    const int i = grouping.indices[0];
    const int j = grouping.indices[1];
    const std::array<PackedProjectionSpec, 2> grouped = {{
        {plans[i].raw, plans[i].output, plans[i].output_cols},
        {plans[j].raw, plans[j].output, plans[j].output_cols},
    }};
    if (FusedQuantGemm::GemvQ8_1Pair(grouped, act_q8_1, M, K, stream,
                                     policy)) {
      completed[i] = true;
      completed[j] = true;
      used_grouped = true;
    }
  }

  for (size_t i = 0; i < GroupSize; ++i) {
    if (completed[i]) {
      continue;
    }
    const auto &plan = plans[i];
    if (!FusedQuantGemm::GemvQ8_1(plan.raw, act_q8_1, plan.output, M,
                                  plan.output_cols, K, stream, policy)) {
      return false;
    }
  }
  if constexpr (GroupSize > 1) {
    if (used_grouped && grouping.grouped_count < static_cast<int>(GroupSize)) {
      LogPackedGemmPath("q8_1_mixed", "using Q8_1 grouped+individual GEMV");
    } else if (used_grouped) {
      LogPackedGemmPath("q8_1_pair", "using Q8_1 grouped pair GEMV");
    } else {
      LogPackedGemmPath("q8_1_individual",
                        "using Q8_1 individual GEMV (shared quantization)");
    }
  }
  return true;
}

// Standalone Q8_1 GEMV for single projections without norm fusion.
template <typename T>
bool TryMmqGemv(const MmqWeightInfo &, const T *, T *, void *, int, int, int,
                cudaStream_t, const char * = nullptr,
                const NativeExecutionPolicy * = nullptr) {
  return false;
}

template <>
bool TryMmqGemv<half>(const MmqWeightInfo &weight, const half *input,
                      half *output, void *act_q8_1, int M, int N, int K,
                      cudaStream_t stream, const char *proj_name,
                      const NativeExecutionPolicy *policy) {
  if (!weight.data || !input || !output || !act_q8_1 || M <= 0 || N <= 0 ||
      K <= 0 || N != weight.rows || K != weight.cols) {
    return false;
  }

  FusedQuantGemm::QuantizeRowQ8_1(input, act_q8_1, M, K, stream);
  const bool ok =
      FusedQuantGemm::DownProjMmq(weight, act_q8_1, output, M, N, K, stream,
                                  policy);
  if (ok && proj_name) {
    LogPackedGemmPath(proj_name, "using MMQ-style tiled Q8_1 down-proj");
  }
  return ok;
}

template <typename T>
bool TryMmqSiluMulGemv(const MmqWeightInfo &, const T *, const T *, T *, void *,
                       int, int, int, cudaStream_t,
                       const char * = nullptr,
                       const NativeExecutionPolicy * = nullptr) {
  return false;
}

template <>
bool TryMmqSiluMulGemv<half>(const MmqWeightInfo &weight, const half *gate,
                             const half *up, half *output, void *act_q8_1,
                             int M, int N, int K, cudaStream_t stream,
                             const char *proj_name,
                             const NativeExecutionPolicy *policy) {
  if (!weight.data || !gate || !up || !output || !act_q8_1 || M <= 0 ||
      N <= 0 || K <= 0 || N != weight.rows || K != weight.cols) {
    return false;
  }

  FusedQuantGemm::SiluMulQuantizeQ8_1(gate, up, act_q8_1, M, K, stream);
  const bool ok =
      FusedQuantGemm::DownProjMmq(weight, act_q8_1, output, M, N, K, stream,
                                  policy);
  if (ok && proj_name) {
    LogPackedGemmPath(proj_name,
                      "using MMQ-style tiled fused SwiGLU down-proj");
  }
  return ok;
}

template <typename T>
bool TryQ8_1Gemv(const QuantizedWeightInfo &, const T *, T *, void *, int, int,
                 int, cudaStream_t, const char * = nullptr,
                 const NativeExecutionPolicy * = nullptr) {
  return false;
}

template <>
bool TryQ8_1Gemv<half>(const QuantizedWeightInfo &raw, const half *input,
                       half *output, void *act_q8_1, int M, int N, int K,
                       cudaStream_t stream, const char *proj_name,
                       const NativeExecutionPolicy *policy) {
  const auto &policy_ref = ResolveNativeExecutionPolicy(policy);
  if (Q81ActivationsDisabled(policy_ref) || !raw.data || !input || !output ||
      !act_q8_1 || M <= 0 ||
      N <= 0 || K <= 0 ||
      !FusedQuantGemm::SupportsQ8_1Activations(raw.quant_type) ||
      !FusedQuantGemm::ShouldUseFusedPath(
          raw.quant_type, FusedDispatchGeometry{M, N, K, 1, true, false})) {
    return false;
  }

  FusedQuantGemm::QuantizeRowQ8_1(input, act_q8_1, M, K, stream);
  bool ok =
      FusedQuantGemm::GemvQ8_1(raw, act_q8_1, output, M, N, K, stream, policy);
  if (ok && proj_name) {
    LogPackedGemmPath(proj_name, "using Q8_1 pre-quantized GEMV");
  }
  return ok;
}

template <typename T>
bool TryPackedSiluMulGemv(const QuantizedWeightInfo &, const T *, const T *,
                          T *, int8_t *, float *, int, int, int, cudaStream_t,
                          const char * = nullptr,
                          const NativeExecutionPolicy * = nullptr) {
  return false;
}

template <>
bool TryPackedSiluMulGemv<half>(const QuantizedWeightInfo &raw,
                                const half *gate, const half *up, half *output,
                                int8_t *packed_activation, float *packed_scales,
                                int M, int N, int K, cudaStream_t stream,
                                const char *proj_name,
                                const NativeExecutionPolicy *policy) {
  const auto &policy_ref = ResolveNativeExecutionPolicy(policy);
  if (PackedActivationsDisabled(policy_ref) || !raw.data || !gate || !up ||
      !output || !packed_activation ||
      !packed_scales || M <= 0 || N <= 0 || K <= 0 ||
      !FusedQuantGemm::SupportsPackedActivations(raw.quant_type) ||
      !FusedQuantGemm::ShouldUseFusedPath(
          raw.quant_type, FusedDispatchGeometry{M, N, K, 1, true, false})) {
    return false;
  }

  if (cuda_kernel::SiluMulQuantizeRowsSymmetric(gate, up, packed_activation,
                                                packed_scales, M, K,
                                                stream) != cudaSuccess) {
    return false;
  }

  PackedActivationInfo packed{packed_activation, packed_scales};
  const bool ok =
      FusedQuantGemm::GemvPacked(raw, packed, output, M, N, K, stream, policy);
  if (ok && proj_name) {
    LogPackedGemmPath(proj_name, "using fused SwiGLU + packed-activation GEMV");
  }
  return ok;
}

template <typename T>
bool TryQ8_1SiluMulGemv(const QuantizedWeightInfo &, const T *, const T *, T *,
                        void *, int, int, int, cudaStream_t,
                        const char * = nullptr,
                        const NativeExecutionPolicy * = nullptr) {
  return false;
}

template <>
bool TryQ8_1SiluMulGemv<half>(const QuantizedWeightInfo &raw, const half *gate,
                              const half *up, half *output, void *act_q8_1,
                              int M, int N, int K, cudaStream_t stream,
                              const char *proj_name,
                              const NativeExecutionPolicy *policy) {
  const auto &policy_ref = ResolveNativeExecutionPolicy(policy);
  if (Q81ActivationsDisabled(policy_ref) || !raw.data || !gate || !up ||
      !output || !act_q8_1 || M <= 0 || N <= 0 || K <= 0 ||
      !FusedQuantGemm::SupportsQ8_1Activations(raw.quant_type) ||
      !FusedQuantGemm::ShouldUseFusedPath(
          raw.quant_type, FusedDispatchGeometry{M, N, K, 1, true, false})) {
    return false;
  }

  FusedQuantGemm::SiluMulQuantizeQ8_1(gate, up, act_q8_1, M, K, stream);
  const bool ok =
      FusedQuantGemm::GemvQ8_1(raw, act_q8_1, output, M, N, K, stream, policy);
  if (ok && proj_name) {
    LogPackedGemmPath(proj_name, "using fused SwiGLU + Q8_1 GEMV");
  }
  return ok;
}

} // namespace

template <typename T> LlamaForwardTyped<T>::~LlamaForwardTyped() {
  FreeScratchBuffers();
}

template <typename T> bool LlamaForwardTyped<T>::AllocateScratch() {
  auto alloc = [](T **ptr, size_t count) -> bool {
    cudaError_t err = cudaMalloc(ptr, count * sizeof(T));
    return err == cudaSuccess;
  };
  cudaError_t err;

  // Scratch buffers must fit both:
  //   - Single long sequence: max_seq_len_ tokens (prefill)
  //   - Batched decode: max_batch_size_ sequences x 1 token each
  size_t rows = static_cast<size_t>(std::max(max_seq_len_, max_batch_size_));

  if (!alloc(&d_hidden_, rows * hidden_size_))
    return false;
  if (!alloc(&d_residual_, rows * hidden_size_))
    return false;
  if (!alloc(&d_norm_out_, rows * hidden_size_))
    return false;
  if (!alloc(&d_q_, rows * num_heads_ * head_dim_))
    return false;
  if (!alloc(&d_k_new_, rows * num_kv_heads_ * head_dim_))
    return false;
  if (!alloc(&d_v_new_, rows * num_kv_heads_ * head_dim_))
    return false;
  if (!alloc(&d_attn_out_, rows * num_heads_ * head_dim_))
    return false;
  if (!alloc(&d_ffn_gate_, rows * intermediate_size_))
    return false;
  if (!alloc(&d_ffn_up_, rows * intermediate_size_))
    return false;
  if (!alloc(&d_ffn_down_, rows * hidden_size_))
    return false;
  const size_t packed_width = static_cast<size_t>(
      PackedActivationWidth(hidden_size_, intermediate_size_));
  err = cudaMalloc(&d_packed_activation_, rows * packed_width * sizeof(int8_t));
  if (err != cudaSuccess)
    return false;
  err = cudaMalloc(&d_packed_activation_scales_, rows * sizeof(float));
  if (err != cudaSuccess)
    return false;
  // Pre-quantized Q8_1 activation buffer: ceil(max_dim/32) blocks per row
  {
    const int max_dim = std::max(hidden_size_, intermediate_size_);
    const size_t blocks_per_row = (max_dim + 31) / 32;
    // block_q8_1 = 36 bytes (sizeof(half2) + 32)
    const size_t q8_1_block_size = 36;
    err = cudaMalloc(&d_act_q8_1_, rows * blocks_per_row * q8_1_block_size);
    if (err != cudaSuccess)
      return false;
  }
  // Logits buffer sized for batched decode: [max_batch_size, vocab_size]
  if (!alloc(&d_logits_typed_,
             static_cast<size_t>(max_batch_size_) * vocab_size_))
    return false;

  err = cudaMalloc(&d_token_ids_, rows * sizeof(int));
  if (err != cudaSuccess)
    return false;

  // Batch metadata buffers for batched decode
  size_t bsz = static_cast<size_t>(max_batch_size_);
  err = cudaMalloc(&d_batch_n_past_, bsz * sizeof(int));
  if (err != cudaSuccess)
    return false;
  err = cudaMalloc(&d_batch_seq_ids_, bsz * sizeof(int));
  if (err != cudaSuccess)
    return false;
  err = cudaMalloc(&d_batch_kv_lens_, bsz * sizeof(int));
  if (err != cudaSuccess)
    return false;

  // Device pointer arrays for batched KV append and attention
  err = cudaMalloc(&d_k_ptrs_, bsz * sizeof(T *));
  if (err != cudaSuccess)
    return false;
  err = cudaMalloc(&d_v_ptrs_, bsz * sizeof(T *));
  if (err != cudaSuccess)
    return false;
  err = cudaMalloc(&d_k_append_ptrs_, bsz * sizeof(T *));
  if (err != cudaSuccess)
    return false;
  err = cudaMalloc(&d_v_append_ptrs_, bsz * sizeof(T *));
  if (err != cudaSuccess)
    return false;

  // Bulk KV pointer arrays for all layers (CUDA graph capture)
  size_t kv_ptr_total = static_cast<size_t>(num_layers_) * bsz;
  err = cudaMalloc(&d_all_k_append_ptrs_, kv_ptr_total * sizeof(T *));
  if (err != cudaSuccess)
    return false;
  err = cudaMalloc(&d_all_v_append_ptrs_, kv_ptr_total * sizeof(T *));
  if (err != cudaSuccess)
    return false;
  err = cudaMalloc(&d_all_k_read_ptrs_, kv_ptr_total * sizeof(T *));
  if (err != cudaSuccess)
    return false;
  err = cudaMalloc(&d_all_v_read_ptrs_, kv_ptr_total * sizeof(T *));
  if (err != cudaSuccess)
    return false;

  return true;
}

template <typename T> void LlamaForwardTyped<T>::FreeScratchBuffers() {
  auto free_buf = [](T **ptr) {
    if (*ptr) {
      cudaFree(*ptr);
      *ptr = nullptr;
    }
  };

  free_buf(&d_hidden_);
  free_buf(&d_residual_);
  free_buf(&d_norm_out_);
  free_buf(&d_q_);
  free_buf(&d_k_new_);
  free_buf(&d_v_new_);
  free_buf(&d_attn_out_);
  free_buf(&d_ffn_gate_);
  free_buf(&d_ffn_up_);
  free_buf(&d_ffn_down_);
  free_buf(&d_logits_typed_);
  if (d_packed_activation_) {
    cudaFree(d_packed_activation_);
    d_packed_activation_ = nullptr;
  }
  if (d_packed_activation_scales_) {
    cudaFree(d_packed_activation_scales_);
    d_packed_activation_scales_ = nullptr;
  }
  if (d_act_q8_1_) {
    cudaFree(d_act_q8_1_);
    d_act_q8_1_ = nullptr;
  }

  if (d_token_ids_) {
    cudaFree(d_token_ids_);
    d_token_ids_ = nullptr;
  }

  auto free_int = [](int **ptr) {
    if (*ptr) {
      cudaFree(*ptr);
      *ptr = nullptr;
    }
  };
  free_int(&d_batch_n_past_);
  free_int(&d_batch_seq_ids_);
  free_int(&d_batch_kv_lens_);

  auto free_void = [](void **ptr) {
    if (*ptr) {
      cudaFree(*ptr);
      *ptr = nullptr;
    }
  };
  free_void(&d_k_ptrs_);
  free_void(&d_v_ptrs_);
  free_void(&d_k_append_ptrs_);
  free_void(&d_v_append_ptrs_);
  free_void(&d_all_k_append_ptrs_);
  free_void(&d_all_v_append_ptrs_);
  free_void(&d_all_k_read_ptrs_);
  free_void(&d_all_v_read_ptrs_);

  if (decode_graph_exec_) {
    cudaGraphExecDestroy(decode_graph_exec_);
    decode_graph_exec_ = nullptr;
  }
  if (decode_graph_) {
    cudaGraphDestroy(decode_graph_);
    decode_graph_ = nullptr;
  }
}

template <typename T>
bool LlamaForwardTyped<T>::Initialize(
    const SafetensorsLoader::ModelConfig &config, const WeightMap &weights,
    IKvCacheGpu *kv_cache, CublasGemm *gemm, cudaStream_t stream) {
  hidden_size_ = config.hidden_size;
  num_layers_ = config.num_hidden_layers;
  num_heads_ = config.num_attention_heads;
  num_kv_heads_ = config.num_key_value_heads;
  head_dim_ = config.head_dim;
  intermediate_size_ = config.intermediate_size;
  vocab_size_ = config.vocab_size;
  max_seq_len_ = config.max_position_embeddings;
  rope_freq_base_ = config.rope_freq_base;
  rms_norm_eps_ = config.rms_norm_eps;
  rope_type_ =
      static_cast<int>(runtime::cuda::native::InferRopeType(config.model_type));

  if (max_seq_len_ > 4096) {
    max_seq_len_ = 4096;
  }

  weights_ = &weights;
  kv_cache_ = kv_cache;
  gemm_ = gemm;
  stream_ = stream;

  // Match scratch buffer sizing to KV cache limits (not model max)
  if (kv_cache) {
    max_batch_size_ = kv_cache->MaxBatchSize();
    if (kv_cache->MaxSeqLen() > 0 && kv_cache->MaxSeqLen() < max_seq_len_) {
      max_seq_len_ = kv_cache->MaxSeqLen();
    }
  }

  if (!AllocateScratch()) {
    log::Error("llama_forward", "Failed to allocate scratch buffers");
    FreeScratchBuffers();
    return false;
  }

  log::Info("llama_forward",
            "Initialized (" + std::string(DtypeTraits<T>::name) +
                "): hidden=" + std::to_string(hidden_size_) +
                ", layers=" + std::to_string(num_layers_) +
                ", heads=" + std::to_string(num_heads_) + "/" +
                std::to_string(num_kv_heads_) +
                ", head_dim=" + std::to_string(head_dim_) +
                ", vocab=" + std::to_string(vocab_size_) +
                ", max_seq=" + std::to_string(max_seq_len_) +
                ", rope_type=" + (rope_type_ == 2 ? "neox" : "norm") +
                ", model=" + config.model_type);
  return true;
}

template <typename T>
bool LlamaForwardTyped<T>::Forward(const std::vector<int> &token_ids,
                                   int n_past, int sequence_id,
                                   float *d_logits) {
  NVTX_SCOPE("Forward");
  int seq_len = static_cast<int>(token_ids.size());
  if (seq_len == 0)
    return false;
  if (seq_len > max_seq_len_) {
    log::Error("llama_forward", "seq_len " + std::to_string(seq_len) +
                                    " exceeds max " +
                                    std::to_string(max_seq_len_));
    return false;
  }
  const bool allow_fused_quantized_matmul =
      !weights_ || weights_->AllowFusedQuantizedMatmul();
  ScopedFusedMatmulPolicy fused_policy(allow_fused_quantized_matmul);

  int kv_len = n_past + seq_len;
  cudaError_t err;
  PhaseTiming pt;
  pt.Begin(stream_, execution_policy_.phase_timing_enabled);

  // Step 1: Upload token_ids to GPU
  err = cudaMemcpyAsync(d_token_ids_, token_ids.data(), seq_len * sizeof(int),
                        cudaMemcpyHostToDevice, stream_);
  if (err != cudaSuccess) {
    log::Error("llama_forward", "Failed to upload token_ids");
    return false;
  }

  // Step 2: Embedding lookup
  // WeightMap is always WeightMapTyped<half> currently, but the embed_tokens
  // pointer points to the same GPU data regardless of type. We cast it.
  const T *embed = reinterpret_cast<const T *>(weights_->EmbedTokens());
  {
    NVTX_SCOPE("Embedding");
    err = cuda_kernel::EmbeddingLookup<T>(embed, d_token_ids_, d_hidden_,
                                          seq_len, hidden_size_, stream_);
  }
  if (err != cudaSuccess) {
    log::Error("llama_forward", "EmbeddingLookup failed");
    return false;
  }

  DebugDumpHidden("after_embedding", d_hidden_, seq_len * hidden_size_,
                  stream_, &execution_policy_);

  // Step 3: Copy to residual stream
  err = cudaMemcpyAsync(d_residual_, d_hidden_,
                        (size_t)seq_len * hidden_size_ * sizeof(T),
                        cudaMemcpyDeviceToDevice, stream_);
  if (err != cudaSuccess) {
    log::Error("llama_forward", "Residual copy failed");
    return false;
  }
  pt.embed_ms += pt.Mark();

  // Step 4: Transformer layers
  for (int layer = 0; layer < num_layers_; layer++) {
    NVTX_SCOPE("Layer");
    // Norm weights are small (F32/F16), always fetch eagerly
    const T *input_norm =
        reinterpret_cast<const T *>(weights_->LayerInputNorm(layer));
    const T *post_attn_norm =
        reinterpret_cast<const T *>(weights_->LayerPostAttnNorm(layer));

    // 4a-d: RMSNorm + Q/K/V projections + optional bias
    // Try fused RmsNorm+GEMV first (eliminates standalone RmsNorm kernel).
    // Each fused kernel independently normalizes d_residual_ — the norm
    // re-computation is ~1% of GEMV cost and amortized across 8 warps.
    {
      NVTX_SCOPE("QKV_Projection");
      auto q_raw = weights_->LayerQProjRaw(layer);
      auto k_raw = weights_->LayerKProjRaw(layer);
      auto v_raw = weights_->LayerVProjRaw(layer);
      const std::array<PackedProjectionPlan<T>, 3> qkv_plans = {
          {{q_raw, d_q_, num_heads_ * head_dim_},
           {k_raw, d_k_new_, num_kv_heads_ * head_dim_},
           {v_raw, d_v_new_, num_kv_heads_ * head_dim_}}};
      // Priority: Q8_1 > packed per-row > fused RmsNorm+GEMV > cuBLAS
      const bool used_q8_1_qkv = TryQ8_1ProjectionGroup(
          qkv_plans, d_residual_, input_norm, d_act_q8_1_, seq_len,
          hidden_size_, rms_norm_eps_, stream_, &execution_policy_);
      const bool used_packed_qkv =
          !used_q8_1_qkv &&
          TryPackedProjectionGroup(qkv_plans, d_residual_, input_norm,
                                   d_norm_out_, d_packed_activation_,
                                   d_packed_activation_scales_, seq_len,
                                   hidden_size_, rms_norm_eps_, stream_,
                                   &execution_policy_);
      if (!used_q8_1_qkv && !used_packed_qkv) {
        bool norm_computed = false;

        if (!TryFusedRmsNormGemv<T>(q_raw, d_residual_, input_norm, d_q_,
                                    seq_len, num_heads_ * head_dim_,
                                    hidden_size_, rms_norm_eps_, stream_,
                                    "q_proj", &execution_policy_)) {
          // Fallback: standalone RmsNorm + GEMV/cuBLAS
          if (!norm_computed) {
            err = cuda_kernel::RmsNorm<T>(d_residual_, input_norm, d_norm_out_,
                                          seq_len, hidden_size_, rms_norm_eps_,
                                          stream_);
            if (err != cudaSuccess) {
              log::Error("llama_forward",
                         "RmsNorm failed at layer " + std::to_string(layer));
              return false;
            }
            norm_computed = true;
          }
          if (!TryFusedGemv<T>(q_raw, d_norm_out_, d_q_, seq_len,
                               num_heads_ * head_dim_, hidden_size_, stream_,
                               "q_proj", &execution_policy_)) {
            const T *q_proj =
                reinterpret_cast<const T *>(weights_->LayerQProj(layer));
            if (!gemm_->GemmTyped<T>(seq_len, num_heads_ * head_dim_,
                                     hidden_size_, d_norm_out_, q_proj, d_q_)) {
              log::Error("llama_forward", "Q projection failed");
              return false;
            }
          }
        }

        if (!TryFusedRmsNormGemv<T>(k_raw, d_residual_, input_norm, d_k_new_,
                                    seq_len, num_kv_heads_ * head_dim_,
                                    hidden_size_, rms_norm_eps_, stream_,
                                    "k_proj", &execution_policy_)) {
          if (!norm_computed) {
            err = cuda_kernel::RmsNorm<T>(d_residual_, input_norm, d_norm_out_,
                                          seq_len, hidden_size_, rms_norm_eps_,
                                          stream_);
            if (err != cudaSuccess) {
              log::Error("llama_forward",
                         "RmsNorm failed at layer " + std::to_string(layer));
              return false;
            }
            norm_computed = true;
          }
          if (!TryFusedGemv<T>(k_raw, d_norm_out_, d_k_new_, seq_len,
                               num_kv_heads_ * head_dim_, hidden_size_, stream_,
                               "k_proj", &execution_policy_)) {
            const T *k_proj =
                reinterpret_cast<const T *>(weights_->LayerKProj(layer));
            if (!gemm_->GemmTyped<T>(seq_len, num_kv_heads_ * head_dim_,
                                     hidden_size_, d_norm_out_, k_proj,
                                     d_k_new_)) {
              log::Error("llama_forward", "K projection failed");
              return false;
            }
          }
        }

        if (!TryFusedRmsNormGemv<T>(v_raw, d_residual_, input_norm, d_v_new_,
                                    seq_len, num_kv_heads_ * head_dim_,
                                    hidden_size_, rms_norm_eps_, stream_,
                                    "v_proj", &execution_policy_)) {
          if (!norm_computed) {
            err = cuda_kernel::RmsNorm<T>(d_residual_, input_norm, d_norm_out_,
                                          seq_len, hidden_size_, rms_norm_eps_,
                                          stream_);
            if (err != cudaSuccess) {
              log::Error("llama_forward",
                         "RmsNorm failed at layer " + std::to_string(layer));
              return false;
            }
            norm_computed = true;
          }
          if (!TryFusedGemv<T>(v_raw, d_norm_out_, d_v_new_, seq_len,
                               num_kv_heads_ * head_dim_, hidden_size_, stream_,
                               "v_proj", &execution_policy_)) {
            const T *v_proj =
                reinterpret_cast<const T *>(weights_->LayerVProj(layer));
            if (!gemm_->GemmTyped<T>(seq_len, num_kv_heads_ * head_dim_,
                                     hidden_size_, d_norm_out_, v_proj,
                                     d_v_new_)) {
              log::Error("llama_forward", "V projection failed");
              return false;
            }
          }
        }
      }

      // Add biases if present (Qwen2 has q/k/v biases)
      const T *q_bias =
          reinterpret_cast<const T *>(weights_->LayerQProjBias(layer));
      const T *k_bias =
          reinterpret_cast<const T *>(weights_->LayerKProjBias(layer));
      const T *v_bias =
          reinterpret_cast<const T *>(weights_->LayerVProjBias(layer));
      if (q_bias) {
        err = cuda_kernel::BiasAdd<T>(d_q_, q_bias, seq_len,
                                      num_heads_ * head_dim_, stream_);
        if (err != cudaSuccess) {
          log::Error("llama_forward", "Q bias add failed");
          return false;
        }
      }
      if (k_bias) {
        err = cuda_kernel::BiasAdd<T>(d_k_new_, k_bias, seq_len,
                                      num_kv_heads_ * head_dim_, stream_);
        if (err != cudaSuccess) {
          log::Error("llama_forward", "K bias add failed");
          return false;
        }
      }
      if (v_bias) {
        err = cuda_kernel::BiasAdd<T>(d_v_new_, v_bias, seq_len,
                                      num_kv_heads_ * head_dim_, stream_);
        if (err != cudaSuccess) {
          log::Error("llama_forward", "V bias add failed");
          return false;
        }
      }
    }

    pt.qkv_ms += pt.Mark();

    if (layer == 0) {
      DebugDumpHidden("layer0_after_q_proj", d_q_,
                      seq_len * num_heads_ * head_dim_, stream_,
                      &execution_policy_);
      DebugDumpHidden("layer0_after_k_proj", d_k_new_,
                      seq_len * num_kv_heads_ * head_dim_, stream_,
                      &execution_policy_);
    }

    // 4e: RoPE in-place
    {
      NVTX_SCOPE("RoPE");
      err = cuda_kernel::RoPE<T>(d_q_, d_k_new_, seq_len, num_heads_,
                                 num_kv_heads_, head_dim_, n_past,
                                 rope_freq_base_, stream_, rope_type_);
      if (err != cudaSuccess) {
        log::Error("llama_forward", "RoPE failed");
        return false;
      }
    }

    pt.rope_ms += pt.Mark();

    // 4f: KV cache append (cache type matches T via KvCacheGpuTyped<T>)
    {
      NVTX_SCOPE("KV_Append");
      auto *typed_cache = static_cast<KvCacheGpuTyped<T> *>(
          static_cast<IKvCacheGpu *>(kv_cache_));
      err = typed_cache->Append(layer, sequence_id, n_past, seq_len, d_k_new_,
                                d_v_new_, stream_);
      if (err != cudaSuccess) {
        log::Error("llama_forward", "KV cache append failed");
        return false;
      }
    }

    pt.kv_ms += pt.Mark();

    // 4g: FlashAttention-2
    {
      NVTX_SCOPE("FlashAttention2");
      auto *typed_cache = static_cast<KvCacheGpuTyped<T> *>(
          static_cast<IKvCacheGpu *>(kv_cache_));
      T *k_cache = typed_cache->GetK(layer, sequence_id);
      T *v_cache = typed_cache->GetV(layer, sequence_id);

      float attn_scale = 1.0f / sqrtf(static_cast<float>(head_dim_));

      err = cuda_kernel::FlashAttention2Typed<T>(
          d_q_, k_cache, v_cache, d_attn_out_, /*batch_size=*/1, seq_len,
          kv_len, num_heads_, num_kv_heads_, head_dim_, attn_scale,
          /*causal=*/true, stream_);
      if (err != cudaSuccess) {
        log::Error("llama_forward", "FlashAttention2 failed");
        return false;
      }
    }

    pt.attn_ms += pt.Mark();

    // 4h: O projection
    {
      NVTX_SCOPE("O_Projection");
      auto o_raw = weights_->LayerOProjRaw(layer);
      if (!TryQ8_1Gemv<T>(o_raw, d_attn_out_, d_norm_out_, d_act_q8_1_, seq_len,
                          hidden_size_, num_heads_ * head_dim_, stream_, "o_proj",
                          &execution_policy_) &&
          !TryPackedGemv<T>(o_raw, d_attn_out_, d_norm_out_,
                            d_packed_activation_, d_packed_activation_scales_,
                            seq_len, hidden_size_, num_heads_ * head_dim_,
                            stream_, "o_proj", &execution_policy_) &&
          !TryFusedGemv<T>(o_raw, d_attn_out_, d_norm_out_, seq_len,
                           hidden_size_, num_heads_ * head_dim_, stream_,
                           "o_proj", &execution_policy_)) {
        const T *o_proj =
            reinterpret_cast<const T *>(weights_->LayerOProj(layer));
        if (!gemm_->GemmTyped<T>(seq_len, hidden_size_, num_heads_ * head_dim_,
                                 d_attn_out_, o_proj, d_norm_out_)) {
          log::Error("llama_forward", "O projection failed");
          return false;
        }
      }
    }

    // 4i: residual += O
    err = cuda_kernel::ResidualAdd<T>(d_residual_, d_norm_out_,
                                      seq_len * hidden_size_, stream_);
    if (err != cudaSuccess) {
      log::Error("llama_forward", "Residual add (attn) failed");
      return false;
    }

    pt.o_proj_ms += pt.Mark();

    // 4j-n: FFN block
    {
      NVTX_SCOPE("FFN");
      // Gate projection: try fused RmsNorm+GEMV (post-attn norm)
      auto gate_raw = weights_->LayerGateProjRaw(layer);
      // Up projection: try fused RmsNorm+GEMV (post-attn norm)
      auto up_raw = weights_->LayerUpProjRaw(layer);
      const std::array<PackedProjectionPlan<T>, 2> ffn_plans = {
          {{gate_raw, d_ffn_gate_, intermediate_size_},
           {up_raw, d_ffn_up_, intermediate_size_}}};
      const char *ffn_phase = seq_len == 1 ? "decode" : "prefill";
      const auto ffn_selected_op = SelectNativeFfnProjOperator(
          gate_raw, up_raw,
          FusedDispatchGeometry{seq_len, intermediate_size_, hidden_size_, 2,
                                true, true},
          g_allow_fused_quantized_matmul, execution_policy_);
      bool used_q8_1_ffn = false;
      bool used_packed_ffn = false;
      auto ffn_actual_op = FusedQuantGemm::FfnProjOperator::kFallback;
      if (ffn_selected_op == FusedQuantGemm::FfnProjOperator::kQ81Group ||
          ffn_selected_op ==
              FusedQuantGemm::FfnProjOperator::kQ81GroupHotQ4K) {
        used_q8_1_ffn = TryQ8_1ProjectionGroup(
            ffn_plans, d_residual_, post_attn_norm, d_act_q8_1_, seq_len,
            hidden_size_, rms_norm_eps_, stream_, &execution_policy_);
        if (used_q8_1_ffn) {
          ffn_actual_op = ffn_selected_op;
        }
      } else if (ffn_selected_op ==
                 FusedQuantGemm::FfnProjOperator::kPackedGroup) {
        used_packed_ffn = TryPackedProjectionGroup(
            ffn_plans, d_residual_, post_attn_norm, d_norm_out_,
            d_packed_activation_, d_packed_activation_scales_, seq_len,
            hidden_size_, rms_norm_eps_, stream_, &execution_policy_);
        if (used_packed_ffn) {
          ffn_actual_op = ffn_selected_op;
        }
      }
      if (!used_q8_1_ffn && !used_packed_ffn &&
          ffn_selected_op != FusedQuantGemm::FfnProjOperator::kPackedGroup) {
        used_packed_ffn = TryPackedProjectionGroup(
            ffn_plans, d_residual_, post_attn_norm, d_norm_out_,
            d_packed_activation_, d_packed_activation_scales_, seq_len,
            hidden_size_, rms_norm_eps_, stream_, &execution_policy_);
        if (used_packed_ffn) {
          ffn_actual_op = FusedQuantGemm::FfnProjOperator::kPackedGroup;
        }
      }
      const std::string ffn_quant =
          ProjectionGroupQuantLabel(gate_raw, up_raw);
      GlobalMetrics().RecordNativeFfnProjOperator(
          ffn_phase, FusedQuantGemm::FfnProjOperatorName(ffn_actual_op));
      GlobalMetrics().RecordNativeFfnProjGeometry(
          ffn_phase, FusedQuantGemm::FfnProjOperatorName(ffn_actual_op),
          ffn_quant, seq_len, intermediate_size_, hidden_size_,
          /*grouped_outputs=*/2);
      if (!used_q8_1_ffn && !used_packed_ffn) {
        bool ffn_norm_computed = false;

        if (!TryFusedRmsNormGemv<T>(gate_raw, d_residual_, post_attn_norm,
                                    d_ffn_gate_, seq_len, intermediate_size_,
                                    hidden_size_, rms_norm_eps_, stream_,
                                    "gate_proj", &execution_policy_)) {
          if (!ffn_norm_computed) {
            err = cuda_kernel::RmsNorm<T>(d_residual_, post_attn_norm,
                                          d_norm_out_, seq_len, hidden_size_,
                                          rms_norm_eps_, stream_);
            if (err != cudaSuccess) {
              log::Error("llama_forward", "Post-attn RmsNorm failed");
              return false;
            }
            ffn_norm_computed = true;
          }
          if (!TryFusedGemv<T>(gate_raw, d_norm_out_, d_ffn_gate_, seq_len,
                               intermediate_size_, hidden_size_, stream_,
                               "gate_proj", &execution_policy_)) {
            const T *gate_proj =
                reinterpret_cast<const T *>(weights_->LayerGateProj(layer));
            if (!gemm_->GemmTyped<T>(seq_len, intermediate_size_, hidden_size_,
                                     d_norm_out_, gate_proj, d_ffn_gate_)) {
              log::Error("llama_forward", "Gate projection failed");
              return false;
            }
          }
        }

        if (!TryFusedRmsNormGemv<T>(up_raw, d_residual_, post_attn_norm,
                                    d_ffn_up_, seq_len, intermediate_size_,
                                    hidden_size_, rms_norm_eps_, stream_,
                                    "up_proj", &execution_policy_)) {
          if (!ffn_norm_computed) {
            err = cuda_kernel::RmsNorm<T>(d_residual_, post_attn_norm,
                                          d_norm_out_, seq_len, hidden_size_,
                                          rms_norm_eps_, stream_);
            if (err != cudaSuccess) {
              log::Error("llama_forward", "Post-attn RmsNorm failed");
              return false;
            }
            ffn_norm_computed = true;
          }
          if (!TryFusedGemv<T>(up_raw, d_norm_out_, d_ffn_up_, seq_len,
                               intermediate_size_, hidden_size_, stream_,
                               "up_proj", &execution_policy_)) {
            const T *up_proj =
                reinterpret_cast<const T *>(weights_->LayerUpProj(layer));
            if (!gemm_->GemmTyped<T>(seq_len, intermediate_size_, hidden_size_,
                                     d_norm_out_, up_proj, d_ffn_up_)) {
              log::Error("llama_forward", "Up projection failed");
              return false;
            }
          }
        }
      }
      pt.ffn_proj_ms += pt.Mark();

      auto down_raw = weights_->LayerDownProjRaw(layer);
      auto down_mmq = weights_->LayerDownProjMmq(layer);
      const auto down_selected_op = SelectNativeDownProjOperator(
          down_raw, down_mmq,
          FusedDispatchGeometry{seq_len, hidden_size_, intermediate_size_, 1,
                                true, false},
          g_allow_fused_quantized_matmul, execution_policy_);
      bool fused_mmq_down = false;
      bool fused_q8_1_down = false;
      bool fused_packed_down = false;
      auto down_actual_op = FusedQuantGemm::DownProjOperator::kFallback;
      if (down_selected_op == FusedQuantGemm::DownProjOperator::kMmq) {
        fused_mmq_down = TryMmqSiluMulGemv<T>(
            down_mmq, d_ffn_gate_, d_ffn_up_, d_ffn_down_, d_act_q8_1_,
            seq_len, hidden_size_, intermediate_size_, stream_, "down_proj",
            &execution_policy_);
        if (fused_mmq_down) {
          down_actual_op = FusedQuantGemm::DownProjOperator::kMmq;
        }
        fused_q8_1_down =
            !fused_mmq_down &&
            TryQ8_1SiluMulGemv<T>(
                down_raw, d_ffn_gate_, d_ffn_up_, d_ffn_down_, d_act_q8_1_,
                seq_len, hidden_size_, intermediate_size_, stream_,
                "down_proj", &execution_policy_);
        if (fused_q8_1_down) {
          down_actual_op = down_selected_op;
        }
      } else if (down_selected_op ==
                     FusedQuantGemm::DownProjOperator::kPackedGemv) {
        fused_packed_down = TryPackedSiluMulGemv<T>(
            down_raw, d_ffn_gate_, d_ffn_up_, d_ffn_down_,
            d_packed_activation_, d_packed_activation_scales_, seq_len,
            hidden_size_, intermediate_size_, stream_, "down_proj",
            &execution_policy_);
        if (fused_packed_down) {
          down_actual_op = down_selected_op;
        }
      } else {
        fused_q8_1_down = TryQ8_1SiluMulGemv<T>(
            down_raw, d_ffn_gate_, d_ffn_up_, d_ffn_down_, d_act_q8_1_,
            seq_len, hidden_size_, intermediate_size_, stream_, "down_proj",
            &execution_policy_);
        if (fused_q8_1_down) {
          down_actual_op = down_selected_op;
        }
        fused_mmq_down =
            !fused_q8_1_down &&
            TryMmqSiluMulGemv<T>(
                down_mmq, d_ffn_gate_, d_ffn_up_, d_ffn_down_, d_act_q8_1_,
                seq_len, hidden_size_, intermediate_size_, stream_,
                "down_proj", &execution_policy_);
        if (fused_mmq_down) {
          down_actual_op = FusedQuantGemm::DownProjOperator::kMmq;
        }
      }
      fused_packed_down =
          !fused_mmq_down && !fused_q8_1_down && !fused_packed_down &&
          TryPackedSiluMulGemv<T>(
              down_raw, d_ffn_gate_, d_ffn_up_, d_ffn_down_,
              d_packed_activation_, d_packed_activation_scales_, seq_len,
              hidden_size_, intermediate_size_, stream_, "down_proj",
              &execution_policy_);
      if (fused_packed_down) {
        down_actual_op = FusedQuantGemm::DownProjOperator::kPackedGemv;
      }
      GlobalMetrics().RecordNativeDownProjOperator(
          ffn_phase, FusedQuantGemm::DownProjOperatorName(down_actual_op));
      GlobalMetrics().RecordNativeDownProjGeometry(
          ffn_phase, FusedQuantGemm::DownProjOperatorName(down_actual_op),
          ProjectionQuantLabel(down_raw), seq_len, hidden_size_,
          intermediate_size_);
      if (down_actual_op != FusedQuantGemm::DownProjOperator::kFallback) {
        const std::string down_label =
            std::string("selected down-proj operator: ") +
            FusedQuantGemm::DownProjOperatorName(down_actual_op);
        LogPackedGemmPath("down_proj", down_label.c_str());
      }
      if (!fused_mmq_down && !fused_q8_1_down && !fused_packed_down) {
        // SwiGLU
        err = cuda_kernel::SiluMul<T>(d_ffn_gate_, d_ffn_up_, d_ffn_gate_,
                                      seq_len * intermediate_size_, stream_);
        if (err != cudaSuccess) {
          log::Error("llama_forward", "SiluMul failed");
          return false;
        }
        pt.ffn_silu_ms += pt.Mark();

        // Down projection (input is activation, not normalized — no fusion)
        bool down_ok = false;
        if (down_selected_op == FusedQuantGemm::DownProjOperator::kMmq) {
          down_ok = TryMmqGemv<T>(down_mmq, d_ffn_gate_, d_ffn_down_,
                                  d_act_q8_1_, seq_len, hidden_size_,
                                  intermediate_size_, stream_, "down_proj",
                                  &execution_policy_) ||
                    TryQ8_1Gemv<T>(down_raw, d_ffn_gate_, d_ffn_down_,
                                   d_act_q8_1_, seq_len, hidden_size_,
                                   intermediate_size_, stream_, "down_proj",
                                   &execution_policy_);
        } else {
          down_ok = TryQ8_1Gemv<T>(down_raw, d_ffn_gate_, d_ffn_down_,
                                   d_act_q8_1_, seq_len, hidden_size_,
                                   intermediate_size_, stream_, "down_proj",
                                   &execution_policy_) ||
                    TryMmqGemv<T>(down_mmq, d_ffn_gate_, d_ffn_down_,
                                  d_act_q8_1_, seq_len, hidden_size_,
                                  intermediate_size_, stream_, "down_proj",
                                  &execution_policy_);
        }
        if (!down_ok &&
            !TryPackedGemv<T>(down_raw, d_ffn_gate_, d_ffn_down_,
                              d_packed_activation_, d_packed_activation_scales_,
                              seq_len, hidden_size_, intermediate_size_,
                              stream_, "down_proj", &execution_policy_) &&
            !TryFusedGemv<T>(down_raw, d_ffn_gate_, d_ffn_down_, seq_len,
                             hidden_size_, intermediate_size_, stream_,
                             "down_proj", &execution_policy_)) {
          const T *down_proj =
              reinterpret_cast<const T *>(weights_->LayerDownProj(layer));
          if (!gemm_->GemmTyped<T>(seq_len, hidden_size_, intermediate_size_,
                                   d_ffn_gate_, down_proj, d_ffn_down_)) {
            log::Error("llama_forward", "Down projection failed");
            return false;
          }
        }
      }
    }

    // 4o: residual += down
    err = cuda_kernel::ResidualAdd<T>(d_residual_, d_ffn_down_,
                                      seq_len * hidden_size_, stream_);
    if (err != cudaSuccess) {
      log::Error("llama_forward", "Residual add (FFN) failed");
      return false;
    }
    pt.ffn_down_ms += pt.Mark();
  }

  // Step 5: Final RMSNorm + LM head (last token only)
  {
    NVTX_SCOPE("LM_Head");
    T *last_hidden = d_residual_ + (seq_len - 1) * hidden_size_;
    const T *final_norm = reinterpret_cast<const T *>(weights_->FinalNorm());

    // Try Q8_1 fused RmsNorm+Quantize+GEMV for LM head first
    auto lm_raw = weights_->LmHeadRaw();
    bool lm_q8_1_ok = false;
    {
      if constexpr (std::is_same_v<T, half>) {
        if (!Q81ActivationsDisabled(execution_policy_) && lm_raw.data &&
            d_act_q8_1_ &&
            FusedQuantGemm::SupportsQ8_1Activations(lm_raw.quant_type) &&
            FusedQuantGemm::ShouldUseFusedPath(
                lm_raw.quant_type,
                FusedDispatchGeometry{1, vocab_size_, hidden_size_, 1, true,
                                      true})) {
          FusedQuantGemm::FusedRmsNormQuantizeQ8_1(last_hidden, final_norm,
                                                   d_act_q8_1_, 1, hidden_size_,
                                                   rms_norm_eps_, stream_);
          lm_q8_1_ok =
              FusedQuantGemm::GemvQ8_1(lm_raw, d_act_q8_1_, d_logits_typed_, 1,
                                       vocab_size_, hidden_size_, stream_,
                                       &execution_policy_);
          if (lm_q8_1_ok) {
            LogPackedGemmPath("lm_head",
                              "using Q8_1 fused RmsNorm+Quantize+GEMV");
          }
        }
      }
    }
    if (!lm_q8_1_ok &&
        !TryPackedRmsNormGemv<T>(
            lm_raw, last_hidden, final_norm, d_norm_out_, d_packed_activation_,
            d_packed_activation_scales_, d_logits_typed_, 1, vocab_size_,
            hidden_size_, rms_norm_eps_, stream_, "lm_head",
            &execution_policy_) &&
        !TryFusedRmsNormGemv<T>(lm_raw, last_hidden, final_norm,
                                d_logits_typed_, 1, vocab_size_, hidden_size_,
                                rms_norm_eps_, stream_, "lm_head",
                                &execution_policy_)) {
      // Fallback: standalone RmsNorm + GEMV/cuBLAS
      err = cuda_kernel::RmsNorm<T>(last_hidden, final_norm, d_norm_out_, 1,
                                    hidden_size_, rms_norm_eps_, stream_);
      if (err != cudaSuccess) {
        log::Error("llama_forward", "Final RmsNorm failed");
        return false;
      }
      if (!TryFusedGemv<T>(lm_raw, d_norm_out_, d_logits_typed_, 1, vocab_size_,
                           hidden_size_, stream_, "lm_head",
                           &execution_policy_)) {
        const T *lm_head = reinterpret_cast<const T *>(weights_->LmHead());
        if (!gemm_->GemmTyped<T>(1, vocab_size_, hidden_size_, d_norm_out_,
                                 lm_head, d_logits_typed_)) {
          log::Error("llama_forward", "LM head projection failed");
          return false;
        }
      }
    }

    // Step 7: Typed -> FP32 conversion (always float* output)
    err = cuda_kernel::HalfToFloat<T>(d_logits_typed_, d_logits, vocab_size_,
                                      stream_);
    if (err != cudaSuccess) {
      log::Error("llama_forward", "HalfToFloat failed");
      return false;
    }

    DebugDumpLogits(d_logits, vocab_size_, token_ids, n_past, stream_,
                    &execution_policy_);
    pt.lm_head_ms += pt.Mark();
  }

  pt.Report(num_layers_, seq_len);
  return true;
}

template <typename T>
void LlamaForwardTyped<T>::SetStream(cudaStream_t stream) {
  stream_ = stream;
}

template <typename T>
void LlamaForwardTyped<T>::SetExecutionPolicy(
    const NativeExecutionPolicy &policy) {
  execution_policy_ = policy;
}

template <typename T>
bool LlamaForwardTyped<T>::BatchForward(const std::vector<int> &token_ids,
                                        const std::vector<int> &n_past,
                                        const std::vector<int> &sequence_ids,
                                        float *d_logits, int batch_size) {
  NVTX_SCOPE("BatchForward");
  if (batch_size <= 0)
    return false;
  const auto &policy = execution_policy_;
  const bool use_batched = policy.enable_batched_decode;
  if (!use_batched) {
    bool all_ok = true;
    for (int b = 0; b < batch_size; ++b) {
      std::vector<int> single = {token_ids[b]};
      float *out = d_logits + b * vocab_size_;
      if (!Forward(single, n_past[b], sequence_ids[b], out))
        all_ok = false;
    }
    return all_ok;
  }

  int B = batch_size;
  const bool allow_fused_quantized_matmul =
      !weights_ || weights_->AllowFusedQuantizedMatmul();
  ScopedFusedMatmulPolicy fused_policy(allow_fused_quantized_matmul);
  cudaError_t err;

  // ===== Phase 1: Upload metadata to fixed device addresses =====
  // All H2D copies happen BEFORE any graph-captured region so that
  // graph replay reads updated data from the same device addresses.
  err = cudaMemcpyAsync(d_token_ids_, token_ids.data(), B * sizeof(int),
                        cudaMemcpyHostToDevice, stream_);
  if (err != cudaSuccess) {
    log::Error("llama_forward", "BatchForward: token upload failed");
    return false;
  }
  cudaMemcpyAsync(d_batch_n_past_, n_past.data(), B * sizeof(int),
                  cudaMemcpyHostToDevice, stream_);
  cudaMemcpyAsync(d_batch_seq_ids_, sequence_ids.data(), B * sizeof(int),
                  cudaMemcpyHostToDevice, stream_);
  {
    std::vector<int> h_kv_lens(B);
    for (int b = 0; b < B; ++b)
      h_kv_lens[b] = n_past[b] + 1;
    cudaMemcpyAsync(d_batch_kv_lens_, h_kv_lens.data(), B * sizeof(int),
                    cudaMemcpyHostToDevice, stream_);
  }

  // ===== Phase 2: Bulk KV pointer pre-computation for all layers =====
  // Replaces num_layers * 4 per-layer H2D copies with 4 bulk copies.
  auto *typed_cache =
      static_cast<KvCacheGpuTyped<T> *>(static_cast<IKvCacheGpu *>(kv_cache_));
  {
    size_t total = static_cast<size_t>(num_layers_) * B;
    std::vector<T *> h_k_ap(total), h_v_ap(total);
    std::vector<const T *> h_k_rd(total), h_v_rd(total);
    for (int l = 0; l < num_layers_; l++) {
      typed_cache->GetBatchAppendPtrs(l, sequence_ids.data(), n_past.data(), B,
                                      &h_k_ap[l * B], &h_v_ap[l * B]);
      typed_cache->GetBatchKVPtrs(l, sequence_ids.data(), B, &h_k_rd[l * B],
                                  &h_v_rd[l * B]);
    }
    size_t ptr_bytes = total * sizeof(T *);
    cudaMemcpyAsync(d_all_k_append_ptrs_, h_k_ap.data(), ptr_bytes,
                    cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_all_v_append_ptrs_, h_v_ap.data(), ptr_bytes,
                    cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_all_k_read_ptrs_, h_k_rd.data(), ptr_bytes,
                    cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_all_v_read_ptrs_, h_v_rd.data(), ptr_bytes,
                    cudaMemcpyHostToDevice, stream_);
  }

  // ===== Phase 3: CUDA graph replay or capture =====
  // CUDA graph capture eliminates per-kernel launch overhead
  // (~5-10μs × 250+ kernels = 1-2ms/token). Enabled by default;
  // disable with INFERFLUX_DISABLE_CUDA_GRAPH=1.
  const bool phase_timing_enabled = policy.phase_timing_enabled;
  const bool graph_disabled = policy.disable_cuda_graph;
  const bool capture_safe =
      DecodeGraphCaptureSafe(weights_, num_layers_, B, hidden_size_, num_heads_,
                             num_kv_heads_, head_dim_, intermediate_size_,
                             vocab_size_, g_allow_fused_quantized_matmul,
                             policy);
  if (graph_enabled_ && !graph_disabled && !phase_timing_enabled &&
      !capture_safe) {
    static bool logged_graph_fallback = false;
    if (!logged_graph_fallback) {
      logged_graph_fallback = true;
      log::Info("llama_forward",
                "CUDA graph disabled for batched decode because at least one "
                "projection requires non-graph-safe fallback");
    }
  }
  bool use_graph = ShouldUseDecodeGraph(graph_enabled_, graph_disabled,
                                        phase_timing_enabled, capture_safe);
  PhaseTiming pt;
  pt.Begin(stream_, phase_timing_enabled);

  // Fast path: replay existing graph if batch size matches
  if (use_graph && decode_graph_exec_ && graph_batch_size_ == B) {
    err = cudaGraphLaunch(decode_graph_exec_, stream_);
    if (err == cudaSuccess)
      return true;
    log::Warn("llama_forward", "CUDA graph replay failed, disabling");
    cudaGraphExecDestroy(decode_graph_exec_);
    decode_graph_exec_ = nullptr;
    cudaGraphDestroy(decode_graph_);
    decode_graph_ = nullptr;
    graph_enabled_ = false;
    // Fall through to non-graph path
  }

  // Destroy stale graph if batch size changed
  if (decode_graph_exec_ && graph_batch_size_ != B) {
    cudaGraphExecDestroy(decode_graph_exec_);
    decode_graph_exec_ = nullptr;
    cudaGraphDestroy(decode_graph_);
    decode_graph_ = nullptr;
  }

  // Begin graph capture if enabled
  bool capturing = false;
  if (use_graph) {
    err = cudaStreamBeginCapture(stream_, cudaStreamCaptureModeRelaxed);
    if (err == cudaSuccess) {
      capturing = true;
    } else {
      log::Warn("llama_forward", "CUDA graph capture begin failed");
      graph_enabled_ = false;
    }
  }

  // ===== Compute section (captured into graph or executed directly) =====
  // All operations below use fixed device addresses. During graph replay,
  // the kernels read updated data uploaded in Phases 1-2.
  //
  // IMPORTANT: cuBLAS calls during CUDA graph capture corrupt the host heap
  // because cuBLAS internally allocates workspace that becomes stale on
  // replay. All cuBLAS fallback paths are guarded with `if (!capturing)`.
  // If fused GEMV can't handle a projection during capture, we set
  // `capture_abort` and fall back to direct execution after ending capture.
  bool capture_abort = false;
  auto RunCompute = [&]() -> bool {
    // Embedding [B, hidden_size]
    {
      NVTX_SCOPE("Embedding");
      const T *embed = reinterpret_cast<const T *>(weights_->EmbedTokens());
      err = cuda_kernel::EmbeddingLookup<T>(embed, d_token_ids_, d_hidden_, B,
                                            hidden_size_, stream_);
      if (err != cudaSuccess)
        return false;
    }

    // Copy to residual stream
    err = cudaMemcpyAsync(d_residual_, d_hidden_,
                          (size_t)B * hidden_size_ * sizeof(T),
                          cudaMemcpyDeviceToDevice, stream_);
    if (err != cudaSuccess)
      return false;
    pt.embed_ms += pt.Mark();

    // Transformer layers
    for (int layer = 0; layer < num_layers_; layer++) {
      NVTX_SCOPE("Layer");
      const T *input_norm =
          reinterpret_cast<const T *>(weights_->LayerInputNorm(layer));
      const T *post_attn_norm =
          reinterpret_cast<const T *>(weights_->LayerPostAttnNorm(layer));

      // Batched Q/K/V projections with fused RmsNorm
      {
        NVTX_SCOPE("QKV_Projection");
        auto q_raw = weights_->LayerQProjRaw(layer);
        auto k_raw = weights_->LayerKProjRaw(layer);
        auto v_raw = weights_->LayerVProjRaw(layer);
        const std::array<PackedProjectionPlan<T>, 3> qkv_plans = {
            {{q_raw, d_q_, num_heads_ * head_dim_},
             {k_raw, d_k_new_, num_kv_heads_ * head_dim_},
             {v_raw, d_v_new_, num_kv_heads_ * head_dim_}}};
        const bool used_q8_1_qkv = TryQ8_1ProjectionGroup(
            qkv_plans, d_residual_, input_norm, d_act_q8_1_, B, hidden_size_,
            rms_norm_eps_, stream_, &execution_policy_);
        const bool used_packed_qkv =
            !used_q8_1_qkv &&
            TryPackedProjectionGroup(qkv_plans, d_residual_, input_norm,
                                     d_norm_out_, d_packed_activation_,
                                     d_packed_activation_scales_, B,
                                     hidden_size_, rms_norm_eps_, stream_,
                                     &execution_policy_);
        if (!used_q8_1_qkv && !used_packed_qkv) {
          bool norm_computed = false;

          if (!TryFusedRmsNormGemv<T>(q_raw, d_residual_, input_norm, d_q_, B,
                                      num_heads_ * head_dim_, hidden_size_,
                                      rms_norm_eps_, stream_, "q_proj",
                                      &execution_policy_)) {
            if (!norm_computed) {
              cuda_kernel::RmsNorm<T>(d_residual_, input_norm, d_norm_out_, B,
                                      hidden_size_, rms_norm_eps_, stream_);
              norm_computed = true;
            }
            if (!TryFusedGemv<T>(q_raw, d_norm_out_, d_q_, B,
                                 num_heads_ * head_dim_, hidden_size_, stream_,
                                 "q_proj", &execution_policy_)) {
              if (capturing) {
                capture_abort = true;
                return false;
              }
              const T *q_proj =
                  reinterpret_cast<const T *>(weights_->LayerQProj(layer));
              gemm_->GemmTyped<T>(B, num_heads_ * head_dim_, hidden_size_,
                                  d_norm_out_, q_proj, d_q_);
            }
          }

          if (!TryFusedRmsNormGemv<T>(k_raw, d_residual_, input_norm, d_k_new_,
                                      B, num_kv_heads_ * head_dim_,
                                      hidden_size_, rms_norm_eps_, stream_,
                                      "k_proj", &execution_policy_)) {
            if (!norm_computed) {
              cuda_kernel::RmsNorm<T>(d_residual_, input_norm, d_norm_out_, B,
                                      hidden_size_, rms_norm_eps_, stream_);
              norm_computed = true;
            }
            if (!TryFusedGemv<T>(k_raw, d_norm_out_, d_k_new_, B,
                                 num_kv_heads_ * head_dim_, hidden_size_,
                                 stream_, "k_proj", &execution_policy_)) {
              if (capturing) {
                capture_abort = true;
                return false;
              }
              const T *k_proj =
                  reinterpret_cast<const T *>(weights_->LayerKProj(layer));
              gemm_->GemmTyped<T>(B, num_kv_heads_ * head_dim_, hidden_size_,
                                  d_norm_out_, k_proj, d_k_new_);
            }
          }

          if (!TryFusedRmsNormGemv<T>(v_raw, d_residual_, input_norm, d_v_new_,
                                      B, num_kv_heads_ * head_dim_,
                                      hidden_size_, rms_norm_eps_, stream_,
                                      "v_proj", &execution_policy_)) {
            if (!norm_computed) {
              cuda_kernel::RmsNorm<T>(d_residual_, input_norm, d_norm_out_, B,
                                      hidden_size_, rms_norm_eps_, stream_);
              norm_computed = true;
            }
            if (!TryFusedGemv<T>(v_raw, d_norm_out_, d_v_new_, B,
                                 num_kv_heads_ * head_dim_, hidden_size_,
                                 stream_, "v_proj", &execution_policy_)) {
              if (capturing) {
                capture_abort = true;
                return false;
              }
              const T *v_proj =
                  reinterpret_cast<const T *>(weights_->LayerVProj(layer));
              gemm_->GemmTyped<T>(B, num_kv_heads_ * head_dim_, hidden_size_,
                                  d_norm_out_, v_proj, d_v_new_);
            }
          }
        }

        // Biases (if present)
        const T *q_bias =
            reinterpret_cast<const T *>(weights_->LayerQProjBias(layer));
        const T *k_bias =
            reinterpret_cast<const T *>(weights_->LayerKProjBias(layer));
        const T *v_bias =
            reinterpret_cast<const T *>(weights_->LayerVProjBias(layer));
        if (q_bias)
          cuda_kernel::BiasAdd<T>(d_q_, q_bias, B, num_heads_ * head_dim_,
                                  stream_);
        if (k_bias)
          cuda_kernel::BiasAdd<T>(d_k_new_, k_bias, B,
                                  num_kv_heads_ * head_dim_, stream_);
        if (v_bias)
          cuda_kernel::BiasAdd<T>(d_v_new_, v_bias, B,
                                  num_kv_heads_ * head_dim_, stream_);
      }
      pt.qkv_ms += pt.Mark();

      // Batched RoPE
      {
        NVTX_SCOPE("RoPE");
        err = cuda_kernel::BatchedRoPE<T>(d_q_, d_k_new_, B, num_heads_,
                                          num_kv_heads_, head_dim_,
                                          d_batch_n_past_, rope_freq_base_,
                                          stream_, rope_type_);
        if (err != cudaSuccess) {
          log::Error("llama_forward",
                     "BatchedRoPE launch failed: " +
                         std::string(cudaGetErrorString(err)));
          return false;
        }
      }
      pt.rope_ms += pt.Mark();

      // KV append: index into pre-computed bulk pointer arrays
      {
        NVTX_SCOPE("KV_Append");
        T **k_ap = static_cast<T **>(d_all_k_append_ptrs_) + layer * B;
        T **v_ap = static_cast<T **>(d_all_v_append_ptrs_) + layer * B;
        int kv_dim = num_kv_heads_ * head_dim_;
        err = cuda_kernel::BatchedKvAppend<T>(d_k_new_, d_v_new_, k_ap, v_ap,
                                              B, kv_dim, stream_);
        if (err != cudaSuccess) {
          log::Error("llama_forward",
                     "BatchedKvAppend launch failed: " +
                         std::string(cudaGetErrorString(err)));
          return false;
        }
      }
      pt.kv_ms += pt.Mark();

      // FlashDecode: index into pre-computed bulk pointer arrays
      {
        NVTX_SCOPE("FlashAttention2");
        const T *const *k_rd =
            static_cast<const T *const *>(d_all_k_read_ptrs_) + layer * B;
        const T *const *v_rd =
            static_cast<const T *const *>(d_all_v_read_ptrs_) + layer * B;
        float attn_scale = 1.0f / sqrtf(static_cast<float>(head_dim_));
        err = cuda_kernel::FlashDecodeMultiSeq<T>(
            d_q_, k_rd, v_rd, d_attn_out_, d_batch_kv_lens_, B, num_heads_,
            num_kv_heads_, head_dim_, attn_scale, stream_);
        if (err != cudaSuccess) {
          log::Error("llama_forward",
                     "FlashDecodeMultiSeq launch failed: " +
                         std::string(cudaGetErrorString(err)));
          return false;
        }
      }
      pt.attn_ms += pt.Mark();

      // O projection
      {
        NVTX_SCOPE("O_Projection");
        auto o_raw = weights_->LayerOProjRaw(layer);
        if (!TryQ8_1Gemv<T>(o_raw, d_attn_out_, d_norm_out_, d_act_q8_1_, B,
                            hidden_size_, num_heads_ * head_dim_, stream_,
                            "o_proj", &execution_policy_) &&
            !TryPackedGemv<T>(o_raw, d_attn_out_, d_norm_out_,
                              d_packed_activation_, d_packed_activation_scales_,
                              B, hidden_size_, num_heads_ * head_dim_, stream_,
                              "o_proj", &execution_policy_) &&
            !TryFusedGemv<T>(o_raw, d_attn_out_, d_norm_out_, B, hidden_size_,
                             num_heads_ * head_dim_, stream_, "o_proj",
                             &execution_policy_)) {
          if (capturing) {
            capture_abort = true;
            return false;
          }
          const T *o_proj =
              reinterpret_cast<const T *>(weights_->LayerOProj(layer));
          gemm_->GemmTyped<T>(B, hidden_size_, num_heads_ * head_dim_,
                              d_attn_out_, o_proj, d_norm_out_);
        }
      }

      // Residual add
      cuda_kernel::ResidualAdd<T>(d_residual_, d_norm_out_, B * hidden_size_,
                                  stream_);
      pt.o_proj_ms += pt.Mark();

      // FFN block
      {
        NVTX_SCOPE("FFN");
        auto gate_raw = weights_->LayerGateProjRaw(layer);
        auto up_raw = weights_->LayerUpProjRaw(layer);
        const std::array<PackedProjectionPlan<T>, 2> ffn_plans = {
            {{gate_raw, d_ffn_gate_, intermediate_size_},
             {up_raw, d_ffn_up_, intermediate_size_}}};
        const auto ffn_selected_op = SelectNativeFfnProjOperator(
            gate_raw, up_raw,
            FusedDispatchGeometry{B, intermediate_size_, hidden_size_, 2, true,
                                  true},
            g_allow_fused_quantized_matmul, execution_policy_);
        bool used_q8_1_ffn = false;
        bool used_packed_ffn = false;
        auto ffn_actual_op = FusedQuantGemm::FfnProjOperator::kFallback;
        if (ffn_selected_op == FusedQuantGemm::FfnProjOperator::kQ81Group ||
            ffn_selected_op ==
                FusedQuantGemm::FfnProjOperator::kQ81GroupHotQ4K) {
          used_q8_1_ffn = TryQ8_1ProjectionGroup(
              ffn_plans, d_residual_, post_attn_norm, d_act_q8_1_, B,
              hidden_size_, rms_norm_eps_, stream_, &execution_policy_);
          if (used_q8_1_ffn) {
            ffn_actual_op = ffn_selected_op;
          }
        } else if (ffn_selected_op ==
                   FusedQuantGemm::FfnProjOperator::kPackedGroup) {
          used_packed_ffn = TryPackedProjectionGroup(
              ffn_plans, d_residual_, post_attn_norm, d_norm_out_,
              d_packed_activation_, d_packed_activation_scales_, B,
              hidden_size_, rms_norm_eps_, stream_, &execution_policy_);
          if (used_packed_ffn) {
            ffn_actual_op = ffn_selected_op;
          }
        }
        if (!used_q8_1_ffn && !used_packed_ffn &&
            ffn_selected_op != FusedQuantGemm::FfnProjOperator::kPackedGroup) {
          used_packed_ffn = TryPackedProjectionGroup(
              ffn_plans, d_residual_, post_attn_norm, d_norm_out_,
              d_packed_activation_, d_packed_activation_scales_, B,
              hidden_size_, rms_norm_eps_, stream_, &execution_policy_);
          if (used_packed_ffn) {
            ffn_actual_op = FusedQuantGemm::FfnProjOperator::kPackedGroup;
          }
        }
        const std::string ffn_quant =
            ProjectionGroupQuantLabel(gate_raw, up_raw);
        GlobalMetrics().RecordNativeFfnProjOperator(
            "decode", FusedQuantGemm::FfnProjOperatorName(ffn_actual_op));
        GlobalMetrics().RecordNativeFfnProjGeometry(
            "decode", FusedQuantGemm::FfnProjOperatorName(ffn_actual_op),
            ffn_quant, B, intermediate_size_, hidden_size_,
            /*grouped_outputs=*/2);
        if (!used_q8_1_ffn && !used_packed_ffn) {
          bool ffn_norm_computed = false;

          if (!TryFusedRmsNormGemv<T>(gate_raw, d_residual_, post_attn_norm,
                                      d_ffn_gate_, B, intermediate_size_,
                                      hidden_size_, rms_norm_eps_, stream_,
                                      "gate_proj", &execution_policy_)) {
            if (!ffn_norm_computed) {
              cuda_kernel::RmsNorm<T>(d_residual_, post_attn_norm, d_norm_out_,
                                      B, hidden_size_, rms_norm_eps_, stream_);
              ffn_norm_computed = true;
            }
            if (!TryFusedGemv<T>(gate_raw, d_norm_out_, d_ffn_gate_, B,
                                 intermediate_size_, hidden_size_, stream_,
                                 "gate_proj", &execution_policy_)) {
              if (capturing) {
                capture_abort = true;
                return false;
              }
              const T *gate_proj =
                  reinterpret_cast<const T *>(weights_->LayerGateProj(layer));
              gemm_->GemmTyped<T>(B, intermediate_size_, hidden_size_,
                                  d_norm_out_, gate_proj, d_ffn_gate_);
            }
          }

          if (!TryFusedRmsNormGemv<T>(up_raw, d_residual_, post_attn_norm,
                                      d_ffn_up_, B, intermediate_size_,
                                      hidden_size_, rms_norm_eps_, stream_,
                                      "up_proj", &execution_policy_)) {
            if (!ffn_norm_computed) {
              cuda_kernel::RmsNorm<T>(d_residual_, post_attn_norm, d_norm_out_,
                                      B, hidden_size_, rms_norm_eps_, stream_);
              ffn_norm_computed = true;
            }
            if (!TryFusedGemv<T>(up_raw, d_norm_out_, d_ffn_up_, B,
                                 intermediate_size_, hidden_size_, stream_,
                                 "up_proj", &execution_policy_)) {
              if (capturing) {
                capture_abort = true;
                return false;
              }
              const T *up_proj =
                  reinterpret_cast<const T *>(weights_->LayerUpProj(layer));
              gemm_->GemmTyped<T>(B, intermediate_size_, hidden_size_,
                                  d_norm_out_, up_proj, d_ffn_up_);
            }
          }
        }
        pt.ffn_proj_ms += pt.Mark();

        auto down_raw = weights_->LayerDownProjRaw(layer);
        auto down_mmq = weights_->LayerDownProjMmq(layer);
        const auto down_selected_op = SelectNativeDownProjOperator(
            down_raw, down_mmq,
            FusedDispatchGeometry{B, hidden_size_, intermediate_size_, 1, true,
                                  false},
            g_allow_fused_quantized_matmul, execution_policy_);
        bool fused_mmq_down = false;
        bool fused_q8_1_down = false;
        bool fused_packed_down = false;
        auto down_actual_op = FusedQuantGemm::DownProjOperator::kFallback;
        if (down_selected_op == FusedQuantGemm::DownProjOperator::kMmq) {
          fused_mmq_down = TryMmqSiluMulGemv<T>(
              down_mmq, d_ffn_gate_, d_ffn_up_, d_ffn_down_, d_act_q8_1_, B,
              hidden_size_, intermediate_size_, stream_, "down_proj",
              &execution_policy_);
          if (fused_mmq_down) {
            down_actual_op = FusedQuantGemm::DownProjOperator::kMmq;
          }
          fused_q8_1_down =
              !fused_mmq_down &&
              TryQ8_1SiluMulGemv<T>(
                  down_raw, d_ffn_gate_, d_ffn_up_, d_ffn_down_, d_act_q8_1_,
                  B, hidden_size_, intermediate_size_, stream_, "down_proj",
                  &execution_policy_);
          if (fused_q8_1_down) {
            down_actual_op = down_selected_op;
          }
        } else if (down_selected_op ==
                   FusedQuantGemm::DownProjOperator::kPackedGemv) {
          fused_packed_down = TryPackedSiluMulGemv<T>(
              down_raw, d_ffn_gate_, d_ffn_up_, d_ffn_down_,
              d_packed_activation_, d_packed_activation_scales_, B,
              hidden_size_, intermediate_size_, stream_, "down_proj",
              &execution_policy_);
          if (fused_packed_down) {
            down_actual_op = down_selected_op;
          }
        } else {
          fused_q8_1_down = TryQ8_1SiluMulGemv<T>(
              down_raw, d_ffn_gate_, d_ffn_up_, d_ffn_down_, d_act_q8_1_, B,
              hidden_size_, intermediate_size_, stream_, "down_proj",
              &execution_policy_);
          if (fused_q8_1_down) {
            down_actual_op = down_selected_op;
          }
          fused_mmq_down =
              !fused_q8_1_down &&
              TryMmqSiluMulGemv<T>(
                  down_mmq, d_ffn_gate_, d_ffn_up_, d_ffn_down_, d_act_q8_1_,
                  B, hidden_size_, intermediate_size_, stream_, "down_proj",
                  &execution_policy_);
          if (fused_mmq_down) {
            down_actual_op = FusedQuantGemm::DownProjOperator::kMmq;
          }
        }
        fused_packed_down =
            !fused_mmq_down && !fused_q8_1_down && !fused_packed_down &&
            TryPackedSiluMulGemv<T>(
                down_raw, d_ffn_gate_, d_ffn_up_, d_ffn_down_,
                d_packed_activation_, d_packed_activation_scales_, B,
                hidden_size_, intermediate_size_, stream_, "down_proj",
                &execution_policy_);
        if (fused_packed_down) {
          down_actual_op = FusedQuantGemm::DownProjOperator::kPackedGemv;
        }
        GlobalMetrics().RecordNativeDownProjOperator(
            "decode", FusedQuantGemm::DownProjOperatorName(down_actual_op));
        GlobalMetrics().RecordNativeDownProjGeometry(
            "decode", FusedQuantGemm::DownProjOperatorName(down_actual_op),
            ProjectionQuantLabel(down_raw), B, hidden_size_,
            intermediate_size_);
        if (down_actual_op != FusedQuantGemm::DownProjOperator::kFallback) {
          const std::string down_label =
              std::string("selected down-proj operator: ") +
              FusedQuantGemm::DownProjOperatorName(down_actual_op);
          LogPackedGemmPath("down_proj", down_label.c_str());
        }
        if (!fused_mmq_down && !fused_q8_1_down && !fused_packed_down) {
          cuda_kernel::SiluMul<T>(d_ffn_gate_, d_ffn_up_, d_ffn_gate_,
                                  B * intermediate_size_, stream_);
          pt.ffn_silu_ms += pt.Mark();

          bool down_ok = false;
          if (down_selected_op == FusedQuantGemm::DownProjOperator::kMmq) {
            down_ok = TryMmqGemv<T>(down_mmq, d_ffn_gate_, d_ffn_down_,
                                    d_act_q8_1_, B, hidden_size_,
                                    intermediate_size_, stream_,
                                    "down_proj", &execution_policy_) ||
                      TryQ8_1Gemv<T>(down_raw, d_ffn_gate_, d_ffn_down_,
                                     d_act_q8_1_, B, hidden_size_,
                                     intermediate_size_, stream_, "down_proj",
                                     &execution_policy_);
          } else {
            down_ok = TryQ8_1Gemv<T>(down_raw, d_ffn_gate_, d_ffn_down_,
                                     d_act_q8_1_, B, hidden_size_,
                                     intermediate_size_, stream_, "down_proj",
                                     &execution_policy_) ||
                      TryMmqGemv<T>(down_mmq, d_ffn_gate_, d_ffn_down_,
                                    d_act_q8_1_, B, hidden_size_,
                                    intermediate_size_, stream_,
                                    "down_proj", &execution_policy_);
          }
          if (!down_ok &&
              !TryPackedGemv<T>(down_raw, d_ffn_gate_, d_ffn_down_,
                                d_packed_activation_,
                                d_packed_activation_scales_, B, hidden_size_,
                                intermediate_size_, stream_, "down_proj",
                                &execution_policy_) &&
              !TryFusedGemv<T>(down_raw, d_ffn_gate_, d_ffn_down_, B,
                               hidden_size_, intermediate_size_, stream_,
                               "down_proj", &execution_policy_)) {
            if (capturing) {
              capture_abort = true;
              return false;
            }
            const T *down_proj =
                reinterpret_cast<const T *>(weights_->LayerDownProj(layer));
            gemm_->GemmTyped<T>(B, hidden_size_, intermediate_size_,
                                d_ffn_gate_, down_proj, d_ffn_down_);
          }
        }
      }

      cuda_kernel::ResidualAdd<T>(d_residual_, d_ffn_down_, B * hidden_size_,
                                  stream_);
      pt.ffn_down_ms += pt.Mark();
    }

    // Final RMSNorm + LM head
    {
      NVTX_SCOPE("LM_Head");
      const T *final_norm = reinterpret_cast<const T *>(weights_->FinalNorm());
      auto lm_raw = weights_->LmHeadRaw();
      // Q8_1 fused RmsNorm+Quantize+GEMV for LM head
      bool lm_q8_1_ok = false;
      if constexpr (std::is_same_v<T, half>) {
        if (!Q81ActivationsDisabled(execution_policy_) && lm_raw.data &&
            d_act_q8_1_ &&
            FusedQuantGemm::SupportsQ8_1Activations(lm_raw.quant_type) &&
            FusedQuantGemm::ShouldUseFusedPath(
                lm_raw.quant_type,
                FusedDispatchGeometry{B, vocab_size_, hidden_size_, 1, true,
                                      true})) {
          FusedQuantGemm::FusedRmsNormQuantizeQ8_1(d_residual_, final_norm,
                                                   d_act_q8_1_, B, hidden_size_,
                                                   rms_norm_eps_, stream_);
          lm_q8_1_ok =
              FusedQuantGemm::GemvQ8_1(lm_raw, d_act_q8_1_, d_logits_typed_, B,
                                       vocab_size_, hidden_size_, stream_,
                                       &execution_policy_);
        }
      }
      if (!lm_q8_1_ok &&
          !TryPackedRmsNormGemv<T>(lm_raw, d_residual_, final_norm, d_norm_out_,
                                   d_packed_activation_,
                                   d_packed_activation_scales_, d_logits_typed_,
                                   B, vocab_size_, hidden_size_, rms_norm_eps_,
                                   stream_, "lm_head", &execution_policy_) &&
          !TryFusedRmsNormGemv<T>(lm_raw, d_residual_, final_norm,
                                  d_logits_typed_, B, vocab_size_, hidden_size_,
                                  rms_norm_eps_, stream_, "lm_head",
                                  &execution_policy_)) {
        cuda_kernel::RmsNorm<T>(d_residual_, final_norm, d_norm_out_, B,
                                hidden_size_, rms_norm_eps_, stream_);
        if (!TryFusedGemv<T>(lm_raw, d_norm_out_, d_logits_typed_, B,
                             vocab_size_, hidden_size_, stream_, "lm_head",
                             &execution_policy_)) {
          if (capturing) {
            capture_abort = true;
            return false;
          }
          const T *lm_head = reinterpret_cast<const T *>(weights_->LmHead());
          gemm_->GemmTyped<T>(B, vocab_size_, hidden_size_, d_norm_out_,
                              lm_head, d_logits_typed_);
        }
      }

      cuda_kernel::HalfToFloat<T>(d_logits_typed_, d_logits, B * vocab_size_,
                                  stream_);
    }
    pt.lm_head_ms += pt.Mark();
    pt.Report(num_layers_, 1);

    return true;
  };

  // Execute compute (captured into graph or run directly)
  bool compute_ok = RunCompute();

  if (capturing) {
    // End capture regardless of compute success — CUDA requires it.
    err = cudaStreamEndCapture(stream_, &decode_graph_);

    if (capture_abort) {
      // cuBLAS fallback was needed but skipped during capture.
      // Discard the incomplete graph and re-execute with cuBLAS.
      if (decode_graph_) {
        cudaGraphDestroy(decode_graph_);
        decode_graph_ = nullptr;
      }
      log::Info("llama_forward",
                "CUDA graph: cuBLAS fallback needed, running without graph");
      graph_enabled_ = false;
      return RunCompute();
    }

    if (err == cudaSuccess && decode_graph_) {
      err = cudaGraphInstantiate(&decode_graph_exec_, decode_graph_, nullptr,
                                 nullptr, 0);
      if (err == cudaSuccess) {
        graph_batch_size_ = B;
        size_t num_nodes = 0;
        cudaGraphGetNodes(decode_graph_, nullptr, &num_nodes);
        log::Info("llama_forward",
                  "CUDA graph captured for B=" + std::to_string(B) + " (" +
                      std::to_string(num_layers_) + " layers, " +
                      std::to_string(num_nodes) + " nodes)");
        err = cudaGraphLaunch(decode_graph_exec_, stream_);
        return err == cudaSuccess;
      }
      log::Warn("llama_forward", "CUDA graph instantiate failed: " +
                                     std::string(cudaGetErrorString(err)));
      cudaGraphDestroy(decode_graph_);
      decode_graph_ = nullptr;
    }
    // Graph capture failed — re-execute without graph
    log::Warn("llama_forward",
              "CUDA graph capture failed, using direct execution");
    graph_enabled_ = false;
    return RunCompute();
  }

  return true;
}

// Explicit template instantiations
template class LlamaForwardTyped<half>;
template class LlamaForwardTyped<__nv_bfloat16>;

} // namespace inferflux
