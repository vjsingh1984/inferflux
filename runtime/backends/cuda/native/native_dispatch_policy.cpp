#include "runtime/backends/cuda/native/native_dispatch_policy.h"

#include "runtime/backends/cuda/native/gguf_util.h"

namespace inferflux {

const NativeExecutionPolicy &
ResolveNativeExecutionPolicy(const NativeExecutionPolicy *policy) {
  if (policy) {
    return *policy;
  }
  static thread_local NativeExecutionPolicy env_policy;
  env_policy = NativeExecutionPolicy::FromEnv();
  return env_policy;
}

std::string ProjectionQuantLabel(const QuantizedWeightInfo &weight) {
  if (!weight.data) {
    return "unknown";
  }
  return runtime::cuda::native::TensorTypeToString(
      static_cast<runtime::cuda::native::GGUF::TensorType>(weight.quant_type));
}

std::string ProjectionGroupQuantLabel(const QuantizedWeightInfo &first,
                                      const QuantizedWeightInfo &second) {
  if (!first.data || !second.data) {
    return "unknown";
  }
  if (first.quant_type == second.quant_type) {
    return ProjectionQuantLabel(first);
  }
  return "mixed";
}

bool ForceCublasRequested(const NativeExecutionPolicy &policy) {
  return policy.force_cublas;
}

bool PackedActivationsDisabled(const NativeExecutionPolicy &policy) {
  return policy.disable_prepacked_activations;
}

bool Q81ActivationsDisabled(const NativeExecutionPolicy &policy) {
  return policy.disable_q81_activations;
}

bool ProjectionHasGraphSafeKernel(const QuantizedWeightInfo &raw,
                                  const FusedDispatchGeometry &geometry,
                                  bool allow_fused_quantized_matmul,
                                  const NativeExecutionPolicy &policy) {
  if (ForceCublasRequested(policy) || !allow_fused_quantized_matmul ||
      !raw.data || raw.quant_type < 0 || geometry.M <= 0 || geometry.N <= 0 ||
      geometry.K <= 0) {
    return false;
  }

  if (!FusedQuantGemm::ShouldUseFusedPath(raw.quant_type, geometry)) {
    return false;
  }

  if (!Q81ActivationsDisabled(policy) &&
      FusedQuantGemm::SupportsQ8_1Activations(raw.quant_type)) {
    return true;
  }
  if (!PackedActivationsDisabled(policy) &&
      FusedQuantGemm::SupportsPackedActivations(raw.quant_type)) {
    return true;
  }

  return true;
}

FusedQuantGemm::DownProjOperator
SelectNativeDownProjOperator(const QuantizedWeightInfo &raw,
                             const MmqWeightInfo &mmq_weight,
                             const FusedDispatchGeometry &geometry,
                             bool allow_fused_quantized_matmul,
                             const NativeExecutionPolicy &policy) {
  if (ForceCublasRequested(policy) || !allow_fused_quantized_matmul) {
    return FusedQuantGemm::DownProjOperator::kFallback;
  }

  const bool allow_q81 = !Q81ActivationsDisabled(policy);
  const bool allow_packed = !PackedActivationsDisabled(policy);
  const bool allow_mmq =
      mmq_weight.data != nullptr && FusedQuantGemm::IsDownProjMmqEnabled(&policy);
  return FusedQuantGemm::SelectDownProjOperator(
      raw.quant_type, geometry, allow_q81, allow_packed, allow_mmq, &policy);
}

FusedQuantGemm::FfnProjOperator
SelectNativeFfnProjOperator(const QuantizedWeightInfo &gate_raw,
                            const QuantizedWeightInfo &up_raw,
                            const FusedDispatchGeometry &geometry,
                            bool allow_fused_quantized_matmul,
                            const NativeExecutionPolicy &policy) {
  if (ForceCublasRequested(policy) || !allow_fused_quantized_matmul) {
    return FusedQuantGemm::FfnProjOperator::kFallback;
  }

  return FusedQuantGemm::SelectFfnProjOperator(
      gate_raw.quant_type, up_raw.quant_type, geometry,
      !Q81ActivationsDisabled(policy), !PackedActivationsDisabled(policy),
      &policy);
}

bool DecodeGraphCaptureSafe(const WeightMap *weights, int num_layers, int M,
                            int hidden_size, int num_heads, int num_kv_heads,
                            int head_dim, int intermediate_size,
                            int vocab_size,
                            bool allow_fused_quantized_matmul,
                            const NativeExecutionPolicy &policy) {
  if (!weights || !weights->HasQuantizedWeights()) {
    return false;
  }

  for (int layer = 0; layer < num_layers; ++layer) {
    if (!ProjectionHasGraphSafeKernel(
            weights->LayerQProjRaw(layer),
            FusedDispatchGeometry{M, num_heads * head_dim, hidden_size, 1, true,
                                  true},
            allow_fused_quantized_matmul, policy) ||
        !ProjectionHasGraphSafeKernel(
            weights->LayerKProjRaw(layer),
            FusedDispatchGeometry{M, num_kv_heads * head_dim, hidden_size, 1,
                                  true, true},
            allow_fused_quantized_matmul, policy) ||
        !ProjectionHasGraphSafeKernel(
            weights->LayerVProjRaw(layer),
            FusedDispatchGeometry{M, num_kv_heads * head_dim, hidden_size, 1,
                                  true, true},
            allow_fused_quantized_matmul, policy) ||
        !ProjectionHasGraphSafeKernel(
            weights->LayerOProjRaw(layer),
            FusedDispatchGeometry{M, hidden_size, num_heads * head_dim, 1,
                                  true, false},
            allow_fused_quantized_matmul, policy) ||
        !ProjectionHasGraphSafeKernel(
            weights->LayerGateProjRaw(layer),
            FusedDispatchGeometry{M, intermediate_size, hidden_size, 1, true,
                                  true},
            allow_fused_quantized_matmul, policy) ||
        !ProjectionHasGraphSafeKernel(
            weights->LayerUpProjRaw(layer),
            FusedDispatchGeometry{M, intermediate_size, hidden_size, 1, true,
                                  true},
            allow_fused_quantized_matmul, policy) ||
        !ProjectionHasGraphSafeKernel(
            weights->LayerDownProjRaw(layer),
            FusedDispatchGeometry{M, hidden_size, intermediate_size, 1, true,
                                  false},
            allow_fused_quantized_matmul, policy)) {
      return false;
    }
  }

  return ProjectionHasGraphSafeKernel(
      weights->LmHeadRaw(),
      FusedDispatchGeometry{M, vocab_size, hidden_size, 1, true, true},
      allow_fused_quantized_matmul, policy);
}

} // namespace inferflux
