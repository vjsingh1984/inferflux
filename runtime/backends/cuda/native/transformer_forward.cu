#include "runtime/backends/cuda/common/dtype_traits.cuh"
#include "runtime/backends/cuda/kernels/flash_attention.cuh"
#include "runtime/backends/cuda/native/cuda_kernels.cuh"
#include "runtime/backends/cuda/native/fused_quant_gemm.h"
#include "runtime/backends/cuda/native/llama_forward.h"
#include "runtime/backends/cuda/native/model_loader.h"
#include "runtime/backends/cuda/native/nvtx_scoped.h"

#include "server/logging/logger.h"

#include <algorithm>
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

// Debug: dump top-K logits to stderr when INFERFLUX_DEBUG_LOGITS=1
void DebugDumpLogits(const float *d_logits, int vocab_size,
                     const std::vector<int> &token_ids, int n_past,
                     cudaStream_t stream) {
  static const bool enabled = std::getenv("INFERFLUX_DEBUG_LOGITS") != nullptr;
  if (!enabled)
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
                     cudaStream_t stream) {
  static const bool enabled = std::getenv("INFERFLUX_DEBUG_LOGITS") != nullptr;
  if (!enabled)
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
  fprintf(stderr, "[DEBUG_HIDDEN] %s: count=%d min=%.6f max=%.6f mean=%.6f nan=%d\n",
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
  log::Info("llama_forward",
            std::string(proj_name) +
                (fused ? ": using fused dequant-GEMV"
                       : ": using cuBLAS (dequantized FP16)"));
}

// Fused dequant-GEMV dispatch: only valid for half (FP16) type.
// Returns true if the fused kernel was launched.
template <typename T>
bool TryFusedGemv(const QuantizedWeightInfo &, const T *, T *, int, int, int,
                  cudaStream_t, const char * = nullptr) {
  return false; // BF16 and other types: no fused path
}

template <>
bool TryFusedGemv<half>(const QuantizedWeightInfo &raw, const half *input,
                        half *output, int M, int N, int K,
                        cudaStream_t stream, const char *proj_name) {
  // If INFERFLUX_FORCE_CUBLAS=1, skip fused kernels entirely
  static const bool force_cublas =
      std::getenv("INFERFLUX_FORCE_CUBLAS") != nullptr;
  if (force_cublas)
    return false;
  bool ok =
      raw.data && FusedQuantGemm::Gemv(raw, input, output, M, N, K, stream);
  if (proj_name)
    LogGemmPath(proj_name, ok);
  return ok;
}

// Fused RmsNorm+GEMV dispatch: computes normalization inside the GEMV kernel,
// eliminating the standalone RmsNorm kernel launch and d_norm_out_ round-trip.
// Only valid for half (FP16) type.
template <typename T>
bool TryFusedRmsNormGemv(const QuantizedWeightInfo &, const T *, const T *,
                         T *, int, int, int, float, cudaStream_t,
                         const char * = nullptr) {
  return false; // BF16 and other types: no fused path
}

template <>
bool TryFusedRmsNormGemv<half>(const QuantizedWeightInfo &raw,
                               const half *residual, const half *norm_weight,
                               half *output, int M, int N, int K, float eps,
                               cudaStream_t stream, const char *proj_name) {
  static const bool force_cublas =
      std::getenv("INFERFLUX_FORCE_CUBLAS") != nullptr;
  if (force_cublas)
    return false;
  bool ok = raw.data && FusedQuantGemm::RmsNormGemv(raw, residual, norm_weight,
                                                    output, M, N, K, eps,
                                                    stream);
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

} // namespace

template <typename T> LlamaForwardTyped<T>::~LlamaForwardTyped() {
  FreeScratchBuffers();
}

template <typename T> bool LlamaForwardTyped<T>::AllocateScratch() {
  auto alloc = [](T **ptr, size_t count) -> bool {
    cudaError_t err = cudaMalloc(ptr, count * sizeof(T));
    return err == cudaSuccess;
  };

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
  // Logits buffer sized for batched decode: [max_batch_size, vocab_size]
  if (!alloc(&d_logits_typed_,
             static_cast<size_t>(max_batch_size_) * vocab_size_))
    return false;

  cudaError_t err = cudaMalloc(&d_token_ids_, rows * sizeof(int));
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
  rope_type_ = static_cast<int>(
      runtime::cuda::native::InferRopeType(config.model_type));

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

  int kv_len = n_past + seq_len;
  cudaError_t err;

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

  DebugDumpHidden("after_embedding", d_hidden_, seq_len * hidden_size_, stream_);

  // Step 3: Copy to residual stream
  err = cudaMemcpyAsync(d_residual_, d_hidden_,
                        (size_t)seq_len * hidden_size_ * sizeof(T),
                        cudaMemcpyDeviceToDevice, stream_);
  if (err != cudaSuccess) {
    log::Error("llama_forward", "Residual copy failed");
    return false;
  }

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
      bool norm_computed = false;

      auto q_raw = weights_->LayerQProjRaw(layer);
      if (!TryFusedRmsNormGemv<T>(q_raw, d_residual_, input_norm, d_q_,
                                  seq_len, num_heads_ * head_dim_,
                                  hidden_size_, rms_norm_eps_, stream_,
                                  "q_proj")) {
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
                             "q_proj")) {
          const T *q_proj =
              reinterpret_cast<const T *>(weights_->LayerQProj(layer));
          if (!gemm_->GemmTyped<T>(seq_len, num_heads_ * head_dim_,
                                   hidden_size_, d_norm_out_, q_proj, d_q_)) {
            log::Error("llama_forward", "Q projection failed");
            return false;
          }
        }
      }

      auto k_raw = weights_->LayerKProjRaw(layer);
      if (!TryFusedRmsNormGemv<T>(k_raw, d_residual_, input_norm, d_k_new_,
                                  seq_len, num_kv_heads_ * head_dim_,
                                  hidden_size_, rms_norm_eps_, stream_,
                                  "k_proj")) {
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
                             "k_proj")) {
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

      auto v_raw = weights_->LayerVProjRaw(layer);
      if (!TryFusedRmsNormGemv<T>(v_raw, d_residual_, input_norm, d_v_new_,
                                  seq_len, num_kv_heads_ * head_dim_,
                                  hidden_size_, rms_norm_eps_, stream_,
                                  "v_proj")) {
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
                             "v_proj")) {
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

    if (layer == 0) {
      DebugDumpHidden("layer0_after_q_proj", d_q_,
                      seq_len * num_heads_ * head_dim_, stream_);
      DebugDumpHidden("layer0_after_k_proj", d_k_new_,
                      seq_len * num_kv_heads_ * head_dim_, stream_);
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

    // 4h: O projection
    {
      NVTX_SCOPE("O_Projection");
      auto o_raw = weights_->LayerOProjRaw(layer);
      if (!TryFusedGemv<T>(o_raw, d_attn_out_, d_norm_out_, seq_len,
                           hidden_size_, num_heads_ * head_dim_, stream_,
                           "o_proj")) {
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

    // 4j-n: FFN block
    {
      NVTX_SCOPE("FFN");
      bool ffn_norm_computed = false;

      // Gate projection: try fused RmsNorm+GEMV (post-attn norm)
      auto gate_raw = weights_->LayerGateProjRaw(layer);
      if (!TryFusedRmsNormGemv<T>(gate_raw, d_residual_, post_attn_norm,
                                  d_ffn_gate_, seq_len, intermediate_size_,
                                  hidden_size_, rms_norm_eps_, stream_,
                                  "gate_proj")) {
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
                             "gate_proj")) {
          const T *gate_proj =
              reinterpret_cast<const T *>(weights_->LayerGateProj(layer));
          if (!gemm_->GemmTyped<T>(seq_len, intermediate_size_, hidden_size_,
                                   d_norm_out_, gate_proj, d_ffn_gate_)) {
            log::Error("llama_forward", "Gate projection failed");
            return false;
          }
        }
      }

      // Up projection: try fused RmsNorm+GEMV (post-attn norm)
      auto up_raw = weights_->LayerUpProjRaw(layer);
      if (!TryFusedRmsNormGemv<T>(up_raw, d_residual_, post_attn_norm,
                                  d_ffn_up_, seq_len, intermediate_size_,
                                  hidden_size_, rms_norm_eps_, stream_,
                                  "up_proj")) {
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
                             "up_proj")) {
          const T *up_proj =
              reinterpret_cast<const T *>(weights_->LayerUpProj(layer));
          if (!gemm_->GemmTyped<T>(seq_len, intermediate_size_, hidden_size_,
                                   d_norm_out_, up_proj, d_ffn_up_)) {
            log::Error("llama_forward", "Up projection failed");
            return false;
          }
        }
      }

      // SwiGLU
      err = cuda_kernel::SiluMul<T>(d_ffn_gate_, d_ffn_up_, d_ffn_gate_,
                                    seq_len * intermediate_size_, stream_);
      if (err != cudaSuccess) {
        log::Error("llama_forward", "SiluMul failed");
        return false;
      }

      // Down projection (input is activation, not normalized — no fusion)
      auto down_raw = weights_->LayerDownProjRaw(layer);
      if (!TryFusedGemv<T>(down_raw, d_ffn_gate_, d_ffn_down_, seq_len,
                           hidden_size_, intermediate_size_, stream_,
                           "down_proj")) {
        const T *down_proj =
            reinterpret_cast<const T *>(weights_->LayerDownProj(layer));
        if (!gemm_->GemmTyped<T>(seq_len, hidden_size_, intermediate_size_,
                                 d_ffn_gate_, down_proj, d_ffn_down_)) {
          log::Error("llama_forward", "Down projection failed");
          return false;
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
  }

  // Step 5: Final RMSNorm + LM head (last token only)
  {
    NVTX_SCOPE("LM_Head");
    T *last_hidden = d_residual_ + (seq_len - 1) * hidden_size_;
    const T *final_norm = reinterpret_cast<const T *>(weights_->FinalNorm());

    // Try fused RmsNorm+GEMV for LM head
    auto lm_raw = weights_->LmHeadRaw();
    if (!TryFusedRmsNormGemv<T>(lm_raw, last_hidden, final_norm,
                                d_logits_typed_, 1, vocab_size_, hidden_size_,
                                rms_norm_eps_, stream_, "lm_head")) {
      // Fallback: standalone RmsNorm + GEMV/cuBLAS
      err = cuda_kernel::RmsNorm<T>(last_hidden, final_norm, d_norm_out_, 1,
                                    hidden_size_, rms_norm_eps_, stream_);
      if (err != cudaSuccess) {
        log::Error("llama_forward", "Final RmsNorm failed");
        return false;
      }
      if (!TryFusedGemv<T>(lm_raw, d_norm_out_, d_logits_typed_, 1,
                           vocab_size_, hidden_size_, stream_, "lm_head")) {
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

    DebugDumpLogits(d_logits, vocab_size_, token_ids, n_past, stream_);
  }

  return true;
}

template <typename T>
void LlamaForwardTyped<T>::SetStream(cudaStream_t stream) {
  stream_ = stream;
}

template <typename T>
bool LlamaForwardTyped<T>::BatchForward(const std::vector<int> &token_ids,
                                        const std::vector<int> &n_past,
                                        const std::vector<int> &sequence_ids,
                                        float *d_logits, int batch_size) {
  NVTX_SCOPE("BatchForward");
  if (batch_size <= 0)
    return false;
  if (batch_size == 1) {
    std::vector<int> single = {token_ids[0]};
    return Forward(single, n_past[0], sequence_ids[0], d_logits);
  }

  // Each decode request has exactly 1 token.
  // Batch GEMMs (M=batch_size), but attention and KV cache are per-sequence.
  int B = batch_size;
  cudaError_t err;

  // Step 1: Upload all token IDs [B] and batch metadata
  err = cudaMemcpyAsync(d_token_ids_, token_ids.data(), B * sizeof(int),
                        cudaMemcpyHostToDevice, stream_);
  if (err != cudaSuccess) {
    log::Error("llama_forward", "BatchForward: token upload failed");
    return false;
  }
  // Upload n_past and sequence_ids for batched RoPE/KV/attention kernels
  cudaMemcpyAsync(d_batch_n_past_, n_past.data(), B * sizeof(int),
                  cudaMemcpyHostToDevice, stream_);
  cudaMemcpyAsync(d_batch_seq_ids_, sequence_ids.data(), B * sizeof(int),
                  cudaMemcpyHostToDevice, stream_);
  // kv_lens = n_past + 1 for each sequence (computed on host, uploaded once)
  {
    int h_kv_lens[64];
    for (int b = 0; b < B; ++b)
      h_kv_lens[b] = n_past[b] + 1;
    cudaMemcpyAsync(d_batch_kv_lens_, h_kv_lens, B * sizeof(int),
                    cudaMemcpyHostToDevice, stream_);
  }

  // Step 2: Batched embedding [B, hidden_size]
  {
    NVTX_SCOPE("Embedding");
    const T *embed = reinterpret_cast<const T *>(weights_->EmbedTokens());
    err = cuda_kernel::EmbeddingLookup<T>(embed, d_token_ids_, d_hidden_, B,
                                          hidden_size_, stream_);
    if (err != cudaSuccess) {
      log::Error("llama_forward", "BatchForward: embedding failed");
      return false;
    }
  }

  // Step 3: Copy to residual stream
  err = cudaMemcpyAsync(d_residual_, d_hidden_,
                        (size_t)B * hidden_size_ * sizeof(T),
                        cudaMemcpyDeviceToDevice, stream_);
  if (err != cudaSuccess) {
    log::Error("llama_forward", "BatchForward: residual copy failed");
    return false;
  }

  // Step 4: Transformer layers
  for (int layer = 0; layer < num_layers_; layer++) {
    NVTX_SCOPE("Layer");
    const T *input_norm =
        reinterpret_cast<const T *>(weights_->LayerInputNorm(layer));
    const T *post_attn_norm =
        reinterpret_cast<const T *>(weights_->LayerPostAttnNorm(layer));

    // Batched Q/K/V projections (M=B) with fused RmsNorm
    {
      NVTX_SCOPE("QKV_Projection");
      bool norm_computed = false;

      auto q_raw = weights_->LayerQProjRaw(layer);
      if (!TryFusedRmsNormGemv<T>(q_raw, d_residual_, input_norm, d_q_, B,
                                  num_heads_ * head_dim_, hidden_size_,
                                  rms_norm_eps_, stream_, "q_proj")) {
        if (!norm_computed) {
          err = cuda_kernel::RmsNorm<T>(d_residual_, input_norm, d_norm_out_, B,
                                        hidden_size_, rms_norm_eps_, stream_);
          if (err != cudaSuccess) {
            log::Error("llama_forward", "BatchForward: RmsNorm failed");
            return false;
          }
          norm_computed = true;
        }
        if (!TryFusedGemv<T>(q_raw, d_norm_out_, d_q_, B,
                             num_heads_ * head_dim_, hidden_size_, stream_,
                             "q_proj")) {
          const T *q_proj =
              reinterpret_cast<const T *>(weights_->LayerQProj(layer));
          if (!gemm_->GemmTyped<T>(B, num_heads_ * head_dim_, hidden_size_,
                                   d_norm_out_, q_proj, d_q_)) {
            log::Error("llama_forward", "BatchForward: Q projection failed");
            return false;
          }
        }
      }

      auto k_raw = weights_->LayerKProjRaw(layer);
      if (!TryFusedRmsNormGemv<T>(k_raw, d_residual_, input_norm, d_k_new_, B,
                                  num_kv_heads_ * head_dim_, hidden_size_,
                                  rms_norm_eps_, stream_, "k_proj")) {
        if (!norm_computed) {
          err = cuda_kernel::RmsNorm<T>(d_residual_, input_norm, d_norm_out_, B,
                                        hidden_size_, rms_norm_eps_, stream_);
          if (err != cudaSuccess) {
            log::Error("llama_forward", "BatchForward: RmsNorm failed");
            return false;
          }
          norm_computed = true;
        }
        if (!TryFusedGemv<T>(k_raw, d_norm_out_, d_k_new_, B,
                             num_kv_heads_ * head_dim_, hidden_size_, stream_,
                             "k_proj")) {
          const T *k_proj =
              reinterpret_cast<const T *>(weights_->LayerKProj(layer));
          if (!gemm_->GemmTyped<T>(B, num_kv_heads_ * head_dim_, hidden_size_,
                                   d_norm_out_, k_proj, d_k_new_)) {
            log::Error("llama_forward", "BatchForward: K projection failed");
            return false;
          }
        }
      }

      auto v_raw = weights_->LayerVProjRaw(layer);
      if (!TryFusedRmsNormGemv<T>(v_raw, d_residual_, input_norm, d_v_new_, B,
                                  num_kv_heads_ * head_dim_, hidden_size_,
                                  rms_norm_eps_, stream_, "v_proj")) {
        if (!norm_computed) {
          err = cuda_kernel::RmsNorm<T>(d_residual_, input_norm, d_norm_out_, B,
                                        hidden_size_, rms_norm_eps_, stream_);
          if (err != cudaSuccess) {
            log::Error("llama_forward", "BatchForward: RmsNorm failed");
            return false;
          }
          norm_computed = true;
        }
        if (!TryFusedGemv<T>(v_raw, d_norm_out_, d_v_new_, B,
                             num_kv_heads_ * head_dim_, hidden_size_, stream_,
                             "v_proj")) {
          const T *v_proj =
              reinterpret_cast<const T *>(weights_->LayerVProj(layer));
          if (!gemm_->GemmTyped<T>(B, num_kv_heads_ * head_dim_, hidden_size_,
                                   d_norm_out_, v_proj, d_v_new_)) {
            log::Error("llama_forward", "BatchForward: V projection failed");
            return false;
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
      if (q_bias) {
        cuda_kernel::BiasAdd<T>(d_q_, q_bias, B, num_heads_ * head_dim_,
                                stream_);
      }
      if (k_bias) {
        cuda_kernel::BiasAdd<T>(d_k_new_, k_bias, B, num_kv_heads_ * head_dim_,
                                stream_);
      }
      if (v_bias) {
        cuda_kernel::BiasAdd<T>(d_v_new_, v_bias, B, num_kv_heads_ * head_dim_,
                                stream_);
      }
    }

    // Batched RoPE, KV append, and FlashDecode for all B sequences
    auto *typed_cache = static_cast<KvCacheGpuTyped<T> *>(
        static_cast<IKvCacheGpu *>(kv_cache_));

    // Batched RoPE: single kernel for all B sequences
    {
      NVTX_SCOPE("RoPE");
      err = cuda_kernel::BatchedRoPE<T>(d_q_, d_k_new_, B, num_heads_,
                                        num_kv_heads_, head_dim_,
                                        d_batch_n_past_, rope_freq_base_,
                                        stream_, rope_type_);
      if (err != cudaSuccess)
        return false;
    }

    // Batched KV cache append: compute dest pointers on host, upload, scatter
    {
      NVTX_SCOPE("KV_Append");
      T *h_k_ptrs[64], *h_v_ptrs[64]; // Stack arrays, B <= max_batch_size_
      typed_cache->GetBatchAppendPtrs(layer, sequence_ids.data(), n_past.data(),
                                      B, h_k_ptrs, h_v_ptrs);
      cudaMemcpyAsync(d_k_append_ptrs_, h_k_ptrs, B * sizeof(T *),
                       cudaMemcpyHostToDevice, stream_);
      cudaMemcpyAsync(d_v_append_ptrs_, h_v_ptrs, B * sizeof(T *),
                       cudaMemcpyHostToDevice, stream_);
      int kv_dim = num_kv_heads_ * head_dim_;
      err = cuda_kernel::BatchedKvAppend<T>(
          d_k_new_, d_v_new_, static_cast<T **>(d_k_append_ptrs_),
          static_cast<T **>(d_v_append_ptrs_), B, kv_dim, stream_);
      if (err != cudaSuccess)
        return false;
    }

    // Batched FlashDecode: single kernel for all B queries
    {
      NVTX_SCOPE("FlashAttention2");
      const T *h_k_ptrs[64], *h_v_ptrs[64];
      typed_cache->GetBatchKVPtrs(layer, sequence_ids.data(), B,
                                  h_k_ptrs, h_v_ptrs);
      cudaMemcpyAsync(d_k_ptrs_, h_k_ptrs, B * sizeof(T *),
                       cudaMemcpyHostToDevice, stream_);
      cudaMemcpyAsync(d_v_ptrs_, h_v_ptrs, B * sizeof(T *),
                       cudaMemcpyHostToDevice, stream_);

      float attn_scale = 1.0f / sqrtf(static_cast<float>(head_dim_));
      err = cuda_kernel::FlashDecodeMultiSeq<T>(
          d_q_, static_cast<const T *const *>(d_k_ptrs_),
          static_cast<const T *const *>(d_v_ptrs_), d_attn_out_,
          d_batch_kv_lens_, B, num_heads_, num_kv_heads_, head_dim_,
          attn_scale, stream_);
      if (err != cudaSuccess)
        return false;
    }

    // Batched O projection (M=B)
    {
      NVTX_SCOPE("O_Projection");
      auto o_raw = weights_->LayerOProjRaw(layer);
      if (!TryFusedGemv<T>(o_raw, d_attn_out_, d_norm_out_, B, hidden_size_,
                           num_heads_ * head_dim_, stream_, "o_proj")) {
        const T *o_proj =
            reinterpret_cast<const T *>(weights_->LayerOProj(layer));
        if (!gemm_->GemmTyped<T>(B, hidden_size_, num_heads_ * head_dim_,
                                 d_attn_out_, o_proj, d_norm_out_)) {
          log::Error("llama_forward", "BatchForward: O projection failed");
          return false;
        }
      }
    }

    // Residual add
    err = cuda_kernel::ResidualAdd<T>(d_residual_, d_norm_out_,
                                      B * hidden_size_, stream_);
    if (err != cudaSuccess)
      return false;

    // FFN block
    {
      NVTX_SCOPE("FFN");
      bool ffn_norm_computed = false;

      auto gate_raw = weights_->LayerGateProjRaw(layer);
      if (!TryFusedRmsNormGemv<T>(gate_raw, d_residual_, post_attn_norm,
                                  d_ffn_gate_, B, intermediate_size_,
                                  hidden_size_, rms_norm_eps_, stream_,
                                  "gate_proj")) {
        if (!ffn_norm_computed) {
          err = cuda_kernel::RmsNorm<T>(d_residual_, post_attn_norm,
                                        d_norm_out_, B, hidden_size_,
                                        rms_norm_eps_, stream_);
          if (err != cudaSuccess)
            return false;
          ffn_norm_computed = true;
        }
        if (!TryFusedGemv<T>(gate_raw, d_norm_out_, d_ffn_gate_, B,
                             intermediate_size_, hidden_size_, stream_,
                             "gate_proj")) {
          const T *gate_proj =
              reinterpret_cast<const T *>(weights_->LayerGateProj(layer));
          if (!gemm_->GemmTyped<T>(B, intermediate_size_, hidden_size_,
                                   d_norm_out_, gate_proj, d_ffn_gate_))
            return false;
        }
      }

      auto up_raw = weights_->LayerUpProjRaw(layer);
      if (!TryFusedRmsNormGemv<T>(up_raw, d_residual_, post_attn_norm,
                                  d_ffn_up_, B, intermediate_size_,
                                  hidden_size_, rms_norm_eps_, stream_,
                                  "up_proj")) {
        if (!ffn_norm_computed) {
          err = cuda_kernel::RmsNorm<T>(d_residual_, post_attn_norm,
                                        d_norm_out_, B, hidden_size_,
                                        rms_norm_eps_, stream_);
          if (err != cudaSuccess)
            return false;
          ffn_norm_computed = true;
        }
        if (!TryFusedGemv<T>(up_raw, d_norm_out_, d_ffn_up_, B,
                             intermediate_size_, hidden_size_, stream_,
                             "up_proj")) {
          const T *up_proj =
              reinterpret_cast<const T *>(weights_->LayerUpProj(layer));
          if (!gemm_->GemmTyped<T>(B, intermediate_size_, hidden_size_,
                                   d_norm_out_, up_proj, d_ffn_up_))
            return false;
        }
      }

      err = cuda_kernel::SiluMul<T>(d_ffn_gate_, d_ffn_up_, d_ffn_gate_,
                                    B * intermediate_size_, stream_);
      if (err != cudaSuccess)
        return false;

      auto down_raw = weights_->LayerDownProjRaw(layer);
      if (!TryFusedGemv<T>(down_raw, d_ffn_gate_, d_ffn_down_, B, hidden_size_,
                           intermediate_size_, stream_, "down_proj")) {
        const T *down_proj =
            reinterpret_cast<const T *>(weights_->LayerDownProj(layer));
        if (!gemm_->GemmTyped<T>(B, hidden_size_, intermediate_size_,
                                 d_ffn_gate_, down_proj, d_ffn_down_))
          return false;
      }
    }

    err = cuda_kernel::ResidualAdd<T>(d_residual_, d_ffn_down_,
                                      B * hidden_size_, stream_);
    if (err != cudaSuccess)
      return false;
  }

  // Step 5: Final RMSNorm + LM head for each sequence
  {
    NVTX_SCOPE("LM_Head");
    const T *final_norm = reinterpret_cast<const T *>(weights_->FinalNorm());

    // Try fused RmsNorm+GEMV for batched LM head
    auto lm_raw = weights_->LmHeadRaw();
    if (!TryFusedRmsNormGemv<T>(lm_raw, d_residual_, final_norm,
                                d_logits_typed_, B, vocab_size_, hidden_size_,
                                rms_norm_eps_, stream_, "lm_head")) {
      // Fallback: standalone RmsNorm + GEMV/cuBLAS
      err = cuda_kernel::RmsNorm<T>(d_residual_, final_norm, d_norm_out_, B,
                                    hidden_size_, rms_norm_eps_, stream_);
      if (err != cudaSuccess)
        return false;
      if (!TryFusedGemv<T>(lm_raw, d_norm_out_, d_logits_typed_, B, vocab_size_,
                           hidden_size_, stream_, "lm_head")) {
        const T *lm_head = reinterpret_cast<const T *>(weights_->LmHead());
        if (!gemm_->GemmTyped<T>(B, vocab_size_, hidden_size_, d_norm_out_,
                                 lm_head, d_logits_typed_))
          return false;
      }
    }

    // Convert [B * vocab_size] typed -> float
    err = cuda_kernel::HalfToFloat<T>(d_logits_typed_, d_logits,
                                      B * vocab_size_, stream_);
    if (err != cudaSuccess)
      return false;
  }

  return true;
}

// Explicit template instantiations
template class LlamaForwardTyped<half>;
template class LlamaForwardTyped<__nv_bfloat16>;

} // namespace inferflux
