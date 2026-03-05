#include "runtime/backends/cuda/native/quantized_forward.h"
#include "runtime/backends/cuda/native/nvtx_scoped.h"
#include "server/logging/logger.h"

#include <cstring>
#include <limits>

namespace inferflux {
namespace {

bool CheckedMul(std::size_t a, std::size_t b, std::size_t *out) {
  if (!out) {
    return false;
  }
  if (a != 0U && b > (std::numeric_limits<std::size_t>::max() / a)) {
    return false;
  }
  *out = a * b;
  return true;
}

bool CheckedBytesForCount(int count, std::size_t element_size,
                          std::size_t *out) {
  if (count < 0) {
    return false;
  }
  return CheckedMul(static_cast<std::size_t>(count), element_size, out);
}

bool CheckedBytesForProduct(int a, int b, std::size_t element_size,
                            std::size_t *out) {
  if (a < 0 || b < 0) {
    return false;
  }
  std::size_t elements = 0;
  if (!CheckedMul(static_cast<std::size_t>(a), static_cast<std::size_t>(b),
                  &elements)) {
    return false;
  }
  return CheckedMul(elements, element_size, out);
}

bool CheckCudaStatus(cudaError_t status, const std::string &operation) {
  if (status == cudaSuccess) {
    return true;
  }
  log::Error("quantized_forward",
             operation + " failed: " + cudaGetErrorString(status));
  return false;
}

} // namespace

QuantizedForward::~QuantizedForward() { FreeScratchBuffersImpl(); }

bool QuantizedForward::Initialize(const ModelInfo &config,
                                  QuantizedWeightMap *weights,
                                  IKvCacheGpu *kv_cache, CublasGemm *gemm,
                                  cudaStream_t stream) {
  if (!weights || !kv_cache || !gemm) {
    log::Error("quantized_forward", "Null dependency");
    return false;
  }

  // Store config
  hidden_size_ = config.hidden_size;
  num_layers_ = config.num_hidden_layers;
  num_heads_ = config.num_attention_heads;
  num_kv_heads_ = config.num_key_value_heads;
  head_dim_ = config.head_dim;
  if (head_dim_ == 0) {
    head_dim_ = hidden_size_ / num_heads_;
  }
  intermediate_size_ = config.intermediate_size;
  vocab_size_ = config.vocab_size;
  max_seq_len_ = config.max_position_embeddings;
  rope_freq_base_ = config.rope_freq_base;
  rms_norm_eps_ = config.rms_norm_eps;
  model_type_ = config.model_type;

  if (hidden_size_ <= 0 || num_layers_ <= 0 || num_heads_ <= 0 ||
      num_kv_heads_ <= 0 || intermediate_size_ <= 0 || vocab_size_ <= 0 ||
      max_seq_len_ <= 0) {
    log::Error("quantized_forward", "Invalid model dimensions in config");
    return false;
  }

  // Store references
  weights_ = weights;
  kv_cache_ = kv_cache;
  gemm_ = gemm;
  stream_ = stream;

  // Get quantization info
  is_quantized_ = weights->IsQuantized();
  quantization_type_ = weights->GetQuantizationType();

  log::Info("quantized_forward",
            "Initializing: " + model_type_ +
                " layers=" + std::to_string(num_layers_) +
                " hidden=" + std::to_string(hidden_size_) +
                " quantized=" + (is_quantized_ ? quantization_type_ : "false"));

  // Allocate scratch buffers
  if (!AllocateScratch()) {
    log::Error("quantized_forward", "Failed to allocate scratch buffers");
    return false;
  }

  return true;
}

// Compatibility shim for ModelForward interface
bool QuantizedForward::Initialize(const SafetensorsLoader::ModelConfig &config,
                                  const WeightMap &weights,
                                  IKvCacheGpu *kv_cache, CublasGemm *gemm,
                                  cudaStream_t stream) {
  // This should not be called for quantized models
  log::Warn("quantized_forward",
            "Initialize() called with non-quantized types - not supported");
  return false;
}

bool QuantizedForward::Forward(const std::vector<int> &token_ids, int n_past,
                               int sequence_id, float *d_logits) {
  NVTX_SCOPE("QuantizedForward::Forward");

  return RunForwardPass(token_ids, n_past, sequence_id, d_logits);
}

bool QuantizedForward::BatchForward(const std::vector<int> &token_ids,
                                    const std::vector<int> &n_past,
                                    const std::vector<int> &sequence_ids,
                                    float *d_logits, int batch_size) {
  // For now, fall back to sequential calls
  // TODO: Implement true batched forward for quantized models
  for (int i = 0; i < batch_size; ++i) {
    std::vector<int> single_token = {token_ids[i]};
    const std::size_t logits_offset =
        static_cast<std::size_t>(i) * static_cast<std::size_t>(vocab_size_);
    if (!Forward(single_token, n_past[i], sequence_ids[i],
                 d_logits + logits_offset)) {
      return false;
    }
  }
  return true;
}

void QuantizedForward::SetStream(cudaStream_t stream) {
  stream_ = stream;
  if (gemm_) {
    gemm_->SetStream(stream);
  }
}

void QuantizedForward::FreeScratchBuffers() { FreeScratchBuffersImpl(); }

void QuantizedForward::FreeScratchBuffersImpl() {
#define CUDA_FREE(ptr)                                                         \
  do {                                                                         \
    if (ptr) {                                                                 \
      if (!CheckCudaStatus(cudaFree(ptr),                                      \
                           std::string("cudaFree(" #ptr ")"))) {               \
        log::Warn("quantized_forward",                                         \
                  "Continuing cleanup after cudaFree failure for " #ptr);      \
      }                                                                        \
      ptr = nullptr;                                                           \
    }                                                                          \
  } while (0)

  CUDA_FREE(d_hidden_);
  CUDA_FREE(d_residual_);
  CUDA_FREE(d_norm_out_);
  CUDA_FREE(d_q_);
  CUDA_FREE(d_k_new_);
  CUDA_FREE(d_v_new_);
  CUDA_FREE(d_attn_out_);
  CUDA_FREE(d_ffn_gate_);
  CUDA_FREE(d_ffn_up_);
  CUDA_FREE(d_ffn_down_);
  CUDA_FREE(d_token_ids_);

#undef CUDA_FREE
}

bool QuantizedForward::AllocateScratch() {
  std::size_t hidden_bytes = 0;
  std::size_t intermediate_bytes = 0;
  std::size_t q_bytes = 0;
  std::size_t kv_bytes = 0;
  std::size_t token_ids_bytes = 0;

  if (!CheckedBytesForCount(hidden_size_, sizeof(half), &hidden_bytes) ||
      !CheckedBytesForCount(intermediate_size_, sizeof(half),
                            &intermediate_bytes) ||
      !CheckedBytesForProduct(hidden_size_, num_heads_, sizeof(half),
                              &q_bytes) ||
      !CheckedBytesForProduct(hidden_size_, num_kv_heads_, sizeof(half),
                              &kv_bytes) ||
      !CheckedBytesForCount(max_seq_len_, sizeof(int), &token_ids_bytes)) {
    log::Error("quantized_forward",
               "Scratch buffer size overflow or invalid dimensions");
    return false;
  }

#define CUDA_ALLOC(ptr, size)                                                  \
  do {                                                                         \
    cudaError_t err = cudaMalloc(&ptr, size);                                  \
    if (err != cudaSuccess) {                                                  \
      log::Error("quantized_forward",                                          \
                 "Failed to allocate " #ptr ": " +                             \
                     std::string(cudaGetErrorString(err)));                    \
      FreeScratchBuffersImpl();                                                \
      return false;                                                            \
    }                                                                          \
  } while (0)

  CUDA_ALLOC(d_hidden_, hidden_bytes);
  CUDA_ALLOC(d_residual_, hidden_bytes);
  CUDA_ALLOC(d_norm_out_, hidden_bytes);
  CUDA_ALLOC(d_q_, q_bytes);
  CUDA_ALLOC(d_k_new_, kv_bytes);
  CUDA_ALLOC(d_v_new_, kv_bytes);
  CUDA_ALLOC(d_attn_out_, hidden_bytes);
  CUDA_ALLOC(d_ffn_gate_, intermediate_bytes);
  CUDA_ALLOC(d_ffn_up_, intermediate_bytes);
  CUDA_ALLOC(d_ffn_down_, hidden_bytes);
  CUDA_ALLOC(d_token_ids_, token_ids_bytes);

#undef CUDA_ALLOC

  log::Info("quantized_forward", "Allocated scratch buffers");
  return true;
}

bool QuantizedForward::RunForwardPass(const std::vector<int> &token_ids,
                                      int n_past, int sequence_id,
                                      float *d_logits) {
  if (token_ids.empty()) {
    log::Error("quantized_forward", "Empty token_ids");
    return false;
  }

  if (token_ids.size() >
      static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    log::Error("quantized_forward", "token_ids size exceeds int range");
    return false;
  }
  std::size_t token_copy_bytes = 0;
  if (!CheckedMul(token_ids.size(), sizeof(int), &token_copy_bytes)) {
    log::Error("quantized_forward", "token_ids byte-size overflow");
    return false;
  }

  // Copy token IDs to GPU
  if (!CheckCudaStatus(cudaMemcpyAsync(d_token_ids_, token_ids.data(),
                                       token_copy_bytes, cudaMemcpyHostToDevice,
                                       stream_),
                       "cudaMemcpyAsync(token_ids)")) {
    return false;
  }

  // Embed tokens
  const half *embed_weights = weights_->EmbedTokens();
  if (!embed_weights) {
    log::Error("quantized_forward", "Missing embed tokens");
    return false;
  }

  // Embedding lookup: d_hidden = embed_weights[token_ids]
  // For simplicity, use a basic implementation
  // TODO: Optimize with fused kernel

  // Run transformer layers
  for (int layer = 0; layer < num_layers_; ++layer) {
    // Input RMS norm
    const half *input_norm_weights = weights_->LayerInputNorm(layer);
    ComputeRMSNorm(d_hidden_, input_norm_weights, d_norm_out_, hidden_size_,
                   rms_norm_eps_);

    // Attention
    if (!ComputeAttention(layer, n_past, sequence_id)) {
      return false;
    }

    // Residual connection
    // d_hidden = d_hidden + d_attn_out

    // Post-attention RMS norm
    const half *post_norm_weights = weights_->LayerPostAttnNorm(layer);
    ComputeRMSNorm(d_hidden_, post_norm_weights, d_norm_out_, hidden_size_,
                   rms_norm_eps_);

    // FFN
    if (!ComputeFFN(layer)) {
      return false;
    }

    // Residual connection
    // d_hidden = d_hidden + d_ffn_down
  }

  // Final RMS norm
  const half *final_norm_weights = weights_->FinalNorm();
  ComputeRMSNorm(d_hidden_, final_norm_weights, d_norm_out_, hidden_size_,
                 rms_norm_eps_);

  // LM head
  const half *lm_head_weights = weights_->LmHead();
  if (!lm_head_weights) {
    log::Error("quantized_forward", "Missing lm_head");
    return false;
  }

  // Compute logits: d_logits = d_hidden * lm_head_weights^T
  if (!gemm_->Gemm(1, vocab_size_, hidden_size_, d_norm_out_, lm_head_weights,
                   reinterpret_cast<half *>(d_logits))) {
    log::Error("quantized_forward", "LM head GEMM failed");
    return false;
  }

  return true;
}

bool QuantizedForward::ComputeAttention(int layer, int n_past,
                                        int sequence_id) {
  const half *q_proj = weights_->LayerQProj(layer);
  const half *k_proj = weights_->LayerKProj(layer);
  const half *v_proj = weights_->LayerVProj(layer);
  const half *o_proj = weights_->LayerOProj(layer);

  if (!q_proj || !k_proj || !v_proj || !o_proj) {
    log::Error("quantized_forward",
               "Missing attention weights for layer " + std::to_string(layer));
    return false;
  }

  // QKV projections
  // d_q = d_hidden * q_proj^T
  // d_k_new = d_hidden * k_proj^T
  // d_v_new = d_hidden * v_proj^T

  // Apply RoPE
  ApplyRoPE(d_q_, d_k_new_, n_past, layer);

  // Store K, V in cache
  // TODO: Implement KV cache storage

  // Attention computation
  // TODO: Implement attention with cached K, V

  // Output projection
  // d_attn_out = attn_output * o_proj^T

  return true;
}

bool QuantizedForward::ComputeFFN(int layer) {
  const half *gate_proj = weights_->LayerGateProj(layer);
  const half *up_proj = weights_->LayerUpProj(layer);
  const half *down_proj = weights_->LayerDownProj(layer);

  if (!gate_proj || !up_proj || !down_proj) {
    log::Error("quantized_forward",
               "Missing FFN weights for layer " + std::to_string(layer));
    return false;
  }

  // Gate projection
  // d_ffn_gate = d_hidden * gate_proj^T

  // Up projection
  // d_ffn_up = d_hidden * up_proj^T

  // SiLU activation
  // d_ffn_gate = silu(d_ffn_gate)

  // Element-wise multiply
  // d_ffn_down = d_ffn_gate * d_ffn_up

  // Down projection
  // d_ffn_down_out = d_ffn_down * down_proj^T

  return true;
}

bool QuantizedForward::ComputeRMSNorm(const half *input, const half *weight,
                                      half *output, int size, float eps) {
  // TODO: Implement RMSNorm kernel
  // For now, this is a placeholder
  // output = input * sqrt(mean(input^2) + eps)^-1 * weight
  return true;
}

void QuantizedForward::ApplyRoPE(half *q, half *k, int n_past, int layer) {
  // TODO: Implement RoPE
  // Apply rotary position embeddings to Q and K
}

// Factory function
std::unique_ptr<QuantizedForward>
CreateQuantizedForward(const std::string &model_type) {
  auto forward = std::make_unique<QuantizedForward>();
  log::Info("quantized_forward_factory",
            "Created QuantizedForward for: " + model_type);
  return forward;
}

} // namespace inferflux
