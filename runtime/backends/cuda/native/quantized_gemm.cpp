#include "runtime/backends/cuda/native/quantized_gemm.h"
#include "runtime/backends/cuda/native/cublas_gemm.h"
#include "server/logging/logger.h"
#include <cstring>

namespace inferflux {

QuantizedGemm::~QuantizedGemm() {
  // Free cache
  for (int i = 0; i < cache_size_; ++i) {
    if (cache_[i].d_weights) {
      cudaFree(cache_[i].d_weights);
    }
  }
}

bool QuantizedGemm::Initialize(cudaStream_t stream) {
  stream_ = stream;
  cache_size_ = 0;
  return true;
}

void QuantizedGemm::SetStream(cudaStream_t stream) { stream_ = stream; }

bool QuantizedGemm::Gemm(int M, int N, int K, const half *A,
                         std::shared_ptr<IWeightAccessor> weight_accessor,
                         half *C) {

  if (!weight_accessor) {
    log::Error("quantized_gemm", "Null weight accessor");
    return false;
  }

  // Check if weights are quantized
  if (!weight_accessor->IsQuantized()) {
    // Non-quantized: direct GEMM
    half *W = weight_accessor->GetDequantizedGpuWeights(stream_);
    if (!W) {
      log::Error("quantized_gemm", "Failed to get weights");
      return false;
    }
    return GemmDirect(M, N, K, A, W, C);
  }

  // Quantized weights
  auto *cache_entry = FindOrCreateCache(weight_accessor);

  // Check if we have cached dequantized weights
  if (cache_entry && cache_entry->is_valid) {
    return GemmDirect(M, N, K, A, cache_entry->d_weights, C);
  }

  // Lazy dequantization
  half *d_dequantized = weight_accessor->GetDequantizedGpuWeights(stream_);
  if (!d_dequantized) {
    log::Error("quantized_gemm", "Failed to dequantize weights");
    return false;
  }

  // Cache if appropriate
  if (cache_entry && ShouldUseCache(weight_accessor)) {
    cache_entry->is_valid = true;
    cache_entry->d_weights = d_dequantized;
  }

  return GemmDirect(M, N, K, A, d_dequantized, C);
}

bool QuantizedGemm::GemmBatched(
    int M, int N, int K, const half *A,
    std::shared_ptr<IWeightAccessor> weight_accessor, half *C, int batch_count,
    long long stride_A, long long stride_C) {

  // For batched operations, typically dequantize once
  half *d_dequantized = weight_accessor->GetDequantizedGpuWeights(stream_);
  if (!d_dequantized) {
    log::Error("quantized_gemm",
               "Failed to dequantize weights for batched GEMM");
    return false;
  }

  // TODO: Implement batched GEMM
  // For now, sequential calls
  for (int i = 0; i < batch_count; ++i) {
    const half *A_i = A + i * stride_A;
    half *C_i = C + i * stride_C;
    if (!GemmDirect(M, N, K, A_i, d_dequantized, C_i)) {
      return false;
    }
  }

  return true;
}

bool QuantizedGemm::ShouldUseCache(
    std::shared_ptr<IWeightAccessor> accessor) const {
  // Cache weights that are:
  // 1. Large (expensive to dequantize repeatedly)
  // 2. Frequently accessed (projections in all layers)

  if (!accessor) {
    return false;
  }

  auto dims = accessor->GetDimensions();
  size_t num_elements = dims.first * dims.second;

  // Cache if > 1M elements (~2MB for FP16)
  return num_elements > 1024 * 1024;
}

QuantizedGemm::DequantizedCache *
QuantizedGemm::FindOrCreateCache(std::shared_ptr<IWeightAccessor> accessor) {
  // Search for existing entry
  for (int i = 0; i < cache_size_; ++i) {
    if (cache_[i].accessor == accessor) {
      return &cache_[i];
    }
  }

  // Create new entry if space available
  if (cache_size_ < MAX_CACHE_SIZE) {
    cache_[cache_size_].accessor = accessor;
    cache_[cache_size_].is_valid = false;
    return &cache_[cache_size_++];
  }

  // Cache full - evict oldest (simple LRU would be better)
  // Shift all entries
  for (int i = 0; i < MAX_CACHE_SIZE - 1; ++i) {
    cache_[i] = cache_[i + 1];
  }
  cache_[MAX_CACHE_SIZE - 1].accessor = accessor;
  cache_[MAX_CACHE_SIZE - 1].is_valid = false;

  if (cache_[MAX_CACHE_SIZE - 1].d_weights) {
    cudaFree(cache_[MAX_CACHE_SIZE - 1].d_weights);
    cache_[MAX_CACHE_SIZE - 1].d_weights = nullptr;
  }

  return &cache_[MAX_CACHE_SIZE - 1];
}

bool QuantizedGemm::GemmDirect(int M, int N, int K, const half *A, half *W,
                               half *C) {
  // Use cuBLAS for actual computation
  // TODO: Integrate with CublasGemm

  // Placeholder: use simple copy for now
  // In production, this would call:
  //   cublasGemmEx(handle_, CUBLAS_OP_N, CUBLAS_OP_T, ...)

  cudaMemcpyAsync(C, A, M * K * sizeof(half), cudaMemcpyDeviceToDevice,
                  stream_);
  return true;
}

// Factory function
std::unique_ptr<QuantizedGemm> CreateQuantizedGemm() {
  return std::make_unique<QuantizedGemm>();
}

} // namespace inferflux
