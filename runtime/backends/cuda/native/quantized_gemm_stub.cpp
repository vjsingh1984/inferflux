#include "runtime/backends/cuda/native/quantized_gemm.h"

namespace inferflux {

QuantizedGemm::~QuantizedGemm() = default;

bool QuantizedGemm::Initialize(cudaStream_t stream) {
  stream_ = stream;
  cache_size_ = 0;
  return true;
}

void QuantizedGemm::SetStream(cudaStream_t stream) { stream_ = stream; }

bool QuantizedGemm::Gemm(int, int, int, const half *,
                         std::shared_ptr<IWeightAccessor> weight_accessor,
                         half *) {
  return static_cast<bool>(weight_accessor);
}

bool QuantizedGemm::GemmBatched(
    int, int, int, const half *,
    std::shared_ptr<IWeightAccessor> weight_accessor, half *, int, long long,
    long long) {
  return static_cast<bool>(weight_accessor);
}

bool QuantizedGemm::ShouldUseCache(
    std::shared_ptr<IWeightAccessor> accessor) const {
  if (!accessor) {
    return false;
  }
  const auto dims = accessor->GetDimensions();
  const size_t num_elements = dims.first * dims.second;
  return num_elements > 1024U * 1024U;
}

QuantizedGemm::DequantizedCache *
QuantizedGemm::FindOrCreateCache(std::shared_ptr<IWeightAccessor> accessor) {
  if (!accessor) {
    return nullptr;
  }
  for (int i = 0; i < cache_size_; ++i) {
    if (cache_[i].accessor == accessor) {
      return &cache_[i];
    }
  }
  if (cache_size_ < MAX_CACHE_SIZE) {
    cache_[cache_size_].accessor = std::move(accessor);
    cache_[cache_size_].is_valid = false;
    return &cache_[cache_size_++];
  }
  return &cache_[MAX_CACHE_SIZE - 1];
}

bool QuantizedGemm::GemmDirect(int, int, int, const half *, half *, half *) {
  return true;
}

std::unique_ptr<QuantizedGemm> CreateQuantizedGemm() {
  return std::make_unique<QuantizedGemm>();
}

} // namespace inferflux
