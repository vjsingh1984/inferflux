#pragma once

#include "runtime/backends/cuda/inferflux_cuda_executor.h"

namespace inferflux {

/// Friend-based accessor for InferfluxCudaExecutor private members.
class ExecutorTestAccess {
public:
  explicit ExecutorTestAccess(InferfluxCudaExecutor &e) : e_(e) {}

  auto &dequantized_cache_policy() { return e_.dequantized_cache_policy_; }
  auto &dequantized_cache_policy_hint() {
    return e_.dequantized_cache_policy_hint_;
  }
  auto &model_loader() { return e_.model_loader_; }

  bool ConfigureDequantizedCachePolicy(const std::string &raw) {
    return e_.ConfigureDequantizedCachePolicy(raw);
  }

  void ReleaseBatchScopedDequantizedCache() {
    e_.ReleaseBatchScopedDequantizedCache();
  }

private:
  InferfluxCudaExecutor &e_;
};

} // namespace inferflux
