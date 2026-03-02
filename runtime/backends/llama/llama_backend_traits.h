#pragma once

#include "runtime/backends/backend_capabilities.h"
#include "runtime/backends/cpu/llama_backend.h"

#include <string>

namespace inferflux {

enum class LlamaBackendTarget {
  kCpu,
  kCuda,
  kMps,
  kRocm,
  kVulkan,
};

struct LlamaBackendTraits {
  std::string label{"cpu"};
  bool gpu_accelerated{false};
  bool supports_flash_attention{false};
  bool supports_phased_execution{true};
  bool supports_unified_batch{true};
  BackendCapabilities capabilities{};
};

LlamaBackendTarget ParseLlamaBackendTarget(const std::string &hint);
LlamaBackendTraits DescribeLlamaBackendTarget(LlamaBackendTarget target);
LlamaBackendConfig
TuneLlamaBackendConfig(LlamaBackendTarget target,
                       const LlamaBackendConfig &base);

} // namespace inferflux
