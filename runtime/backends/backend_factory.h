#pragma once

#include "runtime/backends/cpu/llama_backend.h"
#include "runtime/backends/llama/llama_backend_traits.h"

#include <memory>
#include <string>
#include <vector>

namespace inferflux {

enum class BackendProvider {
  kUniversalLlama,
  kNative,
};

struct BackendExposurePolicy {
  bool prefer_native{true};
  bool allow_universal_fallback{true};
};

struct BackendFactoryResult {
  std::shared_ptr<LlamaCPUBackend> backend;
  std::string backend_label{"cpu"};
  LlamaBackendTarget target{LlamaBackendTarget::kCpu};
  LlamaBackendTraits traits{};
  BackendCapabilities capabilities{};
  BackendProvider provider{BackendProvider::kUniversalLlama};
  bool used_fallback{false};
  std::string fallback_reason;
  LlamaBackendConfig config{};
};

class BackendFactory {
public:
  static BackendFactoryResult Create(const std::string &backend_hint);
  static std::string NormalizeHint(const std::string &backend_hint);
  static std::vector<std::string>
  NormalizeHintList(const std::vector<std::string> &backend_hints,
                    const std::string &default_hint = "cpu");
  static void SetExposurePolicy(const BackendExposurePolicy &policy);
  static BackendExposurePolicy ExposurePolicy();
};

LlamaBackendConfig MergeBackendConfig(const LlamaBackendConfig &defaults,
                                      const BackendFactoryResult &selection);

} // namespace inferflux
