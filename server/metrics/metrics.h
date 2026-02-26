#pragma once

#include <atomic>
#include <cstdint>
#include <string>

namespace inferflux {

class MetricsRegistry {
 public:
  void SetBackend(const std::string& backend);
  void RecordSuccess(int prompt_tokens, int completion_tokens);
  void RecordError();
  std::string RenderPrometheus() const;

 private:
  std::string backend_{"cpu"};
  std::atomic<uint64_t> total_requests_{0};
  std::atomic<uint64_t> total_errors_{0};
  std::atomic<uint64_t> total_prompt_tokens_{0};
  std::atomic<uint64_t> total_completion_tokens_{0};
};

MetricsRegistry& GlobalMetrics();

}  // namespace inferflux
