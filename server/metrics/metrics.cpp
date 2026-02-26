#include "server/metrics/metrics.h"

#include <sstream>

namespace inferflux {

namespace {
MetricsRegistry g_metrics;
}  // namespace

void MetricsRegistry::SetBackend(const std::string& backend) { backend_ = backend; }

void MetricsRegistry::RecordSuccess(int prompt_tokens, int completion_tokens) {
  total_requests_.fetch_add(1, std::memory_order_relaxed);
  total_prompt_tokens_.fetch_add(prompt_tokens, std::memory_order_relaxed);
  total_completion_tokens_.fetch_add(completion_tokens, std::memory_order_relaxed);
}

void MetricsRegistry::RecordError() { total_errors_.fetch_add(1, std::memory_order_relaxed); }

std::string MetricsRegistry::RenderPrometheus() const {
  std::ostringstream out;
  out << "# HELP inferflux_requests_total Total successful generation requests\n";
  out << "# TYPE inferflux_requests_total counter\n";
  out << "inferflux_requests_total{backend=\"" << backend_ << "\"} " << total_requests_.load() << "\n";

  out << "# HELP inferflux_errors_total Total generation errors\n";
  out << "# TYPE inferflux_errors_total counter\n";
  out << "inferflux_errors_total{backend=\"" << backend_ << "\"} " << total_errors_.load() << "\n";

  out << "# HELP inferflux_prompt_tokens_total Total prompt tokens processed\n";
  out << "# TYPE inferflux_prompt_tokens_total counter\n";
  out << "inferflux_prompt_tokens_total{backend=\"" << backend_ << "\"} "
      << total_prompt_tokens_.load() << "\n";

  out << "# HELP inferflux_completion_tokens_total Total completion tokens produced\n";
  out << "# TYPE inferflux_completion_tokens_total counter\n";
  out << "inferflux_completion_tokens_total{backend=\"" << backend_ << "\"} "
      << total_completion_tokens_.load() << "\n";
  return out.str();
}

MetricsRegistry& GlobalMetrics() { return g_metrics; }

}  // namespace inferflux
