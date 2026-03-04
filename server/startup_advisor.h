#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace inferflux {

// Pure-data snapshot of a loaded model for advisor inspection.
struct AdvisorModelInfo {
  std::string id;
  std::string path;
  std::string format{"unknown"};
  std::string backend{"cpu"};
  std::string backend_provider{"universal"};
  bool backend_fallback{false};
  std::string backend_fallback_reason;
  std::uint64_t file_size_bytes{0};
  bool is_moe{false};
  int n_experts{0};
};

// GPU hardware snapshot (populated by ProbeCudaGpu / ProbeRocmGpu).
struct AdvisorGpuInfo {
  bool available{false};
  int device_count{0};
  std::string device_name;
  int compute_major{0};
  int compute_minor{0};
  int sm_count{0};
  std::uint64_t total_vram_bytes{0};
  std::uint64_t free_vram_bytes{0};
};

// Effective config snapshot taken at startup.
struct AdvisorConfig {
  bool cuda_enabled{false};
  bool flash_attention_enabled{false};
  std::string cuda_attention_kernel{"auto"};
  bool phase_overlap_enabled{false};
  int max_batch_size{8};
  int max_batch_tokens{2048};
  std::size_t kv_cpu_pages{32};
  bool prefer_native{true};
  bool allow_universal_fallback{true};
  std::string backend_priority;
  int tp_degree{1};
  bool speculative_enabled{false};
};

// Everything the advisor needs — assembled in main.cpp, no live pointers.
struct StartupAdvisorContext {
  std::vector<AdvisorModelInfo> models;
  AdvisorGpuInfo gpu;
  AdvisorConfig config;
};

// Run all recommendation rules.  Returns the number of recommendations logged.
// Respects INFERFLUX_DISABLE_STARTUP_ADVISOR=true (returns 0 immediately).
int RunStartupAdvisor(const StartupAdvisorContext &ctx);

// Hardware probing helpers — safe to call on any platform.
// On builds without the matching SDK they return a default (unavailable) struct.
AdvisorGpuInfo ProbeCudaGpu();
AdvisorGpuInfo ProbeRocmGpu();

} // namespace inferflux
