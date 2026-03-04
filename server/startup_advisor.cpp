#include "server/startup_advisor.h"
#include "server/logging/logger.h"

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

#ifdef INFERFLUX_HAS_CUDA
#include <cuda_runtime.h>
#endif

#ifdef INFERFLUX_HAS_ROCM
#include <hip/hip_runtime.h>
#endif

namespace inferflux {
namespace {

const char *kComponent = "startup_advisor";

// ---------------------------------------------------------------------------
// Rule 1: Backend mismatch — safetensors on CUDA using universal provider.
// ---------------------------------------------------------------------------
int CheckBackendMismatch(const StartupAdvisorContext &ctx) {
  int count = 0;
  for (const auto &m : ctx.models) {
    if (m.format == "safetensors" && m.backend == "cuda" &&
        m.backend_provider == "universal") {
      log::Info(kComponent,
                "[RECOMMEND] backend: Model '" + m.id +
                    "' uses safetensors on CUDA — set "
                    "INFERFLUX_NATIVE_CUDA_EXECUTOR=native_kernel");
      ++count;
    }
  }
  return count;
}

// ---------------------------------------------------------------------------
// Rule 2: Attention kernel — GPU SM >= 8.0, CUDA on, FA disabled.
// ---------------------------------------------------------------------------
int CheckAttentionKernel(const StartupAdvisorContext &ctx) {
  if (!ctx.config.cuda_enabled || !ctx.gpu.available) return 0;
  if (ctx.gpu.compute_major < 8) return 0;
  if (ctx.config.flash_attention_enabled) return 0;

  log::Info(kComponent,
            "[RECOMMEND] attention: GPU '" + ctx.gpu.device_name + "' (SM " +
                std::to_string(ctx.gpu.compute_major) + "." +
                std::to_string(ctx.gpu.compute_minor) +
                ") supports FA2 — set runtime.cuda.flash_attention.enabled: "
                "true and runtime.cuda.attention.kernel: fa2");
  return 1;
}

// ---------------------------------------------------------------------------
// Rule 3: Batch size vs VRAM — can fit more concurrent sequences.
// ---------------------------------------------------------------------------
int CheckBatchSizeVsVram(const StartupAdvisorContext &ctx) {
  if (!ctx.gpu.available || ctx.gpu.free_vram_bytes == 0) return 0;
  if (ctx.models.empty()) return 0;

  // Estimate per the largest model.
  std::uint64_t max_model_size = 0;
  for (const auto &m : ctx.models) {
    if (m.file_size_bytes > max_model_size) max_model_size = m.file_size_bytes;
  }
  if (max_model_size == 0) return 0;

  // Heuristic: each batch slot costs ~15% of model size in KV/activation.
  double per_slot = static_cast<double>(max_model_size) * 0.15;
  if (per_slot <= 0) return 0;

  int suggested =
      static_cast<int>(static_cast<double>(ctx.gpu.free_vram_bytes) / per_slot);
  if (suggested < 1) suggested = 1;

  if (suggested > ctx.config.max_batch_size * 1.5) {
    log::Info(kComponent,
              "[RECOMMEND] batch_size: " +
                  std::to_string(ctx.gpu.free_vram_bytes / (1024 * 1024)) +
                  " MB VRAM free — increase "
                  "runtime.scheduler.max_batch_size to " +
                  std::to_string(suggested) + " (current: " +
                  std::to_string(ctx.config.max_batch_size) + ")");
    return 1;
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Rule 4: Phase overlap — CUDA on, overlap off, batch_size >= 4.
// ---------------------------------------------------------------------------
int CheckPhaseOverlap(const StartupAdvisorContext &ctx) {
  if (!ctx.config.cuda_enabled) return 0;
  if (ctx.config.phase_overlap_enabled) return 0;
  if (ctx.config.max_batch_size < 4) return 0;

  log::Info(kComponent,
            "[RECOMMEND] phase_overlap: CUDA enabled with batch_size >= 4 — "
            "set runtime.cuda.phase_overlap.enabled: true for mixed-batch "
            "decode prioritization");
  return 1;
}

// ---------------------------------------------------------------------------
// Rule 5: KV cache pages — large free VRAM, low page count.
// ---------------------------------------------------------------------------
int CheckKvCachePages(const StartupAdvisorContext &ctx) {
  if (!ctx.gpu.available || ctx.gpu.free_vram_bytes == 0) return 0;

  // Suggest more pages when free VRAM > 4 GB and pages <= 64.
  constexpr std::uint64_t kFourGb = 4ULL * 1024 * 1024 * 1024;
  if (ctx.gpu.free_vram_bytes < kFourGb) return 0;
  if (ctx.config.kv_cpu_pages > 64) return 0;

  std::size_t suggested =
      static_cast<std::size_t>(ctx.gpu.free_vram_bytes / (64 * 1024 * 1024));
  if (suggested <= ctx.config.kv_cpu_pages) return 0;

  log::Info(kComponent,
            "[RECOMMEND] kv_cache: " +
                std::to_string(ctx.gpu.free_vram_bytes / (1024 * 1024)) +
                " MB VRAM free with only " +
                std::to_string(ctx.config.kv_cpu_pages) +
                " KV pages — increase runtime.paged_kv.cpu_pages to " +
                std::to_string(suggested));
  return 1;
}

// ---------------------------------------------------------------------------
// Rule 6: Tensor parallelism — multi-GPU, TP=1, large model.
// ---------------------------------------------------------------------------
int CheckTensorParallelism(const StartupAdvisorContext &ctx) {
  if (!ctx.gpu.available) return 0;
  if (ctx.gpu.device_count <= 1) return 0;
  if (ctx.config.tp_degree > 1) return 0;

  // Check if any model barely fits (file_size > 70% of single-GPU VRAM).
  for (const auto &m : ctx.models) {
    if (m.file_size_bytes >
        static_cast<std::uint64_t>(ctx.gpu.total_vram_bytes * 0.7)) {
      log::Info(kComponent,
                "[RECOMMEND] tensor_parallel: " +
                    std::to_string(ctx.gpu.device_count) +
                    " GPUs detected but TP=1 — model '" + m.id +
                    "' uses " +
                    std::to_string(m.file_size_bytes / (1024 * 1024)) +
                    " MB, set runtime.tensor_parallel: " +
                    std::to_string(ctx.gpu.device_count));
      return 1;
    }
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Rule 7: Unknown format — model format not detected.
// ---------------------------------------------------------------------------
int CheckUnknownFormat(const StartupAdvisorContext &ctx) {
  int count = 0;
  for (const auto &m : ctx.models) {
    if (m.format == "unknown") {
      log::Info(kComponent,
                "[RECOMMEND] format: Model '" + m.id +
                    "' has unknown format — set models[*].format explicitly "
                    "(gguf, safetensors, or hf)");
      ++count;
    }
  }
  return count;
}

// ---------------------------------------------------------------------------
// Rule 8: GPU unused — GPU available but all models on CPU.
// ---------------------------------------------------------------------------
int CheckGpuUnused(const StartupAdvisorContext &ctx) {
  if (!ctx.gpu.available) return 0;
  if (ctx.models.empty()) return 0;

  bool all_cpu =
      std::all_of(ctx.models.begin(), ctx.models.end(),
                  [](const AdvisorModelInfo &m) { return m.backend == "cpu"; });
  if (!all_cpu) return 0;

  log::Info(kComponent,
            "[RECOMMEND] gpu_unused: GPU '" + ctx.gpu.device_name +
                "' is available but all models use CPU — set "
                "runtime.cuda.enabled: true");
  return 1;
}

} // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

int RunStartupAdvisor(const StartupAdvisorContext &ctx) {
  if (const char *env = std::getenv("INFERFLUX_DISABLE_STARTUP_ADVISOR")) {
    std::string val = env;
    if (val == "true" || val == "1" || val == "yes") return 0;
  }

  if (ctx.models.empty()) return 0;

  log::Info(kComponent, "=== InferFlux Startup Recommendations ===");

  int total = 0;
  total += CheckBackendMismatch(ctx);
  total += CheckAttentionKernel(ctx);
  total += CheckBatchSizeVsVram(ctx);
  total += CheckPhaseOverlap(ctx);
  total += CheckKvCachePages(ctx);
  total += CheckTensorParallelism(ctx);
  total += CheckUnknownFormat(ctx);
  total += CheckGpuUnused(ctx);

  if (total == 0) {
    log::Info(kComponent,
              "=== No recommendations — config looks good! ===");
  } else {
    log::Info(kComponent, "=== End Recommendations (" +
                              std::to_string(total) + " suggestion" +
                              (total != 1 ? "s" : "") + ") ===");
  }
  return total;
}

AdvisorGpuInfo ProbeCudaGpu() {
  AdvisorGpuInfo info;
#ifdef INFERFLUX_HAS_CUDA
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    return info;
  }
  info.available = true;
  info.device_count = device_count;

  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
    info.device_name = prop.name;
    info.compute_major = prop.major;
    info.compute_minor = prop.minor;
    info.sm_count = prop.multiProcessorCount;
    info.total_vram_bytes = static_cast<std::uint64_t>(prop.totalGlobalMem);
  }

  std::size_t free_mem = 0, total_mem = 0;
  if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
    info.free_vram_bytes = static_cast<std::uint64_t>(free_mem);
  }
#endif
  return info;
}

AdvisorGpuInfo ProbeRocmGpu() {
  AdvisorGpuInfo info;
#ifdef INFERFLUX_HAS_ROCM
  int device_count = 0;
  if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
    return info;
  }
  info.available = true;
  info.device_count = device_count;

  hipDeviceProp_t prop{};
  if (hipGetDeviceProperties(&prop, 0) == hipSuccess) {
    info.device_name = prop.name;
    info.compute_major = prop.major;
    info.compute_minor = prop.minor;
    info.sm_count = prop.multiProcessorCount;
    info.total_vram_bytes = static_cast<std::uint64_t>(prop.totalGlobalMem);
  }

  std::size_t free_mem = 0, total_mem = 0;
  if (hipMemGetInfo(&free_mem, &total_mem) == hipSuccess) {
    info.free_vram_bytes = static_cast<std::uint64_t>(free_mem);
  }
#endif
  return info;
}

} // namespace inferflux
