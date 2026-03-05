#include "server/startup_advisor.h"
#include "server/logging/logger.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <filesystem>

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
// Helper: Format bytes to human-readable string
// ---------------------------------------------------------------------------
std::string FormatBytes(std::uint64_t bytes) {
  const char* units[] = {"B", "KB", "MB", "GB", "TB"};
  int unit_index = 0;
  double size = static_cast<double>(bytes);

  while (size >= 1024.0 && unit_index < 4) {
    size /= 1024.0;
    unit_index++;
  }

  char buffer[64];
  snprintf(buffer, sizeof(buffer), "%.2f %s", size, units[unit_index]);
  return std::string(buffer);
}

// ---------------------------------------------------------------------------
// Helper: Detect quantization from GGUF model filename
// ---------------------------------------------------------------------------
QuantizationType DetectQuantizationFromFilename(const std::string& filename) {
  std::string lower = filename;
  std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

  if (lower.find("q4_k_m") != std::string::npos) return QuantizationType::kQ4_K_M;
  if (lower.find("q4_k") != std::string::npos) return QuantizationType::kQ4_K;
  if (lower.find("q5_k_m") != std::string::npos) return QuantizationType::kQ5_K_M;
  if (lower.find("q5_k") != std::string::npos) return QuantizationType::kQ5_K;
  if (lower.find("q6_k") != std::string::npos) return QuantizationType::kQ6_K;
  if (lower.find("q8_0") != std::string::npos) return QuantizationType::kQ8_0;
  if (lower.find("q3_k_m") != std::string::npos) return QuantizationType::kQ3_K_M;
  if (lower.find("q2_k") != std::string::npos) return QuantizationType::kQ2_K;
  if (lower.find("f16") != std::string::npos) return QuantizationType::kFp16;
  if (lower.find("bf16") != std::string::npos) return QuantizationType::kBf16;
  if (lower.find("fp32") != std::string::npos || lower.find("f32") != std::string::npos) {
    return QuantizationType::kFp32;
  }

  return QuantizationType::kUnknown;
}

// ---------------------------------------------------------------------------
// Helper: Estimate model parameters from filename
// ---------------------------------------------------------------------------
int EstimateModelParams(const std::string& filename) {
  std::string lower = filename;
  std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

  // Common model sizes
  if (lower.find("1b") != std::string::npos || lower.find("1b") != std::string::npos) return 1;
  if (lower.find("3b") != std::string::npos) return 3;
  if (lower.find("7b") != std::string::npos) return 7;
  if (lower.find("8b") != std::string::npos) return 8;
  if (lower.find("14b") != std::string::npos) return 14;
  if (lower.find("27b") != std::string::npos) return 27;
  if (lower.find("70b") != std::string::npos) return 70;

  return 0;  // Unknown
}

// ---------------------------------------------------------------------------
// Rule: Dynamic slot allocation based on model size and GPU memory
// ---------------------------------------------------------------------------
int CheckDynamicSlotAllocation(const StartupAdvisorContext &ctx,
                                MemoryAllocationRecommendation& rec) {
  if (!ctx.gpu.available || ctx.gpu.total_vram_bytes == 0) return 0;
  if (ctx.models.empty()) return 0;

  rec = CalculateOptimalSlotAllocation(ctx);
  if (!rec.valid) return 0;

  // Only recommend if significantly different from current config
  double slot_ratio = static_cast<double>(rec.recommended_max_slots) /
                       std::max(1, ctx.config.max_parallel_sequences);

  if (slot_ratio < 0.5 || slot_ratio > 2.0) {
    log::Info(kComponent,
              "[RECOMMEND] slot_allocation: GPU has " +
              FormatBytes(ctx.gpu.total_vram_bytes) + " (" +
              std::to_string(ctx.gpu.total_vram_bytes / (1024*1024)) + " MB)\n" +
              "  Model: " + ctx.models[0].id + " (" +
              FormatBytes(rec.model_size_bytes) + " loaded, " +
              ctx.models[0].quantization_string + ")\n" +
              "  Current: max_parallel_sequences=" +
              std::to_string(ctx.config.max_parallel_sequences) +
              ", n_ctx=" + std::to_string(ctx.config.n_ctx) + "\n" +
              "  Recommended: max_parallel_sequences=" +
              std::to_string(rec.recommended_max_slots) +
              ", n_ctx=" + std::to_string(rec.recommended_n_ctx) + "\n" +
              "  Memory breakdown:\n" +
              "    - Model: " + FormatBytes(rec.model_size_bytes) + "\n" +
              "    - Overhead: " + FormatBytes(rec.overhead_bytes) + "\n" +
              "    - KV cache: " + FormatBytes(rec.total_kv_bytes) +
              " (" + std::to_string(rec.recommended_max_slots) + " slots × " +
              FormatBytes(rec.per_slot_kv_bytes) + " per slot)\n" +
              "    - Total: " + FormatBytes(rec.total_needed_bytes) +
              " (" + std::to_string(rec.utilization_percent) + "% of GPU)\n" +
              "  Config:\n" + rec.config_yaml_snippet);

    // Log warnings if any
    for (const auto& warning : rec.warnings) {
      log::Warn(kComponent, "[WARNING] " + warning);
    }

    return 1;
  }

  return 0;
}

// ---------------------------------------------------------------------------
// Rule: Validate quantization format matches model file
// ---------------------------------------------------------------------------
int CheckQuantizationMismatch(const StartupAdvisorContext &ctx) {
  int count = 0;
  for (const auto& m : ctx.models) {
    if (m.format == "gguf" && m.quantization == QuantizationType::kUnknown) {
      log::Info(kComponent,
                "[RECOMMEND] quantization: Model '" + m.id +
                "' is GGUF but quantization type unknown — ensure filename "
                "contains quantization (e.g., q4_k_m, q5_k_m)");
      ++count;
    }
  }
  return count;
}

} // namespace

// ---------------------------------------------------------------------------
// Public API: Calculate optimal slot allocation
// ---------------------------------------------------------------------------
MemoryAllocationRecommendation CalculateOptimalSlotAllocation(
    const StartupAdvisorContext &ctx) {

  MemoryAllocationRecommendation rec;
  if (ctx.models.empty() || !ctx.gpu.available) {
    return rec;
  }

  // Use largest model for calculation
  const AdvisorModelInfo& model = *std::max_element(
      ctx.models.begin(), ctx.models.end(),
      [](const auto& a, const auto& b) {
        return a.file_size_bytes < b.file_size_bytes;
      });

  // Step 1: Estimate loaded model size
  rec.model_size_bytes = EstimateLoadedModelSize(model);
  if (rec.model_size_bytes == 0) {
    // Fallback to file size if estimation fails
    rec.model_size_bytes = model.file_size_bytes;
  }

  // Step 2: Calculate available memory (target % of total VRAM)
  // Configurable via INFERFLUX_GPU_UTILIZATION_PCT (0-100, default: 85)
  constexpr double kDefaultTargetUtilization = 0.85;
  double target_utilization = kDefaultTargetUtilization;

  if (const char* env_util = std::getenv("INFERFLUX_GPU_UTILIZATION_PCT")) {
    try {
      int util_pct = std::stoi(env_util);
      if (util_pct >= 50 && util_pct <= 98) {
        target_utilization = util_pct / 100.0;
      }
    } catch (const std::exception&) {
      // Invalid value, use default
    }
  }

  std::uint64_t target_vram = static_cast<std::uint64_t>(
      ctx.gpu.total_vram_bytes * target_utilization);

  // Step 3: Estimate overhead (CUDA context, fragmentation, activation tensors, etc.)
  // Configurable via INFERFLUX_OVERHEAD_GB (default: 1 GB)
  constexpr std::uint64_t kDefaultOverheadBytes = 1024ULL * 1024 * 1024;  // 1 GB
  std::uint64_t overhead_bytes = kDefaultOverheadBytes;

  if (const char* env_overhead = std::getenv("INFERFLUX_OVERHEAD_GB")) {
    try {
      int overhead_gb = std::stoi(env_overhead);
      if (overhead_gb >= 0 && overhead_gb <= 16) {
        overhead_bytes = static_cast<std::uint64_t>(overhead_gb) * 1024ULL * 1024 * 1024;
      }
    } catch (const std::exception&) {
      // Invalid value, use default
    }
  }

  rec.overhead_bytes = overhead_bytes;

  // Step 4: Calculate memory available for KV cache
  std::uint64_t available_for_kv = target_vram - rec.model_size_bytes - rec.overhead_bytes;

  if (rec.model_size_bytes >= target_vram) {
    rec.warnings.push_back("Model size exceeds 85% of GPU VRAM - will likely fail to load");
    return rec;
  }

  // Step 5: Determine model architecture (defaults for Qwen 2.5 3B)
  int n_params = model.n_params > 0 ? model.n_params : EstimateModelParams(model.path);
  int n_layers = model.n_layers > 0 ? model.n_layers : 36;
  int hidden_dim = model.hidden_dim > 0 ? model.hidden_dim : 128;

  // Step 6: Get or estimate context window
  int n_ctx = ctx.config.n_ctx > 0 ? ctx.config.n_ctx : 2048;

  // Step 7: Calculate per-slot KV cache size
  rec.per_slot_kv_bytes = CalculatePerSlotKvSize(n_ctx, hidden_dim, n_layers, 32, sizeof(float) * 2);

  // Step 8: Calculate max slots based on available KV memory
  int max_slots_by_memory = static_cast<int>(available_for_kv / rec.per_slot_kv_bytes);

  // Step 9: Apply practical limits
  // Minimum 10 slots for reasonable concurrent operation
  // Maximum 256 slots for large contexts
  // Both can be overridden via environment variables
  int min_slots = 10;
  int max_slots = 256;

  if (const char* env_min = std::getenv("INFERFLUX_MIN_SLOTS")) {
    min_slots = std::max(4, std::stoi(env_min));  // Allow 4-256 range
  }
  if (const char* env_max = std::getenv("INFERFLUX_MAX_SLOTS")) {
    max_slots = std::min(512, std::stoi(env_max));  // Allow up to 512
  }

  rec.recommended_max_slots = std::max(min_slots, std::min(max_slots, max_slots_by_memory));
  rec.recommended_n_ctx = n_ctx;

  // Step 10: Calculate totals
  rec.total_kv_bytes = rec.per_slot_kv_bytes * rec.recommended_max_slots;
  rec.total_needed_bytes = rec.model_size_bytes + rec.overhead_bytes + rec.total_kv_bytes;
  rec.available_bytes = ctx.gpu.total_vram_bytes;
  rec.utilization_percent = (static_cast<double>(rec.total_needed_bytes) /
                               ctx.gpu.total_vram_bytes) * 100.0;

  rec.valid = true;

  // Step 11: Generate config YAML snippet
  std::ostringstream config;
  config << "runtime:\n";
  config << "  llama:\n";
  config << "    max_parallel_sequences: " << rec.recommended_max_slots << "\n";
  config << "    n_ctx: " << rec.recommended_n_ctx;
  rec.config_yaml_snippet = config.str();

  return rec;
}

// ---------------------------------------------------------------------------
// Public API: Detect quantization type
// ---------------------------------------------------------------------------
QuantizationType DetectQuantization(const std::string& model_path,
                                   const std::string& format) {
  if (format == "gguf") {
    return DetectQuantizationFromFilename(
        std::filesystem::path(model_path).filename().string());
  }
  // TODO: Detect from safetensors metadata
  return QuantizationType::kUnknown;
}

// ---------------------------------------------------------------------------
// Public API: Get quantization string
// ---------------------------------------------------------------------------
std::string GetQuantizationString(QuantizationType q) {
  switch (q) {
    case QuantizationType::kFp32: return "FP32";
    case QuantizationType::kFp16: return "FP16";
    case QuantizationType::kBf16: return "BF16";
    case QuantizationType::kQ8_0: return "Q8_0";
    case QuantizationType::kQ6_K: return "Q6_K";
    case QuantizationType::kQ5_K_M: return "Q5_K_M";
    case QuantizationType::kQ5_K: return "Q5_K";
    case QuantizationType::kQ4_K_M: return "Q4_K_M";
    case QuantizationType::kQ4_K: return "Q4_K";
    case QuantizationType::kQ3_K_M: return "Q3_K_M";
    case QuantizationType::kQ2_K: return "Q2_K";
    default: return "Unknown";
  }
}

// ---------------------------------------------------------------------------
// Public API: Estimate loaded model size
// ---------------------------------------------------------------------------
std::uint64_t EstimateLoadedModelSize(const AdvisorModelInfo& model) {
  // Base size from file
  std::uint64_t file_size = model.file_size_bytes;
  if (file_size == 0) return 0;

  // Adjust based on quantization compression ratio
  double compression = 1.0;

  switch (model.quantization) {
    case QuantizationType::kQ4_K_M:
    case QuantizationType::kQ4_K:
      compression = 4.5;  // ~4.5x compression from FP16
      break;
    case QuantizationType::kQ5_K_M:
    case QuantizationType::kQ5_K:
      compression = 3.5;
      break;
    case QuantizationType::kQ6_K:
      compression = 3.0;
      break;
    case QuantizationType::kQ8_0:
      compression = 2.0;
      break;
    case QuantizationType::kFp16:
    case QuantizationType::kBf16:
      compression = 2.0;  // 2x from FP32
      break;
    case QuantizationType::kFp32:
      compression = 1.0;
      break;
    default:
      // Estimate from filename if unknown
      if (model.format == "gguf") {
        QuantizationType detected = DetectQuantizationFromFilename(model.path);
        compression = (detected == QuantizationType::kQ4_K_M ||
                      detected == QuantizationType::kQ4_K) ? 4.5 : 2.0;
      }
      break;
  }

  // For GGUF, file size is already compressed
  // For safetensors, need to account for quantization
  if (model.format == "gguf") {
    // GGUF file is the actual size when loaded
    // Add ~10% for overhead (CUDA context, etc.)
    return static_cast<std::uint64_t>(file_size * 1.1);
  } else {
    // Safetensors: file contains FP32/BF16 weights
    // Calculate based on compression
    return static_cast<std::uint64_t>((file_size / compression) * 1.1);
  }
}

// ---------------------------------------------------------------------------
// Public API: Calculate per-slot KV cache size
// ---------------------------------------------------------------------------
std::uint64_t CalculatePerSlotKvSize(
    int n_ctx,
    int hidden_dim,
    int n_layers,
    int n_heads,
    size_t element_size) {

  // KV cache for one sequence:
  // - K cache: n_ctx × hidden_dim × n_layers × element_size
  // - V cache: n_ctx × hidden_dim × n_layers × element_size
  // Total: 2 × n_ctx × hidden_dim × n_layers × element_size

  return 2ULL * n_ctx * hidden_dim * n_layers * element_size;
}

// ---------------------------------------------------------------------------
// Existing rules (preserved)
// ---------------------------------------------------------------------------

// Rule 1: Backend mismatch — safetensors on CUDA using universal provider.
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

// Rule 2: Attention kernel — GPU SM >= 8.0, CUDA on, FA disabled.
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

// Rule 3: Batch size vs VRAM — can fit more concurrent sequences.
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

// Rule 4: Phase overlap — CUDA on, overlap off, batch_size >= 4.
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

// Rule 5: KV cache pages — large free VRAM, low page count.
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

// Rule 6: Tensor parallelism — multi-GPU, TP=1, large model.
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

// Rule 7: Unknown format — model format not detected.
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

// Rule 8: GPU unused — GPU available but all models on CPU.
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

  // NEW: Dynamic slot allocation
  MemoryAllocationRecommendation mem_rec;
  total += CheckDynamicSlotAllocation(ctx, mem_rec);
  total += CheckQuantizationMismatch(ctx);

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

  // NEW: Calculate recommended reserve (15%)
  info.recommended_reserve_bytes = static_cast<std::uint64_t>(
      info.total_vram_bytes * 0.15);
  info.usable_vram_bytes = info.total_vram_bytes - info.recommended_reserve_bytes;
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

  // NEW: Calculate recommended reserve (15%)
  info.recommended_reserve_bytes = static_cast<std::uint64_t>(
      info.total_vram_bytes * 0.15);
  info.usable_vram_bytes = info.total_vram_bytes - info.recommended_reserve_bytes;
#endif
  return info;
}

} // namespace inferflux
