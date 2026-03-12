#include "runtime/backends/rocm/rocm_backend.h"
#include "runtime/backends/llama/llama_backend_traits.h"
#include "server/logging/logger.h"
#include "server/metrics/metrics.h"

#ifdef INFERFLUX_HAS_ROCM
#include <hip/hip_runtime_api.h>
#endif

namespace inferflux {

namespace {

#ifdef INFERFLUX_HAS_ROCM
bool ParseBoolEnv(const char *name, bool default_value) {
  const char *raw = std::getenv(name);
  if (!raw) {
    return default_value;
  }
  std::string lowered = raw;
  std::transform(
      lowered.begin(), lowered.end(), lowered.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return lowered == "1" || lowered == "true" || lowered == "yes" ||
         lowered == "on";
}

bool DetectHipDevice() {
  int device_count = 0;
  hipError_t err = hipGetDeviceCount(&device_count);
  return (err == hipSuccess && device_count > 0);
}

std::string GetArchName(int device_id) {
  hipDeviceProp_t prop;
  hipError_t err = hipGetDeviceProperties(&prop, device_id);
  if (err != hipSuccess) {
    return "unknown";
  }

  // gcnArchName format: "gfx90a", "gfx942", etc.
  std::string arch(prop.gcnArchName);

  // Convert to uppercase for consistency
  std::transform(arch.begin(), arch.end(), arch.begin(), ::toupper);

  return arch;
}

bool SupportsFlashAttentionForArch(const std::string &arch) {
  // GFX9+ (MI200, MI250X, MI300X) supports FlashAttention-2
  // The support comes through llama.cpp's HIP backend which reuses
  // CUDA FlashAttention kernels
  if (arch.find("GFX9") != std::string::npos) {
    return true; // MI200 (GFX90A), MI250X (GFX90A), MI300X (GFX942)
  }
  if (arch.find("GFX10") != std::string::npos) {
    return true; // RX 7000 series (experimental)
  }
  return false;
}

#endif // INFERFLUX_HAS_ROCM

} // anonymous namespace

RocmBackend::RocmBackend() = default;

RocmBackend::~RocmBackend() {
#ifdef INFERFLUX_HAS_ROCM
  if (hip_initialized_) {
    // Cleanup HIP resources if needed
    hip_initialized_ = false;
  }
#endif
}

bool RocmBackend::InitializeHip() {
#ifdef INFERFLUX_HAS_ROCM
  if (!DetectHipDevice()) {
    log::Error("rocm_backend", "No HIP devices found");
    return false;
  }

  // Get device count and select device 0
  int device_count = 0;
  hipError_t err = hipGetDeviceCount(&device_count);
  if (err != hipSuccess || device_count == 0) {
    log::Error("rocm_backend", "No HIP devices available");
    return false;
  }

  // Set device
  err = hipSetDevice(hip_device_);
  if (err != hipSuccess) {
    log::Error("rocm_backend", "Failed to set HIP device: " +
                                   std::to_string(static_cast<int>(err)));
    return false;
  }

  // Get device properties
  hipDeviceProp_t prop;
  err = hipGetDeviceProperties(&prop, hip_device_);
  if (err != hipSuccess) {
    log::Error("rocm_backend", "Failed to get HIP device properties");
    return false;
  }

  device_arch_ = GetArchName(hip_device_);
  total_memory_mb_ = prop.totalGlobalMem / (1024 * 1024);

  hip_initialized_ = true;

  log::Info("rocm_backend",
            "HIP device initialized: " + std::string(prop.name) +
                " (Arch: " + device_arch_ +
                ", Memory: " + std::to_string(total_memory_mb_) + " MB)");

  // Record ROCm device metrics
  GlobalMetrics().RecordRocmDeviceProperties(hip_device_, device_arch_);

  return true;
#else
  log::Error("rocm_backend", "ROCm support not compiled in");
  return false;
#endif
}

bool RocmBackend::DetectFlashAttentionSupport() {
#ifdef INFERFLUX_HAS_ROCM
  supports_flash_attention_ = SupportsFlashAttentionForArch(device_arch_);

  if (supports_flash_attention_) {
    log::Info("rocm_backend", "FlashAttention-2 supported on " + device_arch_);
  } else {
    log::Info("rocm_backend", "FlashAttention not supported on " +
                                  device_arch_ +
                                  " (requires GFX9+ architecture)");
  }

  return supports_flash_attention_;
#else
  return false;
#endif
}

bool RocmBackend::ConfigureFlashAttention(const LlamaBackendConfig &config) {
  // ROCm uses FlashAttention-2 (FA2) algorithms on supported AMD GPUs
  // This is provided through llama.cpp's HIP backend which reuses
  // the CUDA FlashAttention kernels

  if (!config.use_flash_attention) {
    selected_attention_kernel_ = "standard";
    return true;
  }

  if (!supports_flash_attention_) {
    log::Warn("rocm_backend", "FlashAttention requested but not supported on " +
                                  device_arch_ +
                                  ", falling back to standard attention");
    selected_attention_kernel_ = "standard";

    GlobalMetrics().RecordCudaAttentionKernelFallback("auto", "standard",
                                                      "rocm_arch_unsupported");
    return true;
  }

  // ROCm uses FlashAttention-2 (not FA3 which is Hopper-only)
  selected_attention_kernel_ = "fa2";

  log::Info("rocm_backend",
            "FlashAttention enabled for ROCm (kernel=fa2, arch=" +
                device_arch_ + ")");

  // Record FlashAttention metrics
  GlobalMetrics().SetCudaAttentionKernel("fa2");
  GlobalMetrics().RecordFlashAttentionRequest("fa2");
  GlobalMetrics().SetFlashAttentionEnabled(true);

  return true;
}

bool RocmBackend::LoadModel(const std::filesystem::path &model_path,
                            const LlamaBackendConfig &config) {
#ifdef INFERFLUX_HAS_ROCM
  log::Info("rocm_backend",
            "Loading model with ROCm backend: " + model_path.string());

  // Initialize HIP
  if (!InitializeHip()) {
    return false;
  }

  // Detect FlashAttention support
  if (!DetectFlashAttentionSupport()) {
    // Continue without FA
  }

  // Configure FlashAttention
  if (!ConfigureFlashAttention(config)) {
    return false;
  }

  // Tune config for ROCm
  LlamaBackendConfig tuned =
      TuneLlamaBackendConfig(LlamaBackendTarget::kRocm, config);

  // Load model using base class (which will use llama.cpp HIP backend)
  if (!LlamaCppBackend::LoadModel(model_path, tuned)) {
    log::Error("rocm_backend", "Failed to load model with ROCm backend");
    return false;
  }

  log::Info("rocm_backend",
            "Model loaded successfully with ROCm backend (kernel=" +
                selected_attention_kernel_ + ")");

  return true;
#else
  (void)model_path;
  (void)config;
  log::Error("rocm_backend", "ROCm support not compiled in (ENABLE_ROCM=OFF)");
  return false;
#endif
}

bool RocmBackend::IsReady() const {
#ifdef INFERFLUX_HAS_ROCM
  return hip_initialized_ && LlamaCppBackend::IsReady();
#else
  return false;
#endif
}

bool RocmBackend::SupportsFlashAttention() const {
  return supports_flash_attention_;
}

std::string RocmBackend::GetFlashAttentionVersion() const {
  if (supports_flash_attention_) {
    return "fa2"; // ROCm uses FA2 (via HIP)
  }
  return "none";
}

} // namespace inferflux