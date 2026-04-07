#include "runtime/core/gguf/igguf_executor.h"
#include "server/logging/logger.h"

#include <cstring>
#include <vector>

namespace inferflux {
namespace runtime {
namespace core {
namespace gguf {

/**
 * @brief CPU-only GGUF executor
 *
 * Handles tensor operations on CPU. No GPU memory allocation.
 * Dequantization happens on CPU using quantization handlers.
 */
class CpuGgufExecutor : public IGGUFExecutor {
public:
  bool UploadTensor(const GgufTensorInfo &info, const void *cpu_data,
                    void **gpu_ptr) override {
    // CPU executor: no actual upload, just return the CPU pointer
    // The gpu_ptr is a misnomer here - it's actually a CPU pointer
    *gpu_ptr = const_cast<void *>(cpu_data);
    return true;
  }

  bool Dequantize(const void *quantized_data, void *output,
                  const GgufTensorInfo &info,
                  std::shared_ptr<IQuantizationHandler> handler) override {
    if (!quantized_data || !output) {
      log::Error("cpu_executor", "Invalid parameters for dequantization");
      return false;
    }

    // TODO: Phase 3 will implement CPU dequantization
    // For now, return true to allow compilation
    // Actual dequantization will be added in Phase 3
    log::Warn("cpu_executor", "CPU dequantization not yet implemented, "
                              "will be added in Phase 3");
    return true;
  }

  bool FreeGPUMemory(void *ptr) override {
    // CPU executor: nothing to free, memory is managed by caller
    (void)ptr;
    return true;
  }

  ExecutorType GetType() const override { return ExecutorType::CPU; }

  bool IsAvailable() const override { return true; }

  const char *GetName() const override { return "CPU"; }
};

//==============================================================================
// Factory Functions
//==============================================================================

std::unique_ptr<IGGUFExecutor> CreateExecutor(ExecutorType type) {
#ifdef INFERFLUX_HAS_CUDA
  if (type == ExecutorType::CUDA) {
    // TODO: Return CudaGgufExecutor in Phase 2b
    log::Warn("cpu_executor",
              "CUDA executor requested but not yet implemented, "
              "falling back to CPU");
  }
#endif

#ifdef INFERFLUX_HAS_ROCM
  if (type == ExecutorType::ROCm) {
    // TODO: Return RocmGgufExecutor in Phase 2b
    log::Warn("cpu_executor",
              "ROCm executor requested but not yet implemented, "
              "falling back to CPU");
  }
#endif

  // Default to CPU executor
  return std::make_unique<CpuGgufExecutor>();
}

std::unique_ptr<IGGUFExecutor> CreateBestExecutor() {
#ifdef INFERFLUX_HAS_CUDA
  auto cuda_executor = CreateExecutor(ExecutorType::CUDA);
  if (cuda_executor && cuda_executor->IsAvailable()) {
    return cuda_executor;
  }
#endif

#ifdef INFERFLUX_HAS_ROCM
  auto rocm_executor = CreateExecutor(ExecutorType::ROCm);
  if (rocm_executor && rocm_executor->IsAvailable()) {
    return rocm_executor;
  }
#endif

  // Fall back to CPU
  return std::make_unique<CpuGgufExecutor>();
}

} // namespace gguf
} // namespace core
} // namespace runtime
} // namespace inferflux
