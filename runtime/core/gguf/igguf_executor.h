#pragma once

#include "runtime/core/gguf/igguf_parser.h"
#include <cstddef>
#include <memory>

namespace inferflux {
namespace runtime {
namespace cuda {
namespace native {
class IQuantizationHandler;
}
} // namespace cuda
} // namespace runtime
} // namespace inferflux

namespace inferflux {
namespace runtime {
namespace core {
namespace gguf {

// Forward declarations
class IQuantizationHandler;

/**
 * @brief Executor type enumeration
 */
enum class ExecutorType { CPU, CUDA, ROCm };

/**
 * @brief Interface for GGUF execution (CPU or GPU)
 *
 * Handles tensor upload, dequantization, and memory management.
 * Separates parsing logic (IGGUFParser) from execution logic.
 */
class IGGUFExecutor {
public:
  virtual ~IGGUFExecutor() = default;

  /**
   * @brief Upload tensor data to device memory
   * @param info Tensor information
   * @param cpu_data Source data in CPU memory
   * @param gpu_ptr Output device pointer (allocated by executor)
   * @return true on success
   */
  virtual bool UploadTensor(const GgufTensorInfo &info, const void *cpu_data,
                            void **gpu_ptr) = 0;

  /**
   * @brief Dequantize tensor data
   * @param quantized_data Input quantized data
   * @param output Output buffer (dequantized)
   * @param info Tensor information
   * @param handler Quantization handler
   * @return true on success
   */
  virtual bool Dequantize(const void *quantized_data, void *output,
                          const GgufTensorInfo &info,
                          std::shared_ptr<IQuantizationHandler> handler) = 0;

  /**
   * @brief Free device memory
   * @param ptr Pointer to free
   * @return true on success
   */
  virtual bool FreeGPUMemory(void *ptr) = 0;

  /**
   * @brief Get executor type
   * @return Executor type (CPU, CUDA, ROCm)
   */
  virtual ExecutorType GetType() const = 0;

  /**
   * @brief Check if executor is available
   * @return true if executor can be used
   */
  virtual bool IsAvailable() const = 0;

  /**
   * @brief Get executor name (for logging)
   * @return Human-readable executor name
   */
  virtual const char *GetName() const = 0;
};

/**
 * @brief Factory function to create executor based on type
 * @param type Desired executor type
 * @return Executor instance (falls back to CPU if requested type unavailable)
 */
std::unique_ptr<IGGUFExecutor> CreateExecutor(ExecutorType type);

/**
 * @brief Create best available executor
 * @return Best available executor (CUDA > ROCm > CPU)
 */
std::unique_ptr<IGGUFExecutor> CreateBestExecutor();

} // namespace gguf
} // namespace core
} // namespace runtime
} // namespace inferflux
