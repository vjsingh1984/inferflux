#pragma once

#include "runtime/core/gguf/igguf_executor.h"

namespace inferflux {
namespace runtime {
namespace core {
namespace gguf {

/**
 * @brief CPU-only GGUF executor
 *
 * Handles tensor operations on CPU without any GPU dependencies.
 * UploadTensor is a no-op (returns CPU pointer directly).
 * Dequantize uses CPU quantization handlers.
 * FreeGPUMemory is a no-op (caller manages memory).
 */
class CpuGgufExecutor : public IGGUFExecutor {
public:
  bool UploadTensor(const GgufTensorInfo &info, const void *cpu_data,
                    void **gpu_ptr) override;

  bool Dequantize(const void *quantized_data, void *output,
                  const GgufTensorInfo &info,
                  std::shared_ptr<IQuantizationHandler> handler) override;

  bool FreeGPUMemory(void *ptr) override;

  ExecutorType GetType() const override;

  bool IsAvailable() const override;

  const char *GetName() const override;
};

} // namespace gguf
} // namespace core
} // namespace runtime
} // namespace inferflux
