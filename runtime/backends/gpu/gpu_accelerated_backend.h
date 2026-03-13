#pragma once

#include "runtime/backends/gpu/gpu_device_strategy.h"
#include "runtime/backends/llama/llama_cpp_backend.h"

#include <memory>
#include <string>

namespace inferflux {

class GpuAcceleratedBackend : public LlamaCppBackend {
public:
  explicit GpuAcceleratedBackend(std::unique_ptr<GpuDeviceStrategy> strategy);

  bool LoadModel(const std::filesystem::path &model_path,
                 const LlamaBackendConfig &config = {}) override;

  bool IsReady() const override;

  const GpuDeviceInfo &DeviceInfo() const { return device_info_; }

#ifdef INFERFLUX_USE_COMMON_BACKEND_TYPES
  std::string Name() const override;
#endif

protected:
  std::unique_ptr<GpuDeviceStrategy> strategy_;
  GpuDeviceInfo device_info_;
  bool device_initialized_{false};
};

} // namespace inferflux
