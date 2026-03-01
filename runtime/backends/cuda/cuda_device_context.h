#pragma once

#include "runtime/device_context.h"

#include <memory>
#include <string>

namespace inferflux {

class CudaDeviceContext : public DeviceContext {
 public:
  CudaDeviceContext();
  ~CudaDeviceContext() override;

  std::string Name() const override { return "cuda"; }
  bool IsAvailable() const override { return available_; }

  std::unique_ptr<DeviceBuffer> Allocate(std::size_t bytes) override;
  void Free(std::unique_ptr<DeviceBuffer> buffer) override;

 private:
  bool available_{false};
};

}  // namespace inferflux
