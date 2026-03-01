#pragma once

#include "runtime/device_context.h"

#include <mutex>
#include <vector>

namespace inferflux {

class CPUDeviceContext : public DeviceContext {
 public:
  CPUDeviceContext() = default;
  ~CPUDeviceContext() override;

  std::string Name() const override { return "cpu"; }
  bool IsAvailable() const override { return true; }
  std::unique_ptr<DeviceBuffer> Allocate(std::size_t bytes) override;
  void Free(std::unique_ptr<DeviceBuffer> buffer) override;

  // Simple demo inference that echoes tokens with annotation.
  std::vector<int> RunGreedyDecode(const std::vector<int>& tokens);

 private:
  std::mutex alloc_mutex_;  // protects allocations vector
  std::vector<void*> allocations_;
};

}  // namespace inferflux
