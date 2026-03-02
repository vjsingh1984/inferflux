#pragma once

#include <cstddef>
#include <memory>
#include <string>

namespace inferflux {

class DeviceBuffer {
public:
  DeviceBuffer() = default;
  DeviceBuffer(void *ptr, std::size_t bytes) : ptr_(ptr), bytes_(bytes) {}
  void *data() const { return ptr_; }
  std::size_t size() const { return bytes_; }

private:
  void *ptr_{nullptr};
  std::size_t bytes_{0};
};

class DeviceContext {
public:
  virtual ~DeviceContext() = default;
  virtual std::string Name() const = 0;
  virtual bool IsAvailable() const = 0;
  virtual std::unique_ptr<DeviceBuffer> Allocate(std::size_t bytes) = 0;
  virtual void Free(std::unique_ptr<DeviceBuffer> buffer) = 0;
};

} // namespace inferflux
