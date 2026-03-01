#include "runtime/backends/cpu/cpu_backend.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace inferflux {

CPUDeviceContext::~CPUDeviceContext() {
  std::lock_guard<std::mutex> lock(alloc_mutex_);
  for (void* ptr : allocations_) {
    std::free(ptr);
  }
  allocations_.clear();
}

std::unique_ptr<DeviceBuffer> CPUDeviceContext::Allocate(std::size_t bytes) {
  void* ptr = std::malloc(bytes);
  if (!ptr) {
    throw std::bad_alloc();
  }
  {
    std::lock_guard<std::mutex> lock(alloc_mutex_);
    allocations_.push_back(ptr);
  }
  return std::make_unique<DeviceBuffer>(ptr, bytes);
}

void CPUDeviceContext::Free(std::unique_ptr<DeviceBuffer> buffer) {
  if (!buffer) {
    return;
  }
  void* ptr = buffer->data();
  if (!ptr) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(alloc_mutex_);
    auto it = std::find(allocations_.begin(), allocations_.end(), ptr);
    if (it != allocations_.end()) {
      allocations_.erase(it);
    }
  }
  std::free(ptr);
}

std::vector<int> CPUDeviceContext::RunGreedyDecode(const std::vector<int>& tokens) {
  // For the MVP we simply echo the incoming token ids and append a terminator token id (0).
  std::vector<int> output = tokens;
  output.push_back(0);
  return output;
}

}  // namespace inferflux
