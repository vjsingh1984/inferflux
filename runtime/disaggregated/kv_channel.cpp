#include "runtime/disaggregated/kv_channel.h"

#include <algorithm>

namespace inferflux {
namespace disaggregated {

KVChannel::KVChannel(std::size_t capacity) : capacity_(std::max<std::size_t>(1, capacity)) {}

void KVChannel::SetCapacity(std::size_t capacity) {
  std::lock_guard<std::mutex> lock(mutex_);
  capacity_ = std::max<std::size_t>(1, capacity);
  while (queue_.size() > capacity_) {
    queue_.pop_front();
  }
}

std::size_t KVChannel::Capacity() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return capacity_;
}

bool KVChannel::Enqueue(KVPacket packet) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (queue_.size() >= capacity_) {
    return false;
  }
  if (packet.kv_page >= 0) {
    packet.metadata += "|kv_page=" + std::to_string(packet.kv_page);
  }
  queue_.push_back(std::move(packet));
  return true;
}

std::optional<KVPacket> KVChannel::TryDequeue() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (queue_.empty()) {
    return std::nullopt;
  }
  KVPacket packet = std::move(queue_.front());
  queue_.pop_front();
  return packet;
}

std::size_t KVChannel::Size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return queue_.size();
}

void KVChannel::Clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  queue_.clear();
}

}  // namespace disaggregated
}  // namespace inferflux
