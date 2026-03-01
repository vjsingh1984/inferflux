#pragma once

#include <chrono>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace inferflux {
namespace disaggregated {

struct KVPacket {
  uint64_t request_id{0};
  std::vector<uint8_t> prompt_tokens;
  std::vector<uint8_t> kv_blob;
  int kv_page{-1};
  int n_past{-1};       // Filled after phased Prefill(); -1 if not yet prefilled.
  int sequence_id{-1};  // KV cache sequence slot used during Prefill(); -1 = unassigned.
  std::string metadata;
  std::chrono::steady_clock::time_point enqueue_time{std::chrono::steady_clock::now()};
};

// Thread-safe queue that will evolve into the shared-memory/RDMA transport.
class KVChannel {
 public:
  explicit KVChannel(std::size_t capacity = 64);

  void SetCapacity(std::size_t capacity);
  std::size_t Capacity() const;

  bool Enqueue(KVPacket packet);
  std::optional<KVPacket> TryDequeue();
  std::size_t Size() const;
  void Clear();

 private:
  mutable std::mutex mutex_;
  std::deque<KVPacket> queue_;
  std::size_t capacity_;
};

}  // namespace disaggregated
}  // namespace inferflux
