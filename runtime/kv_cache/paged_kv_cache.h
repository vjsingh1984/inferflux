#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include "io/async_file_writer.h"

#include <memory>
#include <vector>

namespace inferflux {

struct KVPage {
  std::vector<float> data;
  bool in_use{false};
  std::uint64_t last_used{0};
  bool dirty{false};
};

class PagedKVCache {
 public:
  enum class EvictionPolicy { kLRU, kClock };

  explicit PagedKVCache(std::size_t pages,
                        std::size_t page_size_bytes,
                        EvictionPolicy policy = EvictionPolicy::kLRU);

  int ReservePage();
  void ReleasePage(int page_id);
  void Write(int page_id, const std::vector<float>& values);
  std::vector<float> Read(int page_id) const;

  void SetOffloadPath(const std::string& path);
  void SetEvictionPolicy(EvictionPolicy policy);
  void ConfigureAsyncWriter(std::size_t workers, std::size_t queue_depth);

  std::size_t PageSizeBytes() const { return page_size_bytes_; }

 private:
  std::size_t page_size_bytes_;
  mutable std::vector<KVPage> pages_;
  mutable std::mutex mutex_;
  std::string offload_path_;
  std::shared_ptr<AsyncFileWriter> writer_;
  mutable std::uint64_t usage_counter_{0};
  EvictionPolicy eviction_policy_;
  mutable std::vector<bool> clock_ref_bits_;
  mutable std::size_t clock_hand_{0};
  std::size_t writer_workers_{1};
  std::size_t writer_queue_depth_{256};

  void PersistPage(int page_id, const std::vector<float>& values) const;
  std::vector<float> LoadPage(int page_id) const;
  int SelectEvictionCandidate() const;
};

}  // namespace inferflux
