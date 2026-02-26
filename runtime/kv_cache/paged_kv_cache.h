#pragma once

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace inferflux {

struct KVPage {
  std::vector<float> data;
  bool in_use{false};
};

class PagedKVCache {
 public:
  explicit PagedKVCache(std::size_t pages, std::size_t page_size_bytes);

  int ReservePage();
  void ReleasePage(int page_id);
  void Write(int page_id, const std::vector<float>& values);
  std::vector<float> Read(int page_id) const;

  std::size_t PageSizeBytes() const { return page_size_bytes_; }

 private:
  std::size_t page_size_bytes_;
  std::vector<KVPage> pages_;
  mutable std::mutex mutex_;
};

}  // namespace inferflux
