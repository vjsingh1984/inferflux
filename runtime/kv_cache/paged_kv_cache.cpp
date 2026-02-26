#include "runtime/kv_cache/paged_kv_cache.h"

#include <stdexcept>

namespace inferflux {

PagedKVCache::PagedKVCache(std::size_t pages, std::size_t page_size_bytes)
    : page_size_bytes_(page_size_bytes), pages_(pages) {}

int PagedKVCache::ReservePage() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (std::size_t i = 0; i < pages_.size(); ++i) {
    if (!pages_[i].in_use) {
      pages_[i].in_use = true;
      pages_[i].data.assign(page_size_bytes_ / sizeof(float), 0.0f);
      return static_cast<int>(i);
    }
  }
  throw std::runtime_error("KV cache exhausted");
}

void PagedKVCache::ReleasePage(int page_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (page_id < 0 || static_cast<std::size_t>(page_id) >= pages_.size()) {
    return;
  }
  pages_[page_id].in_use = false;
  pages_[page_id].data.clear();
}

void PagedKVCache::Write(int page_id, const std::vector<float>& values) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (page_id < 0 || static_cast<std::size_t>(page_id) >= pages_.size()) {
    throw std::out_of_range("invalid page id");
  }
  pages_[page_id].data = values;
}

std::vector<float> PagedKVCache::Read(int page_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (page_id < 0 || static_cast<std::size_t>(page_id) >= pages_.size()) {
    throw std::out_of_range("invalid page id");
  }
  return pages_[page_id].data;
}

}  // namespace inferflux
