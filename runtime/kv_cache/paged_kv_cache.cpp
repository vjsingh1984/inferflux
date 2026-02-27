#include "runtime/kv_cache/paged_kv_cache.h"

#include <filesystem>
#include <stdexcept>
#include <fstream>

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
  if (!offload_path_.empty() && !pages_[page_id].data.empty()) {
    PersistPage(page_id, pages_[page_id].data);
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
  if (!offload_path_.empty()) {
    PersistPage(page_id, values);
  }
}

std::vector<float> PagedKVCache::Read(int page_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (page_id < 0 || static_cast<std::size_t>(page_id) >= pages_.size()) {
    throw std::out_of_range("invalid page id");
  }
  if (pages_[page_id].data.empty() && !offload_path_.empty()) {
    return LoadPage(page_id);
  }
  return pages_[page_id].data;
}

void PagedKVCache::SetOffloadPath(const std::string& path) {
  std::lock_guard<std::mutex> lock(mutex_);
  offload_path_ = path;
  if (!offload_path_.empty()) {
    std::error_code ec;
    std::filesystem::create_directories(offload_path_, ec);
  }
}

void PagedKVCache::PersistPage(int page_id, const std::vector<float>& values) const {
  if (offload_path_.empty()) {
    return;
  }
  std::filesystem::path file = std::filesystem::path(offload_path_) /
                               ("page_" + std::to_string(page_id) + ".bin");
  std::ofstream out(file, std::ios::binary | std::ios::trunc);
  if (!out.good()) {
    return;
  }
  out.write(reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(float)));
}

std::vector<float> PagedKVCache::LoadPage(int page_id) const {
  std::vector<float> data;
  if (offload_path_.empty()) {
    return data;
  }
  std::filesystem::path file = std::filesystem::path(offload_path_) /
                               ("page_" + std::to_string(page_id) + ".bin");
  if (!std::filesystem::exists(file)) {
    return data;
  }
  auto size = std::filesystem::file_size(file);
  if (size % sizeof(float) != 0) {
    return data;
  }
  data.resize(size / sizeof(float));
  std::ifstream in(file, std::ios::binary);
  if (!in.good()) {
    data.clear();
    return data;
  }
  in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size));
  return data;
}

}  // namespace inferflux
