#include "runtime/kv_cache/paged_kv_cache.h"

#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>

namespace inferflux {

PagedKVCache::PagedKVCache(std::size_t pages,
                           std::size_t page_size_bytes,
                           EvictionPolicy policy)
    : page_size_bytes_(page_size_bytes),
      pages_(pages),
      eviction_policy_(policy),
      clock_ref_bits_(pages, false) {}

int PagedKVCache::ReservePage() {
  std::lock_guard<std::mutex> lock(mutex_);
  int candidate = SelectEvictionCandidate();
  if (candidate < 0) {
    throw std::runtime_error("KV cache exhausted");
  }
  auto& page = pages_[candidate];
  page.last_used = usage_counter_++;
  page.in_use = true;
  page.dirty = false;
  if (page.data.size() != page_size_bytes_ / sizeof(float)) {
    page.data.assign(page_size_bytes_ / sizeof(float), 0.0f);
  }
  if (static_cast<std::size_t>(candidate) < clock_ref_bits_.size()) {
    clock_ref_bits_[candidate] = true;
  }
  return candidate;
}

void PagedKVCache::ReleasePage(int page_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (page_id < 0 || static_cast<std::size_t>(page_id) >= pages_.size()) {
    return;
  }
  if (!offload_path_.empty() && !pages_[page_id].data.empty() && pages_[page_id].dirty) {
    PersistPage(page_id, pages_[page_id].data);
  }
  pages_[page_id].in_use = false;
  pages_[page_id].data.clear();
  pages_[page_id].dirty = false;
  if (static_cast<std::size_t>(page_id) < clock_ref_bits_.size()) {
    clock_ref_bits_[page_id] = false;
  }
}

void PagedKVCache::Write(int page_id, const std::vector<float>& values) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (page_id < 0 || static_cast<std::size_t>(page_id) >= pages_.size()) {
    throw std::out_of_range("invalid page id");
  }
  pages_[page_id].data = values;
  pages_[page_id].dirty = true;
  pages_[page_id].last_used = usage_counter_++;
  if (static_cast<std::size_t>(page_id) < clock_ref_bits_.size()) {
    clock_ref_bits_[page_id] = true;
  }
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
    auto data = LoadPage(page_id);
    pages_[page_id].data = data;
  }
  pages_[page_id].last_used = usage_counter_++;
  if (static_cast<std::size_t>(page_id) < clock_ref_bits_.size()) {
    clock_ref_bits_[page_id] = true;
  }
  return pages_[page_id].data;
}

void PagedKVCache::SetOffloadPath(const std::string& path) {
  std::lock_guard<std::mutex> lock(mutex_);
  offload_path_ = path;
  if (!offload_path_.empty()) {
    std::error_code ec;
    std::filesystem::create_directories(offload_path_, ec);
    if (!writer_) {
      writer_ = std::make_shared<AsyncFileWriter>(writer_queue_depth_);
    }
    writer_->Configure(writer_workers_, writer_queue_depth_);
    writer_->Start(writer_workers_);
  } else if (writer_) {
    writer_->Stop();
    writer_.reset();
  }
}

void PagedKVCache::SetEvictionPolicy(EvictionPolicy policy) {
  std::lock_guard<std::mutex> lock(mutex_);
  eviction_policy_ = policy;
}

void PagedKVCache::ConfigureAsyncWriter(std::size_t workers, std::size_t queue_depth) {
  std::lock_guard<std::mutex> lock(mutex_);
  writer_workers_ = workers == 0 ? 1 : workers;
  writer_queue_depth_ = queue_depth == 0 ? 64 : queue_depth;
  if (writer_) {
    writer_->Configure(writer_workers_, writer_queue_depth_);
    writer_->Start(writer_workers_);
  }
}

void PagedKVCache::PersistPage(int page_id, const std::vector<float>& values) const {
  if (offload_path_.empty()) {
    return;
  }
  std::filesystem::path file = std::filesystem::path(offload_path_) /
                               ("page_" + std::to_string(page_id) + ".bin");
  std::vector<char> buffer(reinterpret_cast<const char*>(values.data()),
                           reinterpret_cast<const char*>(values.data()) +
                               values.size() * sizeof(float));
  if (writer_) {
    writer_->Enqueue(AsyncWriteTask{file, std::move(buffer)});
  } else {
    std::ofstream out(file, std::ios::binary | std::ios::trunc);
    if (!out.good()) {
      return;
    }
    out.write(buffer.data(), static_cast<std::streamsize>(buffer.size()));
  }
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

int PagedKVCache::SelectEvictionCandidate() const {
  if (pages_.empty()) {
    return -1;
  }
  if (eviction_policy_ == EvictionPolicy::kClock) {
    for (std::size_t scanned = 0; scanned < pages_.size() * 2; ++scanned) {
      if (clock_hand_ >= pages_.size()) {
        clock_hand_ = 0;
      }
      std::size_t idx = clock_hand_;
      clock_hand_ = (clock_hand_ + 1) % pages_.size();
      if (pages_[idx].in_use) {
        continue;
      }
      if (!clock_ref_bits_[idx]) {
        return static_cast<int>(idx);
      }
      clock_ref_bits_[idx] = false;
    }
    for (std::size_t i = 0; i < pages_.size(); ++i) {
      if (!pages_[i].in_use) {
        return static_cast<int>(i);
      }
    }
    return -1;
  }
  std::size_t candidate = pages_.size();
  std::uint64_t oldest = std::numeric_limits<std::uint64_t>::max();
  for (std::size_t i = 0; i < pages_.size(); ++i) {
    if (!pages_[i].in_use && pages_[i].last_used <= oldest) {
      oldest = pages_[i].last_used;
      candidate = i;
    }
  }
  return candidate == pages_.size() ? -1 : static_cast<int>(candidate);
}

}  // namespace inferflux
