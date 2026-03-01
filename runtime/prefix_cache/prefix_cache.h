#pragma once

#include <chrono>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace inferflux {

struct PrefixCacheEntry {
  std::string completion;
  int completion_tokens{0};
  std::chrono::steady_clock::time_point last_used;
  std::size_t hits{0};
};

class PrefixCache {
 public:
  explicit PrefixCache(std::size_t capacity = 256);

  bool Lookup(const std::vector<int>& tokens,
              std::string* completion,
              int* completion_tokens);

  void Insert(const std::vector<int>& tokens,
              const std::string& completion,
              int completion_tokens);

  std::size_t Capacity() const { return capacity_; }

 private:
  std::string Serialize(const std::vector<int>& tokens) const;

  struct EntryState {
    PrefixCacheEntry entry;
    std::list<std::string>::iterator lru_it;
  };

  std::size_t capacity_;
  std::unordered_map<std::string, EntryState> table_;
  std::list<std::string> lru_;
  mutable std::mutex mutex_;
};

}  // namespace inferflux
