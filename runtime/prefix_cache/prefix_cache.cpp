#include "runtime/prefix_cache/prefix_cache.h"

#include <sstream>

namespace inferflux {

PrefixCache::PrefixCache(std::size_t capacity) : capacity_(capacity) {}

bool PrefixCache::Lookup(const std::vector<int>& tokens,
                         std::string* completion,
                         int* completion_tokens) {
  if (capacity_ == 0) {
    return false;
  }
  auto key = Serialize(tokens);
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = table_.find(key);
  if (it == table_.end()) {
    return false;
  }
  it->second.entry.hits++;
  it->second.entry.last_used = std::chrono::steady_clock::now();
  lru_.splice(lru_.begin(), lru_, it->second.lru_it);
  if (completion) {
    *completion = it->second.entry.completion;
  }
  if (completion_tokens) {
    *completion_tokens = it->second.entry.completion_tokens;
  }
  return true;
}

void PrefixCache::Insert(const std::vector<int>& tokens,
                         const std::string& completion,
                         int completion_tokens) {
  if (capacity_ == 0) {
    return;
  }
  auto key = Serialize(tokens);
  std::lock_guard<std::mutex> lock(mutex_);
  auto now = std::chrono::steady_clock::now();
  auto it = table_.find(key);
  if (it != table_.end()) {
    it->second.entry.completion = completion;
    it->second.entry.completion_tokens = completion_tokens;
    it->second.entry.last_used = now;
    lru_.splice(lru_.begin(), lru_, it->second.lru_it);
    return;
  }
  if (table_.size() >= capacity_ && !lru_.empty()) {
    auto victim = lru_.back();
    lru_.pop_back();
    table_.erase(victim);
  }
  lru_.push_front(key);
  EntryState state;
  state.entry.completion = completion;
  state.entry.completion_tokens = completion_tokens;
  state.entry.last_used = now;
  state.entry.hits = 0;
  state.lru_it = lru_.begin();
  table_[key] = std::move(state);
}

std::string PrefixCache::Serialize(const std::vector<int>& tokens) const {
  std::ostringstream oss;
  for (int token : tokens) {
    oss << token << ',';
  }
  return oss.str();
}

}  // namespace inferflux
