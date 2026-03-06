#include "scheduler/session_handle_manager.h"

#include <algorithm>

namespace inferflux {
namespace scheduler {

SessionHandleManager::SessionHandleManager() : SessionHandleManager(Config{}) {}

SessionHandleManager::SessionHandleManager(const Config &config)
    : config_(config) {
  if (config_.max_sessions == 0) {
    config_.max_sessions = 1;
  }
  if (config_.ttl.count() <= 0) {
    config_.ttl = std::chrono::milliseconds(1);
  }
}

SessionHandleManager::EntryMap::iterator
SessionHandleManager::FindEvictionCandidateLocked() {
  auto best = entries_.end();
  for (auto it = entries_.begin(); it != entries_.end(); ++it) {
    if (it->second.in_use) {
      continue;
    }
    if (best == entries_.end() ||
        it->second.last_access < best->second.last_access) {
      best = it;
    }
  }
  return best;
}

SessionHandleManager::LeaseResult
SessionHandleManager::AcquireLease(const std::string &session_id) {
  LeaseResult result;
  if (session_id.empty()) {
    return result;
  }

  const auto now = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = entries_.find(session_id);
  if (it == entries_.end()) {
    if (entries_.size() >= config_.max_sessions) {
      auto evict_it = FindEvictionCandidateLocked();
      if (evict_it == entries_.end()) {
        return result;
      }
      if (evict_it->second.has_state) {
        result.cleanup_states.push_back(std::move(evict_it->second.state));
      }
      entries_.erase(evict_it);
    }

    Entry entry;
    entry.in_use = true;
    entry.last_access = now;
    entry.expires_at = now + config_.ttl;
    entries_.emplace(session_id, std::move(entry));

    result.status = LeaseResult::Status::kAcquired;
    return result;
  }

  Entry &entry = it->second;
  if (entry.in_use) {
    return result;
  }

  if (entry.has_state && now >= entry.expires_at) {
    result.cleanup_states.push_back(std::move(entry.state));
    entry.has_state = false;
    entry.state = SessionHandleState{};
  }

  entry.in_use = true;
  entry.last_access = now;
  entry.expires_at = now + config_.ttl;

  result.status = LeaseResult::Status::kAcquired;
  if (entry.has_state) {
    result.has_state = true;
    result.state = entry.state;
  }
  return result;
}

void SessionHandleManager::CommitLease(const std::string &session_id,
                                       const SessionHandleState &state) {
  if (session_id.empty()) {
    return;
  }
  const auto now = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex> lock(mutex_);
  Entry &entry = entries_[session_id];
  entry.state = state;
  entry.has_state = true;
  entry.in_use = false;
  entry.last_access = now;
  entry.expires_at = now + config_.ttl;
}

void SessionHandleManager::ReleaseLease(const std::string &session_id) {
  if (session_id.empty()) {
    return;
  }
  const auto now = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = entries_.find(session_id);
  if (it == entries_.end()) {
    return;
  }
  it->second.in_use = false;
  it->second.last_access = now;
  if (it->second.has_state) {
    it->second.expires_at = now + config_.ttl;
    return;
  }
  entries_.erase(it);
}

void SessionHandleManager::DiscardLeasedState(const std::string &session_id) {
  if (session_id.empty()) {
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = entries_.find(session_id);
  if (it == entries_.end()) {
    return;
  }
  it->second.state = SessionHandleState{};
  it->second.has_state = false;
}

std::vector<SessionHandleState> SessionHandleManager::CollectExpired() {
  std::vector<SessionHandleState> expired;
  const auto now = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto it = entries_.begin(); it != entries_.end();) {
    if (it->second.in_use || now < it->second.expires_at) {
      ++it;
      continue;
    }
    if (it->second.has_state) {
      expired.push_back(std::move(it->second.state));
    }
    it = entries_.erase(it);
  }
  return expired;
}

std::vector<SessionHandleState> SessionHandleManager::DrainAll() {
  std::vector<SessionHandleState> drained;
  std::lock_guard<std::mutex> lock(mutex_);
  drained.reserve(entries_.size());
  for (auto &entry : entries_) {
    if (entry.second.has_state) {
      drained.push_back(std::move(entry.second.state));
    }
  }
  entries_.clear();
  return drained;
}

std::size_t SessionHandleManager::SessionCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return entries_.size();
}

} // namespace scheduler
} // namespace inferflux
