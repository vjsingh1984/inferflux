#include "server/auth/rate_limiter.h"

namespace inferflux {

RateLimiter::RateLimiter(int tokens_per_minute)
    : tokens_per_minute_(tokens_per_minute),
      refill_per_second_(tokens_per_minute > 0 ? tokens_per_minute / 60.0 : 0.0) {}

bool RateLimiter::Allow(const std::string& key) {
  if (tokens_per_minute_ <= 0) {
    return true;
  }
  auto now = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex> lock(mutex_);
  auto& entry = entries_[key];
  if (entry.last.time_since_epoch().count() == 0) {
    entry.tokens = tokens_per_minute_;
    entry.last = now;
  }
  auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(now - entry.last).count();
  entry.tokens = std::min<double>(tokens_per_minute_, entry.tokens + elapsed * refill_per_second_);
  entry.last = now;
  if (entry.tokens >= 1.0) {
    entry.tokens -= 1.0;
    return true;
  }
  return false;
}

void RateLimiter::UpdateLimit(int tokens_per_minute) {
  std::lock_guard<std::mutex> lock(mutex_);
  tokens_per_minute_ = tokens_per_minute;
  refill_per_second_ = tokens_per_minute > 0 ? tokens_per_minute / 60.0 : 0.0;
  entries_.clear();
}

bool RateLimiter::Enabled() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return tokens_per_minute_ > 0;
}

int RateLimiter::CurrentLimit() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return static_cast<int>(tokens_per_minute_);
}

}  // namespace inferflux
