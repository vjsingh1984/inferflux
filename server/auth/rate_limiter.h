#pragma once

#include <chrono>
#include <mutex>
#include <string>
#include <unordered_map>

namespace inferflux {

class RateLimiter {
 public:
  explicit RateLimiter(int tokens_per_minute);

  bool Allow(const std::string& key);
  bool Enabled() const { return tokens_per_minute_ > 0; }
  void UpdateLimit(int tokens_per_minute);
  int CurrentLimit() const { return tokens_per_minute_; }

 private:
  struct Entry {
    double tokens{0.0};
    std::chrono::steady_clock::time_point last;
  };

  double tokens_per_minute_;
  double refill_per_second_;
  std::unordered_map<std::string, Entry> entries_;
  std::mutex mutex_;
};

}  // namespace inferflux
