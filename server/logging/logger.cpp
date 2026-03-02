#include "server/logging/logger.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>

using json = nlohmann::json;

namespace inferflux {
namespace log {

namespace {

std::atomic<bool> g_json_mode{false};
std::mutex g_mutex;

const char *LevelString(Level level) {
  switch (level) {
  case Level::DEBUG:
    return "DEBUG";
  case Level::INFO:
    return "INFO";
  case Level::WARN:
    return "WARN";
  case Level::ERROR:
    return "ERROR";
  }
  return "UNKNOWN";
}

} // namespace

void SetJsonMode(bool enabled) { g_json_mode.store(enabled); }
bool IsJsonMode() { return g_json_mode.load(); }

void Log(Level level, const std::string &component, const std::string &message,
         const std::string &extra) {
  auto now = std::chrono::system_clock::now();
  auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch())
                .count();

  std::string line;
  if (g_json_mode.load()) {
    json j;
    j["ts"] = ts;
    j["level"] = LevelString(level);
    j["component"] = component;
    j["message"] = message;
    if (!extra.empty()) {
      j["extra"] = extra;
    }
    line = j.dump();
  } else {
    line = std::string("[") + LevelString(level) + "] " + component + ": " +
           message;
    if (!extra.empty()) {
      line += " | " + extra;
    }
  }

  std::lock_guard<std::mutex> lock(g_mutex);
  std::cerr << line << "\n";
}

} // namespace log
} // namespace inferflux
