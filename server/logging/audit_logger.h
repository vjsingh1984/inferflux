#pragma once

#include <fstream>
#include <mutex>
#include <string>

namespace inferflux {

class AuditLogger {
 public:
  AuditLogger() = default;
  explicit AuditLogger(const std::string& path);

  bool Enabled() const { return stream_.is_open(); }
  void Log(const std::string& subject,
           const std::string& model,
           const std::string& status,
           const std::string& message);

 private:
  std::ofstream stream_;
  std::mutex mutex_;
};

}  // namespace inferflux
