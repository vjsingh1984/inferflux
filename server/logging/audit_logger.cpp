#include "server/logging/audit_logger.h"

#include <chrono>
#include <ctime>

namespace inferflux {

AuditLogger::AuditLogger(const std::string& path) {
  if (!path.empty()) {
    stream_.open(path, std::ios::app);
  }
}

void AuditLogger::Log(const std::string& subject,
                      const std::string& model,
                      const std::string& status,
                      const std::string& message) {
  if (!Enabled()) {
    return;
  }
  auto now = std::chrono::system_clock::now();
  auto ts = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
  std::lock_guard<std::mutex> lock(mutex_);
  stream_ << "{\"timestamp\":" << ts << ",\"subject\":\"" << subject << "\",\"model\":\"" << model
          << "\",\"status\":\"" << status << "\",\"message\":\"" << message << "\"}\n";
  stream_.flush();
}

}  // namespace inferflux
