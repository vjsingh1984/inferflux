#pragma once

#include <fstream>
#include <mutex>
#include <string>

namespace inferflux {

class AuditLogger {
 public:
  AuditLogger() = default;

  // path: log file path; debug_mode: when true, log raw prompt/response text
  // instead of SHA-256 hashes (Architecture.md §Security requirement).
  explicit AuditLogger(const std::string& path, bool debug_mode = false);

  bool Enabled() const { return stream_.is_open(); }

  // General-purpose structured log entry (subject, model, status, metadata message).
  void Log(const std::string& subject,
           const std::string& model,
           const std::string& status,
           const std::string& message);

  // Log a generation request/response pair. In production mode (debug_mode=false),
  // prompt and response are SHA-256 hashed before writing to satisfy the
  // Architecture.md §Security requirement for prompt redaction.
  void LogRequest(const std::string& subject,
                  const std::string& model,
                  const std::string& prompt,
                  const std::string& response,
                  int prompt_tokens,
                  int completion_tokens);

  // Hash a string to its SHA-256 hex representation (64 chars).
  static std::string HashContent(const std::string& content);

 private:
  std::ofstream stream_;
  std::mutex mutex_;
  bool debug_mode_{false};
};

}  // namespace inferflux
