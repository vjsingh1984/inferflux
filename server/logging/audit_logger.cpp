#include "server/logging/audit_logger.h"

#include <nlohmann/json.hpp>
#include <openssl/sha.h>

#include <chrono>
#include <iomanip>
#include <sstream>

using json = nlohmann::json;

namespace inferflux {

AuditLogger::AuditLogger(const std::string& path, bool debug_mode)
    : debug_mode_(debug_mode) {
  if (!path.empty()) {
    stream_.open(path, std::ios::app);
  }
}

std::string AuditLogger::HashContent(const std::string& content) {
  unsigned char hash[SHA256_DIGEST_LENGTH];
  SHA256(reinterpret_cast<const unsigned char*>(content.data()), content.size(), hash);
  std::ostringstream hex;
  hex << std::hex << std::setfill('0');
  for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
    hex << std::setw(2) << static_cast<int>(hash[i]);
  }
  return hex.str();
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
  json j;
  j["timestamp"] = ts;
  j["subject"] = subject;
  j["model"] = model;
  j["status"] = status;
  j["message"] = message;
  std::lock_guard<std::mutex> lock(mutex_);
  stream_ << j.dump() << "\n";
  stream_.flush();
}

void AuditLogger::LogRequest(const std::string& subject,
                             const std::string& model,
                             const std::string& prompt,
                             const std::string& response,
                             int prompt_tokens,
                             int completion_tokens) {
  if (!Enabled()) {
    return;
  }
  auto now = std::chrono::system_clock::now();
  auto ts = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
  json j;
  j["timestamp"] = ts;
  j["subject"] = subject;
  j["model"] = model;
  j["status"] = "success";
  j["prompt_tokens"] = prompt_tokens;
  j["completion_tokens"] = completion_tokens;
  if (debug_mode_) {
    j["prompt"] = prompt;
    j["response"] = response;
  } else {
    // Hash by default — never log raw prompt/response to disk in production.
    j["prompt_sha256"] = HashContent(prompt);
    j["response_sha256"] = HashContent(response);
  }
  std::lock_guard<std::mutex> lock(mutex_);
  stream_ << j.dump() << "\n";
  stream_.flush();
}

}  // namespace inferflux
