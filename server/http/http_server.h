#pragma once

#include "scheduler/scheduler.h"
#include "server/auth/api_key_auth.h"
#include "server/auth/oidc_validator.h"
#include "server/auth/rate_limiter.h"
#include "server/logging/audit_logger.h"
#include "server/metrics/metrics.h"
#include "server/policy/guardrail.h"

#include <atomic>
#include <memory>
#include <string>
#include <thread>

namespace inferflux {

class HttpServer {
 public:
  HttpServer(std::string host,
             int port,
             Scheduler* scheduler,
             std::shared_ptr<ApiKeyAuth> auth,
             MetricsRegistry* metrics,
             OIDCValidator* oidc,
             RateLimiter* rate_limiter,
             Guardrail* guardrail,
             AuditLogger* audit_logger);
  ~HttpServer();

  void Start();
  void Stop();

 private:
  void Run();
  void HandleClient(int client_fd);
  bool ResolveSubject(const std::string& header_blob, std::string* subject) const;

  std::string host_;
  int port_;
  Scheduler* scheduler_;
  std::shared_ptr<ApiKeyAuth> auth_;
  MetricsRegistry* metrics_;
  OIDCValidator* oidc_;
  RateLimiter* rate_limiter_;
  Guardrail* guardrail_;
  AuditLogger* audit_logger_;
  std::atomic<bool> running_{false};
  std::thread worker_;
};

}  // namespace inferflux
