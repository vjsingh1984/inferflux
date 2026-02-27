#pragma once

#include "policy/policy_store.h"
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
#include <unordered_set>

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
             AuditLogger* audit_logger,
             PolicyStore* policy_store);
  ~HttpServer();

  void Start();
  void Stop();

 private:
  void Run();
  void HandleClient(int client_fd);

  struct AuthContext {
    std::string subject{"anonymous"};
    std::unordered_set<std::string> scopes;
  };

  bool ResolveSubject(const std::string& header_blob, AuthContext* ctx) const;
  bool RequireScope(const AuthContext& ctx,
                    const std::string& scope,
                    int client_fd,
                    const std::string& error_message);

  std::string host_;
  int port_;
  Scheduler* scheduler_;
  std::shared_ptr<ApiKeyAuth> auth_;
  MetricsRegistry* metrics_;
  OIDCValidator* oidc_;
  RateLimiter* rate_limiter_;
  Guardrail* guardrail_;
  AuditLogger* audit_logger_;
  PolicyStore* policy_store_;
  std::atomic<bool> running_{false};
  std::thread worker_;
};

}  // namespace inferflux
