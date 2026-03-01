#pragma once

#include "policy/policy_backend.h"
#include "runtime/speculative/speculative_decoder.h"
#include "scheduler/scheduler.h"
#include "server/auth/api_key_auth.h"
#include "server/auth/oidc_validator.h"
#include "server/auth/rate_limiter.h"
#include "server/logging/audit_logger.h"
#include "server/metrics/metrics.h"
#include "server/policy/guardrail.h"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include <openssl/ssl.h>

namespace inferflux {

class HttpServer {
 public:
  struct TlsConfig {
    bool enabled{false};
    std::string cert_path;
    std::string key_path;
  };

  // Disaggregated deployment role (§2.5 item 12).
  // kUnified: combined prefill+decode (default; existing behaviour).
  // kPrefill: this instance only runs prefill; /readyz gates on model load.
  // kDecode:  this instance only runs decode; /readyz gates on decode pool.
  enum class PoolRole { kUnified, kPrefill, kDecode };

  HttpServer(std::string host,
             int port,
             Scheduler* scheduler,
             std::shared_ptr<ApiKeyAuth> auth,
             MetricsRegistry* metrics,
             OIDCValidator* oidc,
             RateLimiter* rate_limiter,
             Guardrail* guardrail,
             AuditLogger* audit_logger,
             PolicyBackend* policy_store,
             std::shared_ptr<SpeculativeDecoder> speculative_decoder,
             TlsConfig tls_config,
             int num_workers = 4);
  ~HttpServer();

  void Start();
  void Stop();
  void SetModelReady(bool ready) { model_ready_.store(ready); }
  void SetRole(PoolRole role) { role_.store(role, std::memory_order_relaxed); }
  void SetDecodePoolReady(bool ready) { decode_pool_ready_.store(ready, std::memory_order_relaxed); }

 private:
  struct ClientSession {
    int fd{-1};
    SSL* ssl{nullptr};
  };

  void Run();
  void HandleClient(ClientSession& session);

  struct AuthContext {
    std::string subject{"anonymous"};
    std::unordered_set<std::string> scopes;
  };

  bool ResolveSubject(const std::string& header_blob, AuthContext* ctx) const;
  bool RequireScope(const AuthContext& ctx,
                    const std::string& scope,
                    ClientSession& session,
                    const std::string& error_message);

  bool SendAll(ClientSession& session, const std::string& payload);
  ssize_t Receive(ClientSession& session, char* buffer, std::size_t length);
  void CloseSession(ClientSession& session);

  std::string host_;
  int port_;
  Scheduler* scheduler_;
  std::shared_ptr<ApiKeyAuth> auth_;
  MetricsRegistry* metrics_;
  OIDCValidator* oidc_;
  RateLimiter* rate_limiter_;
  Guardrail* guardrail_;
  AuditLogger* audit_logger_;
  PolicyBackend* policy_store_;
  std::shared_ptr<SpeculativeDecoder> speculative_decoder_;
  bool tls_enabled_{false};
  SSL_CTX* ssl_ctx_{nullptr};
  std::atomic<bool> running_{false};
  std::atomic<bool> model_ready_{false};
  std::atomic<PoolRole> role_{PoolRole::kUnified};
  std::atomic<bool> decode_pool_ready_{false};
  std::atomic<int> server_fd_{-1};
  int num_workers_;
  std::thread accept_thread_;
  std::vector<std::thread> workers_;
  std::queue<ClientSession> client_queue_;
  std::mutex queue_mutex_;
  std::condition_variable queue_cv_;

  void WorkerLoop();
};

}  // namespace inferflux
