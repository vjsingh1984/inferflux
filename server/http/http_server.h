#pragma once

#include "scheduler/scheduler.h"
#include "server/auth/api_key_auth.h"
#include "server/metrics/metrics.h"

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
             MetricsRegistry* metrics);
  ~HttpServer();

  void Start();
  void Stop();

 private:
  void Run();
  void HandleClient(int client_fd);
  bool CheckAuth(const std::string& header_blob) const;

  std::string host_;
  int port_;
  Scheduler* scheduler_;
  std::shared_ptr<ApiKeyAuth> auth_;
  MetricsRegistry* metrics_;
  std::atomic<bool> running_{false};
  std::thread worker_;
};

}  // namespace inferflux
