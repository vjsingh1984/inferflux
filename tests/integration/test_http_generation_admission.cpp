#include <catch2/catch_amalgamated.hpp>

#include "net/http_client.h"
#include "runtime/disaggregated/kv_channel.h"
#include "runtime/kv_cache/paged_kv_cache.h"
#include "scheduler/scheduler.h"
#include "server/http/http_server.h"
#include "server/metrics/metrics.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <thread>

using namespace inferflux;

namespace {

class ScopedEnvVar {
public:
  ScopedEnvVar(std::string name, std::string value) : name_(std::move(name)) {
    const char *existing = std::getenv(name_.c_str());
    if (existing != nullptr) {
      had_original_ = true;
      original_value_ = existing;
    }
    Set(value);
  }

  ~ScopedEnvVar() {
    if (had_original_) {
      Set(original_value_);
    } else {
      Unset();
    }
  }

private:
  void Set(const std::string &value) {
#ifdef _WIN32
    _putenv_s(name_.c_str(), value.c_str());
#else
    setenv(name_.c_str(), value.c_str(), 1);
#endif
  }

  void Unset() {
#ifdef _WIN32
    _putenv_s(name_.c_str(), "");
#else
    unsetenv(name_.c_str());
#endif
  }

  std::string name_;
  std::string original_value_;
  bool had_original_{false};
};

bool WaitForCondition(
    const std::function<bool()> &predicate,
    std::chrono::milliseconds timeout = std::chrono::milliseconds(2000)) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return predicate();
}

int ReserveFreePort() {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  REQUIRE(fd >= 0);

  int opt = 1;
  ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = inet_addr("127.0.0.1");
  addr.sin_port = htons(0);
  REQUIRE(::bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0);

  socklen_t len = sizeof(addr);
  REQUIRE(::getsockname(fd, reinterpret_cast<sockaddr *>(&addr), &len) == 0);
  const int port = ntohs(addr.sin_port);
  ::close(fd);
  return port;
}

std::unique_ptr<Scheduler> MakeScheduler(SimpleTokenizer &tokenizer) {
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      16, 1024, PagedKVCache::EvictionPolicy::kLRU);
  DisaggregatedConfig disagg_config;
  disagg_config.decode_pool_size = 1;
  disagg_config.kv_transport =
      std::make_shared<disaggregated::KVChannel>(8);
  return std::make_unique<Scheduler>(tokenizer, device, cache, nullptr, nullptr,
                                     nullptr, FairnessConfig{}, disagg_config);
}

} // namespace

TEST_CASE("HTTP generation admission fails closed on degraded distributed KV "
          "transport",
          "[http_server][integration][admission]") {
  ScopedEnvVar fail_closed(
      "INFERFLUX_ADMISSION_FAIL_CLOSED_ON_DISAGG_DEGRADED", "true");

  SimpleTokenizer tokenizer;
  auto scheduler = MakeScheduler(tokenizer);
  MetricsRegistry metrics;
  const int port = ReserveFreePort();
  HttpServer server("127.0.0.1", port, scheduler.get(), nullptr, &metrics,
                    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                    HttpServer::TlsConfig{}, 1);
  server.SetRole(HttpServer::PoolRole::kDecode);
  server.SetModelReady(true);
  server.Start();

  REQUIRE(WaitForCondition([&]() { return scheduler->LiveDecodeWorkers() == 1; }));

  inferflux::HttpClient client;
  REQUIRE(WaitForCondition([&]() {
    try {
      auto resp = client.Get("http://127.0.0.1:" + std::to_string(port) +
                             "/livez");
      return resp.status == 200;
    } catch (...) {
      return false;
    }
  }));

  metrics.RecordDisaggKVTicketStage("timed_out");
  metrics.RecordDisaggKVTicketStage("timed_out");
  metrics.RecordDisaggKVTicketStage("timed_out");

  auto resp = client.Post("http://127.0.0.1:" + std::to_string(port) +
                              "/v1/completions",
                          R"({"prompt":"hello","max_tokens":4})");

  server.Stop();

  REQUIRE(resp.status == 503);
  REQUIRE(resp.body.find("distributed_kv_transport_degraded") !=
          std::string::npos);
}
