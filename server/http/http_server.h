#pragma once

#include "policy/policy_backend.h"
#include "runtime/speculative/speculative_decoder.h"
#include "scheduler/model_selection.h"
#include "scheduler/scheduler.h"
#include "server/auth/api_key_auth.h"
#include "server/auth/oidc_validator.h"
#include "server/auth/rate_limiter.h"
#include "server/logging/audit_logger.h"
#include "server/metrics/metrics.h"
#include "server/policy/guardrail.h"
#include "server/tracing/span.h"

#if INFERFLUX_ENABLE_WEBUI
#include "webui/ui_renderer.h"
#endif

#include <atomic>
#include <algorithm>
#include <cctype>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include <openssl/ssl.h>

namespace inferflux {

// Test-visible header lookup helper. Header-name matching is case-insensitive
// per RFC 9110.
inline std::string LookupHeaderValueForTest(const std::string &headers,
                                            const std::string &name) {
  auto lower_name = name;
  std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(),
                 [](unsigned char c) {
                   return static_cast<char>(std::tolower(c));
                 });

  std::size_t line_start = 0;
  while (line_start < headers.size()) {
    const std::size_t line_end = headers.find("\r\n", line_start);
    const std::size_t current_end =
        line_end == std::string::npos ? headers.size() : line_end;
    const std::size_t colon = headers.find(':', line_start);
    if (colon != std::string::npos && colon < current_end) {
      std::string header_name = headers.substr(line_start, colon - line_start);
      std::transform(header_name.begin(), header_name.end(), header_name.begin(),
                     [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                     });
      if (header_name == lower_name) {
        std::string value =
            headers.substr(colon + 1, current_end - (colon + 1));
        const auto s = value.find_first_not_of(" \t");
        const auto e = value.find_last_not_of(" \t\r\n");
        return (s == std::string::npos) ? "" : value.substr(s, e - s + 1);
      }
    }
    if (line_end == std::string::npos) {
      break;
    }
    line_start = line_end + 2;
  }
  return {};
}

struct HttpRequestMetadata {
  std::string session_id;
  std::string client_request_id;
  std::string trace_id;
};

struct HttpGenerationRequestEnvelopeInput {
  std::string prompt;
  std::string model{"unknown"};
  int max_tokens{256};
  int priority{0};
  bool stream{false};
  bool json_mode{false};
  bool has_response_format{false};
  std::string response_format_type;
  std::string response_format_schema;
  std::string response_format_grammar;
  std::string response_format_root{"root"};
  bool collect_logprobs{false};
  int logprob_top_n{0};
  SamplingParams sampling{};
  std::vector<std::string> stop;
  bool has_images{false};
  std::vector<DecodedImage> images;
};

inline InferenceRequest BuildGenerationRequestEnvelope(
    const HttpGenerationRequestEnvelopeInput &input,
    const HttpRequestMetadata &metadata) {
  InferenceRequest request;
  request.prompt = input.prompt;
  if (input.max_tokens > 0) {
    request.max_tokens = input.max_tokens;
  }
  request.model = input.model;
  request.priority = input.priority;
  request.session_id = metadata.session_id;
  request.client_request_id = metadata.client_request_id;
  request.trace_id = metadata.trace_id;
  request.stream = input.stream;
  request.json_mode = input.json_mode;
  if (input.has_response_format) {
    request.has_response_format = true;
    request.response_format_type = input.response_format_type;
    request.response_format_schema = input.response_format_schema;
    request.response_format_grammar = input.response_format_grammar;
    request.response_format_root = input.response_format_root;
  }
  if (input.collect_logprobs) {
    request.collect_logprobs = true;
    request.logprob_top_n = input.logprob_top_n;
  }
  request.sampling = input.sampling;
  request.stop = input.stop;
  if (input.has_images) {
    request.has_images = true;
    request.images = input.images;
  }
  return request;
}

// Test-visible metadata normalization helper. Payload-provided values take
// precedence over headers; trace_id is derived from an incoming traceparent
// header when present and valid.
inline HttpRequestMetadata ResolveHttpRequestMetadataForTest(
    const std::string &headers, const std::string &payload_session_id,
    const std::string &payload_client_request_id) {
  HttpRequestMetadata metadata;
  metadata.session_id = payload_session_id.empty()
                            ? LookupHeaderValueForTest(headers,
                                                       "x-inferflux-session-id")
                            : payload_session_id;
  metadata.client_request_id =
      payload_client_request_id.empty()
          ? LookupHeaderValueForTest(headers, "x-inferflux-client-request-id")
          : payload_client_request_id;
  const std::string traceparent =
      LookupHeaderValueForTest(headers, "traceparent");
  if (!traceparent.empty()) {
    auto parent_ctx = tracing::ParseTraceparent(traceparent);
    metadata.trace_id = parent_ctx.trace_id;
  }
  return metadata;
}

class HttpServer {
public:
  struct TlsConfig {
    bool enabled{false};
    std::string cert_path;
    std::string key_path;
  };

  struct ReadyStatus {
    bool ready{false};
    bool model_loaded{false};
    bool decode_pool_warm{false};
    bool disagg_transport_degraded{false};
    uint64_t disagg_timeout_debt{0};
    uint64_t disagg_timeout_debt_threshold{0};
    uint64_t disagg_timeout_streak{0};
    uint64_t disagg_timeout_streak_threshold{0};
    std::string role{"unified"};
    std::string reason;
  };

  struct AdminPoolsStatus {
    struct DistributedKVStatus {
      std::optional<int64_t> enqueue_rejections_total;
      std::optional<int64_t> enqueue_exhausted_total;
      std::optional<int64_t> tickets_enqueued_total;
      std::optional<int64_t> tickets_acknowledged_total;
      std::optional<int64_t> tickets_committed_total;
      std::optional<int64_t> tickets_timed_out_total;
    };

    ReadyStatus pool_health;
    std::optional<int64_t> queue_depth;
    std::optional<int64_t> prefill_queue_depth;
    std::optional<int64_t> decode_queue_depth;
    std::optional<int64_t> batch_limit_size;
    std::optional<int64_t> batch_limit_tokens;
    DistributedKVStatus distributed_kv;
  };

  struct AdmissionDecision {
    bool allowed{true};
    int http_status{200};
    std::string error;
    std::string reason;
  };

  // Disaggregated deployment role (§2.5 item 12).
  // kUnified: combined prefill+decode (default; existing behaviour).
  // kPrefill: this instance only runs prefill; /readyz gates on model load.
  // kDecode:  this instance only runs decode; /readyz gates on decode pool.
  enum class PoolRole { kUnified, kPrefill, kDecode };

  HttpServer(std::string host, int port, Scheduler *scheduler,
             std::shared_ptr<ApiKeyAuth> auth, MetricsRegistry *metrics,
             OIDCValidator *oidc, RateLimiter *rate_limiter,
             Guardrail *guardrail, AuditLogger *audit_logger,
             PolicyBackend *policy_store,
             std::shared_ptr<SpeculativeDecoder> speculative_decoder,
             const TlsConfig &tls_config, int num_workers = 4,
             const ModelSelectionOptions &model_selection_options = {});
  ~HttpServer();

  void Start();
  void Stop();
  void SetModelReady(bool ready) { model_ready_.store(ready); }
  void SetRole(PoolRole role) { role_.store(role, std::memory_order_relaxed); }
  void SetDecodePoolReady(bool ready) {
    decode_pool_ready_.store(ready, std::memory_order_relaxed);
  }
  ReadyStatus EvaluateReadyStatus() const;
  AdminPoolsStatus EvaluateAdminPoolsStatus() const;
  AdmissionDecision EvaluateGenerationAdmissionDecision() const;

private:
  struct ClientSession {
    int fd{-1};
    SSL *ssl{nullptr};
  };

  void Run();
  void HandleClient(ClientSession &session);

  struct AuthContext {
    std::string subject{"anonymous"};
    std::unordered_set<std::string> scopes;
  };

  bool ResolveSubject(const std::string &header_blob, AuthContext *ctx) const;
  bool RequireScope(const AuthContext &ctx, const std::string &scope,
                    ClientSession &session, const std::string &error_message);

  bool SendAll(ClientSession &session, const std::string &payload);
  ssize_t Receive(ClientSession &session, char *buffer, std::size_t length);
  void CloseSession(ClientSession &session);

  std::string host_;
  int port_;
  Scheduler *scheduler_;
  std::shared_ptr<ApiKeyAuth> auth_;
  MetricsRegistry *metrics_;
  OIDCValidator *oidc_;
  RateLimiter *rate_limiter_;
  Guardrail *guardrail_;
  AuditLogger *audit_logger_;
  PolicyBackend *policy_store_;
  std::shared_ptr<SpeculativeDecoder> speculative_decoder_;
  mutable std::mutex policy_update_mutex_;
  mutable std::mutex model_selection_mutex_;
  ModelSelectionOptions model_selection_options_;
  bool tls_enabled_{false};
  SSL_CTX *ssl_ctx_{nullptr};
  std::atomic<bool> running_{false};
  std::atomic<bool> model_ready_{false};
  std::atomic<PoolRole> role_{PoolRole::kUnified};
  std::atomic<bool> decode_pool_ready_{false};
  bool admission_fail_closed_on_disagg_degraded_{false};
  int readyz_disagg_timeout_debt_threshold_{0};
  int readyz_disagg_timeout_streak_threshold_{3};
#if INFERFLUX_ENABLE_WEBUI
  std::unique_ptr<WebUiRenderer> webui_renderer_;
#endif
  std::atomic<int> server_fd_{-1};
  int num_workers_;
  std::thread accept_thread_;
  std::vector<std::thread> workers_;
  std::queue<ClientSession> client_queue_;
  std::mutex queue_mutex_;
  std::condition_variable queue_cv_;

  void WorkerLoop();
};

} // namespace inferflux
