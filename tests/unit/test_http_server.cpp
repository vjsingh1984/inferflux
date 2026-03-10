#include <catch2/catch_amalgamated.hpp>

#include "runtime/disaggregated/kv_channel.h"
#include "runtime/kv_cache/paged_kv_cache.h"
#include "scheduler/scheduler.h"
#include "server/http/http_server.h"
#include "server/metrics/metrics.h"

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
    std::chrono::milliseconds timeout = std::chrono::milliseconds(1000)) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return predicate();
}

std::unique_ptr<Scheduler> MakeScheduler(SimpleTokenizer &tokenizer,
                                         bool with_transport) {
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      16, 1024, PagedKVCache::EvictionPolicy::kLRU);
  DisaggregatedConfig disagg_config;
  disagg_config.decode_pool_size = 1;
  if (with_transport) {
    disagg_config.kv_transport =
        std::make_shared<disaggregated::KVChannel>(8);
  }
  return std::make_unique<Scheduler>(tokenizer, device, cache, nullptr, nullptr,
                                     nullptr, FairnessConfig{}, disagg_config);
}

std::unique_ptr<HttpServer> MakeServer(Scheduler *scheduler,
                                       MetricsRegistry *metrics) {
  return std::make_unique<HttpServer>(
      "127.0.0.1", 0, scheduler, nullptr, metrics, nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr, HttpServer::TlsConfig{}, 1);
}

} // namespace

TEST_CASE("LookupHeaderValueForTest matches header names case-insensitively",
          "[http_server]") {
  const std::string headers =
      "POST /v1/completions HTTP/1.1\r\n"
      "Host: 127.0.0.1\r\n"
      "X-InferFlux-Client-Request-Id: bench-5\r\n"
      "traceparent: 00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01\r\n"
      "\r\n";

  REQUIRE(LookupHeaderValueForTest(headers, "x-inferflux-client-request-id") ==
          "bench-5");
  REQUIRE(LookupHeaderValueForTest(headers, "X-INFERFLUX-CLIENT-REQUEST-ID") ==
          "bench-5");
  REQUIRE(LookupHeaderValueForTest(headers, "TraceParent") ==
          "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01");
}

TEST_CASE("ResolveHttpRequestMetadataForTest prefers payload metadata over "
          "headers",
          "[http_server]") {
  const std::string headers =
      "POST /v1/completions HTTP/1.1\r\n"
      "Host: 127.0.0.1\r\n"
      "X-InferFlux-Session-Id: header-session\r\n"
      "X-InferFlux-Client-Request-Id: header-request\r\n"
      "traceparent: 00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01\r\n"
      "\r\n";

  const auto metadata = ResolveHttpRequestMetadataForTest(
      headers, "payload-session", "payload-request");

  REQUIRE(metadata.session_id == "payload-session");
  REQUIRE(metadata.client_request_id == "payload-request");
  REQUIRE(metadata.trace_id == "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
}

TEST_CASE("ResolveHttpRequestMetadataForTest falls back to headers and rejects "
          "invalid traceparent",
          "[http_server]") {
  const std::string headers =
      "POST /v1/completions HTTP/1.1\r\n"
      "Host: 127.0.0.1\r\n"
      "X-InferFlux-Session-Id: header-session\r\n"
      "x-inferflux-client-request-id: header-request\r\n"
      "TraceParent: garbage\r\n"
      "\r\n";

  const auto metadata =
      ResolveHttpRequestMetadataForTest(headers, "", "");

  REQUIRE(metadata.session_id == "header-session");
  REQUIRE(metadata.client_request_id == "header-request");
  REQUIRE(metadata.trace_id.empty());
}

TEST_CASE("BuildGenerationRequestEnvelope projects prompt, sampling, and "
          "metadata into InferenceRequest",
          "[http_server]") {
  HttpGenerationRequestEnvelopeInput input;
  input.prompt = "hello";
  input.model = "qwen";
  input.max_tokens = 32;
  input.priority = 10;
  input.stream = true;
  input.json_mode = true;
  input.has_response_format = true;
  input.response_format_type = "json_object";
  input.response_format_schema = "{\"type\":\"object\"}";
  input.response_format_grammar = "root ::= \"{}\"";
  input.response_format_root = "root";
  input.collect_logprobs = true;
  input.logprob_top_n = 3;
  input.sampling.temperature = 0.2f;
  input.sampling.top_p = 0.95f;
  input.sampling.top_k = 40;
  input.sampling.seed = 7;
  input.stop = {"\n\n"};

  HttpRequestMetadata metadata;
  metadata.session_id = "sess-1";
  metadata.client_request_id = "bench-7";
  metadata.trace_id = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

  const InferenceRequest request =
      BuildGenerationRequestEnvelope(input, metadata);

  REQUIRE(request.prompt == "hello");
  REQUIRE(request.model == "qwen");
  REQUIRE(request.max_tokens == 32);
  REQUIRE(request.priority == 10);
  REQUIRE(request.stream);
  REQUIRE(request.json_mode);
  REQUIRE(request.has_response_format);
  REQUIRE(request.response_format_type == "json_object");
  REQUIRE(request.response_format_schema == "{\"type\":\"object\"}");
  REQUIRE(request.response_format_grammar == "root ::= \"{}\"");
  REQUIRE(request.response_format_root == "root");
  REQUIRE(request.collect_logprobs);
  REQUIRE(request.logprob_top_n == 3);
  REQUIRE(request.sampling.temperature == Catch::Approx(0.2f));
  REQUIRE(request.sampling.top_p == Catch::Approx(0.95f));
  REQUIRE(request.sampling.top_k == 40);
  REQUIRE(request.sampling.seed == 7);
  REQUIRE(request.stop == std::vector<std::string>{"\n\n"});
  REQUIRE(request.session_id == "sess-1");
  REQUIRE(request.client_request_id == "bench-7");
  REQUIRE(request.trace_id == "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
}

TEST_CASE("BuildGenerationRequestEnvelope keeps multimodal payload attached",
          "[http_server]") {
  HttpGenerationRequestEnvelopeInput input;
  input.prompt = "describe";
  input.has_images = true;
  DecodedImage image;
  image.raw_bytes = {0, 1, 2, 3, 4, 5};
  image.source_uri = "data:image/png;base64,AAAA";
  image.image_id = "img-1";
  input.images.push_back(image);

  const InferenceRequest request =
      BuildGenerationRequestEnvelope(input, HttpRequestMetadata{});

  REQUIRE(request.has_images);
  REQUIRE(request.images.size() == 1);
  REQUIRE(request.images.front().raw_bytes ==
          std::vector<uint8_t>{0, 1, 2, 3, 4, 5});
  REQUIRE(request.images.front().source_uri == "data:image/png;base64,AAAA");
  REQUIRE(request.images.front().image_id == "img-1");
}

TEST_CASE("BuildGenerationExecutionContext derives trace header and stream "
          "identity from request",
          "[http_server]") {
  InferenceRequest request;
  request.model = "qwen";
  request.prompt = "hello";
  request.trace_id = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

  const auto context = BuildGenerationExecutionContext(
      request, /*is_legacy_completions=*/true, /*stream_enabled=*/true,
      /*stream_created_at=*/42);

  REQUIRE(context.model == "qwen");
  REQUIRE(context.audit_prompt == "hello");
  REQUIRE(context.stream_id == "chatcmpl-42");
  REQUIRE(context.stream_created_at == 42);
  REQUIRE(context.trace_response_header.find(
              "traceparent: 00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-") !=
          std::string::npos);
  REQUIRE(context.trace_response_header.find("Deprecation: true\r\n") !=
          std::string::npos);
  REQUIRE(context.trace_response_header.find(
              "Link: </v1/chat/completions>; rel=\"successor-version\"\r\n") !=
          std::string::npos);
}

TEST_CASE("Tool stub helpers normalize fallback tool metadata", "[http_server]") {
  REQUIRE(IsToolStubCompletion(true, ""));
  REQUIRE(IsToolStubCompletion(false, "No model backend is loaded"));
  REQUIRE_FALSE(IsToolStubCompletion(false, "regular completion"));

  const ToolCallResult no_hint =
      BuildStubToolCall("lookup_weather", /*include_hint=*/false);
  REQUIRE(no_hint.detected);
  REQUIRE(no_hint.call_id == "call_stub_lookup_weather");
  REQUIRE(no_hint.function_name == "lookup_weather");
  REQUIRE(no_hint.arguments_json == "{\"reason\":\"no_model_available\"}");

  const ToolCallResult with_hint =
      BuildStubToolCall("lookup_weather", /*include_hint=*/true);
  REQUIRE(with_hint.detected);
  REQUIRE(with_hint.arguments_json.find("\"hint\"") != std::string::npos);

  REQUIRE(BuildToolStateLogLine(true, "No model backend is loaded") ==
          "tool_state no_backend=1 contains_stub=1");
  REQUIRE(BuildStubToolLogLine("lookup_weather") ==
          "[tools] stub tool call for lookup_weather");
}

TEST_CASE("HttpServer ready status reports healthy distributed decode pool",
          "[http_server]") {
  SimpleTokenizer tokenizer;
  auto scheduler = MakeScheduler(tokenizer, true);
  MetricsRegistry metrics;
  auto server = MakeServer(scheduler.get(), &metrics);
  server->SetRole(HttpServer::PoolRole::kDecode);
  server->SetModelReady(true);

  REQUIRE(WaitForCondition(
      [&]() { return scheduler->LiveDecodeWorkers() == 1; }));

  const auto status = server->EvaluateReadyStatus();
  REQUIRE(status.ready);
  REQUIRE(status.model_loaded);
  REQUIRE(status.decode_pool_warm);
  REQUIRE_FALSE(status.disagg_transport_degraded);
  REQUIRE(status.disagg_timeout_debt == 0);
  REQUIRE(status.disagg_timeout_debt_threshold == 6);
  REQUIRE(status.disagg_timeout_streak == 0);
  REQUIRE(status.disagg_timeout_streak_threshold == 3);
  REQUIRE(status.role == "decode");
}

TEST_CASE(
    "HttpServer ready status fails when distributed KV timeout streak hits "
    "threshold",
    "[http_server]") {
  SimpleTokenizer tokenizer;
  auto scheduler = MakeScheduler(tokenizer, true);
  MetricsRegistry metrics;
  auto server = MakeServer(scheduler.get(), &metrics);
  server->SetRole(HttpServer::PoolRole::kDecode);
  server->SetModelReady(true);

  REQUIRE(WaitForCondition(
      [&]() { return scheduler->LiveDecodeWorkers() == 1; }));

  metrics.RecordDisaggKVTicketStage("timed_out");
  metrics.RecordDisaggKVTicketStage("timed_out");
  metrics.RecordDisaggKVTicketStage("timed_out");

  const auto status = server->EvaluateReadyStatus();
  REQUIRE_FALSE(status.ready);
  REQUIRE(status.model_loaded);
  REQUIRE(status.decode_pool_warm);
  REQUIRE(status.disagg_transport_degraded);
  REQUIRE(status.disagg_timeout_debt == 3);
  REQUIRE(status.disagg_timeout_debt_threshold == 6);
  REQUIRE(status.disagg_timeout_streak == 3);
  REQUIRE(status.reason == "distributed kv transport degraded");
}

TEST_CASE(
    "HttpServer ready status fails when distributed KV timeout debt hits "
    "threshold despite intervening commit",
    "[http_server]") {
  ScopedEnvVar streak_threshold("INFERFLUX_READYZ_DISAGG_TIMEOUT_STREAK_THRESHOLD",
                                "5");
  ScopedEnvVar debt_threshold("INFERFLUX_READYZ_DISAGG_TIMEOUT_DEBT_THRESHOLD",
                              "2");

  SimpleTokenizer tokenizer;
  auto scheduler = MakeScheduler(tokenizer, true);
  MetricsRegistry metrics;
  auto server = MakeServer(scheduler.get(), &metrics);
  server->SetRole(HttpServer::PoolRole::kDecode);
  server->SetModelReady(true);

  REQUIRE(WaitForCondition(
      [&]() { return scheduler->LiveDecodeWorkers() == 1; }));

  metrics.RecordDisaggKVTicketStage("timed_out");
  metrics.RecordDisaggKVTicketStage("committed");
  metrics.RecordDisaggKVTicketStage("timed_out");
  metrics.RecordDisaggKVTicketStage("timed_out");

  const auto status = server->EvaluateReadyStatus();
  REQUIRE_FALSE(status.ready);
  REQUIRE(status.disagg_transport_degraded);
  REQUIRE(status.disagg_timeout_debt == 2);
  REQUIRE(status.disagg_timeout_debt_threshold == 2);
  REQUIRE(status.disagg_timeout_streak == 2);
  REQUIRE(status.disagg_timeout_streak_threshold == 5);
  REQUIRE(status.reason == "distributed kv transport degraded");
}

TEST_CASE("HttpServer ready status recovers after a committed KV handoff",
          "[http_server]") {
  SimpleTokenizer tokenizer;
  auto scheduler = MakeScheduler(tokenizer, true);
  MetricsRegistry metrics;
  auto server = MakeServer(scheduler.get(), &metrics);
  server->SetRole(HttpServer::PoolRole::kDecode);
  server->SetModelReady(true);

  REQUIRE(WaitForCondition(
      [&]() { return scheduler->LiveDecodeWorkers() == 1; }));

  metrics.RecordDisaggKVTicketStage("timed_out");
  metrics.RecordDisaggKVTicketStage("timed_out");
  metrics.RecordDisaggKVTicketStage("timed_out");
  metrics.RecordDisaggKVTicketStage("committed");

  const auto status = server->EvaluateReadyStatus();
  REQUIRE(status.ready);
  REQUIRE_FALSE(status.disagg_transport_degraded);
  REQUIRE(status.disagg_timeout_debt == 2);
  REQUIRE(status.disagg_timeout_streak == 0);
}

TEST_CASE("HttpServer generation admission stays open by default when "
          "distributed transport is degraded",
          "[http_server]") {
  SimpleTokenizer tokenizer;
  auto scheduler = MakeScheduler(tokenizer, true);
  MetricsRegistry metrics;
  auto server = MakeServer(scheduler.get(), &metrics);
  server->SetRole(HttpServer::PoolRole::kDecode);
  server->SetModelReady(true);

  REQUIRE(WaitForCondition(
      [&]() { return scheduler->LiveDecodeWorkers() == 1; }));

  metrics.RecordDisaggKVTicketStage("timed_out");
  metrics.RecordDisaggKVTicketStage("timed_out");
  metrics.RecordDisaggKVTicketStage("timed_out");

  const auto decision = server->EvaluateGenerationAdmissionDecision();
  REQUIRE(decision.allowed);
  REQUIRE(decision.error.empty());
}

TEST_CASE("HttpServer generation admission fails closed when configured and "
          "distributed transport is degraded",
          "[http_server]") {
  ScopedEnvVar fail_closed(
      "INFERFLUX_ADMISSION_FAIL_CLOSED_ON_DISAGG_DEGRADED", "true");

  SimpleTokenizer tokenizer;
  auto scheduler = MakeScheduler(tokenizer, true);
  MetricsRegistry metrics;
  auto server = MakeServer(scheduler.get(), &metrics);
  server->SetRole(HttpServer::PoolRole::kDecode);
  server->SetModelReady(true);

  REQUIRE(WaitForCondition(
      [&]() { return scheduler->LiveDecodeWorkers() == 1; }));

  metrics.RecordDisaggKVTicketStage("timed_out");
  metrics.RecordDisaggKVTicketStage("timed_out");
  metrics.RecordDisaggKVTicketStage("timed_out");

  const auto decision = server->EvaluateGenerationAdmissionDecision();
  REQUIRE_FALSE(decision.allowed);
  REQUIRE(decision.http_status == 503);
  REQUIRE(decision.error == "distributed_kv_transport_degraded");
  REQUIRE(decision.reason == "distributed kv transport degraded");
}

TEST_CASE("HttpServer generation admission ignores fail-closed policy without "
          "KV transport",
          "[http_server]") {
  ScopedEnvVar fail_closed(
      "INFERFLUX_ADMISSION_FAIL_CLOSED_ON_DISAGG_DEGRADED", "true");

  SimpleTokenizer tokenizer;
  auto scheduler = MakeScheduler(tokenizer, false);
  MetricsRegistry metrics;
  auto server = MakeServer(scheduler.get(), &metrics);
  server->SetRole(HttpServer::PoolRole::kDecode);
  server->SetModelReady(true);

  REQUIRE(WaitForCondition(
      [&]() { return scheduler->LiveDecodeWorkers() == 1; }));

  metrics.RecordDisaggKVTicketStage("timed_out");
  metrics.RecordDisaggKVTicketStage("timed_out");
  metrics.RecordDisaggKVTicketStage("timed_out");

  const auto decision = server->EvaluateGenerationAdmissionDecision();
  REQUIRE(decision.allowed);
  REQUIRE(decision.error.empty());
}

TEST_CASE("HttpServer ignores distributed timeout streak without KV transport",
          "[http_server]") {
  SimpleTokenizer tokenizer;
  auto scheduler = MakeScheduler(tokenizer, false);
  MetricsRegistry metrics;
  auto server = MakeServer(scheduler.get(), &metrics);
  server->SetRole(HttpServer::PoolRole::kDecode);
  server->SetModelReady(true);

  REQUIRE(WaitForCondition(
      [&]() { return scheduler->LiveDecodeWorkers() == 1; }));

  metrics.RecordDisaggKVTicketStage("timed_out");
  metrics.RecordDisaggKVTicketStage("timed_out");
  metrics.RecordDisaggKVTicketStage("timed_out");

  const auto status = server->EvaluateReadyStatus();
  REQUIRE(status.ready);
  REQUIRE_FALSE(status.disagg_transport_degraded);
  REQUIRE(status.disagg_timeout_streak == 0);
}

TEST_CASE("HttpServer admin pools status mirrors readiness and scheduler gauges",
          "[http_server]") {
  SimpleTokenizer tokenizer;
  auto scheduler = MakeScheduler(tokenizer, true);
  MetricsRegistry metrics;
  metrics.SetQueueDepth(7);
  metrics.SetPrefillQueueDepth(2);
  metrics.SetDecodeQueueDepth(5);
  metrics.SetSchedulerBatchLimits(4, 8192);
  metrics.RecordDisaggKVEnqueueRejected(false);
  metrics.RecordDisaggKVEnqueueRejected(true);

  auto server = MakeServer(scheduler.get(), &metrics);
  server->SetRole(HttpServer::PoolRole::kDecode);
  server->SetModelReady(true);

  REQUIRE(WaitForCondition(
      [&]() { return scheduler->LiveDecodeWorkers() == 1; }));

  metrics.RecordDisaggKVTicketStage("enqueued");
  metrics.RecordDisaggKVTicketStage("enqueued");
  metrics.RecordDisaggKVTicketStage("acknowledged");
  metrics.RecordDisaggKVTicketStage("committed");
  metrics.RecordDisaggKVTicketStage("timed_out");
  metrics.RecordDisaggKVTicketStage("timed_out");
  metrics.RecordDisaggKVTicketStage("timed_out");
  REQUIRE(metrics.GetDisaggKVTimeoutStreak() == 3);
  REQUIRE(metrics.GetDisaggKVTimeoutDebt() == 3);

  const auto ready_status = server->EvaluateReadyStatus();
  REQUIRE_FALSE(ready_status.ready);
  REQUIRE(ready_status.disagg_transport_degraded);
  REQUIRE(ready_status.disagg_timeout_debt == 3);
  REQUIRE(ready_status.disagg_timeout_debt_threshold == 6);

  const auto status = server->EvaluateAdminPoolsStatus();
  REQUIRE_FALSE(status.pool_health.ready);
  REQUIRE(status.pool_health.disagg_transport_degraded);
  REQUIRE(status.pool_health.disagg_timeout_debt == 3);
  REQUIRE(status.queue_depth == 7);
  REQUIRE(status.prefill_queue_depth == 2);
  REQUIRE(status.decode_queue_depth == 5);
  REQUIRE(status.batch_limit_size == 4);
  REQUIRE(status.batch_limit_tokens == 8192);
  REQUIRE(status.distributed_kv.enqueue_rejections_total == 2);
  REQUIRE(status.distributed_kv.enqueue_exhausted_total == 1);
  REQUIRE(status.distributed_kv.tickets_enqueued_total == 2);
  REQUIRE(status.distributed_kv.tickets_acknowledged_total == 1);
  REQUIRE(status.distributed_kv.tickets_committed_total == 1);
  REQUIRE(status.distributed_kv.tickets_timed_out_total == 3);
}

TEST_CASE("HttpServer admin pools status tolerates missing metrics registry",
          "[http_server]") {
  SimpleTokenizer tokenizer;
  auto scheduler = MakeScheduler(tokenizer, true);
  auto server = MakeServer(scheduler.get(), nullptr);
  server->SetRole(HttpServer::PoolRole::kDecode);
  server->SetModelReady(true);

  REQUIRE(WaitForCondition(
      [&]() { return scheduler->LiveDecodeWorkers() == 1; }));

  const auto status = server->EvaluateAdminPoolsStatus();
  REQUIRE(status.pool_health.ready);
  REQUIRE(status.pool_health.disagg_timeout_debt == 0);
  REQUIRE_FALSE(status.queue_depth.has_value());
  REQUIRE_FALSE(status.prefill_queue_depth.has_value());
  REQUIRE_FALSE(status.decode_queue_depth.has_value());
  REQUIRE_FALSE(status.batch_limit_size.has_value());
  REQUIRE_FALSE(status.batch_limit_tokens.has_value());
  REQUIRE_FALSE(status.distributed_kv.enqueue_rejections_total.has_value());
  REQUIRE_FALSE(status.distributed_kv.enqueue_exhausted_total.has_value());
  REQUIRE_FALSE(status.distributed_kv.tickets_enqueued_total.has_value());
  REQUIRE_FALSE(
      status.distributed_kv.tickets_acknowledged_total.has_value());
  REQUIRE_FALSE(status.distributed_kv.tickets_committed_total.has_value());
  REQUIRE_FALSE(status.distributed_kv.tickets_timed_out_total.has_value());
}
