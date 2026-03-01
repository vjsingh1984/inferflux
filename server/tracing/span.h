#pragma once

// server/tracing/span.h — Lightweight W3C Trace Context + RAII span timer.
//
// This abstraction satisfies OBS-2 without pulling in the full OpenTelemetry
// C++ SDK. The Span::OnEnd callback is the extension point: when a real OTel
// exporter is wired, replace the callback with an SDK span::End() call.
//
// W3C Trace Context spec: https://www.w3.org/TR/trace-context/
// Format:  traceparent: 00-<trace-id(32 hex)>-<parent-id(16 hex)>-<flags(2 hex)>

#include <chrono>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>

namespace inferflux {

// SpanContext carries the W3C trace-id and span-id for distributed tracing.
// An empty context means no trace is active.
struct SpanContext {
  std::string trace_id;  // 32-char lowercase hex (16 bytes)
  std::string span_id;   // 16-char lowercase hex (8 bytes)

  bool valid() const { return trace_id.size() == 32 && span_id.size() == 16; }

  // Returns the W3C traceparent header value for this context.
  std::string ToTraceparent() const {
    if (!valid()) return {};
    return "00-" + trace_id + "-" + span_id + "-01";
  }
};

namespace tracing {

namespace detail {
inline uint64_t RandomU64() {
  // Per-thread RNG seeded from high-resolution clock to avoid correlation
  // between threads. Using steady_clock rather than random_device to avoid
  // blocking on low-entropy systems (CI containers).
  static thread_local std::mt19937_64 rng{
      static_cast<uint64_t>(
          std::chrono::steady_clock::now().time_since_epoch().count())};
  return rng();
}
}  // namespace detail

// Generates a lowercase hex string encoding `bytes` random bytes.
// `bytes` must be a multiple of 8.
inline std::string RandomHex(std::size_t bytes) {
  std::ostringstream oss;
  for (std::size_t i = 0; i < bytes / 8; ++i) {
    oss << std::hex << std::setfill('0') << std::setw(16) << detail::RandomU64();
  }
  return oss.str();
}

// Returns a fresh root SpanContext (new trace_id + span_id).
inline SpanContext NewContext() {
  return SpanContext{RandomHex(16), RandomHex(8)};
}

// Returns a child SpanContext: inherits trace_id, generates new span_id.
// If parent is empty, starts a new root trace.
inline SpanContext ChildContext(const SpanContext& parent) {
  std::string tid = parent.trace_id.size() == 32 ? parent.trace_id : RandomHex(16);
  return SpanContext{tid, RandomHex(8)};
}

// Parses a W3C traceparent header value into a SpanContext.
// Returns an empty (invalid) context on parse failure.
// Expected format: "00-<trace-id(32)>-<parent-id(16)>-<flags(2)>"
inline SpanContext ParseTraceparent(const std::string& header) {
  SpanContext ctx;
  if (header.size() < 55) return ctx;
  auto p1 = header.find('-');
  if (p1 == std::string::npos) return ctx;
  auto p2 = header.find('-', p1 + 1);
  if (p2 == std::string::npos) return ctx;
  auto p3 = header.find('-', p2 + 1);
  if (p3 == std::string::npos) return ctx;
  ctx.trace_id = header.substr(p1 + 1, p2 - p1 - 1);
  ctx.span_id  = header.substr(p2 + 1, p3 - p2 - 1);
  if (!ctx.valid()) ctx = {};
  return ctx;
}

}  // namespace tracing

// Span — RAII timer that records a named phase duration.
//
// OnEnd is called with (name, context, duration_ms) when the span finishes.
// The caller provides the callback; GlobalMetrics() is the default target.
//
// Extension point: when the OpenTelemetry C++ SDK is integrated, replace the
// callback with sdk_span->End() and export via OTLP.
class Span {
 public:
  using OnEnd = std::function<void(const std::string&, const SpanContext&, double)>;

  Span(std::string name, SpanContext context, OnEnd on_end = nullptr)
      : name_(std::move(name)),
        context_(std::move(context)),
        on_end_(std::move(on_end)),
        start_(std::chrono::steady_clock::now()) {}

  ~Span() { Finish(); }

  // Non-copyable, non-movable (start time is captured at construction).
  Span(const Span&)            = delete;
  Span& operator=(const Span&) = delete;

  // Finish the span early. Safe to call multiple times (idempotent).
  void Finish() {
    if (finished_) return;
    finished_ = true;
    auto end = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start_).count();
    if (on_end_) on_end_(name_, context_, ms);
  }

  double ElapsedMs() const {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(now - start_).count();
  }

  const SpanContext& context() const { return context_; }
  const std::string& name()    const { return name_; }

 private:
  std::string  name_;
  SpanContext  context_;
  OnEnd        on_end_;
  std::chrono::steady_clock::time_point start_;
  bool         finished_{false};
};

}  // namespace inferflux
