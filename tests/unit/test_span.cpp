#include "server/tracing/span.h"

#include <catch2/catch_amalgamated.hpp>

#include <string>
#include <vector>

using namespace inferflux;
using namespace inferflux::tracing;

TEST_CASE("SpanContext: NewContext produces valid 32/16 hex IDs", "[tracing]") {
  auto ctx = NewContext();
  REQUIRE(ctx.valid());
  REQUIRE(ctx.trace_id.size() == 32);
  REQUIRE(ctx.span_id.size() == 16);
  // All chars must be lowercase hex.
  for (char c : ctx.trace_id) {
    REQUIRE(((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')));
  }
  for (char c : ctx.span_id) {
    REQUIRE(((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')));
  }
}

TEST_CASE("SpanContext: ToTraceparent format", "[tracing]") {
  auto ctx = NewContext();
  std::string tp = ctx.ToTraceparent();
  // Expected: "00-{32 hex}-{16 hex}-01"
  REQUIRE(tp.size() == 55);
  REQUIRE(tp.substr(0, 3) == "00-");
  REQUIRE(tp.substr(35, 1) == "-");
  REQUIRE(tp.substr(52, 3) == "-01");
}

TEST_CASE("SpanContext: ToTraceparent empty on invalid context", "[tracing]") {
  SpanContext ctx; // default — invalid
  REQUIRE_FALSE(ctx.valid());
  REQUIRE(ctx.ToTraceparent().empty());
}

TEST_CASE("ParseTraceparent: round-trip", "[tracing]") {
  auto orig = NewContext();
  std::string tp = orig.ToTraceparent();
  auto parsed = ParseTraceparent(tp);
  REQUIRE(parsed.valid());
  REQUIRE(parsed.trace_id == orig.trace_id);
  // span_id in traceparent is the parent-id, not the orig span_id
  // (the format is "00-trace_id-span_id-flags"), so they should match.
  REQUIRE(parsed.span_id == orig.span_id);
}

TEST_CASE("ParseTraceparent: rejects short/malformed input", "[tracing]") {
  REQUIRE_FALSE(ParseTraceparent("").valid());
  REQUIRE_FALSE(ParseTraceparent("00-tooshort-tooshort-01").valid());
  REQUIRE_FALSE(ParseTraceparent("garbage").valid());
}

TEST_CASE("ChildContext: inherits trace_id, generates new span_id",
          "[tracing]") {
  auto parent = NewContext();
  auto child = ChildContext(parent);
  REQUIRE(child.trace_id == parent.trace_id);
  REQUIRE(child.span_id != parent.span_id);
  REQUIRE(child.valid());
}

TEST_CASE("ChildContext: from empty parent starts new trace", "[tracing]") {
  SpanContext empty;
  auto child = ChildContext(empty);
  REQUIRE(child.valid());
  REQUIRE(child.trace_id.size() == 32);
}

TEST_CASE("Span: records duration via callback", "[tracing]") {
  double recorded_ms = -1.0;
  std::string recorded_name;
  {
    Span span("test-phase", NewContext(),
              [&](const std::string &name, const SpanContext &, double ms) {
                recorded_name = name;
                recorded_ms = ms;
              });
    // Span destructor fires on scope exit.
  }
  REQUIRE(recorded_name == "test-phase");
  REQUIRE(recorded_ms >= 0.0);
}

TEST_CASE("Span: Finish() is idempotent", "[tracing]") {
  int call_count = 0;
  Span span(
      "op", NewContext(),
      [&](const std::string &, const SpanContext &, double) { ++call_count; });
  span.Finish();
  span.Finish(); // Second call must not double-fire.
  // Destructor fires but span is already finished.
  // call_count stays at 1 after scope exit.
  REQUIRE(call_count == 1);
}

TEST_CASE("Span: ElapsedMs is non-negative", "[tracing]") {
  Span span("op", NewContext(), nullptr);
  double elapsed = span.ElapsedMs();
  REQUIRE(elapsed >= 0.0);
}

TEST_CASE("NewContext: unique per call", "[tracing]") {
  // Generate several contexts and confirm no duplicates (probabilistic,
  // astronomically unlikely).
  std::vector<std::string> ids;
  for (int i = 0; i < 20; ++i) {
    ids.push_back(NewContext().trace_id);
  }
  for (std::size_t i = 0; i < ids.size(); ++i) {
    for (std::size_t j = i + 1; j < ids.size(); ++j) {
      REQUIRE(ids[i] != ids[j]);
    }
  }
}
