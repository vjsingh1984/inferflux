#include <catch2/catch_amalgamated.hpp>

#include "runtime/backends/cpu/llama_backend.h"
#include "runtime/disaggregated/kv_channel.h"
#include "scheduler/request_batch.h"

using namespace inferflux;

// ---------------------------------------------------------------------------
// PrefillResult struct
// ---------------------------------------------------------------------------

TEST_CASE("PrefillResult defaults: ok=false, n_past=0", "[phased]") {
  LlamaCPUBackend::PrefillResult r;
  REQUIRE_FALSE(r.ok);
  REQUIRE(r.n_past == 0);
}

// ---------------------------------------------------------------------------
// LlamaCPUBackend — unloaded-backend guard paths
// (ForceReadyForTests() sets test_ready_=true but context_ remains null;
//  these guard paths trigger when context_ == nullptr.)
// ---------------------------------------------------------------------------

TEST_CASE("Prefill returns ok=false when context is null", "[phased]") {
  LlamaCPUBackend backend;
  // context_ is null; Prefill must return {ok=false} without crashing.
  auto pr = backend.Prefill("hello world", 0);
  REQUIRE_FALSE(pr.ok);
  REQUIRE(pr.n_past == 0);
}

TEST_CASE("Decode returns empty string when context is null", "[phased]") {
  LlamaCPUBackend backend;
  std::string out = backend.Decode(4, 0, 8);
  REQUIRE(out.empty());
}

TEST_CASE("FreeSequence is a no-op when context is null", "[phased]") {
  LlamaCPUBackend backend;
  // Must not crash or throw.
  REQUIRE_NOTHROW(backend.FreeSequence(0));
  REQUIRE_NOTHROW(backend.FreeSequence(15));
}

TEST_CASE("Prefill returns ok=false for empty prompt", "[phased]") {
  LlamaCPUBackend backend;
  auto pr = backend.Prefill("", 1);
  REQUIRE_FALSE(pr.ok);
}

// ---------------------------------------------------------------------------
// KVPacket carries n_past and sequence_id fields
// ---------------------------------------------------------------------------

TEST_CASE("KVPacket n_past and sequence_id default to -1", "[phased]") {
  disaggregated::KVPacket pkt;
  REQUIRE(pkt.n_past == -1);
  REQUIRE(pkt.sequence_id == -1);
}

TEST_CASE("KVPacket n_past and sequence_id survive enqueue/dequeue round-trip", "[phased]") {
  disaggregated::KVChannel channel(4);
  disaggregated::KVPacket pkt;
  pkt.request_id = 7;
  pkt.n_past = 42;
  pkt.sequence_id = 3;
  REQUIRE(channel.Enqueue(std::move(pkt)));
  auto out = channel.TryDequeue();
  REQUIRE(out.has_value());
  REQUIRE(out->request_id == 7);
  REQUIRE(out->n_past == 42);
  REQUIRE(out->sequence_id == 3);
}

// ---------------------------------------------------------------------------
// InferenceRequest n_past and sequence_id
// ---------------------------------------------------------------------------

TEST_CASE("InferenceRequest n_past and sequence_id default to -1", "[phased]") {
  InferenceRequest req;
  REQUIRE(req.n_past == -1);
  REQUIRE(req.sequence_id == -1);
}

TEST_CASE("InferenceRequest phased fields are assignable", "[phased]") {
  InferenceRequest req;
  req.n_past = 128;
  req.sequence_id = 5;
  REQUIRE(req.n_past == 128);
  REQUIRE(req.sequence_id == 5);
}
