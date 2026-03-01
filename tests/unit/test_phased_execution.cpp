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

// ---------------------------------------------------------------------------
// Bug fix: KVChannel deadlock when channel fills with no consumers (Bug 1)
//
// The scheduler gates enqueue on use_decode_workers_.  We can't test that
// gate here, but we can verify the channel contract that callers rely on:
// Enqueue() returns false when at capacity, so callers MUST handle rejection
// without infinite retry.
// ---------------------------------------------------------------------------

TEST_CASE("KVChannel Enqueue returns false when at capacity (no deadlock contract)", "[phased]") {
  disaggregated::KVChannel channel(2);
  disaggregated::KVPacket p1; p1.request_id = 1;
  disaggregated::KVPacket p2; p2.request_id = 2;
  disaggregated::KVPacket p3; p3.request_id = 3;
  REQUIRE(channel.Enqueue(std::move(p1)));
  REQUIRE(channel.Enqueue(std::move(p2)));
  // At capacity: third enqueue must fail so callers can requeue without looping.
  REQUIRE_FALSE(channel.Enqueue(std::move(p3)));
}

// ---------------------------------------------------------------------------
// Bug fix: sequence_id assignment (Bug 2)
//
// The invariant the Scheduler's free-list provides: if a slot is in use
// (sequence_id >= 0) it must be reset to -1 before the slot is reused.
// Simulate the lifecycle the scheduler enforces: assign → use → clear.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// §2.5: KV serialization guard paths (context_ == nullptr)
// ---------------------------------------------------------------------------

TEST_CASE("SerializeSequence returns empty when context is null", "[phased]") {
  LlamaCPUBackend backend;
  auto blob = backend.SerializeSequence(0);
  REQUIRE(blob.empty());
}

TEST_CASE("HydrateSequence returns false when context is null", "[phased]") {
  LlamaCPUBackend backend;
  std::vector<uint8_t> dummy{1, 2, 3};
  REQUIRE_FALSE(backend.HydrateSequence(0, dummy));
}

TEST_CASE("HydrateSequence returns false for empty blob", "[phased]") {
  LlamaCPUBackend backend;
  std::vector<uint8_t> empty;
  REQUIRE_FALSE(backend.HydrateSequence(0, empty));
}

TEST_CASE("InferenceRequest sequence_id lifecycle: assign then clear on completion", "[phased]") {
  InferenceRequest req;
  REQUIRE(req.sequence_id == -1);  // starts unassigned

  // Scheduler assigns a slot after Prefill().
  req.sequence_id = 3;
  req.n_past = 64;
  REQUIRE(req.sequence_id == 3);

  // Scheduler clears slot after request fully completes (not a fairness yield).
  req.sequence_id = -1;
  req.n_past = -1;
  REQUIRE(req.sequence_id == -1);  // slot returned to the pool
  REQUIRE(req.n_past == -1);
}
