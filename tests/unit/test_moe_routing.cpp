#include <catch2/catch_amalgamated.hpp>

#include "runtime/backends/cpu/llama_backend.h"
#include "runtime/backends/ep_dispatch.h"
#include "scheduler/model_router.h"
#include "scheduler/single_model_router.h"
#include "server/metrics/metrics.h"

using namespace inferflux;

// ---------------------------------------------------------------------------
// LlamaCPUBackend MoE guard paths (model_ == nullptr → 0 / false)
// ---------------------------------------------------------------------------

TEST_CASE("IsMoE returns false when model is not loaded", "[moe]") {
  LlamaCPUBackend backend;
  REQUIRE_FALSE(backend.IsMoE());
}

TEST_CASE("ExpertCount returns 0 when model is not loaded", "[moe]") {
  LlamaCPUBackend backend;
  REQUIRE(backend.ExpertCount() == 0);
}

TEST_CASE("ActiveExperts returns 0 when model is not loaded", "[moe]") {
  LlamaCPUBackend backend;
  REQUIRE(backend.ActiveExperts() == 0);
}

// ---------------------------------------------------------------------------
// ModelInfo MoE fields — defaults
// ---------------------------------------------------------------------------

TEST_CASE("ModelInfo MoE fields default to false/0", "[moe]") {
  ModelInfo info;
  REQUIRE_FALSE(info.is_moe);
  REQUIRE(info.n_experts == 0);
  REQUIRE(info.n_active_experts == 0);
}

TEST_CASE("ModelInfo MoE fields are assignable", "[moe]") {
  ModelInfo info;
  info.is_moe = true;
  info.n_experts = 64;
  info.n_active_experts = 8;
  REQUIRE(info.is_moe);
  REQUIRE(info.n_experts == 64);
  REQUIRE(info.n_active_experts == 8);
}

// ---------------------------------------------------------------------------
// SingleModelRouter populates MoE fields from backend
// (backend has no model loaded → fields stay false/0)
// ---------------------------------------------------------------------------

TEST_CASE("SingleModelRouter RegisterModel populates MoE fields (no model)", "[moe]") {
  auto backend = std::make_shared<LlamaCPUBackend>();
  ModelInfo info;
  info.id = "test-model";
  info.path = "/tmp/test.gguf";
  info.backend = "cpu";

  SingleModelRouter router;
  REQUIRE(router.RegisterModel(info, backend));

  auto* resolved = router.Resolve("test-model");
  REQUIRE(resolved != nullptr);
  // Backend has no model loaded → MoE fields must be false/0.
  REQUIRE_FALSE(resolved->is_moe);
  REQUIRE(resolved->n_experts == 0);
  REQUIRE(resolved->n_active_experts == 0);
}

// ---------------------------------------------------------------------------
// MetricsRegistry MoE counter
// ---------------------------------------------------------------------------

TEST_CASE("RecordMoERequest increments moe_requests counter", "[moe]") {
  MetricsRegistry reg;
  // Render before recording — counter should not appear or be 0.
  std::string before = reg.RenderPrometheus();
  REQUIRE(before.find("inferflux_moe_requests_total 0") != std::string::npos);

  reg.RecordMoERequest();
  reg.RecordMoERequest();

  std::string after = reg.RenderPrometheus();
  REQUIRE(after.find("inferflux_moe_requests_total 2") != std::string::npos);
}

// ---------------------------------------------------------------------------
// EPDispatch stub — LocalEPDispatch
// ---------------------------------------------------------------------------

TEST_CASE("LocalEPDispatch owns all experts with world_size=1", "[moe]") {
  LocalEPDispatch ep(64);
  REQUIRE(ep.LocalRank().world_size == 1);
  REQUIRE(ep.LocalRank().rank == 0);
  REQUIRE(ep.LocalRank().expert_start == 0);
  REQUIRE(ep.LocalRank().expert_end == 64);
  REQUIRE(ep.OwnsExpert(0));
  REQUIRE(ep.OwnsExpert(63));
  REQUIRE(ep.Name() == "local");
}

TEST_CASE("LocalEPDispatch zero experts is valid (non-MoE model)", "[moe]") {
  LocalEPDispatch ep(0);
  REQUIRE(ep.LocalRank().expert_end == 0);
  REQUIRE_FALSE(ep.OwnsExpert(0));
}
