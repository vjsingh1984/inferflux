#include "runtime/backends/ep_dispatch.h"
#include "runtime/execution/parallel_context.h"
#include <catch2/catch_amalgamated.hpp>
#include <vector>

using namespace inferflux;

TEST_CASE("EPDispatch: Local (single-process) logic", "[ep]") {
  LocalEPDispatch dispatch(8); // 8 experts

  REQUIRE(dispatch.Name() == "local");
  REQUIRE(dispatch.LocalRank().world_size == 1);
  REQUIRE(dispatch.LocalRank().expert_start == 0);
  REQUIRE(dispatch.LocalRank().expert_end == 8);

  // Owns everything
  REQUIRE(dispatch.OwnsExpert(0));
  REQUIRE(dispatch.OwnsExpert(7));
  REQUIRE_FALSE(dispatch.OwnsExpert(8));

  std::vector<float> states = {1.0f, 2.0f};
  auto routed = dispatch.Route(states, {0, 1});
  REQUIRE(routed == states); // No-op in local
}

TEST_CASE("EPDispatch: Distributed logic", "[ep]") {
  // Rank 1 of 2, 8 experts total.
  DistributedEPDispatch dispatch(1, 2, 8);

  REQUIRE(dispatch.Name() == "distributed");
  REQUIRE(dispatch.LocalRank().world_size == 2);
  REQUIRE(dispatch.LocalRank().expert_start == 4);
  REQUIRE(dispatch.LocalRank().expert_end == 8);

  REQUIRE_FALSE(dispatch.OwnsExpert(0));
  REQUIRE(dispatch.OwnsExpert(4));
  REQUIRE(dispatch.OwnsExpert(7));

  SECTION("Route simulation") {
    std::vector<float> states = {0.5f, 0.6f};
    // Should use ParallelContext Comm stub (no-op/copy)
    auto routed = dispatch.Route(states, {4, 5});
    REQUIRE(routed.size() == states.size());
  }
}
