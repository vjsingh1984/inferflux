#include <catch2/catch_amalgamated.hpp>
#include <type_traits>

#include "runtime/backends/common/backend_types.h"

using namespace inferflux;

TEST_CASE("UnifiedBatchInput defaults", "[backend_types]") {
  UnifiedBatchInput input;

  REQUIRE(input.sequence_id == 0);
  REQUIRE(input.n_past == 0);
  REQUIRE(input.tokens.empty());
  REQUIRE(input.request_logits == true);
  REQUIRE(input.sampling.temperature == 1.0f);
  REQUIRE(input.sampling.top_p == 1.0f);
  REQUIRE(input.request_id == -1);
  REQUIRE(input.sequence_generation == 0);
}

TEST_CASE("UnifiedBatchInput with values", "[backend_types]") {
  UnifiedBatchInput input;
  input.sequence_id = 42;
  input.n_past = 10;
  input.tokens = {1, 2, 3};
  input.request_logits = false;
  input.request_id = 77;
  input.sequence_generation = 9;

  REQUIRE(input.sequence_id == 42);
  REQUIRE(input.n_past == 10);
  REQUIRE(input.tokens.size() == 3);
  REQUIRE(input.tokens[0] == 1);
  REQUIRE(input.tokens[1] == 2);
  REQUIRE(input.tokens[2] == 3);
  REQUIRE(input.request_logits == false);
  REQUIRE(input.request_id == 77);
  REQUIRE(input.sequence_generation == 9);
}

TEST_CASE("UnifiedBatchOutput defaults", "[backend_types]") {
  UnifiedBatchOutput output;

  REQUIRE(output.token == -1);
  REQUIRE(output.piece.empty());
  REQUIRE(output.ok == false);
}

TEST_CASE("UnifiedBatchOutput with values", "[backend_types]") {
  UnifiedBatchOutput output;
  output.token = 42;
  output.piece = "hello";
  output.ok = true;

  REQUIRE(output.token == 42);
  REQUIRE(output.piece == "hello");
  REQUIRE(output.ok == true);
}

TEST_CASE("UnifiedBatchLane enum values", "[backend_types]") {
  REQUIRE(static_cast<int>(UnifiedBatchLane::kAuto) == 0);
  REQUIRE(static_cast<int>(UnifiedBatchLane::kDecode) == 1);
  REQUIRE(static_cast<int>(UnifiedBatchLane::kPrefill) == 2);
}

TEST_CASE("UnifiedBatchHandle is uint64_t", "[backend_types]") {
  UnifiedBatchHandle handle = 12345;
  REQUIRE(handle == 12345);

  REQUIRE(std::is_same<UnifiedBatchHandle, uint64_t>::value);
}

TEST_CASE("PrefillResult defaults", "[backend_types]") {
  PrefillResult result;

  REQUIRE(result.n_past == 0);
  REQUIRE(result.ok == false);
  REQUIRE(result.first_token == -1);
  REQUIRE(result.first_piece.empty());
}

TEST_CASE("PrefillResult with values", "[backend_types]") {
  PrefillResult result;
  result.n_past = 100;
  result.ok = true;
  result.first_token = 42;
  result.first_piece = "hello";

  REQUIRE(result.n_past == 100);
  REQUIRE(result.ok == true);
  REQUIRE(result.first_token == 42);
  REQUIRE(result.first_piece == "hello");
}

TEST_CASE("UnifiedBatch types basic properties", "[backend_types]") {
  UnifiedBatchInput input;
  REQUIRE(input.tokens.empty());

  UnifiedBatchOutput output;
  REQUIRE(output.token == -1);

  UnifiedBatchLane lane = UnifiedBatchLane::kAuto;
  REQUIRE(static_cast<int>(lane) == 0);

  UnifiedBatchHandle handle = 0;
  REQUIRE(handle == 0);
}

TEST_CASE("MakeUnifiedBatchInput copies canonical request metadata",
          "[backend_types]") {
  InferenceRequest request;
  request.id = 42;
  request.sequence_id = 7;
  request.sequence_generation = 9;
  request.client_request_id = "bench-42";
  request.sampling.temperature = 0.3f;

  auto input = MakeUnifiedBatchInput(request, /*n_past=*/11, {3, 4, 5},
                                     /*request_logits=*/false);

  REQUIRE(input.sequence_id == 7);
  REQUIRE(input.sequence_generation == 9);
  REQUIRE(input.n_past == 11);
  REQUIRE(input.tokens == std::vector<int>{3, 4, 5});
  REQUIRE(input.request_logits == false);
  REQUIRE(input.request_id == 42);
  REQUIRE(input.client_request_id == "bench-42");
  REQUIRE(input.sampling.temperature == Catch::Approx(0.3f));
}

TEST_CASE("MakeUnifiedBatchInput supports explicit sequence lease overrides",
          "[backend_types]") {
  InferenceRequest request;
  request.id = 77;
  request.sequence_id = 1;
  request.sequence_generation = 2;
  request.client_request_id = "prefill-77";

  auto input = MakeUnifiedBatchInput(request, /*sequence_id=*/12,
                                     /*sequence_generation=*/34,
                                     /*n_past=*/5, {8, 9},
                                     /*request_logits=*/true);

  REQUIRE(input.sequence_id == 12);
  REQUIRE(input.sequence_generation == 34);
  REQUIRE(input.n_past == 5);
  REQUIRE(input.tokens == std::vector<int>{8, 9});
  REQUIRE(input.request_logits == true);
  REQUIRE(input.request_id == 77);
  REQUIRE(input.client_request_id == "prefill-77");
}
