#include <catch2/catch_amalgamated.hpp>

#include "runtime/backends/common/batching_utils.h"

using namespace inferflux;

TEST_CASE("BatchAnalyzer detects prefill input", "[batch_analyzer]") {
  UnifiedBatchInput input;
  input.tokens = {1, 2, 3};

  REQUIRE(BatchAnalyzer::IsPrefillLikeInput(input) == true);
  REQUIRE(BatchAnalyzer::IsDecodeLikeInput(input) == false);
}

TEST_CASE("BatchAnalyzer detects decode input", "[batch_analyzer]") {
  UnifiedBatchInput input;
  input.tokens = {42};

  REQUIRE(BatchAnalyzer::IsPrefillLikeInput(input) == false);
  REQUIRE(BatchAnalyzer::IsDecodeLikeInput(input) == true);
}

TEST_CASE("BatchAnalyzer detects empty input", "[batch_analyzer]") {
  UnifiedBatchInput input;
  // No tokens

  // Empty input is neither prefill nor decode
  REQUIRE(BatchAnalyzer::IsPrefillLikeInput(input) == false);
  REQUIRE(BatchAnalyzer::IsDecodeLikeInput(input) == false);
}

TEST_CASE("BatchAnalyzer detects prefill-only batch", "[batch_analyzer]") {
  std::vector<UnifiedBatchInput> inputs(3);
  inputs[0].tokens = {1, 2};      // prefill
  inputs[1].tokens = {3, 4, 5};   // prefill
  inputs[2].tokens = {6, 7, 8};   // prefill

  REQUIRE(BatchAnalyzer::IsPrefillOnlyBatch(inputs) == true);
  REQUIRE(BatchAnalyzer::IsDecodeOnlyBatch(inputs) == false);
  REQUIRE(BatchAnalyzer::HasMixedWorkload(inputs) == false);
}

TEST_CASE("BatchAnalyzer detects decode-only batch", "[batch_analyzer]") {
  std::vector<UnifiedBatchInput> inputs(3);
  inputs[0].tokens = {1};    // decode
  inputs[1].tokens = {2};    // decode
  inputs[2].tokens = {3};    // decode

  REQUIRE(BatchAnalyzer::IsPrefillOnlyBatch(inputs) == false);
  REQUIRE(BatchAnalyzer::IsDecodeOnlyBatch(inputs) == true);
  REQUIRE(BatchAnalyzer::HasMixedWorkload(inputs) == false);
}

TEST_CASE("BatchAnalyzer detects mixed workload", "[batch_analyzer]") {
  std::vector<UnifiedBatchInput> inputs(3);
  inputs[0].tokens = {1, 2};   // prefill
  inputs[1].tokens = {3};      // decode
  inputs[2].tokens = {4, 5};   // prefill

  REQUIRE(BatchAnalyzer::IsPrefillOnlyBatch(inputs) == false);
  REQUIRE(BatchAnalyzer::IsDecodeOnlyBatch(inputs) == false);
  REQUIRE(BatchAnalyzer::HasMixedWorkload(inputs) == true);
}

TEST_CASE("BatchAnalyzer handles empty batch", "[batch_analyzer]") {
  std::vector<UnifiedBatchInput> inputs;

  REQUIRE(BatchAnalyzer::IsPrefillOnlyBatch(inputs) == false);
  REQUIRE(BatchAnalyzer::IsDecodeOnlyBatch(inputs) == false);
  REQUIRE(BatchAnalyzer::HasMixedWorkload(inputs) == false);
}

TEST_CASE("BatchAnalyzer splits batch correctly", "[batch_analyzer]") {
  std::vector<UnifiedBatchInput> inputs(5);
  inputs[0].tokens = {1, 2};    // prefill
  inputs[1].tokens = {3};       // decode
  inputs[2].tokens = {4, 5, 6}; // prefill
  inputs[3].tokens = {7};       // decode
  inputs[4].tokens = {8, 9};    // prefill

  std::vector<size_t> prefill_indices;
  std::vector<size_t> decode_indices;

  BatchAnalyzer::SplitBatchByType(inputs, prefill_indices, decode_indices);

  REQUIRE(prefill_indices.size() == 3);
  REQUIRE(decode_indices.size() == 2);

  REQUIRE(prefill_indices[0] == 0);
  REQUIRE(prefill_indices[1] == 2);
  REQUIRE(prefill_indices[2] == 4);

  REQUIRE(decode_indices[0] == 1);
  REQUIRE(decode_indices[1] == 3);
}

TEST_CASE("BatchAnalyzer counts total tokens", "[batch_analyzer]") {
  std::vector<UnifiedBatchInput> inputs(3);
  inputs[0].tokens = {1, 2, 3};    // 3 tokens
  inputs[1].tokens = {4};          // 1 token
  inputs[2].tokens = {5, 6, 7, 8}; // 4 tokens

  REQUIRE(BatchAnalyzer::CountTotalTokens(inputs) == 8);
}

TEST_CASE("BatchAnalyzer handles empty batch token count", "[batch_analyzer]") {
  std::vector<UnifiedBatchInput> inputs;

  REQUIRE(BatchAnalyzer::CountTotalTokens(inputs) == 0);
}

TEST_CASE("BatchAnalyzer checks token capacity", "[batch_analyzer]") {
  std::vector<UnifiedBatchInput> inputs(2);
  inputs[0].tokens = {1, 2, 3};    // 3 tokens
  inputs[1].tokens = {4, 5};      // 2 tokens

  REQUIRE(BatchAnalyzer::CountTotalTokens(inputs) == 5);
  REQUIRE(BatchAnalyzer::ExceedsTokenCapacity(inputs, 5) == false);
  REQUIRE(BatchAnalyzer::ExceedsTokenCapacity(inputs, 10) == false);
  REQUIRE(BatchAnalyzer::ExceedsTokenCapacity(inputs, 4) == true);
}

TEST_CASE("BatchAnalyzer early exit on mixed workload", "[batch_analyzer]") {
  // Create a large batch where first two are prefill, third is decode
  std::vector<UnifiedBatchInput> inputs(100);
  inputs[0].tokens = {1, 2};    // prefill
  inputs[1].tokens = {3, 4};   // prefill
  inputs[2].tokens = {5};       // decode - should trigger early exit
  // Rest don't matter for early exit test

  // Should return true after finding both prefill and decode
  REQUIRE(BatchAnalyzer::HasMixedWorkload(inputs) == true);
}
