#include "runtime/backends/cpu/llama_backend.h"
#include "runtime/execution/batch_executor.h"
#include "scheduler/single_model_router.h"
#include <catch2/catch_amalgamated.hpp>
#include <memory>
#include <vector>

using namespace inferflux;

// Mock backend to intercept ExecuteUnifiedBatch calls.
class MockUnifiedBackend : public LlamaCPUBackend {
public:
  struct CallInfo {
    std::vector<UnifiedBatchInput> inputs;
  };
  std::vector<CallInfo> calls;

  std::vector<UnifiedBatchOutput>
  ExecuteUnifiedBatch(const std::vector<UnifiedBatchInput> &inputs) override {
    calls.push_back({inputs});
    std::vector<UnifiedBatchOutput> outputs;
    for (const auto &in : inputs) {
      UnifiedBatchOutput out;
      out.ok = true;
      if (in.request_logits) {
        out.token = 42; // Always return token 42
        out.piece = "x";
      }
      outputs.push_back(out);
    }
    return outputs;
  }

  bool IsReady() const override { return true; }
  int TokenCount(const std::string &) const override { return 5; }
};

TEST_CASE("BatchExecutor: Unified Batching & Chunked Prefill",
          "[unified_batch]") {
  SimpleTokenizer tokenizer;
  auto device = std::make_shared<CPUDeviceContext>();
  auto cache = std::make_shared<PagedKVCache>(
      10, 1024, PagedKVCache::EvictionPolicy::kLRU);
  auto prefix_cache =
      std::make_shared<RadixPrefixCache>(cache, [](int) {}, 1024, 12);
  auto router = std::make_shared<SingleModelRouter>();
  auto executor = std::make_unique<BatchExecutor>(&tokenizer, device, cache,
                                                  router, nullptr);

  auto mock_backend = std::make_shared<MockUnifiedBackend>();

  SECTION("Chunked Prefill for a large prompt") {
    InferenceRequest req;
    req.model = "mock";
    req.phase = RequestPhase::kPrefill;
    req.n_past = 0;      // Required for eligibility
    req.sequence_id = 0; // Required for eligibility
    // Large prompt: 1200 tokens (will take 3 chunks of 512, 512, 176)
    req.bpe_prompt_tokens.assign(1200, 1);
    req.max_tokens = 5;
    req.block_table = {1, 2, 3};

    RequestBatch batch;
    batch.requests = {&req};

    auto results = executor->ExecuteBatch(batch, {mock_backend});

    // kMaxPrefillChunkSize is 512.
    // 1200 tokens -> 512 (chunk 0), 512 (chunk 1), 176 (chunk 2 + first token
    // sampled). Then 4 more tokens to reach max_tokens=5. Total steps in while
    // loop: 3 (prefill) + 4 (remaining decode) = 7 steps.
    REQUIRE(mock_backend->calls.size() == 7);

    // First call: 512 tokens, request_logits=false
    REQUIRE(mock_backend->calls[0].inputs[0].tokens.size() == 512);
    REQUIRE_FALSE(mock_backend->calls[0].inputs[0].request_logits);

    // Third call: 176 tokens, request_logits=true (final chunk)
    REQUIRE(mock_backend->calls[2].inputs[0].tokens.size() == 176);
    REQUIRE(mock_backend->calls[2].inputs[0].request_logits);

    REQUIRE(req.phase == RequestPhase::kFinished);
    REQUIRE(req.total_completion_tokens == 5);
  }

  SECTION("Interleaved Prefill and Decode") {
    InferenceRequest req_prefill;
    req_prefill.phase = RequestPhase::kPrefill;
    req_prefill.n_past = 0;
    req_prefill.sequence_id = 1;
    req_prefill.bpe_prompt_tokens = {1, 2, 3};
    req_prefill.max_tokens = 5;
    req_prefill.block_table = {4};

    InferenceRequest req_decode;
    req_decode.phase = RequestPhase::kDecode;
    req_decode.n_past = 10;
    req_decode.first_token = 42;
    req_decode.first_piece = "x";
    req_decode.max_tokens = 5;
    req_decode.sequence_id = 2;
    req_decode.block_table = {5};

    RequestBatch batch;
    batch.requests = {&req_prefill, &req_decode};

    executor->ExecuteBatch(batch, {mock_backend, mock_backend});

    // Step 1: prefill (chunk 0), decode (step 0) -> 2 inputs
    // Step 2: prefill-now-decode (step 1), decode (step 1) -> 2 inputs
    REQUIRE(mock_backend->calls.size() >= 2);
    REQUIRE(mock_backend->calls[0].inputs.size() == 2);
    REQUIRE(mock_backend->calls[1].inputs.size() == 2);
  }

  SECTION("Groups phased unified execution by backend instance") {
    auto mock_backend_a = std::make_shared<MockUnifiedBackend>();
    auto mock_backend_b = std::make_shared<MockUnifiedBackend>();

    InferenceRequest req_a;
    req_a.model = "model-a";
    req_a.phase = RequestPhase::kPrefill;
    req_a.n_past = 0;
    req_a.sequence_id = 3;
    req_a.bpe_prompt_tokens = {1, 2, 3};
    req_a.max_tokens = 2;
    req_a.block_table = {10};

    InferenceRequest req_b;
    req_b.model = "model-b";
    req_b.phase = RequestPhase::kPrefill;
    req_b.n_past = 0;
    req_b.sequence_id = 4;
    req_b.bpe_prompt_tokens = {4, 5, 6};
    req_b.max_tokens = 2;
    req_b.block_table = {11};

    RequestBatch batch;
    batch.requests = {&req_a, &req_b};

    executor->ExecuteBatch(batch, {mock_backend_a, mock_backend_b});

    REQUIRE_FALSE(mock_backend_a->calls.empty());
    REQUIRE_FALSE(mock_backend_b->calls.empty());
    REQUIRE(mock_backend_a->calls.front().inputs.size() == 1);
    REQUIRE(mock_backend_b->calls.front().inputs.size() == 1);
    REQUIRE(req_a.phase == RequestPhase::kFinished);
    REQUIRE(req_b.phase == RequestPhase::kFinished);
  }
}
